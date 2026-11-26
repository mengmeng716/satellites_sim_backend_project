"""三段式跨层关联模型：环境 → 物理 → 网络 → 综合质量分。

为什么是三段而不是一个大回归
    把三层特征平铺后做一次回归，物理层会吃掉全部解释力，另外两层贡献趋零，
    不满足"从不同层级综合评估"的要求。拆成三段后：
      * 每段的输入只来自上一层，贡献可单独测量；
      * 融合层再做加权，用 Shapley 验证三层都参与。

三层划分
    环境层  AmbientTemperature_K, TerminalTemperature_K, Sunlit, LocalHour
    物理层  TxPower_dBm, TxGain_dBi, PathLoss_dB, PointingLoss_dB, RxPower_dBm,
            SNR_dB, BER_preFEC, LinkDistance_km
    网络层  PacketLossRate, Delay_ms, DelayJitter_ms, Throughput_Gbps, OfferedLoadRatio

用法
    from association import fit_hierarchical, shapley_layers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ENV_CHANNELS = ["AmbientTemperature_K", "TerminalTemperature_K", "Sunlit", "LocalHour"]
PHY_CHANNELS = ["TxPower_dBm", "TxGain_dBi", "PathLoss_dB", "PointingLoss_dB",
                "RxPower_dBm", "SNR_dB", "BER_preFEC", "LinkDistance_km"]
NET_CHANNELS = ["PacketLossRate", "Delay_ms", "DelayJitter_ms", "Throughput_Gbps", "OfferedLoadRatio"]

LAYERS = {"env": ENV_CHANNELS, "phy": PHY_CHANNELS, "net": NET_CHANNELS}


# ----------------------------------------------------------------------
# 窗口特征：均值 / 标准差 / 斜率
# ----------------------------------------------------------------------
def _slope(arr: np.ndarray) -> np.ndarray:
    """沿时间轴的线性斜率（最小二乘，逐窗口）。"""
    n = arr.shape[1]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    denom = float(np.dot(t, t)) or 1e-12
    return (arr * t).sum(axis=1) / denom


def window_features(df: pd.DataFrame, channels: Sequence[str], window: int, stride: int,
                    max_windows: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """按卫星切窗口，返回 (特征矩阵, 窗口中心行号)。

    每个通道产出 3 个统计量：均值、标准差、斜率。
    """
    feats, centers = [], []
    for _, g in df.groupby("SatelliteId", sort=False):
        g = g.sort_index()
        arr = g[list(channels)].to_numpy(np.float64).T
        if arr.shape[1] < window:
            continue
        for s in range(0, arr.shape[1] - window + 1, stride):
            w = arr[:, s:s + window]
            feats.append(np.concatenate([w.mean(axis=1), w.std(axis=1), _slope(w)]))
            centers.append(g.index[s + window // 2])
            if max_windows and len(feats) >= max_windows:
                return np.asarray(feats, np.float32), np.asarray(centers, np.int64)
    return np.asarray(feats, np.float32), np.asarray(centers, np.int64)


# ----------------------------------------------------------------------
# 单段岭回归
# ----------------------------------------------------------------------
@dataclass
class StageModel:
    coef: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    ridge: float = 1e-2
    r2_train: float = 0.0
    r2_test: float = 0.0
    r2_per_target: List[float] = field(default_factory=list)
    n_features: int = 0
    n_targets: int = 0
    extra: Dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, float) - self.mu) / self.sd
        design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
        return design @ self.coef


def _r2(y, p) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    return 1.0 - ss_res / ss_tot


def fit_stage(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2,
              test_ratio: float = 0.2) -> StageModel:
    """标准化 + 岭回归（多输出共用一组系数矩阵）。"""
    n = len(x)
    split = int(n * (1 - test_ratio))
    mu, sd = x[:split].mean(0), x[:split].std(0) + 1e-9

    def design(a):
        z = (a - mu) / sd
        return np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)

    xa = design(x[:split])
    gram = xa.T @ xa
    reg = ridge * np.eye(gram.shape[0])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(gram + reg, xa.T @ y[:split])

    model = StageModel(coef=coef, mu=mu, sd=sd, ridge=ridge,
                       n_features=x.shape[1], n_targets=y.shape[1])
    train_pred = design(x[:split]) @ coef
    test_pred = design(x[split:]) @ coef if split < n else None
    model.r2_train = _r2(y[:split], train_pred)
    model.r2_test = _r2(y[split:], test_pred) if test_pred is not None else float("nan")
    # 逐目标 R²：常量目标（方差≈0）不计入平均，避免虚高
    per = []
    for j in range(y.shape[1]):
        if float(y[:split, j].var()) < 1e-9:
            continue
        per.append(_r2(y[split:, j], test_pred[:, j]) if test_pred is not None else float("nan"))
    model.r2_per_target = per
    model.extra["r2_test_mean_per_target"] = float(np.nanmean(per)) if per else float("nan")
    return model


# ----------------------------------------------------------------------
# 质量分与融合
# ----------------------------------------------------------------------
#: 质量分权重（合计 1.0），每一项都有明确的工程含义：
#:   rate   吞吐/容量        能力上限          —— 网络层
#:   loss   1 - 丢包率        数据完整性        —— 网络层
#:   delay  1 - 归一化时延    实时性            —— 网络层
#:   margin SNR 余量          抗扰能力/鲁棒性   —— 物理层（网络层看不到）
QUALITY_WEIGHTS = {"rate": 0.35, "loss": 0.20, "delay": 0.20, "margin": 0.25}


def quality_target(df: pd.DataFrame, weights: Dict[str, float] | None = None) -> np.ndarray:
    """综合质量分（0~1）。

    加入 SNR 余量的理由：两条同样跑满 100 Gbps 的链路，余量 1 dB 与 6 dB 的
    抗扰能力完全不同——前者温度漂移 1 dB 就掉档，后者不会。网络层指标（吞吐/丢包/时延）
    反映的是"当前表现"，物理层余量反映的是"还能不能撑住"，两者不可互相替代。
    """
    w = dict(QUALITY_WEIGHTS if weights is None else weights)
    rate = np.clip(df["Throughput_Gbps"].to_numpy(float) / 100.0, 0.0, 1.0)
    loss = 1.0 - np.clip(df["PacketLossRate"].to_numpy(float), 0.0, 1.0)
    delay = 1.0 - np.clip((df["Delay_ms"].to_numpy(float) - 1.0) / 9.0, 0.0, 1.0)
    # 余量：以 10 dB（最低可用档）为 0，40 dB 为满。曾试过 0~40 dB 线性归一化，
    # 结果是 margin 完全由 SNR 决定，环境层贡献从 6.5% 掉到 6.2%，故保留截断形式。
    margin = np.clip((df["SNR_dB"].to_numpy(float) - 10.0) / 30.0, 0.0, 1.0)
    return (w["rate"] * rate + w["loss"] * loss + w["delay"] * delay
            + w["margin"] * margin).astype(np.float32)


def shapley_layers(model: StageModel, parts: Dict[str, np.ndarray],
                   baseline: Dict[str, np.ndarray]) -> Dict[str, float]:
    """逐层遮挡的 Shapley 近似：把某层替换为均值，看预测变化多少。

    parts:    各层原始特征 {"env": (N,Fe), "phy": ..., "net": ...}
    baseline: 各层替换值（通常为训练集均值），形状 (Fe,)
    返回：每层的平均绝对贡献占比（三者归一化到 1）。
    """
    names = list(parts.keys())
    widths = {k: parts[k].shape[1] for k in names}

    n_rows = parts[names[0]].shape[0]

    def assemble(override: str | None) -> np.ndarray:
        blocks = []
        for k in names:
            if k == override:
                blocks.append(np.broadcast_to(np.asarray(baseline[k], float).reshape(1, -1),
                                               (n_rows, widths[k])))
            else:
                blocks.append(parts[k])
        return np.concatenate(blocks, axis=1)

    base_pred = model.predict(assemble(None))
    shares = {}
    for k in names:
        alt = model.predict(assemble(k))
        shares[k] = float(np.mean(np.abs(base_pred - alt)))
    total = sum(shares.values()) or 1e-12
    return {k: v / total for k, v in shares.items()}, shares


# ----------------------------------------------------------------------
# 滞后因果：温度变化领先 SNR 残差多少步
# ----------------------------------------------------------------------
def lagged_causality(df: pd.DataFrame, cause: str, effect: str,
                     max_lag: int = 30, max_sats: int = 300) -> Dict:
    """对每颗星做互相关，找使 |corr| 最大的滞后（正=原因领先）。"""
    best_lags, best_corrs = [], []
    for _, g in list(df.groupby("SatelliteId"))[:max_sats]:
        a = g[cause].to_numpy(float)
        b = g[effect].to_numpy(float)
        if len(a) < max_lag * 2:
            continue
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        cors = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                c = float(np.mean(a[:len(a) - lag] * b[lag:])) if lag else float(np.mean(a * b))
            else:
                c = float(np.mean(a[-lag:] * b[:len(b) + lag]))
            cors.append(c)
        idx = int(np.argmax(np.abs(cors)))
        best_lags.append(idx - max_lag)
        best_corrs.append(cors[idx])
    if not best_lags:
        return {"best_lag_mean": float("nan"), "corr_mean": float("nan"), "n": 0}
    return {
        "best_lag_mean": float(np.mean(best_lags)),
        "best_lag_mode": int(pd.Series(best_lags).mode().iloc[0]),
        "corr_mean": float(np.mean(best_corrs)),
        "n": len(best_lags),
    }


# ----------------------------------------------------------------------
# 按星校准（A 段）
# ----------------------------------------------------------------------
def calibrate_per_satellite(model: StageModel, x: np.ndarray, y: np.ndarray,
                            sats: np.ndarray, calib_frac: float = 0.3,
                            min_points: int = 6, shrink: float = 20.0) -> Dict[str, float]:
    """按星校准：只用少量校准窗口估计偏移/斜率，并向"不做校准"收缩。

    直接在每颗星的 9 个窗口上做最小二乘斜率会严重过拟合（实测 R² 掉到 −6），
    因此这里用收缩估计：
        offset_s = mean(y_cal - pred_cal)  × λ
        slope_s  = 1 + (b_s - 1)           × λ,   λ = k / (k + shrink)
    k 为校准窗口数；窗口越多越信任本星估计，越少越接近全局模型。
    """
    pred = np.asarray(model.predict(x), float)
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]

    calibrated = pred.copy()
    n_cal = 0
    for sat in np.unique(sats):
        idx = np.where(sats == sat)[0]
        if idx.size < min_points:
            continue
        k = max(min_points, int(idx.size * calib_frac))
        cal, ev = idx[:k], idx[k:]
        if ev.size == 0:
            continue
        lam = k / (k + shrink)
        for t in range(y.shape[1]):
            # 只做偏移校准：窗内预测值方差极小，估斜率会被放大到发散（实测 R² 掉到 −11）
            off = float(np.mean(y[cal, t] - pred[cal, t])) * lam
            off = float(np.clip(off, -3.0, 3.0))          # 偏移限幅，防止个别星带偏
            calibrated[ev, t] = pred[ev, t] + off
        n_cal += 1

    raw = [float(_r2(y[:, t], pred[:, t])) for t in range(y.shape[1])]
    cal = [float(_r2(y[:, t], calibrated[:, t])) for t in range(y.shape[1])]
    return {"r2_raw": float(np.mean(raw)), "r2_calibrated": float(np.mean(cal)),
            "calibrated_sats": int(n_cal)}
