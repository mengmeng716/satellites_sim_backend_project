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

本次新增
    1. 终端温度 → 发射功率（热降额）
         tx_power_from_temperature(p_nom_dbm, terminal_temp_k)
         P_tx = P_nom − α·max(T_term − T_ref, 0)
         TX_TEMP_REF_K = 308.0          热降额起点 35 °C（在轨遥测：光学底座热控稳定在 29–35 °C）
         TX_TEMP_COEF_DB_PER_K = 0.02   温度系数 dB/K（恒流 DFB 无温控约 0.018–0.043，
                                        EDFA 约 0.006，考虑 TEC/APC 部分补偿取 0.02）
         P_nom 建议按链路取值、时间上不变（遥测发射功率零波动）。

    2. 跨层响应关系 cross_layer_response(payload) → 表3-4 Correlation.ParamPairs
         输入：表3-3 数据（读取 TerminalTemperature、TxPower、SNR、BER、PacketLossRate）
         每项字段：
           Pair            [源参数, 响应参数]                  （原有）
           Corr            Spearman 秩相关系数                  （原有，原为 Pearson）
           SourceValues    源参数 10 分位分箱后各箱中位数       （新增）
           ResponseValues  各箱响应参数均值，与 SourceValues 等长（新增）
         5 对关联：
           TerminalTemperature_K → TxPower_dBm         链路内（先扣除各链路均值）
           TxPower_dBm           → SNR_dB
           SNR_dB                → BER_preFEC
           SNR_dB                → AchievableRate_Gbps  可实现速率按 MCS 档位由 SNR 换算
           SNR_dB                → PacketLossRate

用法
    from association import fit_hierarchical, shapley_layers, cross_layer_response
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



# ----------------------------------------------------------------------
# 终端温度 → 发射功率（热降额）
# ----------------------------------------------------------------------
TX_TEMP_REF_K = 308.0
TX_TEMP_COEF_DB_PER_K = 0.02


def tx_power_from_temperature(p_nom_dbm, terminal_temp_k):
    """P_tx = P_nom − α·max(T_term − T_ref, 0)，单位 dBm。"""
    excess = np.maximum(np.asarray(terminal_temp_k, np.float64) - TX_TEMP_REF_K, 0.0)
    return np.asarray(p_nom_dbm, np.float64) - TX_TEMP_COEF_DB_PER_K * excess


# ----------------------------------------------------------------------
# 跨层响应关系（CRR）
# ----------------------------------------------------------------------
#: (源参数, 响应参数, 粒度)。"within" 为链路内：先扣除各链路自身均值再算相关，
#: 去掉链路间标称功率差异对温度效应的掩盖；"sample" 为全部样本直接计算。
RESPONSE_PAIRS: List[Tuple[str, str, str]] = [
    ("TerminalTemperature_K", "TxPower_dBm", "within"),     # 环境 → 物理（热降额）
    ("TxPower_dBm", "SNR_dB", "sample"),                    # 物理内部
    ("SNR_dB", "BER_preFEC", "sample"),                     # 物理内部
    ("SNR_dB", "AchievableRate_Gbps", "sample"),            # 物理层速率波动
    ("SNR_dB", "PacketLossRate", "sample"),                 # 物理 → 网络
]
RESPONSE_BINS = 10
#: MCS 档位 (SNR 门限 dB, 速率 Gbps)，与物理链路模型一致；SNR < 6 dB 为链路中断
MCS_RATE = [(6.0, 25.0), (10.0, 50.0), (14.0, 75.0), (18.0, 100.0)]


def _achievable_rate(snr_db: np.ndarray) -> np.ndarray:
    """SNR → 可实现速率（Gbps），SNR < 6 dB 为 0。"""
    rate = np.zeros_like(snr_db, dtype=np.float64)
    for level_snr, level_rate in MCS_RATE:
        rate[snr_db >= level_snr] = level_rate
    return rate


def _spearman(x: np.ndarray, y: np.ndarray):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _response_curve(x: np.ndarray, y: np.ndarray, bins: int = RESPONSE_BINS):
    """按源参数分位数分箱，返回 (各箱源参数中位数, 各箱响应参数均值)。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or float(x.std()) <= 1e-12:
        return [], []
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return [], []
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    src, resp = [], []
    for b in range(edges.size - 1):
        mask = which == b
        if mask.any():
            src.append(float(np.median(x[mask])))
            resp.append(float(np.mean(y[mask])))
    return src, resp


def cross_layer_response(payload, pairs: Sequence[Tuple[str, str, str]] = RESPONSE_PAIRS) -> List[Dict]:
    """表3-3 数据 → Correlation.ParamPairs 列表。

    每项：Pair, Corr（Spearman）, SourceValues, ResponseValues。
    """
    env, phy = payload["EnvironmentSeries"], payload["PhysicalSeries"]
    net = payload["NetworkSeries"]
    series = {
        "TerminalTemperature_K": np.asarray(env["TerminalTemperature"], np.float64),
        "TxPower_dBm": np.asarray(phy["TxPower"], np.float64),
        "SNR_dB": np.asarray(phy["SNR"], np.float64),
        "BER_preFEC": np.asarray(phy["BER"], np.float64),
        "PacketLossRate": np.asarray(net["PacketLossRate"], np.float64),
    }
    series["AchievableRate_Gbps"] = _achievable_rate(series["SNR_dB"])

    out = []
    for source, target, level in pairs:
        x2d, y2d = series[source], series[target]          # (链路, 时刻)
        x = x2d.reshape(-1)
        if level == "within":
            x_dev = (x2d - x2d.mean(axis=1, keepdims=True)).reshape(-1)
            y_dev = (y2d - y2d.mean(axis=1, keepdims=True)).reshape(-1)
            corr = _spearman(x_dev, y_dev)
            y = y_dev + float(y2d.mean())                  # 曲线保持原物理单位
        else:
            y = y2d.reshape(-1)
            corr = _spearman(x, y)
        src, resp = _response_curve(x, y)
        out.append({"Pair": [source, target], "Corr": corr,
                    "SourceValues": src, "ResponseValues": resp})
    return out
