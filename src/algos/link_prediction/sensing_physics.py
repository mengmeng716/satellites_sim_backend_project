"""星间激光链路物理层模型与多层参数模拟器。

两个用途
--------
1. **正向物理计算**：给定链路几何、环境与终端状态，计算自由空间损耗、指向损耗、
   接收光功率、SNR、误码率、可实现速率等物理层量。引擎每个采样周期调用
   :func:`evaluate_optical_link` 刷新链路物理状态。
2. **多层参数序列生成**：:func:`simulate_parameter_series` 产生环境层/物理层/网络层
   的时间序列，作为**生成式神经网络**、跨层关联模型与多目标预测模型的训练/验证数据源。
   该函数是“物理仿真”（source_kind=physics_simulated）而不是随机数表：所有量都由
   同一组物理关系与相关噪声生成，跨层依赖是显式建模的。

单位约定（与 :mod:`src.sensing.metrics` 一致）
------------------------------------------------
距离 km；功率 dBm；损耗 dB；SNR dB；BER 无量纲；速率 Gbps；时延 ms；
温度 K；振动 g；指向误差 urad；波长 nm；带宽 Hz；负载比例 0~1。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from metrics import CROSS_LAYER_DATASET_PARAMS

# ---------------------------------------------------------------------------
# 终端与信道常量（可作为详细设计中的链路预算基线）
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_M_S = 299_792_458.0

DEFAULT_WAVELENGTH_NM = 1563.0                  # 真实遥测发射波长（ID_3：1563 nm 为主，偶用 1550）
DEFAULT_MODULATION_BANDWIDTH_HZ = 25.0e9        # 25 GHz 调制带宽
DEFAULT_TX_POWER_DBM = 23.0                     # 约 200 mW
DEFAULT_TX_APERTURE_M = 0.15
DEFAULT_RX_APERTURE_M = 0.30
DEFAULT_OPTICS_EFFICIENCY = 0.6                 # 发射/接收光学效率
DEFAULT_BEAM_DIVERGENCE_URAD = 10.0             # 衍射极限约 2.44·λ/D ≈ 12.6 urad
DEFAULT_POINTING_SIGMA_URAD = 3.0               # PAT 稳态跟踪残差
DEFAULT_ATMOSPHERIC_LOSS_DB = 0.0               # 星间链路不经过大气
DEFAULT_MISC_LOSS_DB = 3.0                      # 滤波器、耦合、分光等固定损耗
DEFAULT_RECEIVER_NOISE_DBM = -28.0              # 按 2600 km 最远链路设计，链路余量约 7 dB
DEFAULT_MAX_RATE_GBPS = 100.0                   # 单链路额定速率（技术要求 ≥100Gbps）
DEFAULT_FEC_THRESHOLD = 1.0e-2                  # 前向纠错可纠正的原始误码率上限
DEFAULT_SNR_THRESHOLD_DB = 6.0                  # 建链所需最低 SNR（含编码体制余量）
DEFAULT_BER_THRESHOLD = 1.0e-9                  # 可用链路的最差后纠错误码率


@dataclass(frozen=True)
class LinkBudget:
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    modulation_bandwidth_hz: float = DEFAULT_MODULATION_BANDWIDTH_HZ
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM
    tx_aperture_m: float = DEFAULT_TX_APERTURE_M
    rx_aperture_m: float = DEFAULT_RX_APERTURE_M
    optics_efficiency: float = DEFAULT_OPTICS_EFFICIENCY
    beam_divergence_urad: float = DEFAULT_BEAM_DIVERGENCE_URAD
    misc_loss_db: float = DEFAULT_MISC_LOSS_DB
    receiver_noise_dbm: float = DEFAULT_RECEIVER_NOISE_DBM
    max_rate_gbps: float = DEFAULT_MAX_RATE_GBPS

    @property
    def tx_gain_db(self) -> float:
        """发射天线增益（dB），G_t = (πD/λ)^2 · η。"""
        lam = self.wavelength_nm * 1e-9
        return _linear_to_db((math.pi * self.tx_aperture_m / lam) ** 2 * self.optics_efficiency)

    @property
    def rx_gain_db(self) -> float:
        lam = self.wavelength_nm * 1e-9
        return _linear_to_db((math.pi * self.rx_aperture_m / lam) ** 2 * self.optics_efficiency)


DEFAULT_BUDGET = LinkBudget()


def _linear_to_db(value: float) -> float:
    value = max(float(value), 1e-30)
    return 10.0 * math.log10(value)


def _db_to_linear(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 10.0)


def free_space_loss_db(distance_km: float, wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> float:
    """自由空间损耗：20·log10(4πL/λ)，距离按 m 计算。"""
    distance_m = max(float(distance_km), 1e-3) * 1000.0
    wavelength_m = wavelength_nm * 1e-9
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength_m)


def pointing_loss_db(pointing_error_urad: float, beam_divergence_urad: float) -> float:
    """高斯光束指向损耗：L = 8.686·(θ_err/θ_div)^2（dB）。

    当指向误差达到波束发散角的 2 倍以上时，认为终端失锁，返回一个很大的损耗。
    """
    error = abs(float(pointing_error_urad))
    divergence = max(float(beam_divergence_urad), 1e-6)
    ratio = error / divergence
    if ratio >= 2.0:
        return 99.0
    return 8.686 * ratio * ratio


def ber_from_snr(snr_db: float, coding_gain_db: float = 0.0) -> float:
    """由 SNR 估计误码率（相干 BPSK 近似 + 编码增益）。

    ``BER = 0.5·erfc(sqrt(2·SNR_lin))``，数值下方截断到 1e-15 以保持可表示性。
    """
    snr_linear = _db_to_linear(float(snr_db) + float(coding_gain_db))
    sqrt_term = math.sqrt(max(2.0 * snr_linear, 0.0))
    ber = 0.5 * math.erfc(sqrt_term)
    return float(min(max(ber, 1e-15), 0.5))


def achievable_rate_gbps(
    snr_db: float,
    bandwidth_hz: float = DEFAULT_MODULATION_BANDWIDTH_HZ,
    max_rate_gbps: float = DEFAULT_MAX_RATE_GBPS,
) -> float:
    """香农容量（Gbps），并按终端额定速率截顶。"""
    snr_linear = _db_to_linear(snr_db)
    capacity_bps = float(bandwidth_hz) * math.log2(1.0 + max(snr_linear, 0.0))
    return float(min(capacity_bps / 1e9, float(max_rate_gbps)))


@dataclass
class OpticalLinkResult:
    wavelength_nm: float
    modulation_bandwidth_hz: float
    tx_power_dbm: float
    path_loss_db: float
    rx_power_dbm: float
    snr_db: float
    ber_pre_fec: float
    ber_post_fec: float
    achievable_rate_gbps: float
    propagation_delay_ms: float
    link_available: bool
    unavailable_reason: str

    def as_dict(self) -> Dict[str, float]:
        return {
            "wavelength_nm": self.wavelength_nm,
            "modulation_bandwidth_hz": self.modulation_bandwidth_hz,
            "tx_power_dbm": self.tx_power_dbm,
            "path_loss_db": self.path_loss_db,
            "rx_power_dbm": self.rx_power_dbm,
            "snr_db": self.snr_db,
            "ber_pre_fec": self.ber_pre_fec,
            "ber_post_fec": self.ber_post_fec,
            "achievable_rate_gbps": self.achievable_rate_gbps,
            "link_delay_ms": self.propagation_delay_ms,
        }


def evaluate_optical_link(
    distance_km: float,
    pointing_error_urad: float = DEFAULT_POINTING_SIGMA_URAD,
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
    pat_state: str = "LOCKED",
    budget: LinkBudget = DEFAULT_BUDGET,
    extra_loss_db: float = 0.0,
) -> OpticalLinkResult:
    """计算单条激光链路的物理层状态。

    :param distance_km: 链路空间距离
    :param pointing_error_urad: PAT 指向误差
    :param tx_power_dbm: 发送光功率
    :param pat_state: PAT 状态，非 LOCKED 时链路不可用
    :param extra_loss_db: 额外损耗（如遮挡、偏振失配）
    """
    distance = max(float(distance_km), 1e-3)
    wavelength_nm = budget.wavelength_nm

    fspl = free_space_loss_db(distance, wavelength_nm)
    point_loss = pointing_loss_db(pointing_error_urad, budget.beam_divergence_urad)
    total_loss = fspl - budget.tx_gain_db - budget.rx_gain_db + point_loss + budget.misc_loss_db + float(extra_loss_db)

    rx_power_dbm = float(tx_power_dbm) - total_loss
    snr_db = rx_power_dbm - budget.receiver_noise_dbm

    ber_pre = ber_from_snr(snr_db)
    # 编码增益约 5 dB（对应纠错能力从 1e-2 降到 1e-9 量级）
    ber_post = ber_from_snr(snr_db, coding_gain_db=5.0)

    rate = achievable_rate_gbps(snr_db, budget.modulation_bandwidth_hz, budget.max_rate_gbps)
    propagation_delay_ms = distance / SPEED_OF_LIGHT_M_S * 1e3 * 1e3 / 1e3
    # 上面等价于 distance_km / 299792.458 * 1000 ms，写清楚便于核对
    propagation_delay_ms = distance * 1000.0 / SPEED_OF_LIGHT_M_S * 1000.0

    available = True
    reason = ""
    if str(pat_state).upper() != "LOCKED":
        available = False
        reason = f"PAT_{str(pat_state).upper()}"
    elif point_loss >= 99.0:
        available = False
        reason = "POINTING_LOST"
    elif snr_db < DEFAULT_SNR_THRESHOLD_DB:
        available = False
        reason = "SNR_BELOW_THRESHOLD"
    elif ber_post > DEFAULT_BER_THRESHOLD:
        available = False
        reason = "BER_ABOVE_THRESHOLD"

    return OpticalLinkResult(
        wavelength_nm=wavelength_nm,
        modulation_bandwidth_hz=budget.modulation_bandwidth_hz,
        tx_power_dbm=float(tx_power_dbm),
        path_loss_db=float(total_loss),
        rx_power_dbm=float(rx_power_dbm),
        snr_db=float(snr_db),
        ber_pre_fec=float(ber_pre),
        ber_post_fec=float(ber_post),
        achievable_rate_gbps=float(rate),
        propagation_delay_ms=float(propagation_delay_ms),
        link_available=bool(available),
        unavailable_reason=reason,
    )


# ---------------------------------------------------------------------------
# 多层参数序列模拟器（生成式模型 / 关联模型 / 预测模型的训练数据源）
# ---------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, n_steps: int, phi: float, sigma: float) -> np.ndarray:
    """生成一阶自回归过程，均值为 0、稳态标准差约等于 sigma。"""
    noise = rng.normal(0.0, sigma * math.sqrt(max(1.0 - phi * phi, 1e-6)), size=n_steps)
    series = np.zeros(n_steps, dtype=np.float64)
    for index in range(1, n_steps):
        series[index] = phi * series[index - 1] + noise[index]
    return series


def _ar1_multi(
    rng: np.random.Generator,
    n_entities: int,
    n_steps: int,
    phi: float,
    sigma: float,
) -> np.ndarray:
    """逐实体独立的一阶自回归噪声，形状 (n_entities, n_steps)。

    必须按实体独立生成：若所有实体共用同一条噪声序列，实体间差异会被抹平，
    跨层相关性检验将失去信息量。
    """
    noise = rng.normal(
        0.0,
        sigma * math.sqrt(max(1.0 - phi * phi, 1e-6)),
        size=(n_entities, n_steps),
    )
    series = np.zeros((n_entities, n_steps), dtype=np.float64)
    for index in range(1, n_steps):
        series[:, index] = phi * series[:, index - 1] + noise[:, index]
    return series


def _tracking_mask(
    rng: np.random.Generator,
    n_entities: int,
    n_steps: int,
    dt_s: float,
    mean_gap_s: float = 3600.0,
    min_duration_s: float = 25.0,
    max_duration_s: float = 180.0,
) -> np.ndarray:
    """生成“跟踪工况”掩码（1 表示该时刻正在捕获/跟踪）。

    依据真实遥测：单次捕获约 25 s（建链进度 88 用时 25.4 s），
    跟踪段持续数十秒到数分钟，两次之间间隔从数分钟到数小时不等。
    """
    mask = np.zeros((n_entities, n_steps), dtype=np.float64)
    for entity in range(n_entities):
        step = int(rng.integers(0, max(1, int(min_duration_s / max(dt_s, 1e-6)))))
        while step < n_steps:
            duration = int(max(1, rng.uniform(min_duration_s, max_duration_s) / max(dt_s, 1e-6)))
            mask[entity, step:min(n_steps, step + duration)] = 1.0
            gap = int(max(1, rng.exponential(mean_gap_s) / max(dt_s, 1e-6)))
            step += duration + gap
    return mask


def _thermal_ramp(
    tracking: np.ndarray,
    dt_s: float,
    heat_rate_c_per_s: float = 0.5,
    cool_rate_c_per_s: float = 0.05,
    ceiling_c: float = 200.0,
) -> np.ndarray:
    """由跟踪工况积分出器件温升（一阶热模型）。

    依据真实遥测：跟踪开启 3 分钟光模块温度 +90 °C，平均约 0.5~1.0 °C/s；
    关闭后缓慢回落。这里用简化的“加热/冷却速率”积分，不做热网络仿真。
    """
    rise = np.zeros_like(tracking)
    for step in range(1, tracking.shape[1]):
        delta = np.where(
            tracking[:, step] > 0,
            heat_rate_c_per_s * dt_s,
            -cool_rate_c_per_s * dt_s,
        )
        rise[:, step] = np.clip(rise[:, step - 1] + delta, 0.0, ceiling_c)
    return rise


def simulate_parameter_series(
    n_entities: int = 32,
    n_steps: int = 240,
    dt_s: float = 1.0,
    seed: int = 20260101,
) -> Dict[str, object]:
    """生成多层级参数时间序列。

    返回契约
    --------
    ``{"params": {参数名: ndarray(n_entities, n_steps) float32}, "conditions": {...}}``

    ``params`` 至少包含 :data:`CROSS_LAYER_DATASET_PARAMS` 中的全部 12 个参数，
    另附 ``node_load_ratio``、``throughput_gbps``、``queue_occupancy_ratio`` 供预测与
    评价模块使用。

    ``conditions`` 包含
    ``layer_index``(n_entities)、``altitude_km``(n_entities)、``latitude``(n_entities)、
    ``local_hour``(n_steps)、``time_index``(n_steps)、``sunlit``(n_entities, n_steps)。

    物理关系链
    ----------
    日照/轨道相位 → 环境温度 → 终端温度 → 振动 → 指向误差 → 指向损耗 → 接收功率
    → SNR → BER → 可实现速率 → 排队/丢包/抖动 → 吞吐 → 节点负载。
    每一环都含有超出确定性映射的随机项，避免模型退化为查表。
    """
    rng = np.random.default_rng(int(seed))

    # ---- 实体条件 ----
    layer_index = rng.integers(0, 2, size=n_entities)
    altitude = np.where(layer_index == 0, 500.0, 1150.0).astype(np.float64)
    latitude = rng.uniform(-60.0, 60.0, size=n_entities)
    # 链路距离按实际 Walker 拓扑的分布抽样：多数为同轨/相邻异轨（约 700~1400 km），
    # 少量为长跨距异轨（可达 2600 km），避免均匀分布把中位距离抬到不真实的 1600 km。
    distance_km = 650.0 + 1900.0 * rng.beta(2.2, 4.2, size=n_entities)

    # ---- 时间轴 ----
    time_index = np.arange(n_steps)
    sim_seconds = time_index * float(dt_s)
    # 轨道周期：500 km 约 5680 s，1150 km 约 6420 s
    period = np.where(altitude < 800.0, 5680.0, 6420.0)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_entities)

    # 当地时：一天 86400 s，加上经度偏移
    start_hour = rng.uniform(0.0, 24.0)
    local_hour = (start_hour + sim_seconds / 3600.0) % 24.0

    # 日照：轨道相位决定（简化：一半时间在阴影）
    orbit_phase = np.mod(sim_seconds[None, :] / period[:, None] + phase[:, None], 1.0)
    sunlit = (orbit_phase < 0.62).astype(np.float64)

    # 昼夜因子（与业务项目背景流量口径一致：6~10 上升、10~22 平台、22~24 下降）
    hour_grid = local_hour[None, :]
    day_factor = np.full_like(hour_grid, 0.08)
    day_factor = np.where((hour_grid >= 6) & (hour_grid < 10), 0.08 + (hour_grid - 6) / 4.0 * 0.92, day_factor)
    day_factor = np.where((hour_grid >= 10) & (hour_grid < 22), 1.0, day_factor)
    day_factor = np.where((hour_grid >= 22) & (hour_grid < 24), 1.0 - (hour_grid - 22) / 2.0 * 0.92, day_factor)

    # ---- 环境层（按真实在轨遥测标定） ----
    # 数据来源：《慢遥》遥测库，1,982 条记录 × 109 参数，覆盖 31.9 小时。
    #   光学头底座：中位 32.6 °C，P5–P95 = 30.4–33.9 °C，小时波动 0.1–2 °C
    #   电机（方位/俯仰）：中位 29.1 °C，P5–P95 = 28.5–32.3 °C
    #   双棱镜电机：中位 42.4 °C
    #   光模块：空闲 26 °C，跟踪时约 0.5–1.0 °C/s 升到 116–240 °C
    # 因此这里建的是**热控平台温度**，不是卫星外表面辐射温度（后者会到 -60 °C）。
    tracking = _tracking_mask(rng, n_entities, n_steps, dt_s)
    tracking_heat_c = _thermal_ramp(tracking, dt_s)

    bench_k = (
        305.55                                     # 32.4 °C
        + 0.7 * np.sin(2.0 * math.pi * orbit_phase)
        + 0.35 * _ar1_multi(rng, n_entities, n_steps, 0.95, 1.0)
    )
    motor_k = (
        302.15                                     # 29.0 °C
        + 0.6 * np.sin(2.0 * math.pi * orbit_phase + 0.8)
        + 0.35 * _ar1_multi(rng, n_entities, n_steps, 0.93, 1.0)
        + 0.02 * tracking_heat_c                   # 跟踪时伺服电机自发热，最多 +4 K
    )
    ambient_k = motor_k
    # 终端温度 = 光机平台温度 + 跟踪自热（热容大，最多 +2.4 K）
    terminal_k = bench_k + 0.012 * tracking_heat_c
    # ---- 周期性调度与机械振动（真实卫星工程的确定性周期现象） ----
    # 1) 波束跳变 / 地面站过境调度：周期 30~60 s，逐实体不同周期与相位。
    #    它是业务负载、队列占用与抖动的**可预测**驱动源，而不是随机噪声。
    hop_period = rng.uniform(30.0, 60.0, size=(n_entities, 1))
    hop_phase = rng.uniform(0.0, 2.0 * math.pi, size=(n_entities, 1))
    hop_signal = np.sin(2.0 * math.pi * sim_seconds[None, :] / hop_period + hop_phase)
    # 2) 反作用轮 / 热致振动：周期 20~40 s，驱动指向误差 → 损耗 → SNR → 速率。
    wheel_period = rng.uniform(20.0, 40.0, size=(n_entities, 1))
    wheel_phase = rng.uniform(0.0, 2.0 * math.pi, size=(n_entities, 1))
    wheel_signal = np.sin(2.0 * math.pi * sim_seconds[None, :] / wheel_period + wheel_phase)

    # 振动：真实数据里跟踪工况下电机与伺服动作显著增强振动
    thermal_shock = np.abs(np.gradient(sunlit, axis=1))
    vibration_g = np.clip(
        0.03
        + 0.10 * tracking
        + 0.04 * thermal_shock
        + 0.05 * wheel_signal
        + 0.030 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.80, 1.0)),
        0.005,
        0.5,
    )
    # 指向误差由振动放大：稳态跟踪残差 3 urad，跟踪/机动时明显恶化
    pointing_urad = np.clip(
        3.0
        + 40.0 * vibration_g
        + 0.4 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.85, 1.0)),
        1.0,
        30.0,
    )

    # ---- 物理层 ----
    tx_power_dbm = (
        DEFAULT_TX_POWER_DBM
        - 0.4 * np.maximum(terminal_k - 300.0, 0.0) / 10.0
        + 0.15 * _ar1_multi(rng, n_entities, n_steps, 0.97, 1.0)
    )
    path_loss_db = (
        np.vectorize(free_space_loss_db)(distance_km)[:, None]
        - DEFAULT_BUDGET.tx_gain_db
        - DEFAULT_BUDGET.rx_gain_db
        + DEFAULT_BUDGET.misc_loss_db
        + np.vectorize(pointing_loss_db)(pointing_urad, DEFAULT_BEAM_DIVERGENCE_URAD)
        + 0.3 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.9, 1.0))
    )
    rx_power_dbm = tx_power_dbm - path_loss_db
    snr_db = rx_power_dbm - DEFAULT_RECEIVER_NOISE_DBM
    ber_pre_fec = np.vectorize(ber_from_snr)(snr_db)
    ber_post_fec = np.vectorize(lambda value: ber_from_snr(value, coding_gain_db=5.0))(snr_db)
    rate_gbps = np.vectorize(achievable_rate_gbps)(snr_db)

    # ---- 网络层 ----
    offered_ratio = np.clip(
        0.30
        + 0.45 * day_factor
        + 0.14 * hop_signal
        + 0.22 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.92, 1.0)),
        0.02,
        1.6,
    )
    queue_ratio = np.clip(
        offered_ratio * 0.62
        + 0.07 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.88, 1.0)),
        0.0,
        0.98,
    )
    # 排队时延：M/M/1 形式，服务速率取可实现速率
    link_delay_ms = (
        np.vectorize(lambda d: d * 1000.0 / SPEED_OF_LIGHT_M_S * 1000.0)(distance_km)[:, None]
        + 0.05 * queue_ratio / np.maximum(1.0 - queue_ratio, 0.02)
    )
    packet_loss = np.clip(
        np.maximum(0.0, queue_ratio - 0.7) * 0.5
        + 0.02 * np.maximum(0.0, ber_pre_fec - DEFAULT_FEC_THRESHOLD),
        0.0,
        1.0,
    )
    # 时延抖动的物理主因是瓶颈排队波动与机械振动，二者都随调度周期变化；
    # 独立测量噪声只占很小一部分，因此这里以结构项为主、噪声项为辅。
    delay_jitter = (
        0.04
        + 1.60 * queue_ratio ** 2
        + 0.60 * vibration_g
        + 0.010 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.75, 1.0))
    )
    throughput = np.clip(rate_gbps * (1.0 - packet_loss) * np.minimum(offered_ratio, 1.0), 0.0, DEFAULT_MAX_RATE_GBPS)
    node_load_ratio = np.clip(
        0.18 * offered_ratio
        + 0.10 * queue_ratio
        + 0.10 * hop_signal
        + 0.02 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.9, 1.0)),
        0.0,
        1.0,
    )

    params = {
        "ambient_temperature_k": ambient_k.astype(np.float32),
        "terminal_temperature_k": terminal_k.astype(np.float32),
        "vibration_rms_g": vibration_g.astype(np.float32),
        "pointing_error_urad": pointing_urad.astype(np.float32),
        "tx_power_dbm": tx_power_dbm.astype(np.float32),
        "path_loss_db": path_loss_db.astype(np.float32),
        "rx_power_dbm": rx_power_dbm.astype(np.float32),
        "snr_db": snr_db.astype(np.float32),
        "ber_pre_fec": ber_pre_fec.astype(np.float32),
        "ber_post_fec": ber_post_fec.astype(np.float32),
        "achievable_rate_gbps": rate_gbps.astype(np.float32),
        "link_delay_ms": link_delay_ms.astype(np.float32),
        "packet_loss_rate": packet_loss.astype(np.float32),
        "delay_jitter_ms": delay_jitter.astype(np.float32),
        "throughput_gbps": throughput.astype(np.float32),
        "queue_occupancy_ratio": queue_ratio.astype(np.float32),
        "node_load_ratio": node_load_ratio.astype(np.float32),
        "pat_locked": np.clip(1.0 - pointing_urad / 30.0, 0.0, 1.0).astype(np.float32),
    }

    missing = [name for name in CROSS_LAYER_DATASET_PARAMS if name not in params]
    if missing:  # pragma: no cover - 编程错误保护
        raise RuntimeError(f"参数序列缺少跨层数据集必需字段: {missing}")

    conditions = {
        "layer_index": layer_index.astype(np.int64),
        "altitude_km": altitude.astype(np.float32),
        "latitude": latitude.astype(np.float32),
        "local_hour": local_hour.astype(np.float32),
        "time_index": time_index.astype(np.int32),
        "sunlit": sunlit.astype(np.float32),
        "distance_km": distance_km.astype(np.float32),
    }
    return {"params": params, "conditions": conditions}


def evaluate_optical_links_bulk(
    distances_km: np.ndarray,
    pointing_errors_urad: np.ndarray,
    pat_locked: np.ndarray,
    tx_powers_dbm: Optional[np.ndarray] = None,
    budget: LinkBudget = DEFAULT_BUDGET,
) -> Dict[str, np.ndarray]:
    """向量化计算一批链路的物理层状态（引擎每秒调用）。

    :param distances_km: (N,) 链路距离
    :param pointing_errors_urad: (N,) 链路两端的等效指向误差
    :param pat_locked: (N,) 1.0 表示两端 PAT 均已锁定
    :param tx_powers_dbm: (N,) 发送光功率，缺省使用预算默认值
    :return: 键为 path_loss_db / rx_power_dbm / snr_db / ber_pre_fec / ber_post_fec /
             achievable_rate_gbps / propagation_delay_ms / link_available
    """
    distances = np.maximum(np.asarray(distances_km, dtype=np.float64), 1e-3)
    errors = np.abs(np.asarray(pointing_errors_urad, dtype=np.float64))
    locked = np.asarray(pat_locked, dtype=np.float64) > 0.5
    tx = (
        np.full_like(distances, budget.tx_power_dbm)
        if tx_powers_dbm is None
        else np.asarray(tx_powers_dbm, dtype=np.float64)
    )

    wavelength_m = budget.wavelength_nm * 1e-9
    fspl = 20.0 * np.log10(4.0 * np.pi * distances * 1000.0 / wavelength_m)

    ratio = errors / max(budget.beam_divergence_urad, 1e-6)
    point_loss = np.where(ratio >= 2.0, 99.0, 8.686 * ratio * ratio)

    total_loss = fspl - budget.tx_gain_db - budget.rx_gain_db + point_loss + budget.misc_loss_db
    rx_power = tx - total_loss
    snr = rx_power - budget.receiver_noise_dbm

    snr_linear = np.power(10.0, snr / 10.0)
    ber_pre = 0.5 * np.vectorize(math.erfc)(np.sqrt(np.maximum(2.0 * snr_linear, 0.0)))
    ber_pre = np.clip(ber_pre, 1e-15, 0.5)
    ber_post = 0.5 * np.vectorize(math.erfc)(np.sqrt(np.maximum(2.0 * snr_linear * (10.0 ** 0.5), 0.0)))
    ber_post = np.clip(ber_post, 1e-15, 0.5)

    capacity_bps = budget.modulation_bandwidth_hz * np.log2(1.0 + np.maximum(snr_linear, 0.0))
    rate = np.minimum(capacity_bps / 1e9, budget.max_rate_gbps)

    propagation_delay_ms = distances * 1000.0 / SPEED_OF_LIGHT_M_S * 1000.0

    available = (
        locked
        & (point_loss < 99.0)
        & (snr >= DEFAULT_SNR_THRESHOLD_DB)
        & (ber_post <= DEFAULT_BER_THRESHOLD)
    )

    return {
        "path_loss_db": total_loss,
        "rx_power_dbm": rx_power,
        "snr_db": snr,
        "ber_pre_fec": ber_pre,
        "ber_post_fec": ber_post,
        "achievable_rate_gbps": rate,
        "propagation_delay_ms": propagation_delay_ms,
        "link_available": available.astype(np.float64),
    }


def simulator_metadata(seed: int, n_entities: int, n_steps: int, dt_s: float) -> Dict[str, object]:
    """返回数据来源说明，供接口与报告标注 provenance。"""
    return {
        "source_kind": "physics_simulated",
        "method_id": "laser-link-budget-v1",
        "seed": int(seed),
        "n_entities": int(n_entities),
        "n_steps": int(n_steps),
        "dt_s": float(dt_s),
        "constants": {
            "wavelength_nm": DEFAULT_WAVELENGTH_NM,
            "tx_power_dbm": DEFAULT_TX_POWER_DBM,
            "beam_divergence_urad": DEFAULT_BEAM_DIVERGENCE_URAD,
            "receiver_noise_dbm": DEFAULT_RECEIVER_NOISE_DBM,
            "max_rate_gbps": DEFAULT_MAX_RATE_GBPS,
            "snr_threshold_db": DEFAULT_SNR_THRESHOLD_DB,
            "ber_threshold": DEFAULT_BER_THRESHOLD,
        },
        "note": "本数据由物理模型正向计算生成，不是在轨实测数据。",
    }
