"""链路物理模型：给定 4 个自由量，推导其余 9 个观测量。

生成模型只负责产生自由量，观测量一律由本模块按公式计算，
因此生成数据天然满足链路预算闭合（闭合率 100%）。

自由量（4）  LinkDistance_km, PointingLoss_dB, TxPower_dBm, OfferedLoadRatio
观测量（9）  TxGain_dBi, PathLoss_dB, RxPower_dBm, SNR_dB, BER_preFEC,
             PacketLossRate, Delay_ms, DelayJitter_ms, Throughput_Gbps
"""

from __future__ import annotations

import numpy as np

# ---------------- 物理常量（100 Gbps 配置）----------------
LAMBDA_NM = 1550.0
APERTURE_M = 0.10
OPTICS_EFF = 0.80
TX_POWER_NOMINAL_DBM = 30.0
NOISE_FLOOR_DBM = -40.6          # -35 dBm 灵敏度 - 15.6 dB 所需 SNR + 10 dB（带宽×10）
CAPACITY_GBPS = 100.0
C_KM_S = 299792.458
FRAME_BITS = 12000
FEC_GAIN = 1e3

MIN_LINK_KM = 200.0
MAX_SLANT_KM = 2000.0

MCS = [(0.0, 10.0, 1), (6.0, 25.0, 1), (10.0, 50.0, 2), (14.0, 75.0, 3), (18.0, 100.0, 4)]

FREE_CHANNELS = ["LinkDistance_km", "PointingLoss_dB", "TxPower_dBm", "OfferedLoadRatio"]
DERIVED_CHANNELS = ["TxGain_dBi", "PathLoss_dB", "RxPower_dBm", "SNR_dB", "BER_preFEC",
                    "PacketLossRate", "Delay_ms", "DelayJitter_ms", "Throughput_Gbps"]
ALL_CHANNELS = FREE_CHANNELS + DERIVED_CHANNELS


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-30))


def derive(free: dict) -> dict:
    """由自由量推导全部观测量，返回与输入同形状的数组字典。"""
    d = np.clip(np.abs(np.asarray(free["LinkDistance_km"], float)), MIN_LINK_KM, MAX_SLANT_KM)
    point = np.clip(np.asarray(free["PointingLoss_dB"], float), 0.05, 25.0)
    tx = np.clip(np.asarray(free["TxPower_dBm"], float), 5.0, 40.0)
    load = np.clip(np.asarray(free["OfferedLoadRatio"], float), 0.05, 2.0)

    lam = LAMBDA_NM * 1e-9
    gain_value = 10.0 * np.log10((np.pi * APERTURE_M / lam) ** 2 * OPTICS_EFF)
    gain = np.full_like(d, gain_value, dtype=float)
    fspl = 20.0 * np.log10(4.0 * np.pi * (d * 1e3) / lam)
    rx = tx + 2.0 * gain - fspl - point

    snr_raw = rx - NOISE_FLOOR_DBM
    snr = np.maximum(snr_raw, 0.0)
    snr_lin = 10 ** (snr / 10.0)

    state = np.where(snr >= 10.0, 2, np.where(snr >= 6.0, 1, 0))
    down = state == 0

    rate = np.zeros_like(snr)
    bits = np.zeros(snr.shape, dtype=int)
    for lv_snr, lv_rate, lv_bits in MCS:
        mask = (snr >= lv_snr) & (~down)
        rate[mask], bits[mask] = lv_rate, lv_bits

    ber = 0.5 * np.vectorize(_erfc)(np.minimum(np.sqrt(snr_lin / (2.0 ** bits)), 26.0))
    ber = np.clip(np.maximum(ber, 1e-15), 1e-15, 0.5)
    ber = np.where(down, 0.5, ber)
    ber_post = np.where(down, 0.5, np.clip(ber / FEC_GAIN, 1e-15, 0.5))

    loss = np.clip(1.0 - (1.0 - ber_post) ** FRAME_BITS, 0.0, 1.0)
    loss = np.where(down, 1.0, loss)

    prop = d / C_KM_S * 1e3
    rho = np.minimum(load, 0.95)
    queue = 0.06 * rho / (1.0 - rho)
    jitter = 0.00875 * rho / (1.0 - rho)

    throughput = np.minimum(load * CAPACITY_GBPS, rate) * (1.0 - loss)

    return {
        "TxGain_dBi": gain,
        "PathLoss_dB": fspl,
        "RxPower_dBm": rx,
        "SNR_dB": snr,
        "BER_preFEC": ber,
        "PacketLossRate": loss,
        "Delay_ms": prop + queue + 0.30,
        "DelayJitter_ms": jitter,
        "Throughput_Gbps": throughput,
        "_rate": rate,
        "_state": state,
    }


def _erfc(x: float) -> float:
    import math
    return math.erfc(float(x))


def stack(free: dict, shape) -> np.ndarray:
    """把自由量按 ALL_CHANNELS 顺序拼成 (..., C) 数组（仅供检查）。"""
    derived = derive(free)
    return np.stack([free[n] if n in free else derived[n] for n in ALL_CHANNELS], axis=-1)
