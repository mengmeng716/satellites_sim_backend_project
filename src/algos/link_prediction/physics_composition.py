"""物理合成头：把隐变量预测还原成 7 个契约目标。

背景
----
数据源 ``src.sensing.physics.simulate_parameter_series`` 里，7 个预测目标中
有 6 个是 **(tx_power, path_loss, queue_ratio, vibration, distance)** 的确定性函数：

============  ============================================================
目标            生成式
============  ============================================================
rx_power_dbm   ``tx − path_loss``
snr（内部量）    ``rx − NOISE_DBM``
ber_pre_fec    ``0.5·erfc(sqrt(2·SNR_lin))``
achievable_rate``B·log2(1+SNR_lin) / 1e9``，按额定速率截顶
link_delay_ms  ``d/c + 0.05·q/(1−q)``
packet_loss    ``max(0, q−0.7)·0.5 + 0.02·max(0, ber−FEC_THRESH)``
delay_jitter   ``0.04 + 1.60·q² + 0.60·vib``
============  ============================================================

只有 ``node_load_ratio`` 依赖 offered_ratio / hop_signal 等未建模中间量，
仍由直接输出头预测。

因此本模块让网络只学 3 个隐变量（``path_loss_db``、``queue_occupancy_ratio``、
``vibration_rms_g``），再用上面的公式**合成 6 个目标**。公式常数全部从
:mod:`src.sensing.physics` 读取，与数据源保持同一版本，避免“训练时一套公式、
生成时另一套公式”。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import math
from typing import Dict, Tuple

import torch

from sensing_physics import (
    DEFAULT_FEC_THRESHOLD,
    DEFAULT_MAX_RATE_GBPS,
    DEFAULT_MODULATION_BANDWIDTH_HZ,
    DEFAULT_RECEIVER_NOISE_DBM,
)

#: 光速（km/ms）
LIGHT_SPEED_KM_PER_MS = 299.792458

#: 目标在 TARGET_NAMES 中的固定索引
IDX_RATE = 0
IDX_JITTER = 1
IDX_NODE_LOAD = 2
IDX_RX = 3
IDX_BER = 4
IDX_LOSS = 5
IDX_DELAY = 6

#: 可由物理公式合成的目标索引（node_load 除外）
COMPOSED_TARGET_INDICES: Tuple[int, ...] = (IDX_RATE, IDX_JITTER, IDX_RX, IDX_BER, IDX_LOSS, IDX_DELAY)

#: 输入特征索引（与 dataset.FEATURE_NAMES 顺序一致）
FEAT_TX_POWER = 3
FEAT_LINK_DELAY = 9
FEAT_QUEUE = 15

_EPS = 1e-9


def _db_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.tensor(10.0, dtype=value.dtype, device=value.device), value / 10.0)


def ber_from_snr_torch(snr_db: torch.Tensor) -> torch.Tensor:
    """``BER = 0.5·erfc(sqrt(2·SNR_lin))``，与 physics.ber_from_snr 同式。"""
    snr_linear = _db_to_linear(snr_db).clamp_min(0.0)
    sqrt_term = torch.sqrt(torch.clamp(2.0 * snr_linear, min=0.0))
    ber = 0.5 * torch.erfc(sqrt_term)
    return ber.clamp(1e-15, 0.5)


def rate_gbps_from_snr_torch(snr_db: torch.Tensor) -> torch.Tensor:
    """香农容量（Gbps），按额定速率截顶，与 physics.achievable_rate_gbps 同式。"""
    snr_linear = _db_to_linear(snr_db).clamp_min(0.0)
    capacity = float(DEFAULT_MODULATION_BANDWIDTH_HZ) * torch.log2(1.0 + snr_linear) / 1e9
    return capacity.clamp(0.0, float(DEFAULT_MAX_RATE_GBPS))


def compose_targets(
    path_loss_db: torch.Tensor,
    queue_ratio: torch.Tensor,
    vibration_g: torch.Tensor,
    tx_power_dbm: torch.Tensor,
    propagation_delay_ms: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """按物理公式合成 6 个目标（物理单位）。

    :param path_loss_db: (B, horizon) 链路总损耗
    :param queue_ratio: (B, horizon) 队列占用比例，裁剪到 [0, 0.98]（与数据源一致）
    :param vibration_g: (B, horizon) 振动加速度均方根
    :param tx_power_dbm: (B, 1) 发送光功率，取窗口末观测值并广播
    :param propagation_delay_ms: (B, 1) 传播时延分量，取窗口末时刻解算值并广播
    """
    q = queue_ratio.clamp(0.0, 0.98)
    vib = vibration_g.clamp_min(0.0)

    rx_power = tx_power_dbm - path_loss_db
    snr = rx_power - float(DEFAULT_RECEIVER_NOISE_DBM)
    ber = ber_from_snr_torch(snr)
    rate = rate_gbps_from_snr_torch(snr)
    delay = propagation_delay_ms + 0.05 * q / (1.0 - q).clamp_min(0.02)
    loss = ((q - 0.7).clamp_min(0.0) * 0.5
            + 0.02 * (ber - float(DEFAULT_FEC_THRESHOLD)).clamp_min(0.0)).clamp(0.0, 1.0)
    jitter = 0.04 + 1.60 * q * q + 0.60 * vib

    return {
        "achievable_rate_gbps": rate,
        "delay_jitter_ms": jitter,
        "rx_power_dbm": rx_power,
        "ber_pre_fec": ber,
        "packet_loss_rate": loss,
        "link_delay_ms": delay,
    }


def propagation_delay_ms(link_delay_ms: torch.Tensor, queue_ratio: torch.Tensor) -> torch.Tensor:
    """由观测的 (总时延, 队列占用) 反解传播时延分量（per-entity 常量）。"""
    q = queue_ratio.clamp(0.0, 0.98)
    queue_part = 0.05 * q / (1.0 - q).clamp_min(0.02)
    return (link_delay_ms - queue_part).clamp_min(0.0)
