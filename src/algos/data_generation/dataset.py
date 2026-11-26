"""生成式模块的数据集构建：把物理仿真序列切成条件 VAE 的监督样本。

数据来源
--------
唯一正式来源是 :func:`src.sensing.physics.simulate_parameter_series`
（``source_kind=physics_simulated``）。该模块缺失时退回 :func:`_fallback_series`，
**仅用于本模块自测**，且在 :func:`load_series` 返回的 provenance 里明确标注
``fallback=True`` / ``delivery_valid=False``，不能作为交付依据。

切分策略
--------
按**实体**切分（60% / 20% / 20%），同一实体的全部时间步只属于一个划分，
避免按时间切分造成的泄漏。标准化统计量只在训练实体上拟合，随模型保存到
``models/generator/scaler.json``。

样本定义（滑窗 window=8）
--------------------------
对实体 ``e`` 的每个时刻 ``t``：

* 目标 ``x``：``t`` 时刻 12 个参数的标准化值；
* 条件 ``c``：
  - 静态：``layer_index``、``altitude_km/1000``、``latitude/90``；
  - 时段：``sin/cos(2π·local_hour/24)``、``sunlit``；
  - 几何：``(link_distance_km-1500)/1000``（缺失填 0）；
  - 前一时刻 12 个参数的标准化值 ``prev``（``t=0`` 冷启动为 0）；
  - ``[max(0, t-window+1), t-1]`` 区间的均值与标准差（``t=0`` 为 0）。

条件向量逐维含义见 :func:`condition_layout`，并写进
``models/generator/feature_contract.json``。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import hashlib
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from contracts import SCHEMA_VERSION, SOURCE_PHYSICS, utc_iso
from metrics import CROSS_LAYER_DATASET_PARAMS, PARAM_SPECS

#: 工程根目录（所有路径都基于它解析，禁止写死绝对路径）
PROJECT_ROOT: Path = Path(__file__).resolve().parent
#: 模型产物目录
MODEL_DIR: Path = PROJECT_ROOT / "models" / "generator"
#: 验证证据目录
EVIDENCE_DIR: Path = PROJECT_ROOT / "evidence" / "generative"

#: 参与生成的 12 个跨层参数（顺序即模型输出维度顺序）
PARAM_NAMES: Tuple[str, ...] = tuple(CROSS_LAYER_DATASET_PARAMS)
#: 参数维度
N_PARAMS: int = len(PARAM_NAMES)
#: 参数名到输出维度下标的映射
PARAM_INDEX: Dict[str, int] = {name: index for index, name in enumerate(PARAM_NAMES)}

#: 需要先取 log10 的参数（跨多个数量级，直接标准化会让 MSE 被大值主导）
LOG10_PARAMS: Tuple[str, ...] = ("ber_pre_fec", "packet_loss_rate", "delay_jitter_ms")
#: log10 变换的平移量：``log10(x + floor)``，同时用于把 0 值映射到有限下界
LOG10_FLOORS: Dict[str, float] = {
    "ber_pre_fec": 1.0e-10,
    "packet_loss_rate": 1.0e-6,
    "delay_jitter_ms": 1.0e-3,
}
#: 每个参数的归一化方式，写进 FeatureContract
PARAM_TRANSFORMS: Dict[str, str] = {
    name: ("log10" if name in LOG10_PARAMS else "identity") for name in PARAM_NAMES
}

#: 默认滑窗长度
DEFAULT_WINDOW: int = 8
#: 标准化值裁剪范围（抑制离群点对 MSE 的影响）
CLIP_SIGMA: float = 6.0
#: 标准差下限（常数参数除以 0 的保护）
STD_FLOOR: float = 1.0e-6
#: 多批次采样时 seed 的步长
BATCH_SEED_STRIDE: int = 7919
#: 链路距离条件归一化参数：``(distance_km - 1500) / 1000``
DISTANCE_OFFSET_KM: float = 1500.0
DISTANCE_SCALE_KM: float = 1000.0

#: 物理合理取值区间（生成阶段的边界投影，投影比例会写进 provenance）
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ambient_temperature_k": (120.0, 400.0),
    "vibration_rms_g": (1.0e-6, 2.0),
    "pointing_error_urad": (1.0e-3, 100.0),
    "tx_power_dbm": (-10.0, 50.0),
    "path_loss_db": (1.0e-3, 400.0),
    "rx_power_dbm": (-300.0, 50.0),
    "snr_db": (-60.0, 100.0),
    "ber_pre_fec": (1.0e-15, 0.5),
    "achievable_rate_gbps": (0.0, 1000.0),
    "link_delay_ms": (1.0e-6, 1.0e4),
    "packet_loss_rate": (0.0, 0.999),
    "delay_jitter_ms": (1.0e-6, 1.0e4),
}

#: 物理仿真模块（可能尚未落地，必须优雅降级）
try:  # pragma: no cover - 取决于父模块的落地状态
    from sensing_physics import simulate_parameter_series as _physics_simulate

    PHYSICS_AVAILABLE: bool = True
    PHYSICS_IMPORT_ERROR: str = ""
except ImportError as _import_error:  # 优雅降级：退回内部 _fallback_series
    _physics_simulate = None  # type: ignore[assignment]
    PHYSICS_AVAILABLE = False
    PHYSICS_IMPORT_ERROR = f"{type(_import_error).__name__}: {_import_error}"


# ---------------------------------------------------------------------------
# 条件向量布局（训练与生成共用，避免两处各写一套下标）
# ---------------------------------------------------------------------------
def condition_layout() -> List[Dict[str, Any]]:
    """返回条件向量逐维定义（``index/name/unit/normalization/source``）。"""
    layout: List[Dict[str, Any]] = [
        {
            "name": "layer_index",
            "unit": "enum",
            "normalization": "原值（0=LEO 约 500 km，1=MEO 约 1150 km）",
            "source": "conditions.layer_index",
        },
        {
            "name": "altitude_km_norm",
            "unit": "-",
            "normalization": "altitude_km / 1000",
            "source": "conditions.altitude_km",
        },
        {
            "name": "latitude_norm",
            "unit": "-",
            "normalization": "latitude / 90",
            "source": "conditions.latitude",
        },
        {
            "name": "local_hour_sin",
            "unit": "-",
            "normalization": "sin(2*pi*local_hour/24)",
            "source": "conditions.local_hour",
        },
        {
            "name": "local_hour_cos",
            "unit": "-",
            "normalization": "cos(2*pi*local_hour/24)",
            "source": "conditions.local_hour",
        },
        {
            "name": "sunlit",
            "unit": "bool",
            "normalization": "0/1（0=阴影区，1=日照区）",
            "source": "conditions.sunlit",
        },
        {
            "name": "link_distance_norm",
            "unit": "-",
            "normalization": "(link_distance_km - 1500) / 1000；缺失填 0",
            "source": "conditions.distance_km",
        },
    ]
    for name in PARAM_NAMES:
        layout.append(
            {
                "name": f"prev__{name}__z",
                "unit": PARAM_SPECS[name].unit,
                "normalization": "上一时刻标准化值（冷启动 t=0 填 0）",
                "source": f"params.{name}[t-1]",
            }
        )
    for name in PARAM_NAMES:
        layout.append(
            {
                "name": f"hist_mean__{name}__z",
                "unit": PARAM_SPECS[name].unit,
                "normalization": "[t-window+1, t-1] 区间标准化均值（无历史时填 0）",
                "source": f"params.{name} 历史窗口",
            }
        )
    for name in PARAM_NAMES:
        layout.append(
            {
                "name": f"hist_std__{name}__z",
                "unit": "-",
                "normalization": "[t-window+1, t-1] 区间标准化标准差（无历史时填 0）",
                "source": f"params.{name} 历史窗口",
            }
        )
    for position, item in enumerate(layout):
        item["index"] = position
    return layout


#: 缓存的条件向量布局
CONDITION_LAYOUT: List[Dict[str, Any]] = condition_layout()
#: 条件向量维度
COND_DIM: int = len(CONDITION_LAYOUT)
#: 条件向量逐维名称
COND_NAMES: Tuple[str, ...] = tuple(item["name"] for item in CONDITION_LAYOUT)
#: 条件向量名称到下标的映射
COND_INDEX: Dict[str, int] = {name: index for index, name in enumerate(COND_NAMES)}


# ---------------------------------------------------------------------------
# 内部兜底物理模拟（仅自测用）
# ---------------------------------------------------------------------------
def _day_factor(local_hour: np.ndarray) -> np.ndarray:
    """昼夜业务因子（与工程 `src/config.py` 的 6~10 上升、10~22 平台、22~24 下降一致）。"""
    hour = np.asarray(local_hour, dtype=np.float64)
    factor = np.full_like(hour, 0.08)
    factor = np.where((hour >= 6.0) & (hour < 10.0), 0.08 + (hour - 6.0) / 4.0 * 0.92, factor)
    factor = np.where((hour >= 10.0) & (hour < 22.0), 1.0, factor)
    factor = np.where((hour >= 22.0) & (hour < 24.0), 1.0 - (hour - 22.0) / 2.0 * 0.92, factor)
    return factor


def _ar1_multi(
    rng: np.random.Generator,
    n_entities: int,
    n_steps: int,
    phi: float,
    sigma: float,
) -> np.ndarray:
    """逐实体独立的一阶自回归过程，稳态标准差约等于 ``sigma``。"""
    phi = float(phi)
    sigma = float(sigma)
    innovation = sigma * math.sqrt(max(1.0 - phi * phi, 1.0e-6))
    noise = rng.normal(0.0, innovation, size=(int(n_entities), int(n_steps)))
    series = np.zeros_like(noise)
    for index in range(1, int(n_steps)):
        series[:, index] = phi * series[:, index - 1] + noise[:, index]
    return series


def _fallback_series(
    n_entities: int = 32,
    n_steps: int = 180,
    dt_s: float = 1.0,
    seed: int = 20260101,
) -> Dict[str, Any]:
    """内部兜底模拟：与激光链路预算同口径的确定性 numpy 实现。

    仅在 ``src.sensing.physics`` 不可用时使用，**只用于本模块自测**。
    物理链条：轨道相位/日照 → 环境温度 → 振动 → 指向误差 → 指向损耗
    → 接收功率 → SNR → BER → 可实现速率 → 排队/丢包/抖动，各环带相关随机项。

    返回契约与 :func:`src.sensing.physics.simulate_parameter_series` 完全一致；
    额外附带 ``distance_km``（n_entities,）作为链路几何条件。
    """
    n = int(n_entities)
    steps = int(n_steps)
    dt = float(dt_s)
    if n <= 0 or steps <= 0:
        raise ValueError("n_entities 与 n_steps 必须为正整数")
    rng = np.random.default_rng(int(seed))
    sim_seconds = np.arange(steps, dtype=np.float64) * dt

    wavelength_m = 1550.0e-9
    tx_gain_db = 10.0 * math.log10((math.pi * 0.15 / wavelength_m) ** 2 * 0.6)
    rx_gain_db = 10.0 * math.log10((math.pi * 0.30 / wavelength_m) ** 2 * 0.6)
    divergence_urad = 10.0
    receiver_noise_dbm = -27.0
    bandwidth_hz = 25.0e9
    max_rate_gbps = 100.0

    layer_index = (np.arange(n) % 2).astype(np.int64)
    altitude_km = np.where(layer_index == 0, 500.0, 1150.0) + rng.normal(0.0, 12.0, n)
    latitude = np.clip(rng.uniform(-60.0, 60.0, n), -75.0, 75.0)
    distance_km = 650.0 + 1900.0 * rng.beta(2.2, 4.2, size=n)

    start_hour = float(rng.uniform(0.0, 24.0))
    local_hour = (start_hour + sim_seconds / 3600.0) % 24.0
    day = _day_factor(local_hour)[None, :]

    orbital_period = 2.0 * math.pi * np.sqrt(
        ((6371.0 + altitude_km) * 1e3) ** 3 / 3.986004418e14
    )
    orbit_phase = np.mod(
        rng.uniform(0.0, 1.0, n)[:, None] + sim_seconds[None, :] / orbital_period[:, None], 1.0
    )
    sunlit = (orbit_phase < 0.62).astype(np.float64)
    thermal_shock = (
        np.abs(np.gradient(sunlit, axis=1)) if steps > 1 else np.zeros_like(sunlit)
    )

    vibration = np.clip(
        0.02
        + 0.15 * thermal_shock
        + 0.03 * day
        + 0.05 * np.abs(_ar1_multi(rng, n, steps, 0.80, 1.0)),
        0.0,
        0.45,
    )
    pointing = np.clip(
        2.0 + 45.0 * vibration + 1.2 * np.abs(_ar1_multi(rng, n, steps, 0.85, 1.0)),
        1.5,
        28.0,
    )

    ambient = (
        210.0
        + 90.0 * sunlit
        + 4.0 * np.sin(2.0 * math.pi * orbit_phase)
        + _ar1_multi(rng, n, steps, 0.95, 0.8)
    )
    terminal = ambient + 12.0 * day + 3.0 * _ar1_multi(rng, n, steps, 0.90, 0.5)

    tx_power = (
        22.7
        - 0.4 * np.maximum(terminal - 300.0, 0.0) / 10.0
        + 0.15 * _ar1_multi(rng, n, steps, 0.97, 1.0)
    )
    fspl = 20.0 * np.log10(
        4.0 * math.pi * np.maximum(distance_km, 1.0)[:, None] * 1e3 / wavelength_m
    )
    ratio = pointing / divergence_urad
    pointing_loss = np.where(ratio >= 2.0, 99.0, 8.686 * ratio * ratio)
    path_loss = (
        fspl
        - tx_gain_db
        - rx_gain_db
        + 3.0
        + pointing_loss
        + 0.3 * np.abs(_ar1_multi(rng, n, steps, 0.90, 1.0))
    )
    rx_power = tx_power - path_loss
    snr = rx_power - receiver_noise_dbm
    snr_linear = np.power(10.0, snr / 10.0)
    ber_pre = np.clip(0.5 * np.vectorize(math.erfc)(np.sqrt(2.0 * snr_linear)), 1.0e-15, 0.5)
    rate = np.minimum(
        bandwidth_hz * np.log2(1.0 + np.maximum(snr_linear, 0.0)) / 1e9, max_rate_gbps
    )

    offered = np.clip(
        0.15 + 0.62 * day + 0.45 * np.abs(_ar1_multi(rng, n, steps, 0.92, 1.0)), 0.02, 1.6
    )
    queue = np.clip(
        0.62 * offered + 0.10 * np.abs(_ar1_multi(rng, n, steps, 0.88, 1.0)), 0.0, 0.98
    )
    queue_delay = 0.05 * queue / np.maximum(1.0 - queue, 0.02)
    link_delay = distance_km[:, None] * 1e3 / 299792458.0 * 1e3 + queue_delay
    packet_loss = np.clip(
        np.maximum(0.0, queue - 0.7) * 0.5 + 0.02 * np.maximum(0.0, ber_pre - 1.0e-2),
        0.0,
        0.9,
    )
    jitter = (
        0.05
        + 0.8 * queue ** 2
        + 0.15 * vibration
        + 0.02 * np.abs(_ar1_multi(rng, n, steps, 0.75, 1.0))
    )

    params = {
        "ambient_temperature_k": ambient.astype(np.float32),
        "vibration_rms_g": vibration.astype(np.float32),
        "pointing_error_urad": pointing.astype(np.float32),
        "tx_power_dbm": tx_power.astype(np.float32),
        "path_loss_db": path_loss.astype(np.float32),
        "rx_power_dbm": rx_power.astype(np.float32),
        "snr_db": snr.astype(np.float32),
        "ber_pre_fec": ber_pre.astype(np.float32),
        "achievable_rate_gbps": rate.astype(np.float32),
        "link_delay_ms": link_delay.astype(np.float32),
        "packet_loss_rate": packet_loss.astype(np.float32),
        "delay_jitter_ms": jitter.astype(np.float32),
        "terminal_temperature_k": terminal.astype(np.float32),
    }
    conditions = {
        "layer_index": layer_index.astype(np.int64),
        "altitude_km": altitude_km.astype(np.float32),
        "latitude": latitude.astype(np.float32),
        "local_hour": local_hour.astype(np.float32),
        "time_index": np.arange(steps, dtype=np.int32),
        "sunlit": sunlit.astype(np.float32),
        "distance_km": distance_km.astype(np.float32),
    }
    return {"params": params, "conditions": conditions}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _validate_series(series: Any, n_entities: int, n_steps: int) -> None:
    """校验数据源返回结构，契约不一致时给出明确错误而不是静默算错。"""
    if not isinstance(series, dict):
        raise TypeError(f"数据源必须返回 dict，收到 {type(series).__name__}")
    params = series.get("params")
    conditions = series.get("conditions")
    if not isinstance(params, dict) or not isinstance(conditions, dict):
        raise ValueError("数据源返回缺少 params / conditions")
    missing = [name for name in PARAM_NAMES if name not in params]
    if missing:
        raise ValueError(f"数据源缺少跨层数据集必需参数: {missing}")
    for name in PARAM_NAMES:
        array = np.asarray(params[name])
        if array.shape != (int(n_entities), int(n_steps)):
            raise ValueError(
                f"参数 {name} 形状应为 ({n_entities}, {n_steps})，实际 {array.shape}"
            )
    required = ("layer_index", "altitude_km", "latitude", "local_hour", "time_index", "sunlit")
    absent = [key for key in required if key not in conditions]
    if absent:
        raise ValueError(f"conditions 缺少必需字段: {absent}")
    for key in ("layer_index", "altitude_km", "latitude"):
        if np.asarray(conditions[key]).shape != (int(n_entities),):
            raise ValueError(f"conditions.{key} 形状应为 ({n_entities},)")
    if np.asarray(conditions["local_hour"]).shape not in ((int(n_steps),), (int(n_entities), int(n_steps))):
        raise ValueError("conditions.local_hour 形状应为 (n_steps,) 或 (n_entities, n_steps)")
    if np.asarray(conditions["sunlit"]).shape != (int(n_entities), int(n_steps)):
        raise ValueError(f"conditions.sunlit 形状应为 ({n_entities}, {n_steps})")
    if np.asarray(conditions["time_index"]).shape != (int(n_steps),):
        raise ValueError(f"conditions.time_index 形状应为 ({n_steps},)")


def physics_fingerprint() -> Dict[str, Any]:
    """记录数据源模块（``src.sensing.physics``）的文件指纹。

    物理模型会随项目迭代变化，训练/校验产物必须能追溯到底用了哪个版本，
    否则“同一份权重”无法复现结论。指纹含绝对路径、sha256 与修改时间。
    """
    module = sys.modules.get("sensing_physics")
    module_path = Path(getattr(module, "__file__", "") or "") if module else Path()
    if not module_path.is_file():
        return {"path": "", "sha256": "", "mtime_utc": ""}
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    modified = datetime.fromtimestamp(module_path.stat().st_mtime, tz=timezone.utc)
    return {
        "path": str(module_path),
        "sha256": digest,
        "mtime_utc": utc_iso(modified),
    }


def load_series(
    n_entities: int = 32,
    n_steps: int = 180,
    dt_s: float = 1.0,
    seed: int = 20260101,
    batches: int = 1,
    allow_fallback: bool = False,
    force_fallback: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """加载多批次参数序列，返回 ``(series, provenance)``。

    :param n_entities: 每批次实体数
    :param n_steps: 时间步数
    :param dt_s: 采样间隔（秒）
    :param seed: 随机种子；批次 ``i`` 使用 ``seed + i*BATCH_SEED_STRIDE``
    :param batches: 批次数（不同批次给出不同的实体集合与当地时）
    :param allow_fallback: physics 缺失时是否允许退回内部兜底模拟
    :param force_fallback: 即使 physics 可用也强制使用兜底模拟（仅自测）
    :raises RuntimeError: physics 缺失且未允许 fallback
    """
    n = int(n_entities)
    steps = int(n_steps)
    count = int(batches)
    if n <= 0 or steps <= 1 or count <= 0:
        raise ValueError("n_entities / n_steps / batches 必须为正，且 n_steps >= 2")

    use_physics = bool(PHYSICS_AVAILABLE) and not bool(force_fallback)
    if not use_physics and not (allow_fallback or force_fallback):
        raise RuntimeError(
            "缺少 sensing_physics.simulate_parameter_series，无法构建正式训练数据。"
            "如需仅做本模块自测，请显式允许 fallback（--allow-fallback）。"
            f"导入错误: {PHYSICS_IMPORT_ERROR or '未启用'}"
        )

    params_buffer: Dict[str, List[np.ndarray]] = {name: [] for name in PARAM_NAMES}
    layer_buffer: List[np.ndarray] = []
    altitude_buffer: List[np.ndarray] = []
    latitude_buffer: List[np.ndarray] = []
    sunlit_buffer: List[np.ndarray] = []
    distance_buffer: List[np.ndarray] = []
    distance_missing = False
    blocks: List[Dict[str, Any]] = []
    seeds: List[int] = []
    offset = 0

    for index in range(count):
        batch_seed = int(seed) + index * BATCH_SEED_STRIDE
        if use_physics:
            batch = _physics_simulate(  # type: ignore[misc]
                n_entities=n, n_steps=steps, dt_s=float(dt_s), seed=batch_seed
            )
        else:
            batch = _fallback_series(n_entities=n, n_steps=steps, dt_s=float(dt_s), seed=batch_seed)
        _validate_series(batch, n, steps)
        conditions = batch["conditions"]
        for name in PARAM_NAMES:
            params_buffer[name].append(np.asarray(batch["params"][name], dtype=np.float32))
        layer_buffer.append(np.asarray(conditions["layer_index"], dtype=np.int64))
        altitude_buffer.append(np.asarray(conditions["altitude_km"], dtype=np.float32))
        latitude_buffer.append(np.asarray(conditions["latitude"], dtype=np.float32))
        sunlit_buffer.append(np.asarray(conditions["sunlit"], dtype=np.float32))
        distance = conditions.get("distance_km")
        if distance is None:
            distance_missing = True
        else:
            distance_buffer.append(np.asarray(distance, dtype=np.float32))
        blocks.append(
            {
                "start": offset,
                "stop": offset + n,
                "local_hour": np.asarray(conditions["local_hour"], dtype=np.float32),
                "time_index": np.asarray(conditions["time_index"], dtype=np.int32),
            }
        )
        seeds.append(batch_seed)
        offset += n

    merged_conditions: Dict[str, Any] = {
        "layer_index": np.concatenate(layer_buffer).astype(np.int64),
        "altitude_km": np.concatenate(altitude_buffer).astype(np.float32),
        "latitude": np.concatenate(latitude_buffer).astype(np.float32),
        "sunlit": np.concatenate(sunlit_buffer, axis=0).astype(np.float32),
        "local_hour": np.asarray(blocks[0]["local_hour"], dtype=np.float32),
        "time_index": np.asarray(blocks[0]["time_index"], dtype=np.int32),
        "distance_km": (
            np.concatenate(distance_buffer).astype(np.float32) if not distance_missing else None
        ),
    }
    series: Dict[str, Any] = {
        "params": {
            name: np.concatenate(params_buffer[name], axis=0) for name in PARAM_NAMES
        },
        "conditions": merged_conditions,
        "time_blocks": blocks,
        "n_entities": offset,
        "n_steps": steps,
        "dt_s": float(dt_s),
    }

    if use_physics:
        provenance: Dict[str, Any] = {
            "source_kind": SOURCE_PHYSICS,
            "module": "sensing_physics.simulate_parameter_series",
            "method_id": "laser-link-budget-v1",
            "fallback": False,
            "fallback_reason": "",
            "delivery_valid": True,
            "seeds": seeds,
            "n_entities": offset,
            "n_steps": steps,
            "dt_s": float(dt_s),
            "batches": count,
            "created_at_utc": utc_iso(),
            "physics_module": physics_fingerprint(),
            "note": "物理模型正向计算生成，非在轨实测数据。",
        }
    else:
        provenance = {
            "source_kind": "fallback_internal_demo",
            "module": "src.generative.dataset._fallback_series",
            "method_id": "internal-fallback-v1",
            "fallback": True,
            "fallback_reason": PHYSICS_IMPORT_ERROR or "调用方显式要求 force_fallback",
            "delivery_valid": False,
            "seeds": seeds,
            "n_entities": offset,
            "n_steps": steps,
            "dt_s": float(dt_s),
            "batches": count,
            "created_at_utc": utc_iso(),
            "physics_module": {"path": "", "sha256": "", "mtime_utc": ""},
            "note": "内部兜底模拟，仅用于生成式模块自测；不能作为交付依据，physics 落地后必须重跑正式训练。",
        }
    series["provenance"] = provenance
    return series, provenance


def subset_series(series: Dict[str, Any], entity_indices: Sequence[int]) -> Dict[str, Any]:
    """按实体下标抽取子集（保持时间轴与批次分块结构）。"""
    index = np.asarray(sorted({int(item) for item in entity_indices}), dtype=np.int64)
    if index.size == 0:
        raise ValueError("entity_indices 不能为空")
    total = int(np.asarray(series["params"][PARAM_NAMES[0]]).shape[0])
    if index.min() < 0 or index.max() >= total:
        raise IndexError(f"实体下标越界: [{index.min()}, {index.max()}] 不在 [0, {total})")
    params = {
        name: np.ascontiguousarray(np.asarray(series["params"][name])[index])
        for name in PARAM_NAMES
    }
    conditions: Dict[str, Any] = {}
    for key in ("layer_index", "altitude_km", "latitude", "sunlit", "distance_km"):
        value = series["conditions"].get(key)
        conditions[key] = (
            None if value is None else np.ascontiguousarray(np.asarray(value)[index])
        )
    conditions["local_hour"] = np.asarray(series["conditions"]["local_hour"])
    conditions["time_index"] = np.asarray(series["conditions"]["time_index"])

    blocks: List[Dict[str, Any]] = []
    cursor = 0
    for block in series.get("time_blocks", []):
        start = int(block["start"])
        stop = int(block["stop"])
        positions = np.where((index >= start) & (index < stop))[0]
        if positions.size == 0:
            continue
        blocks.append(
            {
                "start": cursor,
                "stop": cursor + int(positions.size),
                "local_hour": np.asarray(block["local_hour"]),
                "time_index": np.asarray(block["time_index"]),
            }
        )
        cursor += int(positions.size)
    return {
        "params": params,
        "conditions": conditions,
        "time_blocks": blocks,
        "n_entities": cursor,
        "n_steps": int(np.asarray(series["params"][PARAM_NAMES[0]]).shape[1]),
        "provenance": series.get("provenance", {}),
    }


# ---------------------------------------------------------------------------
# 按实体切分
# ---------------------------------------------------------------------------
def split_by_entity(
    n_entities: int,
    seed: int = 20260101,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, np.ndarray]:
    """按实体切分 train/val/test（60% / 20% / 20%），返回实体下标数组。"""
    total = int(n_entities)
    if total < 3:
        raise ValueError("按实体切分至少需要 3 个实体")
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("切分比例非法")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于 1")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(total)
    n_val = max(1, int(round(val_ratio * total)))
    n_test = max(1, int(round((1.0 - train_ratio - val_ratio) * total)))
    n_train = total - n_val - n_test
    if n_train < 1:
        raise ValueError("训练实体数为 0，请增大实体数量")
    return {
        "train": np.sort(order[:n_train]).astype(np.int64),
        "val": np.sort(order[n_train : n_train + n_val]).astype(np.int64),
        "test": np.sort(order[n_train + n_val :]).astype(np.int64),
    }


def entity_mask(entity_index: np.ndarray, entities: Sequence[int]) -> np.ndarray:
    """返回布尔掩码：``entity_index`` 中的样本是否属于给定实体集合。"""
    index = np.asarray(entity_index, dtype=np.int64)
    selected = np.zeros(int(index.max()) + 1 if index.size else 0, dtype=bool)
    for item in entities:
        position = int(item)
        if 0 <= position < selected.size:
            selected[position] = True
    return selected[index]


# ---------------------------------------------------------------------------
# 标准化与变换
# ---------------------------------------------------------------------------
def _stack_params(params: Dict[str, Any]) -> np.ndarray:
    """把参数字典堆成 ``(n_entities, 12, n_steps)`` 的 float64 立方体。"""
    arrays: List[np.ndarray] = []
    for name in PARAM_NAMES:
        array = np.asarray(params[name], dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"参数 {name} 必须是二维 (n_entities, n_steps)，收到 {array.shape}")
        arrays.append(array)
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"各参数形状不一致: {sorted(shapes)}")
    return np.stack(arrays, axis=1)


def apply_transforms(cube: np.ndarray) -> np.ndarray:
    """对 ``(n_entities, 12, n_steps)`` 立方体的参数维做变换（log10 / identity）。

    ``log10`` 采用 ``log10(x + floor)`` 形式：既能把 0 值映射到有限下界，
    又避免对数在 0 附近发散。参数维固定是第 1 维，不允许依赖广播猜测轴向。
    """
    out = np.array(cube, dtype=np.float64, copy=True)
    if out.ndim != 3 or out.shape[1] != N_PARAMS:
        raise ValueError(f"参数立方体形状应为 (n_entities, {N_PARAMS}, n_steps)，收到 {out.shape}")
    for position, name in enumerate(PARAM_NAMES):
        if PARAM_TRANSFORMS[name] == "log10":
            floor = LOG10_FLOORS[name]
            out[:, position, :] = np.log10(np.maximum(out[:, position, :], 0.0) + floor)
    return out


def fit_scaler(series: Dict[str, Any]) -> Dict[str, Any]:
    """在给定实体上拟合变换空间的均值/标准差。"""
    cube = apply_transforms(_stack_params(series["params"]))
    mean = cube.mean(axis=(0, 2))
    std = cube.std(axis=(0, 2))
    constant = [
        name for position, name in enumerate(PARAM_NAMES) if float(std[position]) < 1.0e-8
    ]
    return {
        "param_names": list(PARAM_NAMES),
        "transforms": dict(PARAM_TRANSFORMS),
        "log10_floors": {name: float(LOG10_FLOORS[name]) for name in LOG10_PARAMS},
        "mean": [float(value) for value in mean],
        "std": [float(max(value, STD_FLOOR)) for value in std],
        "raw_std": [float(value) for value in std],
        "constant_params": constant,
        "clip_sigma": CLIP_SIGMA,
        "std_floor": STD_FLOOR,
        "space": "transform",
        "bounds_raw": {name: list(PARAM_BOUNDS[name]) for name in PARAM_NAMES},
        "n_samples": int(cube.shape[0] * cube.shape[2]),
        "n_entities": int(cube.shape[0]),
        "fitted_at_utc": utc_iso(),
    }


def inverse_transform(values_std: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    """把标准化值还原成物理量（逆变换），形状保持 ``(..., 12)``。

    最后一个维度是参数维；``log10`` 变换按 ``10**v - floor`` 还原。
    """
    values = np.asarray(values_std, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    transforms = scaler.get("transforms", PARAM_TRANSFORMS)
    floors = scaler.get("log10_floors", LOG10_FLOORS)
    raw = values * std + mean
    for index, name in enumerate(PARAM_NAMES):
        if transforms.get(name, "identity") == "log10":
            floor = float(floors.get(name, LOG10_FLOORS.get(name, 0.0)))
            raw[..., index] = np.power(10.0, np.clip(raw[..., index], -30.0, 6.0)) - floor
    return raw


def clip_to_bounds(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """按 :data:`PARAM_BOUNDS` 做边界投影，返回 ``(投影后数组, 逐参数修改计数)``。"""
    out = np.array(values, dtype=np.float64, copy=True)
    counts: Dict[str, int] = {}
    for index, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        column = out[..., index]
        invalid = int(
            np.count_nonzero((column < low) | (column > high) | ~np.isfinite(column))
        )
        cleaned = np.nan_to_num(column, nan=low, posinf=high, neginf=low)
        out[..., index] = np.clip(cleaned, low, high)
        counts[name] = invalid
    return out, counts


# ---------------------------------------------------------------------------
# 滑窗样本构建
# ---------------------------------------------------------------------------
def _hour_matrix(hour: np.ndarray, n_batch: int, n_steps: int) -> np.ndarray:
    """把 local_hour 统一成 ``(n_batch, n_steps)``。"""
    array = np.asarray(hour, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != n_steps:
            raise ValueError(f"local_hour 长度应为 {n_steps}，收到 {array.shape[0]}")
        return np.broadcast_to(array[None, :], (n_batch, n_steps)).copy()
    if array.shape == (n_batch, n_steps):
        return array
    raise ValueError(f"local_hour 形状应为 ({n_steps},) 或 ({n_batch}, {n_steps})")


def _iter_blocks(
    series: Dict[str, Any], n_steps: int
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """迭代 ``(start, stop, hour(n_batch,n_steps), time_index)`` 时间块。"""
    blocks = series.get("time_blocks")
    if blocks:
        for block in blocks:
            start = int(block["start"])
            stop = int(block["stop"])
            yield (
                start,
                stop,
                _hour_matrix(block["local_hour"], stop - start, n_steps),
                np.asarray(block["time_index"], dtype=np.int64),
            )
        return
    total = int(np.asarray(series["params"][PARAM_NAMES[0]]).shape[0])
    conditions = series["conditions"]
    yield (
        0,
        total,
        _hour_matrix(conditions["local_hour"], total, n_steps),
        np.asarray(conditions["time_index"], dtype=np.int64),
    )


def _history_statistics(cube: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个时刻 ``t`` 之前（不含 ``t``）最多 ``window-1`` 个历史点的均值与标准差。

    返回 ``(mean, std)``，形状与 ``cube`` 相同 ``(n_entities, 12, n_steps)``；
    ``t=0`` 无历史，均值和标准差都置 0。
    """
    _, _, n_steps = cube.shape
    mean = np.zeros_like(cube)
    std = np.zeros_like(cube)
    cumulative = np.cumsum(cube, axis=2)
    cumulative_sq = np.cumsum(cube * cube, axis=2)
    for step in range(1, int(n_steps)):
        start = max(0, step - (int(window) - 1))
        stop = step - 1
        total = cumulative[:, :, stop] - (cumulative[:, :, start - 1] if start > 0 else 0.0)
        total_sq = cumulative_sq[:, :, stop] - (
            cumulative_sq[:, :, start - 1] if start > 0 else 0.0
        )
        count = stop - start + 1
        block_mean = total / count
        variance = np.maximum(total_sq / count - block_mean * block_mean, 0.0)
        mean[:, :, step] = block_mean
        std[:, :, step] = np.sqrt(variance)
    return mean, std


def _as_matrix(value: Any, n_entities: int, n_steps: int, name: str) -> np.ndarray:
    """把标量/一维/二维条件量统一成 ``(n_entities, n_steps)``。"""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full((n_entities, n_steps), float(array))
    if array.ndim == 1:
        if array.shape[0] == n_entities:
            return np.repeat(array[:, None], n_steps, axis=1)
        if array.shape[0] == n_steps:
            return np.repeat(array[None, :], n_entities, axis=0)
        raise ValueError(f"{name} 长度 {array.shape[0]} 与实体数 {n_entities} / 步数 {n_steps} 都不匹配")
    if array.shape == (n_entities, n_steps):
        return array
    raise ValueError(f"{name} 形状应为 ({n_entities}, {n_steps})，收到 {array.shape}")


def _as_param_cube(value: Any, n_entities: int, n_steps: int, name: str) -> np.ndarray:
    """把 ``(n_entities, 12)`` 或 ``(n_entities, n_steps, 12)`` 统一成三维。"""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (n_entities, N_PARAMS):
        return np.repeat(array[:, None, :], n_steps, axis=1)
    if array.shape == (n_entities, n_steps, N_PARAMS):
        return array
    raise ValueError(
        f"{name} 形状应为 ({n_entities}, 12) 或 ({n_entities}, {n_steps}, 12)，收到 {array.shape}"
    )


def pack_condition_vector(
    *,
    layer_index: Any,
    altitude_km: Any,
    latitude: Any,
    local_hour: Any,
    sunlit: Any,
    prev_std: Any,
    hist_mean: Any,
    hist_std: Any,
    distance_km: Any = None,
) -> np.ndarray:
    """按 :data:`CONDITION_LAYOUT` 打包条件向量，返回 ``(n_entities, n_steps, cond_dim)``。

    训练与生成共用本函数，保证条件语义与维度顺序完全一致。
    """
    local_hour_matrix = np.asarray(local_hour, dtype=np.float64)
    if local_hour_matrix.ndim == 1:
        n_steps = int(local_hour_matrix.shape[0])
    elif local_hour_matrix.ndim == 2:
        n_steps = int(local_hour_matrix.shape[1])
    else:
        raise ValueError(f"local_hour 维度非法: {local_hour_matrix.shape}")
    layer_array = np.asarray(layer_index, dtype=np.float64)
    if layer_array.ndim != 1:
        raise ValueError("layer_index 必须是一维 (n_entities,)")
    n_entities = int(layer_array.shape[0])

    hour = _as_matrix(local_hour_matrix, n_entities, n_steps, "local_hour")
    conditions = np.zeros((n_entities, n_steps, COND_DIM), dtype=np.float64)
    conditions[:, :, COND_INDEX["layer_index"]] = _as_matrix(
        layer_array, n_entities, n_steps, "layer_index"
    )
    conditions[:, :, COND_INDEX["altitude_km_norm"]] = (
        _as_matrix(altitude_km, n_entities, n_steps, "altitude_km") / 1000.0
    )
    conditions[:, :, COND_INDEX["latitude_norm"]] = (
        _as_matrix(latitude, n_entities, n_steps, "latitude") / 90.0
    )
    conditions[:, :, COND_INDEX["local_hour_sin"]] = np.sin(2.0 * math.pi * hour / 24.0)
    conditions[:, :, COND_INDEX["local_hour_cos"]] = np.cos(2.0 * math.pi * hour / 24.0)
    conditions[:, :, COND_INDEX["sunlit"]] = (
        np.zeros((n_entities, n_steps), dtype=np.float64)
        if sunlit is None
        else _as_matrix(sunlit, n_entities, n_steps, "sunlit")
    )
    conditions[:, :, COND_INDEX["link_distance_norm"]] = (
        np.zeros((n_entities, n_steps), dtype=np.float64)
        if distance_km is None
        else (
            _as_matrix(distance_km, n_entities, n_steps, "distance_km") - DISTANCE_OFFSET_KM
        )
        / DISTANCE_SCALE_KM
    )

    prev = _as_param_cube(prev_std, n_entities, n_steps, "prev_std")
    hist_mean_cube = _as_param_cube(hist_mean, n_entities, n_steps, "hist_mean")
    hist_std_cube = _as_param_cube(hist_std, n_entities, n_steps, "hist_std")
    for position, name in enumerate(PARAM_NAMES):
        conditions[:, :, COND_INDEX[f"prev__{name}__z"]] = prev[:, :, position]
        conditions[:, :, COND_INDEX[f"hist_mean__{name}__z"]] = hist_mean_cube[:, :, position]
        conditions[:, :, COND_INDEX[f"hist_std__{name}__z"]] = hist_std_cube[:, :, position]
    return conditions


# ---------------------------------------------------------------------------
# FeatureContract
# ---------------------------------------------------------------------------
@dataclass
class FeatureContract:
    """生成式数据的特征契约（字段名、单位、归一化方式、窗口长度、对齐规则）。

    该对象直接序列化进 ``models/generator/feature_contract.json``，
    供接口层（``generate_multilayer_data``）声明“模型输入是什么、经过什么变换”。
    """

    contract_id: str
    window: int
    params: List[Dict[str, Any]]
    condition_layout: List[Dict[str, Any]]
    scaler: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    target: str = "12 个跨层参数在 t 时刻的标准化取值"
    entity_alignment: str = "按实体切分 60%/20%/20%，同一实体的全部时间步只属于一个划分"
    time_alignment: str = "滑窗样本：目标为 t 时刻，条件为 t-1 时刻观测与 [t-window+1, t-1] 统计量"
    missing_policy: str = "训练数据不允许缺失；出现非有限值直接报错，不做插值掩盖"
    normalization: str = "先按 transform（log10/identity）变换，再减均值除标准差并裁剪到 ±clip_sigma"
    source_kind: str = SOURCE_PHYSICS

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "target": self.target,
            "window": int(self.window),
            "params": self.params,
            "condition_layout": self.condition_layout,
            "condition_dim": len(self.condition_layout),
            "normalization": self.normalization,
            "entity_alignment": self.entity_alignment,
            "time_alignment": self.time_alignment,
            "missing_policy": self.missing_policy,
            "source_kind": self.source_kind,
            "scaler": self.scaler,
        }


def build_feature_contract(
    scaler: Optional[Dict[str, Any]] = None, window: int = DEFAULT_WINDOW
) -> FeatureContract:
    """构建 FeatureContract；``scaler`` 为空时只填结构不填统计量。"""
    mean = list(scaler["mean"]) if scaler else [0.0] * N_PARAMS
    std = list(scaler["std"]) if scaler else [1.0] * N_PARAMS
    transforms = dict(scaler["transforms"]) if scaler else dict(PARAM_TRANSFORMS)
    params: List[Dict[str, Any]] = []
    for position, name in enumerate(PARAM_NAMES):
        spec = PARAM_SPECS[name]
        params.append(
            {
                "index": position,
                "name": name,
                "unit": spec.unit,
                "layer": spec.layer,
                "critical": bool(spec.critical),
                "source_kind": spec.source_kind,
                "description": spec.description,
                "transform": transforms.get(name, "identity"),
                "transform_floor": (
                    float(LOG10_FLOORS.get(name, 0.0))
                    if transforms.get(name, "identity") == "log10"
                    else 0.0
                ),
                "mean": float(mean[position]),
                "std": float(std[position]),
                "valid_range": list(PARAM_BOUNDS[name]),
            }
        )
    return FeatureContract(
        contract_id="generative-multilayer-cvae-v1",
        window=int(window),
        params=params,
        condition_layout=condition_layout(),
        scaler=dict(scaler) if scaler else {},
    )


def build_training_tensors(
    series: Dict[str, Any],
    window: int = DEFAULT_WINDOW,
    scaler: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """把参数序列切成滑窗样本，返回 ``(X, C, meta)``。

    :param series: :func:`load_series` 返回的序列字典
    :param window: 滑窗长度（≥2）
    :param scaler: 标准化统计量；缺省在**传入的全部实体**上拟合（正式训练请先在训练实体上拟合）
    :return:
        * ``X``：``(n_samples, 12)`` float32 标准化目标
        * ``C``：``(n_samples, cond_dim)`` float32 条件向量
        * ``meta``：样本索引、scaler、FeatureContract、条件布局等
    """
    window = int(window)
    if window < 2:
        raise ValueError("window 至少为 2（条件需要前一时刻观测）")
    cube = _stack_params(series["params"])
    n_entities, n_params, n_steps = cube.shape
    if n_params != N_PARAMS:
        raise ValueError(f"参数维度应为 {N_PARAMS}，实际 {n_params}")
    transformed = apply_transforms(cube)
    if scaler is None:
        scaler = fit_scaler(series)
    if len(scaler.get("mean", [])) != N_PARAMS or len(scaler.get("std", [])) != N_PARAMS:
        raise ValueError("scaler 的 mean/std 维度与参数维度不一致")
    mean = np.asarray(scaler["mean"], dtype=np.float64).reshape(1, N_PARAMS, 1)
    std = np.asarray(scaler["std"], dtype=np.float64).reshape(1, N_PARAMS, 1)
    standardized = np.clip((transformed - mean) / std, -CLIP_SIGMA, CLIP_SIGMA)

    conditions = series["conditions"]
    layer = np.asarray(conditions["layer_index"], dtype=np.float64)
    altitude = np.asarray(conditions["altitude_km"], dtype=np.float64)
    latitude = np.asarray(conditions["latitude"], dtype=np.float64)
    sunlit_all = conditions.get("sunlit")
    sunlit_all = (
        None if sunlit_all is None else np.asarray(sunlit_all, dtype=np.float64)
    )
    distance_all = conditions.get("distance_km")
    distance_all = (
        None if distance_all is None else np.asarray(distance_all, dtype=np.float64)
    )

    x_blocks: List[np.ndarray] = []
    c_blocks: List[np.ndarray] = []
    entity_blocks: List[np.ndarray] = []
    time_blocks: List[np.ndarray] = []
    cold_blocks: List[np.ndarray] = []

    for start, stop, hour, time_index in _iter_blocks(series, n_steps):
        target = standardized[start:stop]
        n_batch = stop - start
        if sunlit_all is None:
            sunlit_batch = np.zeros((n_batch, n_steps), dtype=np.float64)
        elif sunlit_all.shape == (n_entities, n_steps):
            sunlit_batch = sunlit_all[start:stop]
        else:
            raise ValueError(
                f"conditions.sunlit 形状应为 ({n_entities}, {n_steps})，收到 {sunlit_all.shape}"
            )
        if distance_all is None:
            distance_batch = None
        elif distance_all.shape == (n_entities,):
            distance_batch = np.broadcast_to(
                distance_all[start:stop].reshape(n_batch, 1), (n_batch, n_steps)
            )
        elif distance_all.shape == (n_entities, n_steps):
            distance_batch = distance_all[start:stop]
        else:
            raise ValueError(f"conditions.distance_km 形状非法: {distance_all.shape}")

        previous = np.zeros_like(target)
        previous[:, :, 1:] = target[:, :, :-1]
        hist_mean, hist_std = _history_statistics(target, window)
        condition_block = pack_condition_vector(
            layer_index=layer[start:stop],
            altitude_km=altitude[start:stop],
            latitude=latitude[start:stop],
            local_hour=hour,
            sunlit=sunlit_batch,
            distance_km=distance_batch,
            prev_std=np.moveaxis(previous, 1, 2),
            hist_mean=np.moveaxis(hist_mean, 1, 2),
            hist_std=np.moveaxis(hist_std, 1, 2),
        )
        x_blocks.append(target.transpose(0, 2, 1).reshape(-1, N_PARAMS))
        c_blocks.append(condition_block.reshape(-1, COND_DIM))
        entity_blocks.append(np.repeat(np.arange(start, stop, dtype=np.int64), n_steps))
        time_blocks.append(np.tile(np.asarray(time_index, dtype=np.int64), n_batch))
        cold_blocks.append(np.tile(np.asarray(time_index) == 0, n_batch))

    x_matrix = np.ascontiguousarray(np.concatenate(x_blocks, axis=0), dtype=np.float32)
    c_matrix = np.ascontiguousarray(np.concatenate(c_blocks, axis=0), dtype=np.float32)
    if not np.isfinite(x_matrix).all():
        raise ValueError("目标矩阵包含非有限值，请检查数据源")
    if not np.isfinite(c_matrix).all():
        raise ValueError("条件矩阵包含非有限值，请检查数据源")

    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "window": window,
        "n_samples": int(x_matrix.shape[0]),
        "n_entities": int(n_entities),
        "n_steps": int(n_steps),
        "x_dim": N_PARAMS,
        "cond_dim": COND_DIM,
        "param_names": list(PARAM_NAMES),
        "transforms": dict(PARAM_TRANSFORMS),
        "log10_floors": {name: float(LOG10_FLOORS[name]) for name in LOG10_PARAMS},
        "scaler": scaler,
        "constant_params": list(scaler.get("constant_params", [])),
        "entity_index": np.concatenate(entity_blocks),
        "time_index": np.concatenate(time_blocks),
        "cold_start": np.concatenate(cold_blocks),
        "condition_layout": condition_layout(),
        "condition_names": list(COND_NAMES),
        "feature_contract": build_feature_contract(scaler, window).to_dict(),
        "created_at_utc": utc_iso(),
    }
    return (
        torch.from_numpy(x_matrix),
        torch.from_numpy(c_matrix),
        meta,
    )
