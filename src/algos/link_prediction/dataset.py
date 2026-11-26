"""多目标预测数据层：特征契约、物理序列、滑窗样本、切分与标准化。

设计原则
--------
1. **标签是物理真值，输入是感知观测**。``src.sensing.physics.simulate_parameter_series``
   产生的是物理正向计算值（source_kind=physics_simulated），本模块把它当作
   “平台已知的物理 trace”，即预测标签；预测器实际拿到的是经过传感/遥测误差
   污染后的观测历史（:class:`ObservationModel`）。这与审计文档要求的
   “独立评价使用原始物理 trace，不能把预测结果再当成真实标签”一致。
   关闭观测误差时使用 ``observation_noise=False``，仅作诊断，不作为验收口径。
2. **实体切分**：链路与节点分别按实体 60/20/20 切分训练/验证/测试；
   另外用**完全不同的 seed** 生成一个跨场景测试集，证明模型不是记住同一条轨迹。
3. **契约固定**：特征顺序、单位、窗口、步长、目标顺序写入
   :class:`FeatureContract`，随模型保存为 ``feature_contract.json``。
4. **标准化统计量只来自训练集**，随模型保存为 ``scaler.json``。

本模块不导入 torch，便于在只有 numpy 的环境里构建数据。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:  # 允许脚本从任意工作目录导入
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import PARAM_SPECS  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "predictor"
EVIDENCE_DIR = PROJECT_ROOT / "evidence" / "prediction"

SECONDS_PER_DAY = 86400.0
#: 500 km 轨道周期（秒），与 physics 模块的 5680 s 保持一致；仅用于归一化时间特征。
ORBIT_PERIOD_S = 5680.0

ENTITY_LINK = "link"
ENTITY_NODE = "node"

#: 固定 7 个预测目标，顺序不可变（技术要求 P2.1-5 / F1.1-4）。
TARGET_NAMES: Tuple[str, ...] = (
    "achievable_rate_gbps",
    "delay_jitter_ms",
    "node_load_ratio",
    "rx_power_dbm",
    "ber_pre_fec",
    "packet_loss_rate",
    "link_delay_ms",
)
TARGET_UNITS: Tuple[str, ...] = tuple(PARAM_SPECS[name].unit for name in TARGET_NAMES)
TARGET_ENTITY_TYPES: Tuple[str, ...] = (
    ENTITY_LINK,
    ENTITY_LINK,
    ENTITY_NODE,
    ENTITY_LINK,
    ENTITY_LINK,
    ENTITY_LINK,
    ENTITY_LINK,
)

#: 辅助预测目标（不是主契约目标，仅用于物理合成的中间量）。
#: 三者均为 metrics.PARAM_SPECS 已登记的感知参数，标签取 physics 真值，不构成泄漏。
#: 顺序与 :mod:`src.prediction.physics_composition` 的取用顺序一致：
#: 0=queue_occupancy_ratio，1=vibration_rms_g，2=path_loss_db。
AUX_TARGET_NAMES: Tuple[str, ...] = (
    "queue_occupancy_ratio",
    "vibration_rms_g",
    "path_loss_db",
)

#: 12 个跨层参数（环境/物理/网络三层），顺序取自 metrics.CROSS_LAYER_DATASET_PARAMS。
CROSS_LAYER_FEATURES: Tuple[str, ...] = (
    "ambient_temperature_k",
    "vibration_rms_g",
    "pointing_error_urad",
    "tx_power_dbm",
    "path_loss_db",
    "rx_power_dbm",
    "snr_db",
    "ber_pre_fec",
    "achievable_rate_gbps",
    "link_delay_ms",
    "packet_loss_rate",
    "delay_jitter_ms",
)
#: 4 个上下文参数（目标自身历史 + 网络层辅助量），保证模型能看到被预测量与拥塞状态。
CONTEXT_FEATURES: Tuple[str, ...] = (
    "node_load_ratio",
    "node_load_gbps",
    "throughput_gbps",
    "queue_occupancy_ratio",
)
#: 观测通道（16 维），预测器输入必须包含这些量。
OBSERVED_FEATURES: Tuple[str, ...] = CROSS_LAYER_FEATURES + CONTEXT_FEATURES
#: 派生特征（1 维）：队列占用平方。抖动的主因是瓶颈排队波动，q^2 是其显式非线性项。
DERIVED_FEATURES: Tuple[str, ...] = ("queue_occupancy_ratio_sq",)
#: 归一化时间特征（5 维）。
TIME_FEATURES: Tuple[str, ...] = (
    "age_norm",
    "tod_sin",
    "tod_cos",
    "orbit_sin",
    "orbit_cos",
)
#: 模型输入总维度：16 个物理/网络通道 + 1 个派生特征 + 5 个时间特征 = 22。
FEATURE_NAMES: Tuple[str, ...] = OBSERVED_FEATURES + DERIVED_FEATURES + TIME_FEATURES


def _unit_of(name: str) -> str:
    """查询参数字典中的单位；时间特征使用无量纲。"""
    if name == "queue_occupancy_ratio_sq":
        return "ratio^2"
    if name in PARAM_SPECS:
        return PARAM_SPECS[name].unit
    if name == "node_load_gbps":
        return "Gbps"
    return "ratio"


FEATURE_UNITS: Tuple[str, ...] = tuple(_unit_of(name) for name in FEATURE_NAMES)
DERIVED_UNITS: Tuple[str, ...] = tuple(_unit_of(name) for name in DERIVED_FEATURES)

#: 物理量程裁剪，模型与所有基线使用同一规则，避免负速率/负丢包等非物理输出。
TARGET_CLIP: Dict[str, Tuple[float, float]] = {
    "achievable_rate_gbps": (0.0, 100.0),
    "delay_jitter_ms": (0.0, 100.0),
    "node_load_ratio": (0.0, 1.0),
    "rx_power_dbm": (-160.0, 20.0),
    "ber_pre_fec": (0.0, 0.5),
    "packet_loss_rate": (0.0, 1.0),
    "link_delay_ms": (0.0, 1000.0),
}

#: 目标损失权重：7 个目标同权（速率/抖动/负载三项重点目标权重相同；其余也不压低，
#: 避免“用平均值掩盖失败项”的优化偏向）。
TARGET_LOSS_WEIGHTS: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

#: 多步残差预测的基准窗口（步数），按目标顺序：速率、抖动、负载、收功率、BER、丢包、时延。
#: 抖动/速率等快变量用最后值保留相位；节点负载/传播时延等慢变量用 8 步均值稳住水平。
TARGET_BASE_WINDOWS: Tuple[int, ...] = (1, 1, 8, 1, 1, 1, 8)


@dataclass(frozen=True)
class FeatureContract:
    """预测特征契约：字段顺序、单位、窗口、步长、目标、实体类型。

    该对象是数据层、训练、推理和基准测试之间唯一的字段定义来源；
    ``contract_id`` 由窗口/步长/特征顺序确定，模型目录与在线推理必须一致。
    """

    window: int = 48
    horizon: int = 5
    step_ms: int = 1000
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    feature_units: Tuple[str, ...] = FEATURE_UNITS
    observed_feature_names: Tuple[str, ...] = OBSERVED_FEATURES
    derived_feature_names: Tuple[str, ...] = DERIVED_FEATURES
    derived_feature_units: Tuple[str, ...] = DERIVED_UNITS
    time_feature_names: Tuple[str, ...] = TIME_FEATURES
    target_names: Tuple[str, ...] = TARGET_NAMES
    target_units: Tuple[str, ...] = TARGET_UNITS
    target_entity_types: Tuple[str, ...] = TARGET_ENTITY_TYPES
    orbit_period_s: float = ORBIT_PERIOD_S

    def __post_init__(self) -> None:
        if tuple(self.target_names) != TARGET_NAMES:
            raise ValueError("目标顺序必须与固定 7 目标一致，不允许重排或改名")
        if len(self.feature_names) != len(self.feature_units):
            raise ValueError("feature_names 与 feature_units 长度不一致")
        if (
            len(self.observed_feature_names)
            + len(self.derived_feature_names)
            + len(self.time_feature_names)
            != len(self.feature_names)
        ):
            raise ValueError("观测通道 + 派生特征 + 时间特征必须等于特征总维度")
        if self.window <= 0 or self.horizon <= 0 or self.step_ms <= 0:
            raise ValueError("window/horizon/step_ms 必须为正")
        observed_count = len(self.observed_feature_names)
        derived_count = len(self.derived_feature_names)
        if tuple(self.feature_names[:observed_count]) != tuple(self.observed_feature_names):
            raise ValueError("前 observed_feature_names 个通道必须是观测通道")
        if tuple(self.feature_names[observed_count : observed_count + derived_count]) != tuple(
            self.derived_feature_names
        ):
            raise ValueError("派生特征必须紧随观测通道")
        if tuple(self.feature_names[observed_count + derived_count :]) != tuple(self.time_feature_names):
            raise ValueError("时间特征必须位于特征末尾")

    @property
    def contract_id(self) -> str:
        """由契约内容确定的稳定标识。"""
        return "mtp-v2-f{}-w{}-h{}-s{}".format(
            len(self.feature_names), self.window, self.horizon, self.step_ms
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    @property
    def offsets_ms(self) -> Tuple[int, ...]:
        """相对预测发起时刻的偏移（毫秒），第 k 步为 (k+1)*step_ms，不含第 0 步。"""
        return tuple(self.step_ms * (index + 1) for index in range(self.horizon))

    def feature_index(self, name: str) -> int:
        try:
            return self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(f"特征契约未登记字段: {name}") from exc

    def target_index(self, name: str) -> int:
        try:
            return self.target_names.index(name)
        except ValueError as exc:
            raise KeyError(f"目标契约未登记字段: {name}") from exc

    def target_feature_indices(self) -> Tuple[int, ...]:
        """每个目标自身的观测通道下标（基线做单变量持续/平滑预测时使用）。"""
        return tuple(self.feature_index(name) for name in self.target_names)

    def entity_target_mask(self) -> np.ndarray:
        """实体类型 -> 7 目标的有效掩码。行 0=link，行 1=node。"""
        mask = np.zeros((2, self.n_targets), dtype=np.float32)
        for index, entity_type in enumerate(self.target_entity_types):
            mask[0 if entity_type == ENTITY_LINK else 1, index] = 1.0
        return mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "mtp-v2",
            "contract_id": self.contract_id,
            "window": int(self.window),
            "horizon": int(self.horizon),
            "step_ms": int(self.step_ms),
            "feature_names": list(self.feature_names),
            "feature_units": list(self.feature_units),
            "observed_feature_names": list(self.observed_feature_names),
            "derived_feature_names": list(self.derived_feature_names),
            "derived_feature_units": list(self.derived_feature_units),
            "time_feature_names": list(self.time_feature_names),
            "target_names": list(self.target_names),
            "target_units": list(self.target_units),
            "target_entity_types": list(self.target_entity_types),
            "target_feature_indices": list(self.target_feature_indices()),
            "target_loss_weights": list(TARGET_LOSS_WEIGHTS),
            "orbit_period_s": float(self.orbit_period_s),
            "time_feature_spec": {
                "day_period_s": SECONDS_PER_DAY,
                "orbit_period_s": float(self.orbit_period_s),
                "tod_reference": "sim_time_s 为场景当地秒数，模 86400",
                "age_norm": "窗口内相对位置，0=最新一步，1=最老一步",
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureContract":
        return cls(
            window=int(payload["window"]),
            horizon=int(payload["horizon"]),
            step_ms=int(payload["step_ms"]),
            feature_names=tuple(payload["feature_names"]),
            feature_units=tuple(payload["feature_units"]),
            observed_feature_names=tuple(payload["observed_feature_names"]),
            derived_feature_names=tuple(payload.get("derived_feature_names", ())),
            derived_feature_units=tuple(
                payload.get("derived_feature_units", ("ratio^2",) * len(payload.get("derived_feature_names", ())))
            ),
            time_feature_names=tuple(payload["time_feature_names"]),
            target_names=tuple(payload["target_names"]),
            target_units=tuple(payload["target_units"]),
            target_entity_types=tuple(payload["target_entity_types"]),
            orbit_period_s=float(payload.get("orbit_period_s", ORBIT_PERIOD_S)),
        )


# ---------------------------------------------------------------------------
# 观测误差模型（输入侧；标签保持物理真值）
# ---------------------------------------------------------------------------
#: 每个观测通道的 1 s 传感/遥测误差。``abs`` 为绝对误差，``rel`` 为相对误差；
#: ``kind=multiplicative`` 用于跨数量级的 BER（乘性对数正态误差）。
#: 量级依据：光功率计 0.35 dB、速率遥测 1.2 Gbps、1 s 探测均值 0.08 ms、
#: IPDV 估计 0.035 ms、负载 1.8 个百分点、丢包计数 0.4 个百分点。
#: 该表在训练前冻结，训练、验证、测试与跨场景使用同一张表。
OBSERVATION_NOISE_SPEC: Dict[str, Dict[str, Any]] = {
    "ambient_temperature_k": {"kind": "additive", "abs": 0.8, "rel": 0.0},
    "vibration_rms_g": {"kind": "additive", "abs": 0.006, "rel": 0.0},
    "pointing_error_urad": {"kind": "additive", "abs": 0.15, "rel": 0.0},
    "tx_power_dbm": {"kind": "additive", "abs": 0.15, "rel": 0.0},
    "path_loss_db": {"kind": "additive", "abs": 0.10, "rel": 0.0},
    "rx_power_dbm": {"kind": "additive", "abs": 0.35, "rel": 0.0},
    "snr_db": {"kind": "additive", "abs": 0.35, "rel": 0.0},
    "ber_pre_fec": {"kind": "multiplicative", "abs": 0.0, "rel": 0.35},
    "achievable_rate_gbps": {"kind": "additive", "abs": 1.2, "rel": 0.005},
    "link_delay_ms": {"kind": "additive", "abs": 0.08, "rel": 0.002},
    "packet_loss_rate": {"kind": "additive", "abs": 0.004, "rel": 0.0},
    "delay_jitter_ms": {"kind": "additive", "abs": 0.035, "rel": 0.02},
    "node_load_ratio": {"kind": "additive", "abs": 0.018, "rel": 0.0},
    "node_load_gbps": {"kind": "additive", "abs": 2.0, "rel": 0.0},
    "throughput_gbps": {"kind": "additive", "abs": 0.8, "rel": 0.0},
    "queue_occupancy_ratio": {"kind": "additive", "abs": 0.015, "rel": 0.0},
}


@dataclass(frozen=True)
class ObservationModel:
    """把物理真值转换成“可观测历史”的传感误差模型。

    :param enabled: False 时不做任何扰动（仅作诊断）。
    :param scale: 统一缩放因子，默认 1.0；协议冻结后不应改动。
    :param seed: 观测噪声随机种子。
    """

    enabled: bool = True
    scale: float = 1.0
    seed: int = 0
    spec: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: OBSERVATION_NOISE_SPEC)

    def apply(self, params: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """返回观测通道的含噪副本；未登记的通道原样透传。"""
        rng = np.random.default_rng(int(self.seed) + 7919)
        observed: Dict[str, np.ndarray] = {}
        for name, values in params.items():
            array = np.asarray(values, dtype=np.float64)
            config = self.spec.get(name, {})
            if not self.enabled or not config:
                observed[name] = array.astype(np.float32)
                continue
            kind = str(config.get("kind", "additive"))
            if kind == "multiplicative":
                sigma = self.scale * float(config.get("rel", 0.0))
                noise = rng.normal(0.0, sigma, size=array.shape) if sigma > 0 else 0.0
                observed[name] = (array * np.exp(noise)).astype(np.float32)
            else:
                sigma = self.scale * (
                    float(config.get("abs", 0.0)) + float(config.get("rel", 0.0)) * np.abs(array)
                )
                noise = rng.normal(0.0, 1.0, size=array.shape) * sigma
                observed[name] = (array + noise).astype(np.float32)
        return observed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "scale": float(self.scale),
            "seed": int(self.seed),
            "spec": {key: dict(value) for key, value in self.spec.items()},
            "note": "输入侧传感/遥测误差；标签为 physics 物理真值。",
        }


# ---------------------------------------------------------------------------
# 物理序列加载（优先 src.sensing.physics，缺失时用内部 fallback）
# ---------------------------------------------------------------------------
_PHYSICS_SERIES = None
_PHYSICS_IMPORT_ERROR = ""
try:  # pragma: no cover - 取决于外部模块是否已落地
    from sensing_physics import simulate_parameter_series as _PHYSICS_SERIES  # type: ignore
except Exception as exc:  # noqa: BLE001 - 任何导入失败都应可降级
    _PHYSICS_SERIES = None
    _PHYSICS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def physics_available() -> bool:
    """当前环境是否可导入 physics 序列生成器。"""
    return _PHYSICS_SERIES is not None


def _sha256_of_module(module_name: str) -> str:
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if not path:
        return ""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _entity_ids(n_links: int, n_nodes: int) -> Tuple[List[str], List[str], np.ndarray]:
    """按契约格式生成实体 ID：节点 "3600"、有向链路 "3600->3601"。"""
    base = 3600
    node_ids = [str(base + index) for index in range(n_nodes)]
    link_ids: List[str] = []
    for index in range(n_nodes):
        for step in range(1, n_nodes):
            if len(link_ids) >= n_links:
                break
            target = (index + step) % n_nodes
            link_ids.append(f"{base + index}->{base + target}")
        if len(link_ids) >= n_links:
            break
    if len(link_ids) < n_links:  # 节点数过少时循环补齐（仍保持可读 ID）
        cursor = 0
        while len(link_ids) < n_links:
            link_ids.append(f"{base + cursor % n_nodes}->{base + (cursor + 1) % n_nodes}")
            cursor += 1
    types = np.asarray([ENTITY_LINK] * n_links + [ENTITY_NODE] * n_nodes, dtype="<U4")
    return link_ids, node_ids, types


def _validate_series(params: Mapping[str, np.ndarray], n_entities: int, n_steps: int) -> None:
    missing = [name for name in (OBSERVED_FEATURES + TARGET_NAMES) if name not in params]
    if missing:
        raise ValueError(f"参数序列缺少字段: {missing}")
    for name in OBSERVED_FEATURES + TARGET_NAMES:
        shape = np.asarray(params[name]).shape
        if shape != (n_entities, n_steps):
            raise ValueError(f"参数 {name} 形状应为 {(n_entities, n_steps)}，实际 {shape}")


def _ensure_derived_channels(params: Dict[str, np.ndarray]) -> Dict[str, str]:
    """补齐 physics 序列未直接输出的派生通道，并返回派生说明。

    * ``node_load_gbps = node_load_ratio * MAX_NODE_FLOW_GBPS``（节点最大处理能力，
      取自 ``src.config.MAX_NODE_FLOW_GBPS``，业务项目口径 400 Gbps）；
    * ``throughput_gbps = achievable_rate_gbps * (1 - packet_loss_rate)``。

    派生只发生在缺失时；physics 若已提供则不覆盖。
    """
    derived: Dict[str, str] = {}
    if "node_load_gbps" not in params and "node_load_ratio" in params:
        try:
            from config import MAX_NODE_FLOW_GBPS  # noqa: PLC0415 - 延迟导入避免耦合

            capacity = float(MAX_NODE_FLOW_GBPS)
        except Exception:  # noqa: BLE001
            capacity = 400.0
        params["node_load_gbps"] = (
            np.asarray(params["node_load_ratio"], dtype=np.float32) * capacity
        ).astype(np.float32)
        derived["node_load_gbps"] = f"node_load_ratio * {capacity:.1f} Gbps（src.config.MAX_NODE_FLOW_GBPS）"
    if (
        "throughput_gbps" not in params
        and "achievable_rate_gbps" in params
        and "packet_loss_rate" in params
    ):
        params["throughput_gbps"] = (
            np.asarray(params["achievable_rate_gbps"], dtype=np.float32)
            * (1.0 - np.clip(np.asarray(params["packet_loss_rate"], dtype=np.float32), 0.0, 1.0))
        ).astype(np.float32)
        derived["throughput_gbps"] = "achievable_rate_gbps * (1 - packet_loss_rate)"
    return derived


def load_scenario(
    n_links: int,
    n_nodes: int,
    n_steps: int,
    dt_s: float,
    seed: int,
    source: str = "auto",
    observation_noise: bool = True,
    observation_scale: float = 1.0,
) -> Dict[str, Any]:
    """加载一个场景的物理真值与观测历史。

    返回 ``{"params_clean", "params_observed", "conditions", "provenance"}``。
    ``source="auto"`` 时优先使用 ``src.sensing.physics``；导入失败才使用
    内部 ``_fallback_series``，并在 provenance 中明确标注 ``fallback=True``。
    """
    n_total = int(n_links) + int(n_nodes)
    if n_total <= 0 or int(n_steps) <= 0:
        raise ValueError("n_links + n_nodes 与 n_steps 必须为正")
    source = str(source).lower()
    if source not in ("auto", "physics", "fallback"):
        raise ValueError(f"未知数据源: {source}")

    fallback = False
    if source in ("auto", "physics") and _PHYSICS_SERIES is not None:
        payload = _PHYSICS_SERIES(n_entities=n_total, n_steps=int(n_steps), dt_s=float(dt_s), seed=int(seed))
        if not isinstance(payload, Mapping) or "params" not in payload:
            raise ValueError("simulate_parameter_series 返回结构不符合契约")
        params = {str(key): np.asarray(value) for key, value in payload["params"].items()}
        conditions = {str(key): np.asarray(value) for key, value in dict(payload.get("conditions", {})).items()}
        derived_channels = _ensure_derived_channels(params)
        _validate_series(params, n_total, int(n_steps))
        source_kind = "physics_simulated"
        module_name = "sensing_physics"
        import_error = ""
    else:
        if source == "physics":
            raise ImportError(f"physics 序列生成器不可用: {_PHYSICS_IMPORT_ERROR}")
        payload = _fallback_series(n_entities=n_total, n_steps=int(n_steps), dt_s=float(dt_s), seed=int(seed))
        params = {str(key): np.asarray(value) for key, value in payload["params"].items()}
        conditions = {str(key): np.asarray(value) for key, value in dict(payload["conditions"]).items()}
        derived_channels = _ensure_derived_channels(params)
        _validate_series(params, n_total, int(n_steps))
        fallback = True
        source_kind = "fallback_synthetic"
        module_name = "src.prediction.dataset._fallback_series"
        import_error = _PHYSICS_IMPORT_ERROR

    link_ids, node_ids, entity_types = _entity_ids(int(n_links), int(n_nodes))
    entity_ids = link_ids + node_ids
    conditions["entity_ids"] = np.asarray(entity_ids, dtype=object)
    conditions["entity_types"] = entity_types

    observation_model = ObservationModel(
        enabled=bool(observation_noise), scale=float(observation_scale), seed=int(seed)
    )
    params_observed = observation_model.apply(params) if observation_noise else {
        name: np.asarray(params[name], dtype=np.float32) for name in params
    }

    provenance = {
        "source_kind": source_kind,
        "module": module_name,
        "module_sha256": _sha256_of_module(module_name) if not fallback else "",
        "physics_import_error": import_error,
        "fallback": bool(fallback),
        "fallback_note": (
            "内部简化生成器，仅用于 physics 模块缺失时的自测，不得作为验收证据。"
            if fallback
            else ""
        ),
        "seed": int(seed),
        "n_entities": n_total,
        "n_links": int(n_links),
        "n_nodes": int(n_nodes),
        "n_steps": int(n_steps),
        "dt_s": float(dt_s),
        "entity_type_mapping": "前 n_links 行=link，后 n_nodes 行=node（物理序列按行号对应实体）",
        "derived_channels": derived_channels,
        "observation_model": observation_model.to_dict(),
    }
    return {
        "params_clean": {name: np.asarray(value, dtype=np.float32) for name, value in params.items()},
        "params_observed": params_observed,
        "conditions": conditions,
        "provenance": provenance,
    }


def _fallback_series(n_entities: int, n_steps: int, dt_s: float, seed: int) -> Dict[str, Any]:
    """physics 模块缺失时的简化生成器（仅自测，明确标注 fallback）。

    关系链：轨道相位 -> 振动 -> 指向误差 -> 链路损耗 -> 收功率/SNR -> BER -> 速率
    -> 队列/时延/抖动/丢包 -> 节点负载。所有量均为简化公式，不能替代物理模型。
    """
    rng = np.random.default_rng(int(seed))
    time_index = np.arange(int(n_steps), dtype=np.float64)
    seconds = time_index * float(dt_s)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=int(n_entities))
    period = rng.uniform(5400.0, 6400.0, size=int(n_entities))
    distance = rng.uniform(700.0, 2600.0, size=int(n_entities))
    orbit_phase = np.mod(seconds[None, :] / period[:, None] + phase[:, None] / (2.0 * math.pi), 1.0)
    sunlit = (orbit_phase < 0.62).astype(np.float64)
    local_hour = (8.0 + seconds / 3600.0) % 24.0
    day_factor = np.clip(0.15 + 0.85 * np.exp(-0.5 * ((local_hour - 15.0) / 4.5) ** 2), 0.05, 1.0)

    thermal = np.abs(np.gradient(sunlit, axis=1))
    vibration = np.clip(
        0.02 + 0.15 * thermal + 0.02 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.80, 1.0)),
        0.0,
        1.5,
    )
    pointing = np.clip(
        3.0 + 9.0 * vibration + 0.6 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.85, 1.0)),
        0.5,
        40.0,
    )
    tx_power = 23.0 + 0.15 * _ar1_multi(rng, n_entities, n_steps, 0.97, 1.0)
    path_loss = (
        20.0 * np.log10(4.0 * math.pi * distance[:, None] * 1e3 / 1550e-9)
        - 220.9
        + 3.0
        + 8.686 * (pointing / 10.0) ** 2
        + 0.3 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.90, 1.0))
    )
    rx_power = tx_power - path_loss
    snr = rx_power + 27.0
    ber = np.clip(0.5 * np.exp(-1.15 * (snr - 4.0)) + 1e-9, 1e-15, 0.5)
    rate = np.clip(25.0 * np.log2(1.0 + 10.0 ** (snr / 10.0)), 0.0, 100.0)
    offered = np.clip(0.18 + 0.5 * day_factor[None, :] + 0.25 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.92, 1.0)), 0.02, 1.5)
    queue = np.clip(0.55 * offered + 0.05 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.88, 1.0)), 0.0, 0.98)
    delay = distance[:, None] * 1000.0 / 299_792_458.0 * 1000.0 + 0.05 * queue / np.maximum(1.0 - queue, 0.02)
    jitter = 0.05 + 0.8 * queue ** 2 + 0.15 * vibration + 0.02 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.75, 1.0))
    loss = np.clip(np.maximum(0.0, queue - 0.8) * 0.35 + 0.02 * np.maximum(0.0, ber - 0.01), 0.0, 1.0)
    throughput = np.clip(rate * (1.0 - loss) * np.minimum(offered, 1.0), 0.0, 100.0)
    node_load = np.clip(0.25 * offered + 0.1 * queue + 0.05 * np.abs(_ar1_multi(rng, n_entities, n_steps, 0.90, 1.0)), 0.0, 1.0)
    ambient = 210.0 + 90.0 * sunlit + 4.0 * np.sin(2.0 * math.pi * orbit_phase) + 0.8 * _ar1_multi(rng, n_entities, n_steps, 0.95, 0.8)
    terminal = ambient + 12.0 * day_factor[None, :] + 3.0 * _ar1_multi(rng, n_entities, n_steps, 0.90, 0.5)

    params = {
        "ambient_temperature_k": ambient,
        "terminal_temperature_k": terminal,
        "vibration_rms_g": vibration,
        "pointing_error_urad": pointing,
        "tx_power_dbm": tx_power,
        "path_loss_db": path_loss,
        "rx_power_dbm": rx_power,
        "snr_db": snr,
        "ber_pre_fec": ber,
        "ber_post_fec": np.clip(ber * 1e-3, 1e-15, 0.5),
        "achievable_rate_gbps": rate,
        "link_delay_ms": delay,
        "packet_loss_rate": loss,
        "delay_jitter_ms": jitter,
        "throughput_gbps": throughput,
        "queue_occupancy_ratio": queue,
        "node_load_ratio": node_load,
        "node_load_gbps": node_load * 240.0,
        "pat_locked": np.clip(1.0 - pointing / 30.0, 0.0, 1.0),
    }
    conditions = {
        "layer_index": rng.integers(0, 2, size=int(n_entities)).astype(np.int64),
        "altitude_km": np.where(rng.integers(0, 2, size=int(n_entities)) == 0, 500.0, 1150.0).astype(np.float32),
        "latitude": rng.uniform(-60.0, 60.0, size=int(n_entities)).astype(np.float32),
        "local_hour": local_hour.astype(np.float32),
        "time_index": time_index.astype(np.int32),
        "sunlit": sunlit.astype(np.float32),
        "distance_km": distance.astype(np.float32),
    }
    return {
        "params": {name: np.asarray(value, dtype=np.float32) for name, value in params.items()},
        "conditions": conditions,
    }


def _ar1_multi(
    rng: np.random.Generator, n_entities: int, n_steps: int, phi: float, sigma: float
) -> np.ndarray:
    """逐实体独立的一阶自回归噪声，稳态标准差约等于 ``sigma``。"""
    innovation = rng.normal(0.0, sigma * math.sqrt(max(1.0 - phi * phi, 1e-6)), size=(int(n_entities), int(n_steps)))
    series = np.empty_like(innovation)
    series[:, 0] = rng.normal(0.0, sigma, size=int(n_entities))
    for step in range(1, int(n_steps)):
        series[:, step] = phi * series[:, step - 1] + innovation[:, step]
    return series


# ---------------------------------------------------------------------------
# 滑窗样本
# ---------------------------------------------------------------------------
@dataclass
class WindowedDataset:
    """滑窗样本集合：X 观测历史、Y 物理真值、mask 目标有效位、C 条件、meta 元数据。"""

    X: np.ndarray
    Y: np.ndarray
    mask: np.ndarray
    C: Dict[str, Any]
    meta: Dict[str, Any]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, entity_indices: Sequence[int], name: str = "") -> "WindowedDataset":
        """按实体行号切分子集（实体级切分，窗口不跨实体）。"""
        keep = np.isin(self.C["entity_index"], np.asarray(entity_indices, dtype=np.int64))
        C: Dict[str, Any] = {}
        for key, value in self.C.items():
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[:1] == (len(self),)
                and key not in ("entity_types", "entity_ids")
            ):
                C[key] = value[keep]
            else:
                C[key] = value
        meta = dict(self.meta)
        meta["subset"] = name
        meta["n_windows"] = int(keep.sum())
        return WindowedDataset(X=self.X[keep], Y=self.Y[keep], mask=self.mask[keep], C=C, meta=meta)

    def entity_split(self, ratios: Sequence[float], seed: int) -> Dict[str, np.ndarray]:
        """按实体类型分层做 60/20/20 切分，返回各子集的实体行号。"""
        return split_entity_indices(self.C["entity_types"], ratios, seed)


def build_time_channels(
    n_steps: int, dt_s: float, contract: FeatureContract, t0_s: float
) -> np.ndarray:
    """构造全局时间通道 (n_steps, 4)：日周期 sin/cos 与轨道参考 sin/cos。"""
    seconds = float(t0_s) + np.arange(int(n_steps), dtype=np.float64) * float(dt_s)
    tod = np.mod(seconds, SECONDS_PER_DAY) / SECONDS_PER_DAY * 2.0 * math.pi
    orbit = np.mod(seconds / float(contract.orbit_period_s), 1.0) * 2.0 * math.pi
    return np.stack([np.sin(tod), np.cos(tod), np.sin(orbit), np.cos(orbit)], axis=1)


def build_derived_features(segment: np.ndarray, contract: FeatureContract) -> np.ndarray:
    """由观测窗口计算派生特征块；当前仅 ``queue_occupancy_ratio_sq``。

    :param segment: (window, n_observed) 观测通道
    :return: (window, n_derived)
    """
    blocks: List[np.ndarray] = []
    for name in contract.derived_feature_names:
        if name == "queue_occupancy_ratio_sq":
            index = contract.observed_feature_names.index("queue_occupancy_ratio")
            blocks.append(np.asarray(segment[:, index : index + 1], dtype=np.float32) ** 2)
        else:
            raise ValueError(f"未实现的派生特征: {name}")
    if not blocks:
        return np.empty((segment.shape[0], 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def make_windows(
    params_observed: Mapping[str, np.ndarray],
    params_clean: Mapping[str, np.ndarray],
    conditions: Mapping[str, np.ndarray],
    contract: FeatureContract,
    provenance: Mapping[str, Any],
    scenario_seed: int,
    dt_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """把逐实体序列切成 ``(X, Y, C, meta)``。

    * ``X``：``(N, window, feat_dim)``，前 16 维为含噪观测，后 5 维为时间特征；
    * ``Y``：``(N, horizon, 7)``，物理真值（未加噪）；
    * ``C``：实体类型/ID、窗口结束步、场景时间、目标有效掩码、实体行号；
    * ``meta``：契约与数据来源。
    """
    observed = np.stack([np.asarray(params_observed[name], dtype=np.float32) for name in contract.observed_feature_names], axis=2)
    clean = np.stack([np.asarray(params_clean[name], dtype=np.float32) for name in contract.target_names], axis=2)
    aux_clean = np.stack(
        [np.asarray(params_clean[name], dtype=np.float32) for name in AUX_TARGET_NAMES], axis=2
    )
    n_entities, n_steps = observed.shape[0], observed.shape[1]
    horizon = int(contract.horizon)
    window = int(contract.window)
    if n_steps < window + horizon:
        raise ValueError(f"n_steps={n_steps} 不足以构造 window={window}+horizon={horizon} 的样本")

    local_hour = conditions.get("local_hour")
    t0_s = float(np.asarray(local_hour)[0]) * 3600.0 if local_hour is not None else 0.0
    time_global = build_time_channels(n_steps, dt_s, contract, t0_s)
    # 各实体行对应的目标有效掩码
    entity_types = np.asarray(conditions.get("entity_types"), dtype="<U4")
    if entity_types.shape[0] != n_entities:
        raise ValueError("conditions['entity_types'] 长度必须等于实体数")
    entity_ids = [str(value) for value in conditions.get("entity_ids")]
    if len(entity_ids) != n_entities:
        raise ValueError("conditions['entity_ids'] 长度必须等于实体数")
    type_mask = contract.entity_target_mask()  # (2, 7)

    n_windows_per_entity = n_steps - window - horizon + 1
    total = n_entities * n_windows_per_entity
    feature_dim = contract.feature_dim
    X = np.empty((total, window, feature_dim), dtype=np.float32)
    Y = np.empty((total, horizon, contract.n_targets), dtype=np.float32)
    mask = np.empty((total, contract.n_targets), dtype=np.float32)
    aux = np.empty((total, horizon, len(AUX_TARGET_NAMES)), dtype=np.float32)

    age = ((window - 1 - np.arange(window, dtype=np.float32)) / float(max(window - 1, 1)))[:, None]
    cursor = 0
    for entity in range(n_entities):
        entity_type = str(entity_types[entity])
        row_mask = type_mask[0] if entity_type == ENTITY_LINK else type_mask[1]
        for end_step in range(window - 1, n_steps - horizon):
            segment = observed[entity, end_step - window + 1 : end_step + 1]
            time_block = np.broadcast_to(time_global[end_step], (window, 4))
            observed_count = len(contract.observed_feature_names)
            derived_count = len(contract.derived_feature_names)
            X[cursor, :, :observed_count] = segment
            X[cursor, :, observed_count : observed_count + derived_count] = build_derived_features(
                segment, contract
            )
            X[cursor, :, observed_count + derived_count :] = np.concatenate(
                [age, time_block], axis=1
            )
            Y[cursor] = clean[entity, end_step + 1 : end_step + 1 + horizon]
            aux[cursor] = aux_clean[entity, end_step + 1 : end_step + 1 + horizon]
            mask[cursor] = row_mask
            cursor += 1

    C: Dict[str, Any] = {
        "entity_index": np.repeat(np.arange(n_entities, dtype=np.int32), n_windows_per_entity),
        "entity_type": np.repeat(entity_types.astype("<U4"), n_windows_per_entity),
        "entity_id": np.repeat(np.asarray(entity_ids, dtype=object), n_windows_per_entity),
        "entity_types": entity_types.astype("<U4"),
        "entity_ids": np.asarray(entity_ids, dtype=object),
        "end_step": np.tile(np.arange(window - 1, n_steps - horizon, dtype=np.int32), n_entities),
        "sim_time_s": np.tile(t0_s + np.arange(window - 1, n_steps - horizon) * float(dt_s), n_entities),
        "target_mask": mask.copy(),
        "aux": aux,
        "aux_mask": np.ones((total, len(AUX_TARGET_NAMES)), dtype=np.float32),
        "scenario_seed": int(scenario_seed),
        "source_kind": str(provenance.get("source_kind", "")),
    }
    meta = {
        "contract": contract.to_dict(),
        "provenance": dict(provenance),
        "scenario_seed": int(scenario_seed),
        "n_entities": int(n_entities),
        "n_windows": int(total),
        "n_windows_per_entity": int(n_windows_per_entity),
        "n_steps": int(n_steps),
        "dt_s": float(dt_s),
        "window": window,
        "horizon": horizon,
        "t0_s": t0_s,
        "entity_type_counts": {
            ENTITY_LINK: int(np.sum(entity_types == ENTITY_LINK)),
            ENTITY_NODE: int(np.sum(entity_types == ENTITY_NODE)),
        },
        "target_std": {name: float(np.std(clean[:, :, index])) for index, name in enumerate(contract.target_names)},
    }
    return X, Y, C, meta


def split_entity_indices(
    entity_types: np.ndarray, ratios: Sequence[float], seed: int
) -> Dict[str, np.ndarray]:
    """按实体类型分层切分实体行号，保证节点目标在三个子集都有样本。"""
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios 必须是和为 1 的三元组")
    rng = np.random.default_rng(int(seed) + 104729)
    result: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for entity_type in (ENTITY_LINK, ENTITY_NODE):
        rows = np.where(np.asarray(entity_types) == entity_type)[0]
        rows = rng.permutation(rows)
        count = len(rows)
        if count == 0:
            continue
        n_train = int(round(count * ratios[0]))
        n_val = int(round(count * ratios[1]))
        n_train = min(max(n_train, 1), count)
        n_val = min(max(n_val, 1), count - n_train)
        result["train"].extend(int(value) for value in rows[:n_train])
        result["val"].extend(int(value) for value in rows[n_train : n_train + n_val])
        result["test"].extend(int(value) for value in rows[n_train + n_val :])
    return {key: np.sort(np.asarray(values, dtype=np.int64)) for key, values in result.items()}


# ---------------------------------------------------------------------------
# 标准化
# ---------------------------------------------------------------------------
@dataclass
class Standardizer:
    """按特征/目标分别保存均值与标准差；统计量只来自训练集。"""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    constant_targets: Tuple[str, ...] = ()
    aux_names: Tuple[str, ...] = AUX_TARGET_NAMES
    aux_mean: np.ndarray = field(default_factory=lambda: np.zeros(len(AUX_TARGET_NAMES), dtype=np.float64))
    aux_std: np.ndarray = field(default_factory=lambda: np.ones(len(AUX_TARGET_NAMES), dtype=np.float64))

    @classmethod
    def fit(cls, dataset: WindowedDataset, contract: FeatureContract) -> "Standardizer":
        X = dataset.X.astype(np.float64)
        feature_mean = X.mean(axis=(0, 1))
        feature_std = X.std(axis=(0, 1))
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        target_mean = np.zeros(contract.n_targets, dtype=np.float64)
        target_std = np.ones(contract.n_targets, dtype=np.float64)
        constants: List[str] = []
        for index, name in enumerate(contract.target_names):
            valid = dataset.mask[:, index] > 0.5
            values = dataset.Y[valid, :, index].astype(np.float64).reshape(-1)
            if values.size == 0:
                constants.append(name)
                continue
            mean = float(values.mean())
            std = float(values.std())
            if std < 1e-9:
                constants.append(name)
                std = 1.0
            target_mean[index] = mean
            target_std[index] = std
        aux_mean = np.zeros(len(AUX_TARGET_NAMES), dtype=np.float64)
        aux_std = np.ones(len(AUX_TARGET_NAMES), dtype=np.float64)
        aux_values = dataset.C.get("aux")
        if aux_values is not None:
            aux_array = np.asarray(aux_values, dtype=np.float64).reshape(-1, len(AUX_TARGET_NAMES))
            aux_mean = aux_array.mean(axis=0)
            aux_std = np.where(aux_array.std(axis=0) < 1e-9, 1.0, aux_array.std(axis=0))
        return cls(
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            feature_names=tuple(contract.feature_names),
            target_names=tuple(contract.target_names),
            constant_targets=tuple(constants),
            aux_names=AUX_TARGET_NAMES,
            aux_mean=aux_mean,
            aux_std=aux_std,
        )

    def transform_aux(self, aux: np.ndarray) -> np.ndarray:
        """标准化辅助目标（队列占用、振动）。"""
        array = np.asarray(aux, dtype=np.float32)
        return ((array - self.aux_mean.astype(np.float32)) / self.aux_std.astype(np.float32)).astype(np.float32)

    def inverse_aux(self, aux_scaled: np.ndarray) -> np.ndarray:
        array = np.asarray(aux_scaled, dtype=np.float32)
        return (array * self.aux_std.astype(np.float32) + self.aux_mean.astype(np.float32)).astype(np.float32)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(X, dtype=np.float32) - self.feature_mean.astype(np.float32)) / self.feature_std.astype(np.float32)
        return np.clip(scaled, -10.0, 10.0).astype(np.float32)

    def transform_targets(self, Y: np.ndarray) -> np.ndarray:
        return ((np.asarray(Y, dtype=np.float32) - self.target_mean.astype(np.float32)) / self.target_std.astype(np.float32)).astype(np.float32)

    def inverse_targets(self, Y_scaled: np.ndarray) -> np.ndarray:
        return (np.asarray(Y_scaled, dtype=np.float32) * self.target_std.astype(np.float32) + self.target_mean.astype(np.float32)).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
            "feature_mean": [float(value) for value in self.feature_mean],
            "feature_std": [float(value) for value in self.feature_std],
            "target_mean": [float(value) for value in self.target_mean],
            "target_std": [float(value) for value in self.target_std],
            "aux_names": list(self.aux_names),
            "aux_mean": [float(value) for value in self.aux_mean],
            "aux_std": [float(value) for value in self.aux_std],
            "constant_targets": list(self.constant_targets),
            "fit_split": "train",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Standardizer":
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
            target_mean=np.asarray(payload["target_mean"], dtype=np.float64),
            target_std=np.asarray(payload["target_std"], dtype=np.float64),
            feature_names=tuple(payload["feature_names"]),
            target_names=tuple(payload["target_names"]),
            constant_targets=tuple(payload.get("constant_targets", ())),
            aux_names=tuple(payload.get("aux_names", AUX_TARGET_NAMES)),
            aux_mean=np.asarray(payload.get("aux_mean", [0.0] * len(AUX_TARGET_NAMES)), dtype=np.float64),
            aux_std=np.asarray(payload.get("aux_std", [1.0] * len(AUX_TARGET_NAMES)), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# 数据集配置与组装
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """数据集装配参数（默认值与计划一致，可被 CLI 覆盖）。"""

    n_links: int = 80
    n_nodes: int = 24
    n_steps: int = 720
    dt_s: float = 1.0
    step_ms: int = 1000
    window: int = 48
    horizon: int = 5
    seed: int = 20260101
    cross_seed: int = 20260215
    split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    source: str = "auto"
    observation_noise: bool = True
    observation_scale: float = 1.0

    @property
    def contract(self) -> FeatureContract:
        return FeatureContract(
            window=int(self.window),
            horizon=int(self.horizon),
            step_ms=int(self.step_ms),
        )


@dataclass
class DatasetBundle:
    """训练/验证/测试/跨场景四个子集 + 标准化器 + 契约 + 数据来源。"""

    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset
    cross: WindowedDataset
    scaler: Standardizer
    contract: FeatureContract
    provenance: Dict[str, Any]
    config: DatasetConfig
    split_entity_ids: Dict[str, List[str]]

    def summary(self) -> Dict[str, Any]:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_cross": len(self.cross),
            "n_entities": {
                "train": int(len(self.split_entity_ids["train"])),
                "val": int(len(self.split_entity_ids["val"])),
                "test": int(len(self.split_entity_ids["test"])),
            },
            "split_entity_ids": {key: list(value) for key, value in self.split_entity_ids.items()},
            "contract_id": self.contract.contract_id,
            "window": self.contract.window,
            "horizon": self.contract.horizon,
            "targets": list(self.contract.target_names),
            "provenance": self.provenance,
            "config": {
                "n_links": self.config.n_links,
                "n_nodes": self.config.n_nodes,
                "n_steps": self.config.n_steps,
                "dt_s": self.config.dt_s,
                "seed": self.config.seed,
                "cross_seed": self.config.cross_seed,
                "observation_noise": self.config.observation_noise,
                "observation_scale": self.config.observation_scale,
            },
            "constant_targets": list(self.scaler.constant_targets),
        }


def prepare_datasets(config: Optional[DatasetConfig] = None) -> DatasetBundle:
    """装配训练/验证/测试/跨场景数据集，并在训练集上拟合标准化器。"""
    config = config or DatasetConfig()
    contract = config.contract
    scenario = load_scenario(
        n_links=config.n_links,
        n_nodes=config.n_nodes,
        n_steps=config.n_steps,
        dt_s=config.dt_s,
        seed=config.seed,
        source=config.source,
        observation_noise=config.observation_noise,
        observation_scale=config.observation_scale,
    )
    X, Y, C, meta = make_windows(
        scenario["params_observed"],
        scenario["params_clean"],
        scenario["conditions"],
        contract,
        scenario["provenance"],
        scenario_seed=config.seed,
        dt_s=config.dt_s,
    )
    dataset = WindowedDataset(X=X, Y=Y, mask=C["target_mask"], C=C, meta=meta)
    splits = dataset.entity_split(config.split_ratios, config.seed)
    train = dataset.subset(splits["train"], "train")
    val = dataset.subset(splits["val"], "val")
    test = dataset.subset(splits["test"], "test")

    cross_scenario = load_scenario(
        n_links=config.n_links,
        n_nodes=config.n_nodes,
        n_steps=config.n_steps,
        dt_s=config.dt_s,
        seed=config.cross_seed,
        source=config.source,
        observation_noise=config.observation_noise,
        observation_scale=config.observation_scale,
    )
    Xc, Yc, Cc, metac = make_windows(
        cross_scenario["params_observed"],
        cross_scenario["params_clean"],
        cross_scenario["conditions"],
        contract,
        cross_scenario["provenance"],
        scenario_seed=config.cross_seed,
        dt_s=config.dt_s,
    )
    cross = WindowedDataset(X=Xc, Y=Yc, mask=Cc["target_mask"], C=Cc, meta=metac)
    scaler = Standardizer.fit(train, contract)

    entity_ids_by_split = {
        key: [str(dataset.C["entity_ids"][row]) for row in rows] for key, rows in splits.items()
    }
    provenance = dict(scenario["provenance"])
    provenance["cross_scenario_seed"] = int(config.cross_seed)
    provenance["cross_scenario_source"] = cross_scenario["provenance"]["source_kind"]
    provenance["entity_split"] = {
        "strategy": "按实体类型分层、实体级 60/20/20；窗口不跨实体",
        "ratios": list(config.split_ratios),
        "counts": {key: int(len(rows)) for key, rows in splits.items()},
    }
    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        cross=cross,
        scaler=scaler,
        contract=contract,
        provenance=provenance,
        config=config,
        split_entity_ids=entity_ids_by_split,
    )


def save_contract(contract: FeatureContract, path: Path) -> None:
    """保存特征契约 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_contract(path: Path) -> FeatureContract:
    """读取特征契约 JSON。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureContract.from_dict(payload)


def save_scaler(scaler: Standardizer, path: Path) -> None:
    """保存标准化统计量 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_scaler(path: Path) -> Standardizer:
    """读取标准化统计量 JSON。"""
    return Standardizer.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def clip_predictions(predictions: np.ndarray, contract: FeatureContract) -> np.ndarray:
    """按物理量程裁剪预测值；模型与所有基线共用同一规则。"""
    result = np.asarray(predictions, dtype=np.float32).copy()
    for index, name in enumerate(contract.target_names):
        low, high = TARGET_CLIP[name]
        result[..., index] = np.clip(result[..., index], low, high)
    return result


def target_base_from_features(
    X: np.ndarray,
    scaler: "Standardizer",
    contract: FeatureContract,
    windows: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """取每个目标在窗口末端的**平滑基准**并标准化，作为多步残差预测的 base。

    默认按 :data:`TARGET_BASE_WINDOWS`：快变量（速率/抖动）取最后值保留相位，
    慢变量（节点负载/传播时延）取最近 8 步均值稳住实体水平。
    返回值形状 ``(N, 7)``，标准化空间；模型输出 ``base + 残差``。
    对实体类型不适用的目标，基准仍取对应观测通道（随后被有效掩码屏蔽）。
    """
    indices = contract.target_feature_indices()
    widths = tuple(int(value) for value in (windows or TARGET_BASE_WINDOWS))
    if len(widths) != contract.n_targets:
        raise ValueError("基准窗口数必须等于目标数")
    features = np.asarray(X, dtype=np.float32)
    base_raw = np.empty((features.shape[0], contract.n_targets), dtype=np.float32)
    for index, width in enumerate(widths):
        base_raw[:, index] = features[:, -max(width, 1) :, indices[index]].mean(axis=1)
    base = (base_raw - scaler.target_mean.astype(np.float32)) / scaler.target_std.astype(np.float32)
    return base.astype(np.float32)
