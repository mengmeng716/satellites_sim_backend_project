"""从训练好的条件 VAE 生成多层参数数据集，并做物理合理性校验。

生成流程（全部由网络推理完成）
------------------------------
1. 按 ``spec`` 采样静态条件（层、高度、纬度、起始当地时、链路距离）与时段条件
   （``local_hour`` 的 sin/cos、``sunlit``）；
2. 每个时刻把条件向量（静态条件 + 前一时刻 12 参数 + 历史窗口统计量）送入解码器，
   从先验 ``N(0, I)`` 采样隐变量并解码，得到 12 个参数的标准化输出；
3. 逆标准化/逆 log10 变换还原物理量，并按 :data:`~src.generative.dataset.PARAM_BOUNDS`
   做边界投影（投影比例写进 provenance，不隐藏失真）；
4. 把上一时刻生成值写回条件，逐时刻自回归自由演化。

生成阶段不调用 ``src.sensing.physics``，也不使用任何解析物理公式；
``sunlit`` 等条件由场景给定，属于**生成器输入**，不属于生成结果。
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from contracts import SCHEMA_VERSION, SOURCE_SYNTHETIC, utc_iso
from cvae import DEFAULT_PRIOR_CLIP, CVAEConfig, MultilayerCVaE
from dataset import (
    COND_DIM,
    DEFAULT_WINDOW,
    EVIDENCE_DIR,
    MODEL_DIR,
    N_PARAMS,
    PARAM_NAMES,
    clip_to_bounds,
    condition_layout,
    inverse_transform,
    pack_condition_vector,
    physics_fingerprint,
)

#: 生成条件的默认时间段（当地时，小时）
DEFAULT_HOURS: Tuple[float, ...] = (0.0, 6.0, 12.0, 18.0)
#: 默认场景的链路距离区间（km）。
#: 取“同一壳层内距离相近的 ISL 群”（LEO 同轨道面/相邻面 ISL 量级），
#: 这样路径损耗的方差由指向误差而非链路几何差异主导，物理关系才可检验；
#: 需要覆盖大范围链路几何时用 ``spec["distance_range_km"]`` 显式放宽。
DEFAULT_DISTANCE_RANGE_KM: Tuple[float, float] = (1000.0, 1400.0)
#: 宽距离覆盖场景（覆盖性测试用）
WIDE_DISTANCE_RANGE_KM: Tuple[float, float] = (650.0, 2600.0)
#: 默认纬度区间（度）
DEFAULT_LATITUDE_RANGE_DEG: Tuple[float, float] = (-60.0, 60.0)
#: 层索引与标称轨道高度（km）
LAYER_ALTITUDE_KM: Dict[int, float] = {0: 500.0, 1: 1150.0}
#: 层标识别名
LAYER_ALIASES: Dict[str, int] = {
    "0": 0, "LEO": 0, "L0": 0, "LAYER0": 0, "LOW": 0,
    "1": 1, "MEO": 1, "L1": 1, "LAYER1": 1, "MID": 1,
}
#: 校验阈值（与需求书一致）
CORRELATION_THRESHOLDS: Dict[str, float] = {
    "snr_vs_rx_power": 0.5,
    "ber_vs_snr": -0.3,
    "rate_vs_snr": 0.5,
    "pointing_vs_path_loss": 0.3,
    "vibration_vs_pointing": 0.3,
}


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------
class MultilayerGenerator:
    """已训练的条件 VAE 生成器（含标准化统计量、FeatureContract 与流形摘要）。"""

    def __init__(
        self,
        model: MultilayerCVaE,
        scaler: Dict[str, Any],
        condition_layout_payload: Sequence[Dict[str, Any]],
        device: torch.device,
        model_dir: Path,
        model_version: str,
        weight_sha256: str,
        window: int = DEFAULT_WINDOW,
        training_meta: Optional[Dict[str, Any]] = None,
        manifold: Optional[Dict[str, Any]] = None,
        feature_contract_id: str = "",
    ) -> None:
        self.model = model
        self.scaler = dict(scaler)
        self.condition_layout = list(condition_layout_payload)
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_version = str(model_version)
        self.weight_sha256 = str(weight_sha256)
        self.window = int(window)
        self.training_meta = dict(training_meta or {})
        self.manifold = dict(manifold or {})
        self.feature_contract_id = str(feature_contract_id or "generative-multilayer-cvae-v1")
        self.param_names: Tuple[str, ...] = tuple(self.scaler.get("param_names", PARAM_NAMES))
        self.transforms: Dict[str, str] = dict(self.scaler.get("transforms", {}))
        self.x_dim = int(model.config.x_dim)
        self.cond_dim = int(model.config.cond_dim)
        self.latent_dim = int(model.config.latent_dim)

    @property
    def n_parameters(self) -> int:
        """网络可训练参数总量。"""
        return int(self.model.count_parameters())

    @property
    def training_data_provenance(self) -> Dict[str, Any]:
        """训练数据来源（physics_simulated 或 fallback_internal_demo）。"""
        return dict(self.training_meta.get("data_provenance", {}))

    def describe(self) -> Dict[str, Any]:
        """生成器自描述，用于报告与模型注册。"""
        return {
            "model_version": self.model_version,
            "model_dir": str(self.model_dir),
            "weight_sha256": self.weight_sha256,
            "n_parameters": self.n_parameters,
            "model_config": self.model.config.to_dict(),
            "window": self.window,
            "x_dim": self.x_dim,
            "cond_dim": self.cond_dim,
            "latent_dim": self.latent_dim,
            "feature_contract_id": self.feature_contract_id,
            "device": str(self.device),
            "latent_manifold": {
                "kl_nats_mean": self.manifold.get("kl_nats_mean"),
                "active_dims": self.manifold.get("active_dims"),
                "prior_clip_sigma": self.manifold.get("prior_clip_sigma", DEFAULT_PRIOR_CLIP),
            },
            "training_data_provenance": self.training_data_provenance,
        }

    def generate(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        """按场景 spec 生成多层参数数据集（见 :func:`generate_dataset`）。"""
        return generate_dataset(spec, generator=self)


def _sha256_file(path: Path) -> str:
    """计算文件 sha256（大文件分块读取）。"""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_generator(
    model_dir: Optional[Any] = None, device: Optional[Any] = None
) -> MultilayerGenerator:
    """加载训练产物，返回 :class:`MultilayerGenerator`。

    :param model_dir: 模型目录，缺省 ``<工程根>/models/generator``
    :param device: ``torch.device`` / 字符串，缺省自动选择 GPU
    :raises FileNotFoundError: 目录或权重文件不存在（提示先训练）
    """
    path = Path(model_dir) if model_dir else MODEL_DIR
    if not path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {path}，请先运行 scripts/train_generator.py")
    checkpoint_path = path / "cvae_multilayer.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"缺少权重文件 {checkpoint_path}，请先运行 scripts/train_generator.py"
        )
    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = CVAEConfig.from_dict(payload.get("model_config", {}))
    model = MultilayerCVaE(config)
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)
    model.eval()

    scaler = dict(payload["scaler"])
    training_meta: Dict[str, Any] = {}
    meta_path = path / "train_meta.json"
    if meta_path.is_file():
        training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        training_meta = {"data_provenance": payload.get("training", {}).get("data_provenance", {})}
    manifold: Dict[str, Any] = {}
    manifold_path = path / "manifold.json"
    if manifold_path.is_file():
        manifold = json.loads(manifold_path.read_text(encoding="utf-8"))
    feature_contract = payload.get("feature_contract", {}) or {}
    return MultilayerGenerator(
        model=model,
        scaler=scaler,
        condition_layout_payload=payload.get("condition_layout", condition_layout()),
        device=resolved_device,
        model_dir=path,
        model_version=str(payload.get("model_version", f"cvae-{_sha256_file(checkpoint_path)[:12]}")),
        weight_sha256=str(payload.get("sha256_weights", _sha256_file(checkpoint_path))),
        window=int(payload.get("window", DEFAULT_WINDOW)),
        training_meta=training_meta,
        manifold=manifold,
        feature_contract_id=str(feature_contract.get("contract_id", "generative-multilayer-cvae-v1")),
    )


# ---------------------------------------------------------------------------
# 场景 spec 解析
# ---------------------------------------------------------------------------
def _as_float_tuple(value: Any, name: str) -> Tuple[float, ...]:
    """把标量或序列统一成 float 元组。"""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value),)
    try:
        items = tuple(float(item) for item in value)
    except TypeError as error:
        raise ValueError(f"{name} 必须是数字或数字序列，收到 {value!r}") from error
    if not items:
        raise ValueError(f"{name} 不能为空")
    return items


def _as_range(value: Any, name: str) -> Tuple[float, float]:
    """把二元区间统一成 ``(low, high)``。"""
    items = _as_float_tuple(value, name)
    if len(items) != 2:
        raise ValueError(f"{name} 必须是 (low, high)，收到 {items}")
    low, high = float(items[0]), float(items[1])
    if high <= low:
        raise ValueError(f"{name} 的 high 必须大于 low，收到 {items}")
    return low, high


def _normalize_layers(value: Any) -> Tuple[int, ...]:
    """解析层标识（支持 0/1 与 LEO/MEO 等别名）。"""
    if isinstance(value, (int, np.integer, str)):
        value = (value,)
    layers: List[int] = []
    for item in value:
        key = str(item).strip().upper()
        if key not in LAYER_ALIASES:
            raise ValueError(f"无法识别的层标识: {item!r}")
        layers.append(LAYER_ALIASES[key])
    if not layers:
        raise ValueError("layers 不能为空")
    return tuple(layers)


def _normalize_spec(spec: Optional[Mapping[str, Any]], generator: MultilayerGenerator) -> Dict[str, Any]:
    """解析并校验生成场景 spec，返回带默认值的规范化配置。"""
    payload: Mapping[str, Any] = spec or {}
    n_samples = int(payload.get("n_samples", 32))
    n_steps = int(payload.get("n_steps", 180))
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")
    if n_steps < 2:
        raise ValueError("n_steps 至少为 2")
    dt_s = float(payload.get("dt_s", 1.0))
    if dt_s <= 0:
        raise ValueError("dt_s 必须为正数")
    temperature = float(payload.get("temperature", 1.0))
    if temperature < 0:
        raise ValueError("temperature 不能为负")
    return {
        "n_samples": n_samples,
        "n_steps": n_steps,
        "dt_s": dt_s,
        "layers": _normalize_layers(payload.get("layers", (0, 1))),
        "hours": tuple(item % 24.0 for item in _as_float_tuple(payload.get("hours", DEFAULT_HOURS), "hours")),
        "seed": int(payload.get("seed", 20260101)),
        "temperature": temperature,
        "prior_clip": float(payload.get("prior_clip", DEFAULT_PRIOR_CLIP)),
        "prior": str(payload.get("prior", "standard")).lower(),
        "distance_range_km": _as_range(
            payload.get("distance_range_km", DEFAULT_DISTANCE_RANGE_KM), "distance_range_km"
        ),
        "latitude_range_deg": _as_range(
            payload.get("latitude_range_deg", DEFAULT_LATITUDE_RANGE_DEG), "latitude_range_deg"
        ),
        "window": int(generator.window),
        "sunlit": payload.get("sunlit"),
        "latitude": payload.get("latitude"),
        "distance_km": payload.get("distance_km"),
        "start_hours": payload.get("start_hours"),
    }


def _orbit_sunlit(
    altitude_km: np.ndarray,
    n_steps: int,
    dt_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """按简化轨道相位生成日照条件（0/1）。

    这是**生成器的条件输入**（描述场景的昼夜/阴影状态），不是生成结果本身；
    调用方可以用 ``spec["sunlit"]`` 直接指定更精确的时段。
    """
    altitude = np.asarray(altitude_km, dtype=np.float64)
    period = 2.0 * math.pi * np.sqrt(((6371.0 + altitude) * 1e3) ** 3 / 3.986004418e14)
    phase0 = rng.uniform(0.0, 1.0, altitude.size)
    phase = np.mod(
        phase0[:, None] + (np.arange(int(n_steps), dtype=np.float64) * float(dt_s))[None, :]
        / period[:, None],
        1.0,
    )
    return (phase < 0.62).astype(np.float32)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def generate_dataset(
    spec: Mapping[str, Any], generator: Optional[MultilayerGenerator] = None
) -> Dict[str, Any]:
    """按场景 spec 生成多层参数数据集。

    :param spec: 生成场景，支持字段

        * ``n_samples``（实体条数，默认 32）
        * ``n_steps``（每条序列时间步，默认 180）
        * ``dt_s``（采样间隔秒，默认 1.0）
        * ``layers``（层列表，``0/1`` 或 ``LEO/MEO``，默认 ``(0, 1)``）
        * ``hours``（起始当地时列表，默认 ``(0, 6, 12, 18)``）
        * ``seed``、``temperature``（先验采样温度）、``prior_clip``
        * ``distance_range_km``、``latitude_range_deg``
        * ``sunlit``：可选 ``(n_samples, n_steps)`` 显式日照条件
        * ``latitude`` / ``distance_km`` / ``start_hours``：可选的逐实体覆盖
        * ``model_dir``：缺省时用工程内默认模型目录

    :param generator: 已加载的生成器；缺省时按 ``spec['model_dir']`` 或默认目录加载
    :return: ``{"params", "conditions", "provenance", "spec", "shapes"}``
    """
    resolved = generator or load_generator((spec or {}).get("model_dir"))
    config = _normalize_spec(spec, resolved)
    n_samples = int(config["n_samples"])
    n_steps = int(config["n_steps"])
    dt_s = float(config["dt_s"])
    rng = np.random.default_rng(int(config["seed"]))

    layers_pool = config["layers"]
    layer_index = np.array(
        [layers_pool[index % len(layers_pool)] for index in range(n_samples)], dtype=np.int64
    )
    altitude_km = np.array(
        [LAYER_ALTITUDE_KM[int(item)] for item in layer_index], dtype=np.float64
    ) + rng.normal(0.0, 12.0, n_samples)
    if config["latitude"] is not None:
        latitude = np.asarray(config["latitude"], dtype=np.float64)
        if latitude.shape != (n_samples,):
            raise ValueError(f"spec['latitude'] 形状应为 ({n_samples},)")
    else:
        latitude = rng.uniform(*config["latitude_range_deg"], n_samples)
    if config["distance_km"] is not None:
        distance_km = np.asarray(config["distance_km"], dtype=np.float64)
        if distance_km.shape != (n_samples,):
            raise ValueError(f"spec['distance_km'] 形状应为 ({n_samples},)")
        distance_km = np.clip(distance_km, 1.0, 20000.0)
    else:
        distance_km = rng.uniform(*config["distance_range_km"], n_samples)
    hours_pool = config["hours"]
    if config["start_hours"] is not None:
        start_hours = np.asarray(config["start_hours"], dtype=np.float64)
        if start_hours.shape != (n_samples,):
            raise ValueError(f"spec['start_hours'] 形状应为 ({n_samples},)")
        start_hours = np.mod(start_hours, 24.0)
    else:
        start_hours = np.array(
            [hours_pool[index % len(hours_pool)] % 24.0 for index in range(n_samples)],
            dtype=np.float64,
        )
    local_hour = np.mod(
        start_hours[:, None] + np.arange(n_steps, dtype=np.float64)[None, :] * dt_s / 3600.0,
        24.0,
    )
    if config["sunlit"] is not None:
        sunlit = np.asarray(config["sunlit"], dtype=np.float32)
        if sunlit.shape != (n_samples, n_steps):
            raise ValueError(f"spec['sunlit'] 形状应为 ({n_samples}, {n_steps})")
    else:
        sunlit = _orbit_sunlit(altitude_km, n_steps, dt_s, rng)

    model = resolved.model
    model.eval()
    latent_dim = int(resolved.latent_dim)
    prior_clip = float(config["prior_clip"])
    temperature = float(config["temperature"])
    window = int(config["window"])

    # 先验选择：标准 N(0, I) 或“有效先验”（训练集聚合后验 N(mu_mean, mu_std^2)）。
    # 后者的作用是校准采样尺度：后验均值方差通常小于 1，直接用标准正态采样会把
    # 方法外推到没见过的流形区域，生成数据出现过度离散；聚合后验尺度更接近训练分布。
    prior_mode = str(config.get("prior", "standard")).lower()
    prior_mean_values: Optional[np.ndarray] = None
    prior_scale_values: Optional[np.ndarray] = None
    manifold_mean = resolved.manifold.get("mu_mean")
    manifold_std = resolved.manifold.get("mu_std")
    if prior_mode == "manifold" and manifold_mean and manifold_std:
        prior_mean_values = np.asarray(manifold_mean, dtype=np.float32)
        prior_scale_values = np.maximum(np.asarray(manifold_std, dtype=np.float32), 1.0e-3)
    else:
        prior_mode = "standard"
    prior_mean_tensor = (
        torch.from_numpy(prior_mean_values).to(resolved.device)
        if prior_mean_values is not None
        else None
    )
    prior_scale_tensor = (
        torch.from_numpy(prior_scale_values).to(resolved.device)
        if prior_scale_values is not None
        else None
    )

    standardized = np.zeros((n_samples, N_PARAMS, n_steps), dtype=np.float64)
    generated = np.zeros((n_samples, N_PARAMS, n_steps), dtype=np.float64)
    clip_counts: Dict[str, int] = {name: 0 for name in PARAM_NAMES}
    # 固定 torch 随机源，保证同一 seed 下生成结果可复现
    torch_generator = torch.Generator(device=resolved.device.type)
    torch_generator.manual_seed(int(config["seed"]))

    with torch.no_grad():
        for step in range(n_steps):
            if step == 0:
                previous = np.zeros((n_samples, N_PARAMS), dtype=np.float64)
                hist_mean = np.zeros((n_samples, N_PARAMS), dtype=np.float64)
                hist_std = np.zeros((n_samples, N_PARAMS), dtype=np.float64)
            else:
                start = max(0, step - (window - 1))
                history = standardized[:, :, start:step]
                previous = standardized[:, :, step - 1]
                hist_mean = history.mean(axis=2)
                hist_std = history.std(axis=2)
            condition = pack_condition_vector(
                layer_index=layer_index,
                altitude_km=altitude_km,
                latitude=latitude,
                local_hour=local_hour[:, step : step + 1],
                sunlit=sunlit[:, step : step + 1],
                distance_km=distance_km,
                prev_std=previous,
                hist_mean=hist_mean,
                hist_std=hist_std,
            )
            condition_tensor = torch.from_numpy(
                np.ascontiguousarray(condition[:, 0, :], dtype=np.float32)
            ).to(resolved.device)
            if step == 0:
                # 首帧：从先验采样（sample() 内部完成噪声采样与解码）
                sampled = model.sample(
                    condition_tensor,
                    n_samples=1,
                    device=resolved.device,
                    temperature=temperature,
                    prior_clip=prior_clip,
                    prior_mean=prior_mean_tensor,
                    prior_scale=prior_scale_tensor,
                    generator=torch_generator,
                )
                values = sampled.reshape(n_samples, N_PARAMS)
            else:
                noise = torch.randn(
                    n_samples, latent_dim, device=resolved.device, generator=torch_generator
                ) * temperature
                if prior_clip > 0:
                    noise = noise.clamp(-prior_clip, prior_clip)
                if prior_scale_tensor is not None:
                    noise = noise * prior_scale_tensor
                if prior_mean_tensor is not None:
                    noise = noise + prior_mean_tensor
                values = model.decode(noise, condition_tensor)
            standardized[:, :, step] = values.detach().cpu().numpy().astype(np.float64)
            raw = inverse_transform(standardized[:, :, step], resolved.scaler)
            clipped, counts = clip_to_bounds(raw)
            generated[:, :, step] = clipped
            for name in PARAM_NAMES:
                clip_counts[name] += int(counts[name])

    params = {
        name: generated[:, position, :].astype(np.float32)
        for position, name in enumerate(PARAM_NAMES)
    }
    conditions: Dict[str, np.ndarray] = {
        "layer_index": layer_index.astype(np.int64),
        "altitude_km": altitude_km.astype(np.float32),
        "latitude": latitude.astype(np.float32),
        "local_hour": local_hour.astype(np.float32),
        "time_index": np.arange(n_steps, dtype=np.int32),
        "sunlit": sunlit.astype(np.float32),
        "link_distance_km": np.repeat(distance_km[:, None], n_steps, axis=1).astype(np.float32),
        "start_hour": start_hours.astype(np.float32),
    }
    total_values = int(n_samples * n_steps * N_PARAMS)
    modified = int(sum(clip_counts.values()))
    training_provenance = resolved.training_data_provenance
    provenance: Dict[str, Any] = {
        "source_kind": SOURCE_SYNTHETIC,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_iso(),
        "generator_version": resolved.model_version,
        "generator_model_dir": str(resolved.model_dir),
        "generator_weight_sha256": resolved.weight_sha256,
        "generator_class": "MultilayerCVaE",
        "generator_n_parameters": resolved.n_parameters,
        "generator_config": resolved.model.config.to_dict(),
        "feature_contract_id": resolved.feature_contract_id,
        "latent_dim": resolved.latent_dim,
        "window": window,
        "seed": int(config["seed"]),
        "torch_generator_seed": int(config["seed"]),
        "temperature": temperature,
        "prior_clip": prior_clip,
        "prior": {
            "mode": prior_mode,
            "mu_mean": prior_mean_values.tolist() if prior_mean_values is not None else None,
            "mu_std": prior_scale_values.tolist() if prior_scale_values is not None else None,
            "note": (
                "可用有效先验 N(mu_mean, mu_std^2)（训练集聚合后验）校准采样尺度；"
                "standard 表示标准正态先验 N(0, I)。"
            ),
        },
        "device": str(resolved.device),
        "training_data_provenance": {
            "source_kind": training_provenance.get("source_kind", ""),
            "module": training_provenance.get("module", ""),
            "fallback": bool(training_provenance.get("fallback", False)),
            "seeds": training_provenance.get("seeds", []),
            "delivery_valid": bool(training_provenance.get("delivery_valid", False)),
            "physics_module": training_provenance.get("physics_module", {}),
        },
        "fallback_training_data": bool(training_provenance.get("fallback", False)),
        "bound_projection": {
            "modified_values": modified,
            "total_values": total_values,
            "rate": round(modified / float(total_values), 8) if total_values else 0.0,
            "per_param": clip_counts,
            "note": "边界投影按 PARAM_BOUNDS 执行；修改率如实记录，不隐藏失真。",
        },
        "condition_note": "sunlit / 当地时 / 链路距离等为场景条件输入，不属于生成结果。",
        "generation_note": "生成阶段不调用 src.sensing.physics，全部数值由 CVAE 解码器自回归输出。",
    }
    spec_payload = {
        "n_samples": n_samples,
        "n_steps": n_steps,
        "dt_s": dt_s,
        "layers": list(config["layers"]),
        "hours": list(config["hours"]),
        "seed": int(config["seed"]),
        "temperature": temperature,
        "prior_clip": prior_clip,
        "prior": prior_mode,
        "distance_range_km": list(config["distance_range_km"]),
        "latitude_range_deg": list(config["latitude_range_deg"]),
    }
    return {
        "params": params,
        "conditions": conditions,
        "provenance": provenance,
        "spec": spec_payload,
        "shapes": {
            "n_entities": n_samples,
            "n_steps": n_steps,
            "n_params": N_PARAMS,
            "cond_dim": COND_DIM,
        },
    }


# ---------------------------------------------------------------------------
# 物理合理性校验
# ---------------------------------------------------------------------------
def _as_flat(value: Any) -> np.ndarray:
    """转为一维 float64 数组。"""
    return np.asarray(value, dtype=np.float64).ravel()


def _pearson(x: Any, y: Any, min_std: float = 1.0e-9) -> Tuple[float, str]:
    """计算 Pearson 相关系数，无方差/样本不足时返回 ``(nan, 原因)``。"""
    first = _as_flat(x)
    second = _as_flat(y)
    if first.size != second.size:
        return float("nan"), f"长度不一致: {first.size} vs {second.size}"
    mask = np.isfinite(first) & np.isfinite(second)
    if int(mask.sum()) < 3:
        return float("nan"), "有效样本不足 3 个"
    usable_x = first[mask]
    usable_y = second[mask]
    if float(usable_x.std()) < float(min_std):
        return float("nan"), "前一个序列无方差（常数），相关系数不可定义"
    if float(usable_y.std()) < float(min_std):
        return float("nan"), "后一个序列无方差（常数），相关系数不可定义"
    return float(np.corrcoef(usable_x, usable_y)[0, 1]), ""


def _param_matrix(dataset: Mapping[str, Any], name: str) -> np.ndarray:
    """取参数矩阵并统一成 ``(n_entities, n_steps)``。"""
    params = dataset.get("params", {})
    if name not in params:
        raise KeyError(f"生成数据集缺少参数: {name}")
    array = np.asarray(params[name], dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"参数 {name} 形状应为 (n_entities, n_steps)，收到 {array.shape}")
    return array


def _build_check(
    check_id: str,
    group: str,
    description: str,
    metric: str,
    value: float,
    threshold: float,
    comparison: str,
    reason: str = "",
) -> Dict[str, Any]:
    """构造单条校验记录（比较符与阈值决定通过与否）。"""
    numeric = float(value) if value is not None else float("nan")
    if not math.isfinite(numeric):
        return {
            "id": check_id,
            "group": group,
            "description": description,
            "metric": metric,
            "value": None,
            "threshold": float(threshold),
            "comparison": comparison,
            "status": "NOT_MEASURABLE",
            "reason": reason or "无法计算（数值非有限）",
        }
    if comparison == "<=":
        passed = numeric <= float(threshold)
    elif comparison == "<":
        passed = numeric < float(threshold)
    elif comparison == ">=":
        passed = numeric >= float(threshold)
    elif comparison == ">":
        passed = numeric > float(threshold)
    elif comparison == "==":
        passed = abs(numeric - float(threshold)) <= 1.0e-12
    else:
        raise ValueError(f"未知比较符: {comparison}")
    return {
        "id": check_id,
        "group": group,
        "description": description,
        "metric": metric,
        "value": numeric,
        "threshold": float(threshold),
        "comparison": comparison,
        "status": "PASS" if passed else "FAIL",
        "reason": reason,
    }


def _per_entity_correlations(first: np.ndarray, second: np.ndarray) -> Dict[str, Any]:
    """逐实体（逐链路）计算 Pearson，再统计均值和下分位。

    物理关系（例如指向误差增大导致路径损耗增大）是**同一链路内**的关系；
    把所有链路并列在一起算 Pearson 会被链路几何差异（距离、高度）混杂，
    因此以逐链路均值作为判定口径，并列样本值只作为参考报告。
    """
    values: List[float] = []
    for index in range(int(first.shape[0])):
        value, _ = _pearson(first[index], second[index])
        if math.isfinite(value):
            values.append(float(value))
    if not values:
        return {
            "mean": None,
            "min": None,
            "p05": None,
            "n_entities": 0,
            "skipped_entities": int(first.shape[0]),
            "reason": "所有实体的相关系数都不可定义",
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "n_entities": int(array.size),
        "skipped_entities": int(first.shape[0]) - int(array.size),
        "reason": "",
    }


def validate_physical_bounds(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """校验生成数据集的物理合理性。

    覆盖需求书要求的 6 组检查：

    1. ``rx_power_dbm <= tx_power_dbm``，且 ``path_loss_db > 0``；
    2. ``snr_db`` 与 ``rx_power_dbm`` 正相关（Pearson > 0.5）；
    3. ``ber_pre_fec`` 在 (0, 1) 内，且与 ``snr_db`` 负相关（Pearson < -0.3）；
    4. ``achievable_rate_gbps >= 0``，且与 ``snr_db`` 正相关（Pearson > 0.5）；
    5. 跨层：``pointing_error_urad`` 与 ``path_loss_db`` 正相关（Pearson > 0.3）；
    6. 跨层：``vibration_rms_g`` 与 ``pointing_error_urad`` 正相关（Pearson > 0.3）。

    无方差导致相关系数不可定义时，该项标记为 ``NOT_MEASURABLE`` 而不是伪造数值。
    另外给出按实体去均值的补充相关系数（剔除实体间混杂，作为诊断参考）。
    """
    arrays = {name: _param_matrix(dataset, name) for name in PARAM_NAMES}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"各参数形状不一致: {sorted(shapes)}")
    n_entities, n_steps = next(iter(shapes))

    tx = arrays["tx_power_dbm"]
    rx = arrays["rx_power_dbm"]
    path_loss = arrays["path_loss_db"]
    snr = arrays["snr_db"]
    ber = arrays["ber_pre_fec"]
    rate = arrays["achievable_rate_gbps"]
    pointing = arrays["pointing_error_urad"]
    vibration = arrays["vibration_rms_g"]

    checks: List[Dict[str, Any]] = []
    violation_rates: Dict[str, float] = {}

    non_finite = int(sum(np.count_nonzero(~np.isfinite(array)) for array in arrays.values()))
    total_values = int(n_entities * n_steps * N_PARAMS)
    violation_rates["non_finite_values"] = non_finite / float(total_values) if total_values else 0.0
    checks.append(
        _build_check(
            "finite_values", "data_validity", "所有参数均为有限值",
            "非有限值比例", violation_rates["non_finite_values"], 0.0, "<=",
        )
    )

    rate_rx = float(np.count_nonzero(rx > tx)) / float(rx.size)
    violation_rates["rx_power_gt_tx_power"] = rate_rx
    checks.append(
        _build_check(
            "rx_power_le_tx_power", "power_conservation",
            "接收功率不超过发送功率（rx_power_dbm <= tx_power_dbm）",
            "违规率", rate_rx, 0.0, "<=",
        )
    )

    rate_path_loss = float(np.count_nonzero(path_loss <= 0.0)) / float(path_loss.size)
    violation_rates["path_loss_non_positive"] = rate_path_loss
    checks.append(
        _build_check(
            "path_loss_positive", "power_conservation",
            "链路损耗为正（path_loss_db > 0）", "违规率", rate_path_loss, 0.0, "<=",
        )
    )

    per_entity_summary: Dict[str, Any] = {}

    def add_correlation_check(
        check_id: str,
        group: str,
        description: str,
        threshold: float,
        comparison: str,
        first: np.ndarray,
        second: np.ndarray,
    ) -> None:
        """登记一条相关性校验：以逐链路 Pearson 均值为判定值，并列样本值另行报告。"""
        per_entity = _per_entity_correlations(first, second)
        pooled_value, pooled_reason = _pearson(first, second)
        value = per_entity["mean"] if per_entity.get("mean") is not None else float("nan")
        record = _build_check(
            check_id,
            group,
            description,
            "逐链路 Pearson 均值",
            value,
            threshold,
            comparison,
            "" if per_entity.get("mean") is not None else per_entity.get("reason", "无法计算"),
        )
        record["pooled_value"] = (
            None if not math.isfinite(pooled_value) else float(pooled_value)
        )
        record["pooled_reason"] = pooled_reason
        record["pooled_criterion_met"] = bool(
            math.isfinite(pooled_value)
            and (
                (comparison == ">" and pooled_value > float(threshold))
                or (comparison == "<" and pooled_value < float(threshold))
            )
        )
        record["criterion"] = "per_link_pearson_mean"
        record["per_entity"] = per_entity
        checks.append(record)
        per_entity_summary[check_id] = per_entity

    add_correlation_check(
        "snr_rx_positive_correlation",
        "physical_consistency",
        "SNR 与接收功率正相关（逐链路 Pearson 均值 > 0.5）",
        CORRELATION_THRESHOLDS["snr_vs_rx_power"],
        ">",
        snr,
        rx,
    )

    rate_ber_range = float(np.count_nonzero((ber <= 0.0) | (ber >= 1.0))) / float(ber.size)
    violation_rates["ber_out_of_unit_interval"] = rate_ber_range
    checks.append(
        _build_check(
            "ber_in_unit_interval", "physical_consistency",
            "纠错前误码率在 (0, 1) 内", "违规率", rate_ber_range, 0.0, "<=",
        )
    )

    add_correlation_check(
        "ber_snr_negative_correlation",
        "physical_consistency",
        "BER 与 SNR 负相关（逐链路 Pearson 均值 < -0.3）",
        CORRELATION_THRESHOLDS["ber_vs_snr"],
        "<",
        ber,
        snr,
    )

    rate_rate_negative = float(np.count_nonzero(rate < 0.0)) / float(rate.size)
    violation_rates["achievable_rate_negative"] = rate_rate_negative
    checks.append(
        _build_check(
            "achievable_rate_non_negative", "physical_consistency",
            "可实现速率非负（achievable_rate_gbps >= 0）", "违规率", rate_rate_negative, 0.0, "<=",
        )
    )

    add_correlation_check(
        "rate_snr_positive_correlation",
        "physical_consistency",
        "可实现速率与 SNR 正相关（逐链路 Pearson 均值 > 0.5）",
        CORRELATION_THRESHOLDS["rate_vs_snr"],
        ">",
        rate,
        snr,
    )

    add_correlation_check(
        "pointing_path_loss_positive_correlation",
        "cross_layer",
        "跨层：指向误差与链路损耗正相关（逐链路 Pearson 均值 > 0.3；"
        "并列样本值受链路距离混杂，见 pooled_value）",
        CORRELATION_THRESHOLDS["pointing_vs_path_loss"],
        ">",
        pointing,
        path_loss,
    )

    add_correlation_check(
        "vibration_pointing_positive_correlation",
        "cross_layer",
        "跨层：振动均方根与指向误差正相关（逐链路 Pearson 均值 > 0.3）",
        CORRELATION_THRESHOLDS["vibration_vs_pointing"],
        ">",
        vibration,
        pointing,
    )

    def within_entity(first: np.ndarray, second: np.ndarray) -> Tuple[float, str]:
        """按实体去均值后的 Pearson（剔除实体间固定差异）。"""
        first_centered = first - first.mean(axis=1, keepdims=True)
        second_centered = second - second.mean(axis=1, keepdims=True)
        return _pearson(first_centered, second_centered)

    within: Dict[str, Any] = {}
    for key, (first, second) in {
        "snr_vs_rx_power": (snr, rx),
        "ber_vs_snr": (ber, snr),
        "rate_vs_snr": (rate, snr),
        "pointing_vs_path_loss": (pointing, path_loss),
        "vibration_vs_pointing": (vibration, pointing),
    }.items():
        value, reason = within_entity(first, second)
        within[key] = {
            "pearson": None if not math.isfinite(value) else float(value),
            "reason": reason,
        }

    pooled_correlations = {
        check["id"]: check.get("pooled_value") for check in checks if "pooled_value" in check
    }
    check_to_correlation = {
        "snr_rx_positive_correlation": "snr_vs_rx_power",
        "ber_snr_negative_correlation": "ber_vs_snr",
        "rate_snr_positive_correlation": "rate_vs_snr",
        "pointing_path_loss_positive_correlation": "pointing_vs_path_loss",
        "vibration_pointing_positive_correlation": "vibration_vs_pointing",
    }
    correlations = {
        name: pooled_correlations.get(check_id)
        for check_id, name in check_to_correlation.items()
    }
    correlations_per_entity_mean = {
        name: (per_entity_summary.get(check_id, {}) or {}).get("mean")
        for check_id, name in check_to_correlation.items()
    }

    statuses = [check["status"] for check in checks]
    measurable = [check for check in checks if check["status"] != "NOT_MEASURABLE"]
    not_measurable = [check["id"] for check in checks if check["status"] == "NOT_MEASURABLE"]
    failed = [check["id"] for check in measurable if check["status"] == "FAIL"]
    passed = not failed and not not_measurable
    if passed:
        status = "PASS"
    elif failed:
        status = "FAIL"
    else:
        status = "NOT_MEASURABLE"

    provenance = dataset.get("provenance", {}) or {}
    bound_projection = provenance.get("bound_projection", {}) or {}
    hard_violation_rates = [
        value for value in violation_rates.values() if isinstance(value, float)
    ]
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validated_at_utc": utc_iso(),
        "status": status,
        "passed": bool(passed),
        "n_samples": int(n_entities * n_steps),
        "n_entities": int(n_entities),
        "n_steps": int(n_steps),
        "n_params": int(N_PARAMS),
        "violation_rate": float(max(hard_violation_rates)) if hard_violation_rates else 0.0,
        "violation_rates": {key: float(value) for key, value in violation_rates.items()},
        "correlations": correlations,
        "correlations_per_entity_mean": correlations_per_entity_mean,
        "thresholds": dict(CORRELATION_THRESHOLDS),
        "checks": checks,
        "failed_checks": failed,
        "not_measurable_checks": not_measurable,
        "within_entity_correlations": within,
        "criterion_note": (
            "相关性判定口径为“逐链路 Pearson 均值”：物理关系发生在同一条链路内部，"
            "把所有链路并列计算会被链路几何差异（距离、高度）混杂；"
            "correlations 为并列样本值（pooled），correlations_per_entity_mean 为判定值，"
            "within_entity_correlations 为按实体去均值后的并列值，三者并列报告便于复核。"
        ),
        "postprocess": {
            "bound_projection_rate": bound_projection.get("rate", 0.0),
            "bound_projection_modified_values": bound_projection.get("modified_values", 0),
            "bound_projection_per_param": bound_projection.get("per_param", {}),
        },
        "physics_module_current": physics_fingerprint(),
        "provenance": {
            "source_kind": provenance.get("source_kind", ""),
            "generator_version": provenance.get("generator_version", ""),
            "feature_contract_id": provenance.get("feature_contract_id", ""),
            "seed": provenance.get("seed"),
            "fallback_training_data": provenance.get("fallback_training_data", False),
            "physics_module": (
                provenance.get("physics_module")
                or (provenance.get("training_data_provenance", {}) or {}).get("physics_module", {})
            ),
            "physics_module_sha256": (
                (provenance.get("training_data_provenance", {}) or {})
                .get("physics_module", {})
                .get("sha256", "")
            ),
            "physics_module_mtime_utc": (
                (provenance.get("training_data_provenance", {}) or {})
                .get("physics_module", {})
                .get("mtime_utc", "")
            ),
        },
    }
    return result


# ---------------------------------------------------------------------------
# 分布对比（生成数据 vs 物理参考数据）
# ---------------------------------------------------------------------------
def _ks_statistic(sample: np.ndarray, reference: np.ndarray, max_points: int = 20000) -> float:
    """两样本 Kolmogorov-Smirnov 统计量 D（基于 ECDF 的最大偏差）。"""
    first = np.sort(np.asarray(sample, dtype=np.float64).ravel())
    second = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if first.size == 0 or second.size == 0:
        return float("nan")
    if first.size > max_points:
        first = np.random.default_rng(0).choice(first, size=max_points, replace=False)
        first.sort()
    if second.size > max_points:
        second = np.random.default_rng(1).choice(second, size=max_points, replace=False)
        second.sort()
    grid = np.concatenate([first, second])
    cdf_first = np.searchsorted(first, grid, side="right") / float(first.size)
    cdf_second = np.searchsorted(second, grid, side="right") / float(second.size)
    return float(np.max(np.abs(cdf_first - cdf_second)))


def compare_distributions(
    generated: Mapping[str, Any], reference: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """比较生成数据与参考（物理仿真）数据的逐参数分布，返回统计对比字典。"""
    if reference is None or not reference.get("params"):
        return None
    per_param: Dict[str, Any] = {}
    std_ratios: List[float] = []
    within_ratios: List[float] = []
    max_mean_diff = 0.0
    worst_param = ""
    for name in PARAM_NAMES:
        sample = _as_flat(generated["params"][name])
        ref = _as_flat(reference["params"][name])
        sample = sample[np.isfinite(sample)]
        ref = ref[np.isfinite(ref)]
        if sample.size == 0 or ref.size == 0:
            continue
        sample_mean = float(sample.mean())
        ref_mean = float(ref.mean())
        sample_std = float(sample.std())
        ref_std = float(ref.std())
        denominator = max(abs(ref_mean), 1.0e-12)
        mean_rel_diff = float((sample_mean - ref_mean) / denominator)
        std_ratio = float(sample_std / ref_std) if ref_std > 1.0e-12 else float("nan")
        if math.isfinite(std_ratio):
            std_ratios.append(std_ratio)
        if abs(mean_rel_diff) > max_mean_diff:
            max_mean_diff = abs(mean_rel_diff)
            worst_param = name
        # 按实体去均值后的标准差比值：剔除实体间差异（如链路距离），与场景设定无关
        sample_matrix = np.asarray(generated["params"][name], dtype=np.float64)
        ref_matrix = np.asarray(reference["params"][name], dtype=np.float64)
        if sample_matrix.ndim == 2 and ref_matrix.ndim == 2:
            sample_within_std = float(
                (sample_matrix - sample_matrix.mean(axis=1, keepdims=True)).std()
            )
            ref_within_std = float((ref_matrix - ref_matrix.mean(axis=1, keepdims=True)).std())
        else:
            sample_within_std = float("nan")
            ref_within_std = float("nan")
        within_ratio = (
            float(sample_within_std / ref_within_std)
            if ref_within_std > 1.0e-12
            else float("nan")
        )
        if math.isfinite(within_ratio):
            within_ratios.append(within_ratio)
        per_param[name] = {
            "generated_mean": sample_mean,
            "reference_mean": ref_mean,
            "mean_relative_difference": mean_rel_diff,
            "generated_std": sample_std,
            "reference_std": ref_std,
            "std_ratio": None if not math.isfinite(std_ratio) else std_ratio,
            "generated_within_std": sample_within_std,
            "reference_within_std": ref_within_std,
            "within_std_ratio": None if not math.isfinite(within_ratio) else within_ratio,
            "ks_statistic": _ks_statistic(sample, ref),
            "generated_p05": float(np.percentile(sample, 5)),
            "generated_p95": float(np.percentile(sample, 95)),
            "reference_p05": float(np.percentile(ref, 5)),
            "reference_p95": float(np.percentile(ref, 95)),
        }
    return {
        "params": per_param,
        "summary": {
            "max_abs_mean_relative_difference": float(max_mean_diff),
            "worst_param": worst_param,
            "median_std_ratio": float(np.median(std_ratios)) if std_ratios else float("nan"),
            "median_within_std_ratio": (
                float(np.median(within_ratios)) if within_ratios else float("nan")
            ),
            "max_ks_statistic": float(
                max(item["ks_statistic"] for item in per_param.values())
            ) if per_param else float("nan"),
        },
        "note": (
            "参考数据为同口径 physics 仿真序列；std_ratio 含场景设定带来的实体间差异，"
            "within_std_ratio 为按实体去均值后的比值，更接近“同一链路内动态是否被还原”。"
            "差异用于判断生成器是否压平了分布，不作为判定项。"
        ),
    }


# ---------------------------------------------------------------------------
# 证据落盘
# ---------------------------------------------------------------------------
def _format_value(value: Any, digits: int = 6) -> str:
    """把数值格式化成适合 Markdown 的字符串。"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        if abs(value) >= 1.0e4 or (abs(value) > 0.0 and abs(value) < 1.0e-3):
            return f"{value:.4e}"
        return f"{value:.{digits}f}"
    return str(value)


def render_validation_markdown(
    validation: Mapping[str, Any],
    dataset: Optional[Mapping[str, Any]] = None,
    distribution: Optional[Mapping[str, Any]] = None,
) -> str:
    """把校验结果渲染成中文 Markdown 报告。"""
    provenance = (dataset or {}).get("provenance", {}) or validation.get("provenance", {})
    lines: List[str] = [
        "# 生成式神经网络数据校验报告",
        "",
        f"- 校验时间（UTC）：{validation.get('validated_at_utc', '')}",
        f"- 生成器版本：{provenance.get('generator_version', validation.get('provenance', {}).get('generator_version', ''))}",
        f"- 生成器权重 sha256：{provenance.get('generator_weight_sha256', '')}",
        f"- FeatureContract：{provenance.get('feature_contract_id', validation.get('provenance', {}).get('feature_contract_id', ''))}",
        f"- 数据来源：source_kind={provenance.get('source_kind', 'synthetic')}，seed={provenance.get('seed', validation.get('provenance', {}).get('seed'))}",
        f"- 训练数据来源：{provenance.get('training_data_provenance', {})}",
        f"- 对应 physics 版本 sha256（训练数据）：{(provenance.get('training_data_provenance', {}) or {}).get('physics_module', {}).get('sha256', '') or validation.get('provenance', {}).get('physics_module_sha256', '')}",
        f"- 本次校验所用 physics sha256（参考数据）：{(validation.get('physics_module_current', {}) or {}).get('sha256', '')}",
        f"- 规模：{validation.get('n_entities')} 个实体系列 × {validation.get('n_steps')} 步 × {validation.get('n_params')} 参数",
        f"- 结论：**{validation.get('status', '')}**（passed={validation.get('passed')}）",
        f"- 最大违规率：{_format_value(validation.get('violation_rate'))}",
        f"- 边界投影修改率：{_format_value(validation.get('postprocess', {}).get('bound_projection_rate'))}",
        "",
        "## 一、逐项校验",
        "",
        "| 编号 | 分组 | 说明 | 指标 | 实测 | 阈值 | 结论 |",
        "|---|---|---|---|---|---|---|",
    ]
    for check in validation.get("checks", []):
        threshold = f"{check.get('comparison', '')} {_format_value(check.get('threshold'))}"
        lines.append(
            "| {id} | {group} | {desc} | {metric} | {value} | {threshold} | {status} |".format(
                id=check.get("id", ""),
                group=check.get("group", ""),
                desc=check.get("description", ""),
                metric=check.get("metric", ""),
                value=_format_value(check.get("value")),
                threshold=threshold,
                status=check.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 二、参考数据（physics）自查（对照栏）",
            "",
        ]
    )
    reference_validation = validation.get("reference_validation")
    if reference_validation:
        reference_provenance = reference_validation.get("provenance", {}) or {}
        lines.extend(
            [
                f"- 参考数据来源：source_kind={reference_provenance.get('source_kind', '')}，"
                f"seed={reference_provenance.get('seed')}，"
                f"规模={reference_validation.get('n_entities')} × {reference_validation.get('n_steps')}，"
                f"physics sha256={reference_provenance.get('physics_module', {}).get('sha256', '')}",
                f"- 参考数据结论：**{reference_validation.get('status', '')}**"
                f"（passed={reference_validation.get('passed')}）",
                "",
                "| 编号 | 说明 | 指标 | 实测 | 阈值 | 结论 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for check in reference_validation.get("checks", []):
            lines.append(
                "| {id} | {desc} | {metric} | {value} | {threshold} | {status} |".format(
                    id=check.get("id", ""),
                    desc=check.get("description", ""),
                    metric=check.get("metric", ""),
                    value=_format_value(check.get("value")),
                    threshold=f"{check.get('comparison', '')} {_format_value(check.get('threshold'))}",
                    status=check.get("status", ""),
                )
            )
        lines.extend(
            [
                "",
                "参考数据相关系数：",
                json.dumps(reference_validation.get("correlations", {}), ensure_ascii=False),
            ]
        )
    else:
        lines.append("未获取参考数据，跳过对照。")
    lines.extend(
        [
            "",
            "## 三、相关性系数",
            "",
            "| 关系 | 逐链路 Pearson 均值（判定值） | 并列样本 Pearson | 阈值 | 结论 |",
            "|---|---|---|---|---|",
        ]
    )
    thresholds = validation.get("thresholds", {})
    checks_by_correlation = {
        "snr_vs_rx_power": "snr_rx_positive_correlation",
        "ber_vs_snr": "ber_snr_negative_correlation",
        "rate_vs_snr": "rate_snr_positive_correlation",
        "pointing_vs_path_loss": "pointing_path_loss_positive_correlation",
        "vibration_vs_pointing": "vibration_pointing_positive_correlation",
    }
    status_map = {check["id"]: check["status"] for check in validation.get("checks", [])}
    per_entity_correlations = validation.get("correlations_per_entity_mean", {}) or {}
    for key, value in (validation.get("correlations", {}) or {}).items():
        lines.append(
            "| {key} | {value} | {pooled} | {threshold} | {status} |".format(
                key=key,
                value=_format_value(per_entity_correlations.get(key)),
                pooled=_format_value(value),
                threshold=_format_value(thresholds.get(key)),
                status=status_map.get(checks_by_correlation.get(key, ""), ""),
            )
        )
    lines.extend(["", "## 四、违规率", "", "| 项目 | 违规率 |", "|---|---|"])
    for key, value in (validation.get("violation_rates", {}) or {}).items():
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## 五、按实体去均值的补充相关性（诊断用）",
            "",
            "| 关系 | Pearson（within-entity） | 备注 |",
            "|---|---|---|",
        ]
    )
    for key, payload in (validation.get("within_entity_correlations", {}) or {}).items():
        lines.append(
            f"| {key} | {_format_value(payload.get('pearson'))} | {payload.get('reason', '')} |"
        )
    if distribution:
        lines.extend(
            [
                "",
                "## 六、与物理参考数据的分布对比",
                "",
                f"- 参数均值相对偏差最大：{_format_value(distribution['summary'].get('max_abs_mean_relative_difference'))}"
                f"（{distribution['summary'].get('worst_param', '')}）",
                f"- 标准差比值中位数：{_format_value(distribution['summary'].get('median_std_ratio'))}",
                f"- 按实体去均值的标准差比值中位数：{_format_value(distribution['summary'].get('median_within_std_ratio'))}",
                f"- 最大 KS 统计量：{_format_value(distribution['summary'].get('max_ks_statistic'))}",
                "",
                "| 参数 | 生成均值 | 参考均值 | 相对偏差 | 标准差比 | 实体去均值标准差比 | KS 统计量 |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for name, item in distribution.get("params", {}).items():
            lines.append(
                "| {name} | {gm} | {rm} | {diff} | {ratio} | {within} | {ks} |".format(
                    name=name,
                    gm=_format_value(item.get("generated_mean")),
                    rm=_format_value(item.get("reference_mean")),
                    diff=_format_value(item.get("mean_relative_difference")),
                    ratio=_format_value(item.get("std_ratio")),
                    within=_format_value(item.get("within_std_ratio")),
                    ks=_format_value(item.get("ks_statistic")),
                )
            )
    lines.extend(
        [
            "",
            "## 七、说明",
            "",
            f"- {validation.get('supplementary_note', '')}",
            "- 逐项校验表里的“实测”为判定口径（相关性项为逐链路 Pearson 均值，边界项为违规率）；参考数据对照见第二节。",
            f"- {validation.get('criterion_note', '')}",
            "- 无方差导致相关系数不可定义的项会标记为 NOT_MEASURABLE，不伪造数值。",
            "- 边界投影只按物理可行区间裁剪，修改率已在上表如实记录。",
            "",
        ]
    )
    return "\n".join(lines)

def save_generated_dataset(
    dataset: Mapping[str, Any],
    out_dir: Optional[Any] = None,
    validation: Optional[Mapping[str, Any]] = None,
    distribution: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """把生成数据集与校验结果落到 ``evidence/generative/``。

    产物：``generated_dataset.npz``（参数 + 条件 + provenance）、
    ``validation.json``、``validation.md``。
    """
    output = Path(out_dir) if out_dir else EVIDENCE_DIR
    output.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        arrays[f"param__{name}"] = np.asarray(dataset["params"][name], dtype=np.float32)
    for key, value in (dataset.get("conditions", {}) or {}).items():
        array = np.asarray(value)
        arrays[f"cond__{key}"] = array if array.dtype != object else array.astype(np.float32)
    arrays["meta_json"] = np.array(
        json.dumps(
            {
                "provenance": dataset.get("provenance", {}),
                "spec": dataset.get("spec", {}),
                "shapes": dataset.get("shapes", {}),
            },
            ensure_ascii=False,
        )
    )
    dataset_path = output / "generated_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    paths: Dict[str, str] = {"dataset": str(dataset_path)}

    if validation is not None:
        payload: Dict[str, Any] = dict(validation)
        if distribution is not None:
            payload["distribution_comparison"] = distribution
        validation_path = output / "validation.json"
        validation_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path = output / "validation.md"
        report_path.write_text(
            render_validation_markdown(validation, dataset, distribution), encoding="utf-8"
        )
        paths["validation_json"] = str(validation_path)
        paths["validation_markdown"] = str(report_path)
    return paths
