"""在线推理封装：给引擎/API 调用的多目标批量预测器。

用法
----
>>> predictor = MultiTargetPredictor(model_dir="models/predictor")
>>> result = predictor.predict(history_window, entity_ids, horizon=5)
>>> result["predictions"][0]
{'entity_type': 'link', 'entity_id': '3600->3601', 'target': 'achievable_rate_gbps',
 'unit': 'Gbps', 'values': [...], 'valid_mask': [1, 1, 1, 1, 1]}

输入
----
``history_window`` 形状 ``(实体数, window, feat_dim)``：

* 若 ``feat_dim == 21``：按 :class:`FeatureContract` 的完整特征顺序（16 观测 + 5 时间）；
* 若 ``feat_dim == 16``：只给观测通道，预测器用 ``sim_time_s`` 现算时间特征。

``sim_time_s`` 为场景当地秒数（与训练数据的 ``t0_s + step*dt`` 口径一致）；
不传时按 0 处理，时间特征退化为常量相位，但观测通道仍参与预测。

批量
----
默认 ``max_batch=65536``：全部实体一次前向。若显存不足会自动降级分块，
并在返回结构的 ``oom_fallback`` 中如实标记。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from dataset import (
    MODEL_DIR,
    SECONDS_PER_DAY,
    FeatureContract,
    Standardizer,
    build_derived_features,
    clip_predictions,
    target_base_from_features,
)
from model import MultiTargetSpatioTemporalNet, resolve_device
from train import load_model_bundle


class MultiTargetPredictor:
    """多目标、多步在线预测器（单模型、批量推理）。"""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "auto",
        max_batch: int = 65536,
    ) -> None:
        bundle = load_model_bundle(Path(model_dir) if model_dir else MODEL_DIR, device=device)
        self.model: MultiTargetSpatioTemporalNet = bundle["model"]
        self.contract: FeatureContract = bundle["contract"]
        self.scaler: Standardizer = bundle["scaler"]
        self.meta: Dict[str, Any] = bundle["meta"]
        self.model_version: str = bundle["model_version"]
        self.device: torch.device = bundle["device"]
        self.parameter_count: int = bundle["parameter_count"]
        self.max_batch = int(max_batch)
        self.model.eval()

    # -- 特征组装 -----------------------------------------------------
    def _time_features(self, sim_time_s: Any, n_entities: int, window: int) -> np.ndarray:
        """构造 (N, window, 5) 时间特征；``sim_time_s`` 可为标量或每实体数组。"""
        if sim_time_s is None:
            seconds = np.zeros(n_entities, dtype=np.float64)
        else:
            seconds = np.asarray(sim_time_s, dtype=np.float64).reshape(-1)
            if seconds.size == 1:
                seconds = np.repeat(seconds, n_entities)
            if seconds.size != n_entities:
                raise ValueError(f"sim_time_s 长度 {seconds.size} 与实体数 {n_entities} 不一致")
        tod = np.mod(seconds, SECONDS_PER_DAY) / SECONDS_PER_DAY * 2.0 * math.pi
        orbit = np.mod(seconds / float(self.contract.orbit_period_s), 1.0) * 2.0 * math.pi
        global_part = np.stack(
            [np.sin(tod), np.cos(tod), np.sin(orbit), np.cos(orbit)], axis=1
        )  # (N, 4)
        age = ((window - 1 - np.arange(window, dtype=np.float64)) / float(max(window - 1, 1)))[None, :]
        block = np.empty((n_entities, window, 5), dtype=np.float32)
        block[:, :, 0] = age.astype(np.float32)
        block[:, :, 1:] = global_part[:, None, :].astype(np.float32)
        return block

    def _assemble_features(self, history: np.ndarray, sim_time_s: Any) -> np.ndarray:
        n_entities, window, channels = history.shape
        if window != self.contract.window:
            raise ValueError(f"历史窗口长度必须为 {self.contract.window}，实际 {window}")
        observed_count = len(self.contract.observed_feature_names)
        if channels == self.contract.feature_dim:
            return history.astype(np.float32, copy=False)
        if channels == observed_count:
            derived = build_derived_features(history, self.contract)
            return np.concatenate(
                [
                    history.astype(np.float32, copy=False),
                    derived,
                    self._time_features(sim_time_s, n_entities, window),
                ],
                axis=2,
            )
        raise ValueError(
            f"输入特征维度必须为 {observed_count}（仅观测）或 {self.contract.feature_dim}（完整契约），实际 {channels}"
        )

    # -- 前向 ---------------------------------------------------------
    def _forward(self, scaled: np.ndarray, base: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32)).to(self.device)
            base_tensor = torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)).to(self.device)
            output = self.model(tensor, base_tensor)
            return output.detach().cpu().numpy()

    def _forward_chunked(self, scaled: np.ndarray, base: np.ndarray, chunk: int) -> np.ndarray:
        outputs: List[np.ndarray] = []
        for start in range(0, scaled.shape[0], chunk):
            outputs.append(self._forward(scaled[start : start + chunk], base[start : start + chunk]))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros(
            (0, self.contract.horizon, self.contract.n_targets), dtype=np.float32
        )

    # -- 对外接口 -----------------------------------------------------
    def predict(
        self,
        history_window: np.ndarray,
        entity_ids: Sequence[str],
        horizon: Optional[int] = None,
        entity_types: Optional[Sequence[str]] = None,
        sim_time_s: Any = None,
    ) -> Dict[str, Any]:
        """批量预测。

        :param history_window: ``(N, window, feat_dim)`` 观测历史
        :param entity_ids: 长度 N 的实体 ID（节点 "3600"，链路 "3600->3601"）
        :param horizon: 预测步数，默认取契约 horizon；不得超过契约
        :param entity_types: 可选，显式指定 "link"/"node"；缺省按 ID 是否含 "->" 判断
        :param sim_time_s: 场景当地秒数（标量或每实体数组），用于时间特征
        """
        started = time.perf_counter()
        history = np.asarray(history_window)
        if history.ndim != 3:
            raise ValueError(f"history_window 必须为 (N, window, feat_dim)，实际 {history.shape}")
        n_entities = int(history.shape[0])
        entity_ids = [str(value) for value in entity_ids]
        if len(entity_ids) != n_entities:
            raise ValueError(f"entity_ids 长度 {len(entity_ids)} 与样本数 {n_entities} 不一致")
        horizon_steps = int(horizon if horizon is not None else self.contract.horizon)
        if horizon_steps <= 0 or horizon_steps > self.contract.horizon:
            raise ValueError(f"horizon 必须在 1..{self.contract.horizon} 之间，实际 {horizon_steps}")

        features = self._assemble_features(history, sim_time_s)
        finite_entities = np.isfinite(features).all(axis=(1, 2))
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self.scaler.transform_features(features)
        base = target_base_from_features(features, self.scaler, self.contract)

        forward_started = time.perf_counter()
        oom_fallback = False
        batch_used = n_entities
        if n_entities <= self.max_batch:
            try:
                predictions_scaled = self._forward(scaled, base)
            except RuntimeError as exc:  # 显存不足时降级分块，并如实记录
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                batch_used = min(4096, n_entities)
                predictions_scaled = self._forward_chunked(scaled, base, batch_used)
                oom_fallback = True
        else:
            batch_used = self.max_batch
            predictions_scaled = self._forward_chunked(scaled, base, batch_used)
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        predictions = self.scaler.inverse_targets(predictions_scaled)
        predictions = clip_predictions(predictions, self.contract)
        if entity_types is None:
            types = ["link" if "->" in entity_id else "node" for entity_id in entity_ids]
        else:
            types = [str(value) for value in entity_types]
            if len(types) != n_entities:
                raise ValueError("entity_types 长度与实体数不一致")

        horizon_offsets = list(self.contract.offsets_ms[:horizon_steps])
        prediction_rows: List[Dict[str, Any]] = []
        for entity_index, (entity_id, entity_type) in enumerate(zip(entity_ids, types)):
            if entity_type not in ("link", "node"):
                raise ValueError(f"未知 entity_type={entity_type}")
            valid = 1 if bool(finite_entities[entity_index]) else 0
            for target_index, target in enumerate(self.contract.target_names):
                if self.contract.target_entity_types[target_index] != entity_type:
                    continue
                values = predictions[entity_index, :horizon_steps, target_index]
                prediction_rows.append(
                    {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "target": target,
                        "unit": self.contract.target_units[target_index],
                        "values": [float(value) for value in values],
                        "valid_mask": [valid] * horizon_steps,
                    }
                )
        inference_ms = (time.perf_counter() - started) * 1000.0
        return {
            "model_version": self.model_version,
            "feature_contract_id": self.contract.contract_id,
            "horizon_offsets_ms": horizon_offsets,
            "predictions": prediction_rows,
            "inference_ms": float(inference_ms),
            "forward_ms": float(forward_ms),
            "n_entities": n_entities,
            "n_predictions": len(prediction_rows),
            "device": str(self.device),
            "batch_size_used": int(batch_used),
            "oom_fallback": bool(oom_fallback),
            "step_ms": int(self.contract.step_ms),
        }

    def capabilities(self) -> Dict[str, Any]:
        """模型能力声明，供 API/健康检查使用。"""
        return {
            "model_version": self.model_version,
            "feature_contract_id": self.contract.contract_id,
            "model_type": "LSTM(2x128)+MultiheadAttention(4 heads)+MLP head",
            "parameter_count": int(self.parameter_count),
            "device": str(self.device),
            "window": int(self.contract.window),
            "horizon": int(self.contract.horizon),
            "step_ms": int(self.contract.step_ms),
            "horizon_offsets_ms": list(self.contract.offsets_ms),
            "targets": [
                {
                    "name": name,
                    "unit": unit,
                    "entity_type": entity_type,
                }
                for name, unit, entity_type in zip(
                    self.contract.target_names, self.contract.target_units, self.contract.target_entity_types
                )
            ],
            "feature_names": list(self.contract.feature_names),
            "feature_units": list(self.contract.feature_units),
            "max_batch": int(self.max_batch),
            "supports_missing_mask": True,
            "jitter_mode": "composed" if bool(getattr(self.model, "use_composed_jitter", False)) else "direct",
            "aux_targets": ["queue_occupancy_ratio", "vibration_rms_g"]
            if getattr(self.model, "aux_heads", None) is not None
            else [],
        }
