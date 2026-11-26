"""多目标时空预测网络：LSTM 编码 + 多头自注意力 + 多步多目标输出头。

结构
----
``输入 (B, window, feat_dim) -> LayerNorm -> 2 层 LSTM(hidden=128)``
``-> 以最后一步为 query 对全窗口做多头自注意力(num_heads=4)``
``-> 拼接 [最后隐状态, 注意力上下文] -> MLP 头 -> (B, horizon, 7)``

任务
----
固定 7 个目标（顺序见 :mod:`src.prediction.dataset`），其中 ``node_load_ratio``
是节点目标，其余 6 个是链路目标。训练使用按目标加权的 SmoothL1(Huber) 损失，
速率/抖动/负载三项同权；被掩码的目标（实体类型不适用）不计入损失。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#: 可由物理公式合成的目标：索引 → compose_targets 返回的键名
_COMPOSED_NAME_BY_INDEX: Dict[int, str] = {
    0: "achievable_rate_gbps",
    1: "delay_jitter_ms",
    3: "rx_power_dbm",
    4: "ber_pre_fec",
    5: "packet_loss_rate",
    6: "link_delay_ms",
}


def set_global_seed(seed: int) -> None:
    """固定 Python / NumPy / Torch 随机种子（含 CUDA）。"""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(preference: str = "auto") -> torch.device:
    """解析训练/推理设备；``auto`` 在 CUDA 架构不受支持时回退 CPU。"""
    preference = str(preference).lower()
    if preference == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, minor = torch.cuda.get_device_capability()
        current_arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if supported and current_arch not in supported:
            return torch.device("cpu")
    except Exception:  # noqa: BLE001 - 设备探测失败时保守回退
        return torch.device("cpu")
    return torch.device("cuda")


@dataclass
class ModelConfig:
    """网络超参数；随 checkpoint 一起保存，推理时按保存值重建。"""

    input_dim: int = 21
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    horizon: int = 5
    n_targets: int = 7
    dropout: float = 0.1
    head_hidden: int = 128
    enable_aux: bool = True
    use_composed_jitter: bool = False
    #: 物理合成模式：
    #:   ``"none"``  —— 7 个目标全部由输出头直接回归；
    #:   ``"jitter"``—— 仅抖动由 (q, vib) 合成，其余直接回归；
    #:   ``"full"``  —— 6 个目标由 (path_loss, q, vib) 经物理公式合成，仅 node_load 直接回归。
    composition_mode: str = "full"
    #: 辅助隐变量个数（与 dataset.AUX_TARGET_NAMES 长度一致）
    aux_dim: int = 3
    #: 输入归一化方式：
    #:   ``"layer"`` —— LayerNorm（跨通道归一化，会抹掉物理量之间的量级差）；
    #:   ``"none"``  —— 不做，特征已由 scaler 逐通道标准化，保留物理量纲关系。
    input_norm: str = "none"
    #: 逐目标合成开关（7 个布尔）。为空时按 composition_mode 全开/全关；
    #: 训练结束时用验证集逐目标比较“直接回归 vs 物理合成”，只对确实更优的目标启用。
    composed_target_mask: Tuple[bool, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            horizon=int(payload["horizon"]),
            n_targets=int(payload["n_targets"]),
            dropout=float(payload.get("dropout", 0.1)),
            head_hidden=int(payload.get("head_hidden", 128)),
            enable_aux=bool(payload.get("enable_aux", False)),
            use_composed_jitter=bool(payload.get("use_composed_jitter", False)),
            composition_mode=str(payload.get("composition_mode", "none")),
            aux_dim=int(payload.get("aux_dim", 2)),
            input_norm=str(payload.get("input_norm", "layer")),
            composed_target_mask=tuple(payload.get("composed_target_mask", ())),
        )


class MultiTargetSpatioTemporalNet(nn.Module):
    """多目标、多步时空预测网络（LSTM + 多头自注意力 + 输出头）。"""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")

        self.input_norm = (
            nn.LayerNorm(cfg.input_dim) if str(cfg.input_norm) == "layer" else nn.Identity()
        )
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(cfg.hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        # 每个目标独立输出头：共享 LSTM/注意力表征，降低多目标相互干扰。
        self.output_heads = nn.ModuleList(
            [nn.Linear(cfg.head_hidden, cfg.horizon) for _ in range(cfg.n_targets)]
        )
        # 辅助头：预测物理合成所需的隐变量
        # aux 顺序：0=queue_occupancy_ratio，1=vibration_rms_g，2=path_loss_db
        self.aux_heads: Optional[nn.ModuleList] = None
        if cfg.enable_aux:
            self.aux_heads = nn.ModuleList(
                [nn.Linear(cfg.head_hidden, cfg.horizon) for _ in range(max(1, cfg.aux_dim))]
            )
        self.use_composed_jitter = bool(cfg.use_composed_jitter)
        self.composition_mode = str(cfg.composition_mode)
        self._composition_ready = False
        self._feature_stats_ready = False
        self._feature_mean = torch.zeros(cfg.input_dim)
        self._feature_std = torch.ones(cfg.input_dim)
        self._aux_mean = torch.zeros(max(1, cfg.aux_dim))
        self._aux_std = torch.ones(max(1, cfg.aux_dim))
        self._target_mean = torch.zeros(cfg.n_targets)
        self._target_std = torch.ones(cfg.n_targets)

    def set_composition_stats(
        self,
        aux_mean: Any,
        aux_std: Any,
        target_mean: Any,
        target_std: Any,
        feature_mean: Any = None,
        feature_std: Any = None,
    ) -> None:
        """设置物理合成所需的均值/方差（来自训练集 scaler）。

        :param feature_mean/feature_std: 特征标准化参数。模型输入是**标准化后**的特征，
            而物理合成需要原始物理量（dBm / ms / ratio），因此必须能反标准化。
        """
        aux_count = self._aux_mean.numel()
        self._aux_mean = torch.as_tensor(aux_mean, dtype=torch.float32).reshape(-1)[:aux_count]
        self._aux_std = torch.as_tensor(aux_std, dtype=torch.float32).reshape(-1)[:aux_count].clamp_min(1e-8)
        self._target_mean = torch.as_tensor(target_mean, dtype=torch.float32).reshape(-1)
        self._target_std = torch.as_tensor(target_std, dtype=torch.float32).reshape(-1).clamp_min(1e-8)
        if feature_mean is not None and feature_std is not None:
            self._feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(-1)
            self._feature_std = (
                torch.as_tensor(feature_std, dtype=torch.float32).reshape(-1).clamp_min(1e-8)
            )
            self._feature_stats_ready = True
        self._composition_ready = True

    def _composed_mask_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """逐目标合成掩码；未显式配置时按 composition_mode 全开。"""
        configured = tuple(getattr(self.config, "composed_target_mask", ()) or ())
        if len(configured) == self.config.n_targets:
            return torch.tensor([1.0 if flag else 0.0 for flag in configured], dtype=dtype, device=device)
        mask = torch.zeros(self.config.n_targets, dtype=dtype, device=device)
        for index in _COMPOSED_NAME_BY_INDEX:
            mask[index] = 1.0
        return mask

    def _feature_at(self, features: torch.Tensor, index: int) -> torch.Tensor:
        """取窗口末时刻指定特征的**原始物理值**（反标准化）。"""
        raw = features[:, -1, index]
        if not self._feature_stats_ready:
            raise RuntimeError(
                "物理合成需要特征标准化参数，请用 set_composition_stats 传入 feature_mean/feature_std"
            )
        mean = self._feature_mean.to(raw.device)[index]
        std = self._feature_std.to(raw.device)[index]
        return raw * std + mean

    def _compose_jitter(self, aux_scaled: torch.Tensor) -> torch.Tensor:
        """按 physics 公式由 q_hat / vib_hat 合成 jitter（返回标准化空间）。"""
        device = aux_scaled.device
        aux_mean = self._aux_mean.to(device)
        aux_std = self._aux_std.to(device)
        target_mean = self._target_mean.to(device)
        target_std = self._target_std.to(device)
        q_hat = aux_scaled[..., 0] * aux_std[0] + aux_mean[0]
        vib_hat = aux_scaled[..., 1] * aux_std[1] + aux_mean[1]
        jitter_raw = 0.04 + 1.60 * q_hat * q_hat + 0.60 * vib_hat
        # jitter 在目标顺序中索引为 1（固定 7 目标顺序）
        return (jitter_raw - target_mean[1]) / target_std[1]

    def _compose_all(self, features: torch.Tensor, aux_scaled: torch.Tensor) -> torch.Tensor:
        """用 (path_loss, q, vib) 合成 6 个目标，返回标准化空间的 (B, horizon, n_targets)。

        未被合成的通道（node_load）置 0，由调用方保留直接头结果。
        """
        from physics_composition import (
            COMPOSED_TARGET_INDICES,
            FEAT_LINK_DELAY,
            FEAT_QUEUE,
            FEAT_TX_POWER,
            compose_targets,
            propagation_delay_ms,
        )

        device = aux_scaled.device
        aux_mean = self._aux_mean.to(device)
        aux_std = self._aux_std.to(device)
        target_mean = self._target_mean.to(device)
        target_std = self._target_std.to(device)

        q_hat = aux_scaled[..., 0] * aux_std[0] + aux_mean[0]
        vib_hat = aux_scaled[..., 1] * aux_std[1] + aux_mean[1]
        path_loss_hat = aux_scaled[..., 2] * aux_std[2] + aux_mean[2]

        # 上下文取自窗口末时刻的原始物理量（模型输入是标准化特征，需反标准化）
        tx_power = self._feature_at(features, FEAT_TX_POWER).unsqueeze(1)
        prop_delay = propagation_delay_ms(
            self._feature_at(features, FEAT_LINK_DELAY),
            self._feature_at(features, FEAT_QUEUE),
        ).unsqueeze(1)

        composed = compose_targets(
            path_loss_db=path_loss_hat,
            queue_ratio=q_hat,
            vibration_g=vib_hat,
            tx_power_dbm=tx_power,
            propagation_delay_ms=prop_delay,
        )

        batch, horizon = aux_scaled.shape[0], aux_scaled.shape[1]
        output = torch.zeros(
            (batch, horizon, self.config.n_targets), dtype=aux_scaled.dtype, device=device
        )
        for index in COMPOSED_TARGET_INDICES:
            name = _COMPOSED_NAME_BY_INDEX[index]
            output[..., index] = (composed[name] - target_mean[index]) / target_std[index]
        return output

    def forward(
        self,
        features: torch.Tensor,
        base_values: Optional[torch.Tensor] = None,
        use_composed_jitter: Optional[bool] = None,
        return_aux: bool = False,
    ) -> Any:
        """``features``: (B, window, feat_dim) -> (B, horizon, n_targets)。

        :param base_values: 可选 (B, n_targets)，标准化空间的“最后观测值”。
            给出时输出为 ``base + 残差``（多步残差预测），使模型以 persistence
            为初始化，再由网络学习修正量；不给时输出纯网络结果。
        :param use_composed_jitter: 是否用物理合成头替换直接头的 jitter 通道；
            缺省读取 ``self.use_composed_jitter``（由验证集选择后写入）。
        :param return_aux: 为 True 时返回 ``(main, aux, composed_jitter)``。
        """
        if features.dim() != 3:
            raise ValueError(f"期望输入 (batch, window, feat_dim)，实际 {tuple(features.shape)}")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"特征维度 {features.shape[-1]} 与模型契约 {self.config.input_dim} 不一致"
            )
        normalized = self.input_norm(features)
        hidden, _ = self.lstm(normalized)
        last_step = hidden[:, -1:, :]
        context, _ = self.attention(last_step, hidden, hidden, need_weights=False)
        pooled = torch.cat(
            [last_step, self.attention_norm(last_step + context), hidden.mean(dim=1, keepdim=True)],
            dim=-1,
        )
        representation = self.trunk(pooled.squeeze(1))
        output = torch.stack([head(representation) for head in self.output_heads], dim=-1)
        if base_values is not None:
            if base_values.shape != (features.shape[0], self.config.n_targets):
                raise ValueError(
                    f"base_values 形状应为 {(features.shape[0], self.config.n_targets)}，实际 {tuple(base_values.shape)}"
                )
            output = output + base_values.unsqueeze(1)

        # 直接头结果（未被物理合成覆盖的版本）。物理合成是绝对覆盖，若训练时只对
        # 合成结果求损失，直接头在这些通道上得不到梯度、无法作为“退回方案”。
        # 因此训练期用 self.last_direct_output 取它并额外加一项辅助损失。
        direct_output = output
        self.last_direct_output = direct_output
        aux_scaled: Optional[torch.Tensor] = None
        composed: Optional[torch.Tensor] = None
        if self.aux_heads is not None:
            aux_scaled = torch.stack([head(representation) for head in self.aux_heads], dim=-1)
            if self._composition_ready:
                mode = str(getattr(self, "composition_mode", "none"))
                if mode == "full" and aux_scaled.shape[-1] >= 3:
                    composed_full = self._compose_all(features, aux_scaled)
                    mask = self._composed_mask_tensor(output.device, output.dtype)
                    output = output * (1.0 - mask.view(1, 1, -1)) + composed_full * mask.view(1, 1, -1)
                    composed = composed_full[..., 1]
                else:
                    composed = self._compose_jitter(aux_scaled)
                    use_composed = (
                        self.use_composed_jitter if use_composed_jitter is None else bool(use_composed_jitter)
                    )
                    if use_composed:
                        output = output.clone()
                        output[..., 1] = composed
        if return_aux:
            return output, aux_scaled, composed
        return output


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """统计参数量。"""
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return int(sum(parameter.numel() for parameter in parameters))


def masked_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """按目标加权的掩码 SmoothL1 损失。

    :param predictions: (B, H, T) 标准化空间预测
    :param targets: (B, H, T) 标准化空间标签
    :param mask: (B, T) 目标有效掩码（1 有效 / 0 不适用）
    :param weights: (T,) 目标权重；速率/抖动/负载同权
    """
    elementwise = F.smooth_l1_loss(predictions, targets, reduction="none", beta=beta)
    weight = mask.unsqueeze(1) * weights.view(1, 1, -1)
    return (elementwise * weight).sum() / weight.sum().clamp_min(1.0)


def build_model(config: Optional[ModelConfig] = None) -> MultiTargetSpatioTemporalNet:
    """构造网络。"""
    return MultiTargetSpatioTemporalNet(config)
