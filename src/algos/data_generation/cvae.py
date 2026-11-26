"""条件变分自编码器（Conditional VAE）——多层参数生成网络。

本模块只用 ``torch`` 实现，不依赖工程内其他模块，也不调用任何解析物理公式：
生成阶段由网络在条件向量下自回归自由演化，因此输出是**网络推理结果**，
不是随机规则、查表或均匀噪声。

网络结构（技术要求固定项）
--------------------------
* 编码器：``Linear -> ReLU -> Linear -> ReLU``，再分 ``Linear(mu)`` / ``Linear(logvar)`` 两个头；
* 解码器：``Linear -> ReLU -> Linear -> ReLU -> Linear(x_dim)``；
* 隐变量 ``latent_dim=8``，隐藏层宽度 ``hidden=128``（均可在 :class:`CVAEConfig` 覆盖）；
* 损失：``MSE(x, x_hat) + beta * KL(q(z|x,c) || p(z))``，``beta`` 支持 0→1 线性退火。

条件向量 ``c`` 的逐维含义由 :mod:`src.generative.dataset` 的
:class:`~src.generative.dataset.FeatureContract` 定义，本模块只要求 ``cond_dim`` 与之匹配。
``x`` 与 ``x_hat`` 都是**标准化后**的 12 维跨层参数，还原成物理量由
:mod:`src.generative.generate` 负责。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

#: 默认隐变量维度
DEFAULT_LATENT_DIM = 8
#: 默认隐藏层宽度
DEFAULT_HIDDEN = 128
#: 默认先验采样裁剪范围（防止尾部采样落到未见过的流形外）
DEFAULT_PRIOR_CLIP = 2.5


@dataclass(frozen=True)
class CVAEConfig:
    """网络超参数（会随权重一起落盘，加载时按此重建网络）。"""

    x_dim: int = 12
    cond_dim: int = 43
    latent_dim: int = DEFAULT_LATENT_DIM
    hidden: int = DEFAULT_HIDDEN

    def to_dict(self) -> Dict[str, int]:
        """转为可 JSON 序列化的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CVAEConfig":
        """从权重文件里的超参数字典恢复配置（忽略多余字段）。"""
        allowed = {item.name for item in fields(cls)}
        kwargs: Dict[str, int] = {}
        for key, value in dict(payload or {}).items():
            if key in allowed:
                kwargs[key] = int(value)
        return cls(**kwargs)


class MultilayerCVaE(nn.Module):
    """条件变分自编码器：``q(z|x,c)`` 编码、``p(x|z,c)`` 解码。

    :param config: 网络超参数；``cond_dim`` 必须与数据集一致。
    """

    def __init__(self, config: CVAEConfig = CVAEConfig()) -> None:
        super().__init__()
        self.config = config
        x_dim = int(config.x_dim)
        cond_dim = int(config.cond_dim)
        latent_dim = int(config.latent_dim)
        hidden = int(config.hidden)

        encoder_input = x_dim + cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, latent_dim)
        self.head_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    # ------------------------------------------------------------------
    # 前向与编码/解码
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回后验 ``mu``、``logvar``，形状均为 ``(batch, latent_dim)``。"""
        hidden = self.encoder(torch.cat([x, cond], dim=-1))
        return self.head_mu(hidden), self.head_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化采样：``z = mu + eps * std``，训练时保证梯度可回传。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """由隐变量与条件解码出标准化参数向量 ``x_hat``。"""
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(x_hat, mu, logvar)``。"""
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """逐样本 KL 散度（单位 nats），形状 ``(batch,)``。"""
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    @classmethod
    def loss(
        cls,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        free_bits: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 ``MSE(x, x_hat) + beta * KL``。

        :param free_bits: 每个隐变量维度的 KL 免罚下限（nats）。``>0`` 时只惩罚超过下限的
            那部分 KL，用于防止条件很强时后验完全塌缩、隐变量彻底失效（生成退化为
            确定性映射）。``0`` 表示标准 VAE 目标。
        :return: ``(total, recon, kl)`` 三个标量张量，便于日志分别记录；
            ``recon`` 是逐元素均方误差，``kl`` 是逐样本 KL 在批上的均值。
        """
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if float(free_bits) > 0.0:
            kl_vector = torch.clamp(kl_per_dim - float(free_bits), min=0.0).sum(dim=1)
        else:
            kl_vector = kl_per_dim.sum(dim=1)
        kl = kl_vector.mean()
        total = recon + float(beta) * kl
        return total, recon, kl

    # ------------------------------------------------------------------
    # 先验采样（生成入口）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,
        n_samples: int = 1,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        prior_clip: float = DEFAULT_PRIOR_CLIP,
        prior_mean: Optional[torch.Tensor] = None,
        prior_scale: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """从先验采样并解码。

        默认先验是标准正态 ``p(z)=N(0, I)``；当给出 ``prior_mean`` / ``prior_scale``
        时按“有效先验” ``N(prior_mean, prior_scale^2)`` 采样（生成端可用训练集聚合后验
        校准采样尺度，避免先验空洞导致输出过散）。

        :param conditions: ``(cond_dim,)`` 或 ``(n_cond, cond_dim)``
        :param n_samples: 每个条件的采样条数
        :param device: 采样设备，缺省用模型参数所在设备
        :param temperature: 采样温度（缩放先验噪声标准差）
        :param prior_clip: 先验噪声裁剪范围（以先验标准差为单位）；``<=0`` 表示不裁剪
        :param prior_mean: 可选的逐维先验均值 ``(latent_dim,)``
        :param prior_scale: 可选的逐维先验尺度 ``(latent_dim,)``
        :param generator: 可选的 ``torch.Generator``，用于固定采样随机源以保证可复现
        :return:
            * 输入为 1 维条件时返回 ``(n_samples, x_dim)``；
            * 输入为 2 维条件时返回 ``(n_samples, n_cond, x_dim)``。
        """
        target = device if device is not None else next(self.parameters()).device
        cond = torch.as_tensor(conditions, dtype=torch.float32, device=target)
        if cond.ndim == 1:
            cond_matrix = cond.unsqueeze(0)
            single_condition = True
        elif cond.ndim == 2:
            cond_matrix = cond
            single_condition = False
        else:
            raise ValueError(f"conditions 只支持 1 维或 2 维张量，收到 shape={tuple(cond.shape)}")

        latent_dim = int(self.config.latent_dim)
        batch = int(cond_matrix.shape[0])
        z = torch.randn(
            int(n_samples), batch, latent_dim, dtype=torch.float32, device=target,
            generator=generator,
        ) * float(temperature)
        if prior_clip and float(prior_clip) > 0:
            z = z.clamp(-float(prior_clip), float(prior_clip))
        if prior_scale is not None:
            z = z * torch.as_tensor(prior_scale, dtype=torch.float32, device=target)
        if prior_mean is not None:
            z = z + torch.as_tensor(prior_mean, dtype=torch.float32, device=target)
        z_flat = z.reshape(int(n_samples) * batch, latent_dim)
        cond_flat = cond_matrix.unsqueeze(0).expand(int(n_samples), batch, -1).reshape(
            int(n_samples) * batch, -1
        )
        sampled = self.decode(z_flat, cond_flat).reshape(
            int(n_samples), batch, int(self.config.x_dim)
        )
        if single_condition:
            return sampled.squeeze(0)
        return sampled

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """可训练参数总量。"""
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def describe_layers(self) -> Dict[str, Any]:
        """返回网络结构与形状摘要，用于报告与模型注册。"""
        return {
            "class": type(self).__name__,
            "config": self.config.to_dict(),
            "encoder_layers": [str(layer) for layer in self.encoder],
            "mu_layer": str(self.head_mu),
            "logvar_layer": str(self.head_logvar),
            "decoder_layers": [str(layer) for layer in self.decoder],
            "n_parameters": self.count_parameters(),
        }


#: 语义更明确的别名，便于外部按“条件 VAE”名称引用。
ConditionalVAE = MultilayerCVaE


def save_checkpoint(
    model: MultilayerCVaE,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存权重与超参数（其余产物由训练脚本分别落盘）。"""
    payload: Dict[str, Any] = {
        "model_class": type(model).__name__,
        "model_config": model.config.to_dict(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Tuple[MultilayerCVaE, Dict[str, Any]]:
    """加载权重，返回 ``(model, checkpoint 其余字段)``。"""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"权重文件格式不合法: {path}")
    config = CVAEConfig.from_dict(payload.get("model_config", {}))
    model = MultilayerCVaE(config)
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, payload
