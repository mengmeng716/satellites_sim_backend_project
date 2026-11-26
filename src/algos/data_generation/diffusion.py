"""条件 DDPM（一维时序）——替代 CVAE 的生成模型。

设计要点（针对我们数据的形态）：
  * 输入为多通道一维序列 ``(B, C, L)``，C 为参数通道数、L 为时间窗长；
  * 条件向量由“轨道面角度的正余弦 + 日照占比 + 平均纬度 + 归一化面号”构成，
    作为每个残差块的偏置注入，保证同一条件下去噪方向一致；
  * 线性 β 调度（β: 1e-4 → 0.02，T=200 步），预测噪声 ε；
  * 一维 U-Net：两级下采样，通道 32→64→128。

相比 CVAE 的优势：不依赖重构损失，不会把方差压平；条件在每一步去噪都参与，
不存在“条件被数据项淹没”的问题。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        args = x[:, None].float() * freq[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.cond_mlp = nn.Linear(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        bias = self.time_mlp(t_emb) + self.cond_mlp(c_emb)
        h = h + bias[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet1d(nn.Module):
    def __init__(self, channels: int, cond_dim: int = 4, base: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(channels, base, 3, padding=1)
        self.down1 = ResBlock1d(base, base, time_dim, cond_dim)
        self.down2 = ResBlock1d(base, base * 2, time_dim, cond_dim)
        self.down3 = ResBlock1d(base * 2, base * 4, time_dim, cond_dim)
        self.pool = nn.AvgPool1d(2)
        self.mid = ResBlock1d(base * 4, base * 4, time_dim, cond_dim)
        self.up3 = ResBlock1d(base * 4 + base * 2, base * 2, time_dim, cond_dim)
        self.up2 = ResBlock1d(base * 2 + base, base, time_dim, cond_dim)
        self.up1 = ResBlock1d(base + base, base, time_dim, cond_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv1d(base, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        c_emb = cond
        h0 = self.in_conv(x)                                   # base,   L
        h1 = self.down1(h0, t_emb, c_emb)                      # base,   L
        h2 = self.down2(self.pool(h1), t_emb, c_emb)           # 2base,  L/2
        h3 = self.down3(self.pool(h2), t_emb, c_emb)           # 4base,  L/4
        h = self.mid(h3, t_emb, c_emb)                         # 4base,  L/4

        h = F.interpolate(h, size=h2.shape[-1], mode="nearest")
        h = self.up3(torch.cat([h, h2], dim=1), t_emb, c_emb)  # 2base,  L/2

        h = F.interpolate(h, size=h1.shape[-1], mode="nearest")
        h = self.up2(torch.cat([h, h1], dim=1), t_emb, c_emb)  # base,   L

        h = self.up1(torch.cat([h, h0], dim=1), t_emb, c_emb)  # base,   L
        return self.out_conv(F.silu(self.out_norm(h)))


class GaussianDiffusion:
    """线性 β 调度的 DDPM。"""

    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device) -> "GaussianDiffusion":
        for name in ("betas", "alphas_cumprod", "alphas_cumprod_prev",
                     "sqrt_alphas_cumprod", "sqrt_one_minus", "posterior_variance"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t][:, None, None]
        b = self.sqrt_one_minus[t][:, None, None]
        return a * x0 + b * noise

    def p_losses(self, model: nn.Module, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted = model(x_t, t.float(), cond)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, cond: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        device = cond.device
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.float)
            eps = model(x, t, cond)
            alpha = self.alphas_cumprod[step]
            alpha_prev = self.alphas_cumprod_prev[step]
            beta = self.betas[step]
            coef = (1 - alpha_prev) / (1 - alpha) * beta
            mean = (x - coef / torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
            if step > 0:
                var = self.posterior_variance[step]
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return x
