"""
GRLR 在线推理配置。

本文件只保留在线推理所需的模型结构参数、星座规模参数和路由代价参数。
不包含训练流程、训练轮数、日志路径、TLE 建图配置等内容。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_checkpoint_path(path: str | Path, default_filename: str = "checkpoint_final.pt") -> Path:
    """解析模型权重路径。支持传入目录或具体 .pt 文件。"""
    candidate = Path(path).expanduser()
    candidates: list[Path] = []

    if default_filename and candidate.suffix == "":
        candidates.append(candidate / default_filename)
    candidates.append(candidate)

    for resolved in candidates:
        if resolved.exists():
            return resolved

    return candidates[0] if default_filename and candidate.suffix == "" else candidate


@dataclass
class ConstellationConfig:
    """星座规模配置。在线推理只需要卫星编号范围和总规模。"""

    n_orbits: int = 60
    n_sats_per_orbit: int = 60

    @property
    def total_sats(self) -> int:
        return self.n_orbits * self.n_sats_per_orbit


@dataclass
class ModelConfig:
    """GRLR 模型结构配置，必须与训练 checkpoint 对应。"""

    node_in_dim: int = 6
    edge_in_dim: int = 5
    hidden_dim: int = 64
    n_actions: int = 4
    dropout: float = 0.1
    n_gat_layers: int = 2
    n_heads: int = 4


@dataclass
class RoutingConfig:
    """在线推理路径代价配置。"""

    delay_weight: float = 1.0
    packet_loss_weight: float = 60.0
    congestion_weight: float = 8.0
    capacity_penalty_weight: float = 5.0
    hop_weight: float = 0.5
    max_search_hops: int = 25
    max_search_expansions: int = 4096
    rl_tie_break_eps: float = 1e-3


@dataclass
class TrainingConfig:
    """
    兼容历史 checkpoint 反序列化的空配置。

    在线系统不会使用该类进行训练；保留它只是避免旧 checkpoint 中保存了
    grlr.config.TrainingConfig 对象时 torch.load 失败。
    """

    pass
