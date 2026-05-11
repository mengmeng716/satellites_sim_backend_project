
"""
route_planning 在线推理配置。

本文件只服务于总项目集成阶段的推理，不包含训练逻辑。
模型权重根据接口中的 ConstellationId 自动选择：
    - ConstellationId = 432  -> models-weights/432_delta/checkpoint_final.pt
    - ConstellationId = 3600 -> models-weights/3600_qx4/checkpoint_final.pt

说明：不需要额外增加模型选择字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


# 当前 route_planning 目录
BASE_DIR = Path(__file__).resolve().parent

# GRLR 算法源码目录
GRLR_DIR = BASE_DIR / "src"

# 模型权重目录
MODEL_WEIGHTS_DIR = BASE_DIR / "models-weights"


@dataclass(frozen=True)
class ModelProfile:
    """单个星座规模对应的模型权重配置。"""

    model_key: str
    constellation_size: int
    n_orbits: int
    n_sats_per_orbit: int
    weight_dir_name: str
    checkpoint_filename: str = "checkpoint_final.pt"

    @property
    def model_path(self) -> Path:
        return MODEL_WEIGHTS_DIR / self.weight_dir_name / self.checkpoint_filename


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "432": ModelProfile(
        model_key="432",
        constellation_size=432,
        n_orbits=24,
        n_sats_per_orbit=18,
        weight_dir_name="432_delta",
    ),
    "3600": ModelProfile(
        model_key="3600",
        constellation_size=3600,
        n_orbits=60,
        n_sats_per_orbit=60,
        weight_dir_name="3600_qx4",
    ),
}

# 默认使用 3600 星模型。接口任务中的 ConstellationId 为 432 时，自动切换到 432_delta 权重。
DEFAULT_MODEL_KEY = "3600"
DEFAULT_CONSTELLATION_ID = "3600"

# 为了兼容旧代码，保留 MODEL_PATH，但新逻辑请使用 resolve_model_profile(...).model_path。
MODEL_PATH = MODEL_PROFILES[DEFAULT_MODEL_KEY].model_path

# 在线推理不读取 TLE，不在 route_planning 内生成拓扑。
# 当前拓扑由总项目 data_model 的 newTopology 提供。

# 第一版建议使用 CPU，保证部署环境稳定。
# 如果确认服务器有 CUDA，可以改成 "cuda"。
DEVICE = "cpu"

# 单条任务路径最大跳数。
MAX_ROUTE_HOPS = 500

# 当任务中没有显式流量大小时，使用默认值。
DEFAULT_PACKET_SIZE = 1.0

# 当任务中没有显式持续时间时，使用默认值。
DEFAULT_DURATION = 0


# 兼容旧字段：默认配置仍指向 3600。
N_ORBITS = MODEL_PROFILES[DEFAULT_MODEL_KEY].n_orbits
N_SATS_PER_ORBIT = MODEL_PROFILES[DEFAULT_MODEL_KEY].n_sats_per_orbit
TOTAL_SATS = MODEL_PROFILES[DEFAULT_MODEL_KEY].constellation_size


def resolve_model_key(constellation_id: Any = None) -> str:
    """
    根据接口中的 ConstellationId 自动选择模型权重。

    不额外新增模型选择字段。
    当 ConstellationId 的值为 432 时，使用 432_delta 权重；
    当 ConstellationId 的值为 3600 时，使用 3600_qx4 权重。

    为兼容工程命名，也允许 ConstellationId 中包含 432 或 3600，
    例如 "CONSTELLATION_432" 或 "CONSTELLATION_3600"。
    """
    if constellation_id is None or str(constellation_id).strip() == "":
        return DEFAULT_MODEL_KEY

    text = str(constellation_id).strip().lower()

    if text == "432" or "432" in text:
        return "432"
    if text == "3600" or "3600" in text:
        return "3600"

    supported = ", ".join(sorted(MODEL_PROFILES.keys()))
    raise ValueError(
        f"无法根据 ConstellationId={constellation_id!r} 选择模型。"
        f"ConstellationId 需要为 432 或 3600。当前支持: {supported}。"
    )

def resolve_model_profile(constellation_id: Any = None) -> ModelProfile:
    """返回当前任务应使用的模型配置。"""
    return MODEL_PROFILES[resolve_model_key(constellation_id)]


def validate_route_planning_config(constellation_id: Any = None) -> None:
    """检查 route_planning 推理所需的关键文件。"""
    if not GRLR_DIR.exists():
        raise FileNotFoundError(f"GRLR 算法目录不存在: {GRLR_DIR}")

    # 检查指定模型；如果未指定，默认检查 3600。
    profile = resolve_model_profile(constellation_id)
    if not profile.model_path.exists():
        raise FileNotFoundError(
            f"GRLR 模型权重不存在: {profile.model_path}。"
            f"当前选择模型: {profile.model_key}。"
        )


def validate_all_model_weights() -> None:
    """启动自检时可调用：检查所有已注册模型权重是否存在。"""
    if not GRLR_DIR.exists():
        raise FileNotFoundError(f"GRLR 算法目录不存在: {GRLR_DIR}")

    missing = [str(profile.model_path) for profile in MODEL_PROFILES.values() if not profile.model_path.exists()]
    if missing:
        raise FileNotFoundError("以下 GRLR 模型权重不存在: " + "; ".join(missing))
