from pathlib import Path

# 当前 task_planning 模块根目录
TASK_PLANNING_DIR = Path(__file__).resolve().parent

# 模型权重目录
MODEL_WEIGHTS_DIR = TASK_PLANNING_DIR / "models-weights"

# constellation 开关支持值
SUPPORTED_CONSTELLATIONS = ("3600", "432")
DEFAULT_CONSTELLATION = "3600"

# actor-only 推理模型路径。
# constellation="3600" -> actor_only_3600.pt
# constellation="432"  -> actor_only_432.pt
ACTOR_MODEL_PATHS = {
    "3600": MODEL_WEIGHTS_DIR / "actor_only_3600.pt",
    "432": MODEL_WEIGHTS_DIR / "actor_only_432.pt",
}

# 保留原 SAC 模型路径，仅用于一次性导出 actor_only_3600.pt，不在在线推理中加载。
MODEL_PATH = MODEL_WEIGHTS_DIR / "final_dynamic_model.zip"

# 推理参数
NUM_CANDIDATE_PATHS = 3
DETERMINISTIC = True

# 模型设备
MODEL_DEVICE = "cpu"

# 状态编码参数，必须和训练配置保持一致
HIST_NUM_BINS = 10
HIST_VALUE_MIN = 0.0
HIST_VALUE_MAX = 1.0
DEMAND_SCALE = 1.0
INCLUDE_VALID_MASK = False

# 路径生成参数
HOP_WEIGHT_ATTR = "hop_weight"
CONG_WEIGHT_ATTR = "cong_weight"

# actor-only 输出说明：
# TorchScript actor 输出的是 SAC squashed action，范围 [-1, 1]；
# 需要手动映射到环境动作空间 [0, 1] 后再做 mask/normalize。
ACTION_LOW = 0.0
ACTION_HIGH = 1.0
