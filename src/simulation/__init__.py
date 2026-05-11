"""
仿真引擎包：星座初始化、仿真调度核心模块
"""
# 导入engine.py中的核心引擎类
from .engine import SimulationEngine
# 对外公开接口
__all__ = ["SimulationEngine"]
