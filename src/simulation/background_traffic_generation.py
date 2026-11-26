"""兼容层：背景流量生成算法已迁至 :mod:`src.algos.background_traffic`。

保留本模块是为了不改变 ``engine.py`` 的导入路径与调用方式：

    from src.simulation.background_traffic_generation import background_traffic_generation

函数名、参数、返回值结构均与原实现一致。
"""

from __future__ import annotations

from src.algos.background_traffic import (
    background_traffic_generation,
    background_traffic_snapshot,
)

__all__ = ["background_traffic_generation", "background_traffic_snapshot"]
