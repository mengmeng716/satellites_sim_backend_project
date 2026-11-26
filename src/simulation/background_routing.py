"""兼容层：中继路由算法已迁至 :mod:`src.algos.background_traffic.relay_routing`。

保留本模块是为了不改变历史调用方的导入路径：

    from src.simulation.background_routing import calculate_background_routing
"""

from __future__ import annotations

from src.algos.background_traffic.relay_routing import calculate_background_routing

__all__ = ["calculate_background_routing"]
