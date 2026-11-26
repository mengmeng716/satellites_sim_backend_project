"""背景流量生成算法。

对外入口：:func:`background_traffic_interface.background_traffic_generation`
"""

from .background_traffic_interface import (
    background_traffic_generation,
    background_traffic_snapshot,
)

__all__ = ["background_traffic_generation", "background_traffic_snapshot"]
