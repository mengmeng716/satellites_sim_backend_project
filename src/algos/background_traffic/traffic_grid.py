"""人口流量网格加载与时段因子计算。

数据来源：``config.TRAFFIC_GRID_FILE``（默认 ``<工程根>/data/Internet_area.xlsx``），
行为网格，每格代表该经纬度区域的人口/互联网流量权重。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

#: 网格加载失败时的兜底形状（与历史实现保持一致）
DEFAULT_GRID_SHAPE = (12, 24)

_CACHE = {
    "is_loaded": False,
    "grid_base": None,
    "flat_lat": None,
    "flat_lon": None,
    "source": None,
}


def get_traffic_grid_buffer():
    """加载并缓存全球流量网格，返回 ``(grid_base, flat_lat, flat_lon)``。

    * ``grid_base``：一维数组，各网格的人口/流量权重
    * ``flat_lat`` / ``flat_lon``：一维数组，各网格中心点经纬度（度）

    网格文件缺失或读取失败时退化为 ``DEFAULT_GRID_SHAPE`` 的全零网格，
    并记录 ``source="fallback"``，保证仿真不中断。
    """
    if _CACHE["is_loaded"]:
        return _CACHE["grid_base"], _CACHE["flat_lat"], _CACHE["flat_lon"]

    try:
        df = pd.read_excel(config.TRAFFIC_GRID_FILE, header=None)
        rows, cols = df.shape
        grid = np.zeros((rows, cols), dtype=np.float32)
        grid[:] = df.fillna(0.0).values.astype(np.float32)
        source = str(config.TRAFFIC_GRID_FILE)
    except Exception as exc:  # noqa: BLE001 - 缺文件时降级，不阻断仿真
        print(f"[Warning] 流量网格加载失败: {exc}")
        rows, cols = DEFAULT_GRID_SHAPE
        grid = np.zeros((rows, cols), dtype=np.float32)
        source = "fallback"

    lat_grid = np.linspace(90, -90, rows + 1)
    lon_grid = np.linspace(-180, 180, cols + 1)
    lat_centers = (lat_grid[:-1] + lat_grid[1:]) / 2
    lon_centers = (lon_grid[:-1] + lon_grid[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    _CACHE["grid_base"] = grid.flatten()
    _CACHE["flat_lat"] = lat_mesh.flatten()
    _CACHE["flat_lon"] = lon_mesh.flatten()
    _CACHE["source"] = source
    _CACHE["is_loaded"] = True
    return _CACHE["grid_base"], _CACHE["flat_lat"], _CACHE["flat_lon"]


def grid_source() -> str:
    """返回当前网格来源：文件路径或 ``"fallback"``（用于诊断）。"""
    get_traffic_grid_buffer()
    return str(_CACHE["source"])


def time_factors(local_hours: np.ndarray) -> np.ndarray:
    """按当地时间给出流量时段因子。

    * 06:00–10:00 由 0.05 线性升到 1.0
    * 10:00–22:00 维持 1.0
    * 22:00–24:00 由 1.0 线性降到 0.05
    * 其余时段 0.05
    """
    local_hours = np.asarray(local_hours, dtype=np.float64)
    factors = np.full_like(local_hours, 0.05)

    rising = (local_hours >= 6) & (local_hours < 10)
    factors[rising] = 0.05 + (local_hours[rising] - 6) / 4.0 * 0.95

    daytime = (local_hours >= 10) & (local_hours < 22)
    factors[daytime] = 1.0

    falling = (local_hours >= 22) & (local_hours < 24)
    factors[falling] = 1.0 - (local_hours[falling] - 22) / 2.0 * 0.95

    return factors


def active_cells(timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回当前时刻产生流量的网格（经纬度与流量）。

    :param timestamp: UTC ``datetime``
    :return: ``(lat, lon, flow)``，均为等长一维数组
    """
    grid_base, flat_lat, flat_lon = get_traffic_grid_buffer()
    utc_hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

    valid_mask = grid_base > 0
    if not np.any(valid_mask):
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    valid_grid = grid_base[valid_mask]
    valid_lat = flat_lat[valid_mask]
    valid_lon = flat_lon[valid_mask]

    local_hours = (utc_hour + valid_lon / 15.0) % 24.0
    factors = time_factors(local_hours)
    flows = valid_grid * factors * float(getattr(config, "TRAFFIC_SCALE_FACTOR", 1.0))

    keep = flows > 0
    return valid_lat[keep], valid_lon[keep], flows[keep]
