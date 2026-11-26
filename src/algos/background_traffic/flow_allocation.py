"""把地面网格流量分配到可见卫星。

分配规则（与历史实现一致）：

1. 每个活跃网格按斜距反比权重分配给其可视卫星；
2. 可视判据为斜距 ``<= config.MAX_SLANT_RANGE_KM``；
3. 同一网格的权重归一化后求和，保证该网格的总量守恒。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src import config
from src.utils.orbit_calculator import calculate_slant_range_vectorized


def satellite_positions(sat_items: Iterable) -> tuple[list[str], np.ndarray, np.ndarray]:
    """从 ``node_state.sat_list`` 提取卫星 ID 与经纬度。

    :param sat_items: ``[(sat_id, NodeAttribute), ...]``
    :return: ``(sat_ids, lats, lons)``
    """
    pairs = sorted(sat_items, key=lambda item: int(item[0]))
    sat_ids = [str(item[0]) for item in pairs]
    if not sat_ids:
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    lats = np.array([float(item[1].latitude) for item in pairs], dtype=np.float64)
    lons = np.array([float(item[1].longitude) for item in pairs], dtype=np.float64)
    return sat_ids, lats, lons


def allocate_upload_flows(
    sat_lats: Sequence[float],
    sat_lons: Sequence[float],
    cell_lats: Sequence[float],
    cell_lons: Sequence[float],
    cell_flows: Sequence[float],
    max_slant_range_km: float | None = None,
) -> np.ndarray:
    """按斜距反比把网格流量分配到卫星。

    :return: 每颗卫星的上传流量（Gbps），长度与卫星数一致
    """
    sat_lats = np.asarray(sat_lats, dtype=np.float64)
    sat_lons = np.asarray(sat_lons, dtype=np.float64)
    upload = np.zeros(len(sat_lats), dtype=np.float64)
    if len(sat_lats) == 0:
        return upload

    max_range = float(
        max_slant_range_km
        if max_slant_range_km is not None
        else getattr(config, "MAX_SLANT_RANGE_KM", 1200.0)
    )

    for lat, lon, flow in zip(cell_lats, cell_lons, cell_flows):
        dists = calculate_slant_range_vectorized(lat, lon, sat_lats, sat_lons)
        visible = np.where(dists <= max_range)[0]
        if len(visible) == 0:
            continue
        weights = 1.0 / dists[visible]
        upload[visible] += float(flow) * (weights / np.sum(weights))

    return upload


def topology_edges(topology_state, sat_ids: Sequence[str]) -> list[tuple[int, int]]:
    """把 ``{src: {dst: LinksQualitiesValue}}`` 拓扑转成下标边表。"""
    index = {sid: i for i, sid in enumerate(sat_ids)}
    edges: list[tuple[int, int]] = []
    new_topology = getattr(topology_state, "new_topology", None) or {}
    for src_id, links in new_topology.items():
        if src_id not in index:
            continue
        u = index[src_id]
        for dst_id in links:
            if dst_id in index:
                edges.append((u, index[dst_id]))
    return edges
