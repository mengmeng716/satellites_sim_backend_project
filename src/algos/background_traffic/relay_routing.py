"""星间中继路由与链路流量结算。

做法（与历史实现一致）：

1. 以卫星 + 地面站为节点建**有向图**；星间链路按欧氏距离赋权，星地链路按可视距离赋权；
2. 按上传流量从大到小（大象流优先）逐条 Dijkstra；
3. 边权为**拥塞感知**距离：负载超过 50%/80%/99% 容量时分别放大 2/10/10000 倍；
4. 逐条把流量叠加到沿途星间链路上，同时累加节点总负载。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import rustworkx as rx
from scipy.spatial import cKDTree

import src.config as config
from src.utils.gs_manager import gs_manager

#: 拥塞分级阈值与放大系数
CONGESTION_LEVELS = ((0.99, 10000.0), (0.80, 10.0), (0.50, 2.0))

#: 未配置链路额定带宽时的默认值（Gbps）
DEFAULT_LINK_CAPACITY_GBPS = 10.0


def _satellite_ecef(lats: Sequence[float], lons: Sequence[float]) -> np.ndarray:
    phi = np.radians(np.asarray(lats, dtype=np.float64))
    lam = np.radians(np.asarray(lons, dtype=np.float64))
    radius = float(config.EARTH_RADIUS_KM) + float(config.ORBIT_HEIGHT_KM)
    return np.column_stack((
        radius * np.cos(phi) * np.cos(lam),
        radius * np.cos(phi) * np.sin(lam),
        radius * np.sin(phi),
    ))


def _dynamic_weight_fn(link_capacity_gbps: float):
    def weight_fn(edge_data):
        usage = float(edge_data["load"]) / max(link_capacity_gbps, 1e-9)
        for threshold, penalty in CONGESTION_LEVELS:
            if usage >= threshold:
                return edge_data["dist"] * penalty
        return edge_data["dist"]

    return weight_fn


def calculate_background_routing(
    sat_ids: Sequence[str],
    sat_upload_flows: Sequence[float],
    sat_lat_list: Sequence[float],
    sat_lon_list: Sequence[float],
    topology_edges: Iterable[Tuple[int, int]],
) -> Tuple[List[float], List[List[float]]]:
    """结算中继流量。

    :return: ``(sat_total_flows, link_flows)``
             ``sat_total_flows[i]`` 为第 i 颗卫星的节点总流量（Gbps）；
             ``link_flows`` 为 ``[[src_id, dst_id, gbps], ...]``
    """
    num_sats = len(sat_ids)
    gs_ecef = gs_manager.get_ecef_coordinates()
    num_gs = len(gs_ecef)

    sat_ecef = _satellite_ecef(sat_lat_list, sat_lon_list)

    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(num_sats + num_gs))
    edges = []

    for u_idx, v_idx in topology_edges:
        dist = float(np.linalg.norm(sat_ecef[u_idx] - sat_ecef[v_idx]))
        edges.append((u_idx, v_idx, {"dist": dist, "load": 0.0}))

    gs_indices = None
    has_gs = num_gs > 0
    if has_gs:
        gs_tree = cKDTree(gs_ecef)
        dists, gs_indices = gs_tree.query(sat_ecef, k=1)
        gs_start = num_sats
        max_range = float(getattr(config, "MAX_SLANT_RANGE_KM", 1200.0))
        for sat_idx, (dist, gs_idx) in enumerate(zip(dists, gs_indices)):
            if dist <= max_range:
                edges.append((sat_idx, gs_start + gs_idx,
                              {"dist": float(dist), "load": 0.0}))

    graph.add_edges_from(edges)

    sat_total_loads = list(float(x) for x in sat_upload_flows)
    link_loads: Dict[Tuple[str, str], float] = {}

    if not has_gs:
        return [round(x, 3) for x in sat_total_loads], []

    link_capacity = float(getattr(config, "MAX_LINK_SPEED_GBPS", DEFAULT_LINK_CAPACITY_GBPS))
    weight_fn = _dynamic_weight_fn(link_capacity)

    tasks = [(idx, float(sat_upload_flows[idx])) for idx in range(num_sats)
             if sat_upload_flows[idx] > 0]
    tasks.sort(key=lambda item: item[1], reverse=True)

    gs_start = num_sats
    for sat_idx, upload in tasks:
        target = gs_start + int(gs_indices[sat_idx])
        try:
            paths = rx.dijkstra_shortest_paths(graph, sat_idx, target, weight_fn=weight_fn)
        except rx.exception.NoPathFound:
            continue
        if target not in paths:
            continue
        path = paths[target]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = graph.get_edge_data(u, v)
            edge_data["load"] += upload
            if u < num_sats and v < num_sats:
                key = (sat_ids[u], sat_ids[v])
                link_loads[key] = link_loads.get(key, 0.0) + upload
                sat_total_loads[u] += upload
                sat_total_loads[v] += upload

    link_flows = [[u_id, v_id, round(float(load), 3)] for (u_id, v_id), load in link_loads.items()]
    total_flows = [round(float(x), 3) for x in sat_total_loads]
    return total_flows, link_flows
