"""算法：背景流量生成 —— 对外唯一入口。

职责：按「地面人口网格 × 时段因子 → 可见卫星上传流量 → 星间中继 → 链路流量」
生成每个微周期的背景业务负载，供链路预测与拓扑重构作为输入。

与「多层关联数据模拟参数生成」是两个独立算法：
前者产出**业务流量**（Gbps，在线、每 5 s 一次），后者产出**参数序列**（离线、物理层/网络层观测值）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from .flow_allocation import allocate_upload_flows, satellite_positions, topology_edges
from .relay_routing import calculate_background_routing
from .traffic_grid import active_cells, grid_source


def background_traffic_generation(topology_state, node_state, timestamp):
    """生成本周期的背景流量，并回写节点/链路状态。

    :param topology_state: ``TopologyState``，含 ``new_topology``
    :param node_state: ``SatelliteNodeState``，含 ``sat_list``
    :param timestamp: 当前 UTC ``datetime``
    :return: ``(new_topology, new_node, [sat_flows, link_flows], utc_str)``

        * ``sat_flows``：``[[sat_id, upload_gbps, total_gbps], ...]``
        * ``link_flows``：``[[src_id, dst_id, gbps], ...]``
    """
    sat_items = list((node_state.sat_list or {}).items())
    sat_ids, sat_lats, sat_lons = satellite_positions(sat_items)

    utc_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    if not sat_ids:
        return topology_state, node_state, [[], []], utc_str

    cell_lats, cell_lons, cell_flows = active_cells(timestamp)
    upload = allocate_upload_flows(sat_lats, sat_lons, cell_lats, cell_lons, cell_flows)

    edges = topology_edges(topology_state, sat_ids)
    total_flows, link_flows = calculate_background_routing(
        sat_ids, upload, sat_lats, sat_lons, edges
    )

    sat_flows: List[list] = []
    for i, sat_id in enumerate(sat_ids):
        up = round(float(upload[i]), 3)
        total = total_flows[i]
        node_state.sat_list[sat_id].flow = total
        sat_flows.append([sat_id, up, total])

    new_topology = topology_state.new_topology
    # 本周期没有承载中继流量的链路清零，避免沿用上一周期的流量
    for links in new_topology.values():
        for link_attr in links.values():
            link_attr.current_flow = 0.0
    for src_id, dst_id, flow in link_flows:
        links = new_topology.get(str(src_id))
        if links and str(dst_id) in links:
            links[str(dst_id)].current_flow = flow

    return topology_state, node_state, [sat_flows, link_flows], utc_str


def background_traffic_snapshot(topology_state, node_state, timestamp) -> Dict[str, Any]:
    """只读摘要（不修改状态），用于离线检查或日志落盘。

    :return: ``{"Timestamp", "GridSource", "ActiveCells", "OfferedGbps",
                "SatelliteFlow", "LinkFlow", "ElapsedMs"}``
    """
    started = time.perf_counter()
    sat_items = list((node_state.sat_list or {}).items())
    sat_ids, sat_lats, sat_lons = satellite_positions(sat_items)
    cell_lats, cell_lons, cell_flows = active_cells(timestamp)
    upload = (allocate_upload_flows(sat_lats, sat_lons, cell_lats, cell_lons, cell_flows)
              if sat_ids else [])

    return {
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "GridSource": grid_source(),
        "ActiveCells": int(len(cell_flows)),
        "OfferedGbps": round(float(sum(cell_flows)), 3),
        "SatelliteFlow": [[sid, round(float(up), 3)] for sid, up in zip(sat_ids, upload)],
        "LinkFlow": [],
        "ElapsedMs": round((time.perf_counter() - started) * 1000.0, 3),
    }
