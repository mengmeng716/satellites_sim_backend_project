# src/algos/task_planning/src/graph_adapter.py

from __future__ import annotations

import math
from typing import Any, Dict

import networkx as nx

from src.simulation.data_model import SatelliteNodeState, TopologyState


def _to_node_id(x: Any) -> int | str:
    """卫星编号统一转成 int；无法转换时保留字符串。"""
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)


def build_graph_from_simulation_state(
    topology_state: TopologyState,
    node_state: SatelliteNodeState | None = None,
) -> nx.Graph:
    """
    从项目 data_model 定义的状态对象中读取节点和链路属性，构造任务规划内部图 G。

    读取对象：
        topology_state.new_topology[u][v] -> LinksQualitiesValue
        node_state.sat_list[sid]          -> NodeAttribute
    """

    G = nx.Graph()

    # 1. 读取节点状态表：SatelliteNodeState.sat_list
    if node_state is not None:
        for sid, node_attr in node_state.sat_list.items():
            node_id = _to_node_id(sid)

            G.add_node(
                node_id,
                latitude=float(node_attr.latitude),
                longitude=float(node_attr.longitude),
                flow=float(node_attr.flow),
                energy_ratio=float(node_attr.energy_ratio),
                congestion=float(node_attr.congestion),
                heat_flow=float(node_attr.heat_flow),
            )

    # 2. 读取链路状态表：TopologyState.new_topology
    new_topology = topology_state.new_topology if topology_state is not None else {}

    for u, neighbors in new_topology.items():
        u_id = _to_node_id(u)

        if neighbors is None:
            continue

        for v, link_attr in neighbors.items():
            v_id = _to_node_id(v)

            link_distance = float(link_attr.link_distance)
            link_capacity = float(link_attr.link_capacity)
            left_capacity = float(link_attr.left_capacity)
            current_flow = float(link_attr.current_flow)
            propagation_delay = float(link_attr.link_propagation_delay)
            queue_delay = float(link_attr.queue_delay)
            transmission_delay = float(link_attr.transmission_delay)
            packet_loss_rate = float(link_attr.packet_loss_rate)
            heat_value = float(link_attr.heat_value)

            capacity = max(link_capacity, 1e-12)
            utilization = max(0.0, current_flow / capacity)
            delay_ms = propagation_delay + queue_delay + transmission_delay

            G.add_edge(
                u_id,
                v_id,
                # 项目原始链路属性，保留在边上便于后续输出和扩展。
                link_distance=link_distance,
                link_capacity=link_capacity,
                left_capacity=left_capacity,
                current_flow=current_flow,
                link_propagation_delay=propagation_delay,
                queue_delay=queue_delay,
                transmission_delay=transmission_delay,
                packet_loss_rate=packet_loss_rate,
                heat_value=heat_value,
                # 任务规划内部统一属性。
                capacity=capacity,
                load=current_flow,
                utilization=utilization,
                distance=link_distance,
                delay_ms=delay_ms,
                hop_weight=1.0,
                cong_weight=math.exp(min(utilization, 5.0)),
            )

    return G


def graph_summary(G: nx.Graph) -> Dict[str, Any]:
    """返回图的基本统计信息。"""

    if G.number_of_edges() == 0:
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": 0,
            "avg_utilization": 0.0,
            "max_utilization": 0.0,
        }

    utils = [float(data.get("utilization", 0.0)) for _, _, data in G.edges(data=True)]

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_utilization": sum(utils) / len(utils),
        "max_utilization": max(utils),
    }
