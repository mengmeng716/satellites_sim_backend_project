"""
功能：星间路由与中继流量图计算模块
1. _get_gs_buffer: 内存级加载与缓存地面站物理坐标，阻断微周期中的磁盘 I/O 损耗。
2. calculate_background_routing: 构建全网星间链路 (ISL) 与星地链路 (GSL) 的拓扑图；
   基于 Dijkstra 算法计算各个卫星到最近地面站的最短路径；
   将各节点产生的上传流量叠加至沿途的中继节点与星间链路上，输出全网的负载与链路流量状态。
"""

import rustworkx as rx
import numpy as np
from scipy.spatial import cKDTree
import src.config as config

from src.utils.gs_manager import gs_manager

def calculate_background_routing(sat_ids, sat_upload_flows, sat_lat_list, sat_lon_list, topology_edges):
    """
    Docstring for calculate_background_routing
    
    :param sat_ids: Description
    :param sat_upload_flows: Description
    :param sat_lat_list: Description
    :param sat_lon_list: Description
    :param topology_edges: Description
    """
    num_sats = len(sat_ids)
    gs_ecef = gs_manager.get_ecef_coordinates()
    num_gs = len(gs_ecef)

    phi, lam = np.radians(sat_lat_list), np.radians(sat_lon_list)
    r = config.EARTH_RADIUS_KM + config.ORBIT_HEIGHT_KM
    sat_ecef = np.column_stack((r * np.cos(phi) * np.cos(lam), r * np.cos(phi) * np.sin(lam), r * np.sin(phi)))

    G = rx.PyDiGraph()
    G.add_nodes_from(range(num_sats + num_gs))
    edges = []

    # 1. 构建星间链路 (ISL)，边属性由单一距离浮点数改为包含负载的字典对象
    for u_idx, v_idx in topology_edges:
        dist = float(np.linalg.norm(sat_ecef[u_idx] - sat_ecef[v_idx]))
        edges.append((u_idx, v_idx, {"dist": dist, "load": 0.0}))

    # 2. 构建星地链路 (GSL)
    has_gs = num_gs > 0
    if has_gs:
        gs_tree = cKDTree(gs_ecef)
        dists, gs_indices = gs_tree.query(sat_ecef, k=1)
        gs_start_node = num_sats
        for s_idx, (d, g_idx) in enumerate(zip(dists, gs_indices)):
            if d <= config.MAX_SLANT_RANGE_KM :  # 星地可视距离阈值
                edges.append((s_idx, gs_start_node + g_idx, {"dist": float(d), "load": 0.0}))

    G.add_edges_from(edges)

    sat_total_loads = list(sat_upload_flows)
    link_loads_dict = {}

    # 3. 动态拥塞权重函数 (Congestion-aware Weight)
    # 取 config 中配置的额定带宽，若未配置则默认 10.0 Gbps
    LINK_CAPACITY = getattr(config, 'LINK_CAPACITY_GBPS', 10.0) 
    
    def dynamic_weight_fn(edge_data):
        usage = edge_data["load"] / LINK_CAPACITY
        if usage >= 0.99:
            return edge_data["dist"] * 10000.0  # 接近满载：施加断路级极高惩罚
        elif usage > 0.8:
            return edge_data["dist"] * 10.0     # 重度拥塞：施加10倍绕路惩罚
        elif usage > 0.5:
            return edge_data["dist"] * 2.0      # 中度拥塞：施加2倍绕路惩罚
        return edge_data["dist"]                # 正常空闲：纯按物理距离寻路

    if has_gs:
        # [核心优化]: 按上传流量从大到小降序排序 (大象流优先法)
        # 确保产生极大流量的节点先占用最优最短的主干道，避免被小流量碎片抢占而引发频繁绕路抖动
        routing_tasks = [(s_idx, sat_upload_flows[s_idx]) for s_idx in range(num_sats) if sat_upload_flows[s_idx] > 0]
        routing_tasks.sort(key=lambda x: x[1], reverse=True)

        for s_idx, upload_load in routing_tasks:
            target_gs_node = gs_start_node + gs_indices[s_idx]
            try:
                # 每次执行都用 dynamic_weight_fn 对当前的最新图状态进行探测
                paths_dict = rx.dijkstra_shortest_paths(G, s_idx, target_gs_node, weight_fn=dynamic_weight_fn)
                if target_gs_node in paths_dict:
                    path = paths_dict[target_gs_node]
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        
                        # 即时更新图内边负载，使后续查询感知到拥堵
                        edge_data = G.get_edge_data(u, v)
                        edge_data["load"] += upload_load
                        
                        # 结算并写入返回接口字典 (仅对 ISL 星间链路有效)
                        if u < num_sats and v < num_sats:
                            edge_key = (sat_ids[u], sat_ids[v])
                            link_loads_dict[edge_key] = link_loads_dict.get(edge_key, 0.0) + upload_load
                            sat_total_loads[u] += upload_load
                            sat_total_loads[v] += upload_load
            except rx.exception.NoPathFound:
                pass

    # 4. 封装导出
    link_flows = [[u_id, v_id, round(float(load), 3)] for (u_id, v_id), load in link_loads_dict.items()]
    total_flows = [round(float(load), 3) for load in sat_total_loads]

    return total_flows, link_flows