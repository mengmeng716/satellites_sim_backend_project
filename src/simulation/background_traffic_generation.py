"""
    功能：地面人口
    :param topology_state:
    :param node_state:
    :return: 拓扑状态、节点状态、背景流量结构体,时间戳
    返回生成的背景流量
    background_traffic_obj：卫星上传流量、卫星总流量，链路流量【srcsatid，dstsatid，flow】
        [[sat0，upload_flow,total_flow],...,[sat3599，upload_flow,total_flow]],[[srcsatid，dstsatid，flow],...,[srcsatid，dstsatid，flow]]
"""
"""
功能：背景流量生成主控模块
1. _get_traffic_grid_buffer: 内存级加载与缓存全球互联网人口流量网格。
2. background_traffic_generation: 对接 engine.py 的标准接口；
   根据全球人口网格分布与仿真 UTC 时间，加权计算各可见卫星的初始上传流量；
   调用 background_routing.py 计算中继开销；
   拼装并返回严格符合引擎规范的数据结构 (拓扑状态、节点状态、[卫星流量, 链路流量]复合体、时间戳)。
"""

import numpy as np
import pandas as pd
from datetime import datetime
from src import config
from src.utils.orbit_calculator import calculate_slant_range_vectorized
# 注意此处的同级目录导入路径
from src.simulation.background_routing import calculate_background_routing
from src.simulation.data_model import SatelliteNodeState, TopologyState

_GRID_CACHE = {"is_loaded": False, "grid_base": None, "flat_lat": None, "flat_lon": None}

def _get_traffic_grid_buffer():
    if _GRID_CACHE["is_loaded"]:
        return _GRID_CACHE["grid_base"], _GRID_CACHE["flat_lat"], _GRID_CACHE["flat_lon"]
    try:
        df = pd.read_excel(config.TRAFFIC_GRID_FILE, header=None)
        # 【修改】动态获取传入网格的行数和列数
        rows, cols = df.shape
        grid = np.zeros((rows, cols), dtype=np.float32)
        df_vals = df.fillna(0.0).values.astype(np.float32)
        grid[:] = df_vals
    except Exception as e:
        print(f"[Warning] 流量网格加载失败: {e}")
        # 默认回退为原始结构
        rows, cols = 12, 24
        grid = np.zeros((rows, cols), dtype=np.float32)

    lat_grid = np.linspace(90, -90, rows + 1)
    lon_grid = np.linspace(-180, 180, cols + 1)
    lat_c_vec = (lat_grid[:-1] + lat_grid[1:]) / 2
    lon_c_vec = (lon_grid[:-1] + lon_grid[1:]) / 2
    lon_centers, lat_centers = np.meshgrid(lon_c_vec, lat_c_vec)

    _GRID_CACHE["grid_base"] = grid.flatten()
    _GRID_CACHE["flat_lat"] = lat_centers.flatten()
    _GRID_CACHE["flat_lon"] = lon_centers.flatten()
    _GRID_CACHE["is_loaded"] = True
    return _GRID_CACHE["grid_base"], _GRID_CACHE["flat_lat"], _GRID_CACHE["flat_lon"]


def background_traffic_generation(topology_state: TopologyState, node_state: SatelliteNodeState, timestamp: datetime):
    grid_base, flat_lat, flat_lon = _get_traffic_grid_buffer()
    utc_hour_decimal = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

    sat_list = node_state.sat_list
    sat_ids = sorted(list(sat_list.keys()), key=lambda x: int(x))
    if not sat_ids:
        utc_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return topology_state, node_state, [[], []], utc_str

    # 使用 NodeAttribute Pydantic 对象属性访问
    current_sat_lat = np.array([sat_list[sid].latitude for sid in sat_ids])
    current_sat_lon = np.array([sat_list[sid].longitude for sid in sat_ids])
    new_upload_flows = np.zeros(len(sat_ids), dtype=np.float32)

    # [优化] 全矩阵向量化：消除原先数万次的 Python for 循环，提速百倍
    # 过滤掉没有人口密度的海洋/死区网格
    valid_mask = grid_base > 0
    valid_grid = grid_base[valid_mask]
    valid_lat = flat_lat[valid_mask]
    valid_lon = flat_lon[valid_mask]

    # 一次性向量化计算所有有效网格的当地时间和时间权重因子
    local_hours = (utc_hour_decimal + valid_lon / 15.0) % 24.0
    time_factors = np.full_like(local_hours, 0.05)
    
    m1 = (local_hours >= 6) & (local_hours < 10)
    time_factors[m1] = 0.05 + (local_hours[m1] - 6) / 4.0 * 0.95
    m2 = (local_hours >= 10) & (local_hours < 22)
    time_factors[m2] = 1.0
    m3 = (local_hours >= 22) & (local_hours < 24)
    time_factors[m3] = 1.0 - (local_hours[m3] - 22) / 2.0 * 0.95

    # 找出当前时刻真正产生流量的网格
    area_flows = valid_grid * time_factors * config.TRAFFIC_SCALE_FACTOR
    active_mask = area_flows > 0

    # 仅遍历活跃的有效网格计算可视卫星距离（计算量降低 90% 以上）
    for lat, lon, area_flow in zip(valid_lat[active_mask], valid_lon[active_mask], area_flows[active_mask]):
        dists = calculate_slant_range_vectorized(lat, lon, current_sat_lat, current_sat_lon)
        # config.MAX_SLANT_RANGE_KM (如果未定义则默认为1200.0)
        max_dist = getattr(config, 'MAX_SLANT_RANGE_KM', 1200.0)
        visible_idx = np.where(dists <= max_dist)[0]
        if len(visible_idx) == 0: continue

        weights = 1.0 / dists[visible_idx]
        new_upload_flows[visible_idx] += area_flow * (weights / np.sum(weights))

    # 2. 拓扑图边数据清洗
    # 标准格式：Dict[str, Dict[str, LinksQualitiesValue]]
    topology_edges = []
    sat_id_to_idx = {sid: i for i, sid in enumerate(sat_ids)}
    new_topology = topology_state.new_topology

    for u_id_str, links_dict in new_topology.items():
        if u_id_str in sat_id_to_idx:
            u_idx = sat_id_to_idx[u_id_str]
            for v_id_str, link_attr in links_dict.items():
                if v_id_str in sat_id_to_idx:
                    topology_edges.append((u_idx, sat_id_to_idx[v_id_str]))

    # 3. 中继路由结算
    total_flows, link_flows = calculate_background_routing(
        sat_ids, new_upload_flows, current_sat_lat, current_sat_lon, topology_edges
    )

    # 4. 封装系统状态返回
    sat_flows_obj = []
    new_node = node_state
    for i, sid in enumerate(sat_ids):
        up_flow = round(float(new_upload_flows[i]), 3)
        tot_flow = total_flows[i]

        # 使用 Pydantic 对象属性赋值
        new_node.sat_list[sid].flow = tot_flow

        sat_flows_obj.append([sid, up_flow, tot_flow])

    background_traffic_obj = [sat_flows_obj, link_flows]
    utc_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # 把链路流量写回拓扑，补全 topology_state 的更新
    # 标准格式：{sat_id(str): {target_id(str): LinksQualitiesValue}}
    new_top = topology_state
    new_topology_wb = new_top.new_topology
    for src_id, dst_id, flow in link_flows:
        src_str = str(src_id)
        dst_str = str(dst_id)
        if src_str in new_topology_wb and dst_str in new_topology_wb[src_str]:
            new_topology_wb[src_str][dst_str].current_flow = flow

    return new_top, new_node, background_traffic_obj, utc_str