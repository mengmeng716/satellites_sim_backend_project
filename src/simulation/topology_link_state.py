"""
功能：Walker 星座拓扑生成与状态更新模块
遵循数据结构: TopologyState.new_topology (Dict[str, Dict[str, LinksQualitiesValue]])
"""

from typing import Dict
import numpy as np
from src import config
from src.simulation.data_model import LinksQualitiesValue


def _get_default_link_attr() -> LinksQualitiesValue:
    """生成默认链路属性对象"""
    return LinksQualitiesValue(
        link_distance=0.0,
        link_capacity=10.0,
        left_capacity=10.0,
        current_flow=0.0,
        link_propagation_delay=0.0,
        queue_delay=0.0,
        transmission_delay=0.0,
        packet_loss_rate=0.0,
        heat_value=0.0
    )


def init_walker_topology(constellation) -> Dict[str, Dict[str, LinksQualitiesValue]]:
    """
    初始化 Walker 星座拓扑 (仅确定连接关系)
    返回符合 data_model.TopologyState.new_topology 格式的嵌套字典
    """
    num_planes = constellation.numOrbitPlane
    sats_per_plane = constellation.numSatsInPlane
    P = num_planes
    S = sats_per_plane
    
    new_topology: Dict[str, Dict[str, LinksQualitiesValue]] = {}

    for sat_id in range(P * S):
        p = sat_id // S  # 轨道面索引
        s = sat_id % S   # 面内索引
        links: Dict[str, LinksQualitiesValue] = {}
        
        # 辅助函数：添加邻居
        def add_neighbor(target_id: int):
            links[str(target_id)] = _get_default_link_attr()

        # 1. 面内连接 - 下一个
        s_next = (s + 1) % S
        add_neighbor(p * S + s_next)

        # 2. 面内连接 - 上一个
        s_prev = (s - 1) % S
        add_neighbor(p * S + s_prev)

        # 3. 面间连接 - 下一个轨道面
        p_next = (p + 1) % P
        if p == P - 1 and p_next == 0:
            target_s = (s - 38) % S  # 异向缝隙特化
        else:
            target_s = (s + 59) % S
        add_neighbor(p_next * S + target_s)

        # 4. 面间连接 - 上一个轨道面
        p_prev = (p - 1) % P
        if p == 0 and p_prev == P - 1:
            target_s = (s + 38) % S  # 异向缝隙特化
        else:
            target_s = (s - 59) % S
        add_neighbor(p_prev * S + target_s)

        new_topology[str(sat_id)] = links

    return new_topology


def update_topology_snapshot(constellation, current_utc, topology: Dict[str, Dict[str, LinksQualitiesValue]]):
    """
    基于当前时间更新所有链路的距离与传播时延。
    直接在传入的 topology 字典中原地更新 LinksQualitiesValue 对象
    """
    r = config.EARTH_RADIUS_KM + config.ORBIT_HEIGHT_KM
    SPEED_OF_LIGHT_KM_S = 299792.458

    # 从 constellation 对象获取最新坐标
    coords = {}
    for sat in constellation.satelliteList:
        try:
            sid_int = int(sat.satId)
            phi = np.radians(sat.latitude)
            lam = np.radians(sat.longitude)
            x = r * np.cos(phi) * np.cos(lam)
            y = r * np.cos(phi) * np.sin(lam)
            z = r * np.sin(phi)
            coords[sid_int] = np.array([x, y, z])
        except (ValueError, AttributeError):
            continue

    # 更新距离和时延
    for source_id_str, links in topology.items():
        try:
            source_id = int(source_id_str)
        except ValueError:
            continue
            
        if source_id not in coords:
            continue
        vec_source = coords[source_id]

        for target_id_str, link_attr in links.items():
            try:
                target_id = int(target_id_str)
            except ValueError:
                continue
                
            if target_id not in coords:
                continue
            vec_target = coords[target_id]

            # 计算欧氏距离
            distance = float(np.linalg.norm(vec_source - vec_target))
            
            # 原地更新 LinksQualitiesValue 对象属性
            link_attr.link_distance = distance
            link_attr.link_propagation_delay = (distance / SPEED_OF_LIGHT_KM_S) * 1000.0