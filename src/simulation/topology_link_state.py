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
        link_capacity=config.MAX_LINK_SPEED_GBPS,
        left_capacity=1.0,
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
    自动适配 3600 星座 (60x60) 和 432 星座 (24x18)
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

        # 3. 面间连接 - 下一个轨道面 & 4. 上一个轨道面
        p_next = (p + 1) % P
        p_prev = (p - 1) % P

        if P == 24 and S == 18:
            # --- 432 星座逻辑 (24x18: 奇偶面交替偏移 + 24 面闭环边界) ---
            if p == P - 1:
                target_s_next = (s + 4) % S
            elif p % 2 == 0:
                target_s_next = s
            else:
                target_s_next = (s - 1) % S

            if p == 0:
                target_s_prev = (s - 4) % S
            elif p % 2 == 0:
                target_s_prev = (s + 1) % S
            else:
                target_s_prev = s

            add_neighbor(p_next * S + target_s_next)
            add_neighbor(p_prev * S + target_s_prev)

        elif P == 60 and S == 60:
            # 3600 constellation: same alternating inter-plane pattern as 432,
            # with the 60-plane closing seam shifted by 38 satellites.
            if p == P - 1:
                target_s_next = (s - 38) % S
            elif p % 2 == 0:
                target_s_next = s
            else:
                target_s_next = (s - 1) % S

            if p == 0:
                target_s_prev = (s + 38) % S
            elif p % 2 == 0:
                target_s_prev = (s + 1) % S
            else:
                target_s_prev = s

            add_neighbor(p_next * S + target_s_next)
            add_neighbor(p_prev * S + target_s_prev)

        else:
            # --- 3600 星座及其他标准 Walker 逻辑 (保留原代码精髓) ---
            # 3. 面间连接 - 下一个轨道面
            if p == P - 1 and p_next == 0:
                target_s = (s - 38) % S  # 异向缝隙特化 (针对 3600 星座)
            else:
                target_s = (s + 59) % S  # 常规偏移 (59 等价于 -1 mod 60)
            add_neighbor(p_next * S + target_s)

            # 4. 面间连接 - 上一个轨道面
            if p == 0 and p_prev == P - 1:
                target_s = (s + 38) % S  # 异向缝隙特化 (针对 3600 星座)
            else:
                target_s = (s - 59) % S  # 常规偏移 (-59 等价于 +1 mod 60)
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
            link_attr.link_capacity = config.MAX_LINK_SPEED_GBPS
            link_attr.current_flow = max(0.0, float(link_attr.current_flow))
            utilization = link_attr.current_flow / max(config.MAX_LINK_SPEED_GBPS, 1e-9)
            link_attr.left_capacity = 1.0 - utilization
            link_attr.heat_value = max(0.0, min(1.0, utilization))
