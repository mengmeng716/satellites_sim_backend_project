"""
路由需求生成模块

根据当前拓扑和节点状态，生成路由优化算法所需的输入数据。
严格遵循 3.4 路由优化任务输入接口规范，并与 routing_interface.py 的 RouteTask 类型兼容。

"""

import random
from datetime import datetime
from typing import Dict, List, Any, Tuple

from src.simulation.data_model import SatelliteNodeState, TopologyState
from src.utils.source_destination_constraint import check_src_dst_distance_constraint

def _select_valid_sat_pair(
    node_state: SatelliteNodeState
) -> Tuple[int, int]:
    """
    选择源-目的卫星对：
    1. 源节点：从负载最高的前 20 颗卫星中随机选取。
    2. 目的节点：从全局卫星中完全随机选取。
    3. 约束：二者不能相同，且必须满足距离约束。

    :param node_state: 节点状态对象
    :return: (src_sat_id, dest_sat_id) 或 (None, None) 如果找不到满足条件的对
    """
    sat_list = node_state.sat_list
    if len(sat_list) < 2:
        return None, None

    # 提取所有卫星的 (ID, Flow) 数据
    all_sats = [(sid, sdata.flow) for sid, sdata in sat_list.items()]

    # 1. 按流量降序排序，取前 20 名最高负载的卫星
    sorted_sats = sorted(all_sats, key=lambda x: x[1], reverse=True)
    top_20_sats = sorted_sats[:20]

    # 获取所有卫星的 ID 列表，用于目的节点的完全随机抽样
    all_sat_ids = [sid for sid, _ in all_sats]

    # 为了防止极端情况下找不到合法路径导致死循环，设置最大尝试次数
    max_attempts = 100

    for _ in range(max_attempts):
        # 2. 从前 20 名中随机取一个作为源卫星
        src_candidate_id = random.choice(top_20_sats)[0]

        # 3. 完全随机取一个作为目的卫星
        dest_candidate_id = random.choice(all_sat_ids)

        # 约束 1：源和目的不能是同一颗卫星
        if src_candidate_id == dest_candidate_id:
            continue

        src_id_int = int(src_candidate_id)
        dest_id_int = int(dest_candidate_id)

        # 约束 2：检查距离约束 (调用原有的距离检查函数)
        is_valid = check_src_dst_distance_constraint(
            node_state,
            src_id_int,
            dest_id_int
        )

        if is_valid:
            return src_id_int, dest_id_int

    # 如果达到最大尝试次数仍未找到合法对，则放弃本次生成
    return None, None


def routing_demand_generation(
    topology_state: TopologyState,
    node_state: SatelliteNodeState,
    current_utc: datetime
) -> Dict[str, Any]:
    """
    主入口：生成当前时刻的路由需求任务

    :param topology_state: 当前拓扑状态
    :param node_state: 当前节点状态
    :param current_utc: 当前 UTC 时间
    :return: 路由需求字典
    """

    # 提取核心信息
    constellation_id = topology_state.constellation_id
    timestamp_str = current_utc.isoformat()  # ISO 格式字符串，符合 Datetime 类型要求

    # 构建 RouteTaskList - 模拟业务流需求
    route_task_list = []

    # 调用新的随机匹配逻辑
    src_sat_id, dest_sat_id = _select_valid_sat_pair(node_state)

    if src_sat_id is not None and dest_sat_id is not None:
        # 成功找到了满足条件的源-目的卫星对，生成任务
        task_attributes = {
            "TaskId": f"RouteGen_{src_sat_id}_{dest_sat_id}_{int(current_utc.timestamp())}",
            "SrcSatId": src_sat_id,
            "DestSatId": dest_sat_id,
            "PacketSize": 500.0,    # 默认业务包大小，可根据需求修改
            "StartTime": timestamp_str,
            "TaskPriority": 5,      # 默认优先级
            "Duration": 10          # 默认持续时间，决定了占用流的时长
        }
        route_task_list.append(task_attributes)

    return {
        "Timestamp": timestamp_str,
        "ConstellationId": constellation_id,
        "RouteTaskList": route_task_list
    }