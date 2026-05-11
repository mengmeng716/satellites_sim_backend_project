"""
路由需求生成模块

根据当前拓扑和节点状态，生成路由优化算法所需的输入数据。
严格遵循 3.4 路由优化任务输入接口规范，并与 routing_interface.py 的 RouteTask 类型兼容。
"""

from datetime import datetime
from typing import Dict, List, Any
from src.simulation.data_model import SatelliteNodeState, TopologyState


def routing_demand_generation(
    topology_state: TopologyState,
    node_state: SatelliteNodeState,
    current_utc: datetime
) -> Dict[str, Any]:
    """
    生成路由优化任务的输入需求
    
    :param topology_state: 当前拓扑状态（来自 engine.get_current_states()）
    :param node_state: 当前节点状态（来自 engine.get_current_states()）
    :param current_utc: 当前 UTC 时间（datetime 对象）
    :return: 符合规范的路由需求字典
    """
    
    # 1. 提取核心信息
    constellation_id = topology_state.constellation_id
    timestamp_str = current_utc.isoformat()  # ISO 格式字符串，符合 Datetime 类型要求

    # 2. 构建 RouteTaskList - 模拟业务流需求
    route_task_list = []
    
    # 获取所有卫星的流量信息
    sat_list = node_state.sat_list
    
    # 筛选高负载卫星作为潜在源/目的
    high_load_sats = [
        (sid, sdata.flow) 
        for sid, sdata in sat_list.items() 
        if sdata.flow > 0.1  # 阈值过滤，避免噪声
    ]
    
    # 如果有足够多的高负载卫星，生成任务对
    if len(high_load_sats) >= 2:
        # 简单策略：选择流量最高的两颗星作为源和目的
        sorted_sats = sorted(high_load_sats, key=lambda x: x[1], reverse=True)
        src_sat_id = int(sorted_sats[0][0])
        dest_sat_id = int(sorted_sats[1][0])
        
        # 生成一个任务
        task_attributes = {
            "TaskId": f"AutoGen_{src_sat_id}_{dest_sat_id}",
            "SrcSatId": src_sat_id,
            "DestSatId": dest_sat_id,
            "PacketSize": 10.0,  # 10 Mbits
            "StartTime": current_utc,  # 使用当前时间作为开始时间
            "TaskPriority": 10  # 中等优先级
        }
        
        route_task_list.append(task_attributes)
    
    # 3. 构造最终输出结构
    routing_demand = {
        "Timestamp": timestamp_str,  # ISO 格式字符串
        "ConstellationId": constellation_id,
        "RouteTaskList": route_task_list
    }
    
    return routing_demand