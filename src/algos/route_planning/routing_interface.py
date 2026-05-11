"""
路由优化模块对外接口。

本文件只保留路由优化模块的对外调用入口，并直接返回接口文档要求的展开字段。

输入：
    Timestamp, ConstellationId, TaskId, SrcSatId, DestSatId,
    PacketSize, StartTime, TaskPriority

输出：
    TaskId, RoutePath, TotalHopCount, PathTotalCost,
    EndToEndDelay, ISLValidRate, StartTime, EndTime

说明：
    1. 本文件不训练模型；
    2. 本文件不生成拓扑；
    3. 拓扑状态和节点状态由总项目 data_model 维护；
    4. 本文件通过 route_planning_context 读取当前 topology_state 和 node_state；
    5. 根据 ConstellationId 自动选择 432 或 3600 对应的模型权重。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .inference import route_planning_inference


def route_planning_execution(
    topology_state: Any,      # [新增] 显式传入拓扑状态
    node_state: Any,          # [新增] 显式传入节点状态
    Timestamp: Any,
    ConstellationId: str,
    TaskId: str,
    SrcSatId: int,
    DestSatId: int,
    PacketSize: float,
    StartTime: Any,
    TaskPriority: int,
) -> Dict[str, Any]:
    """
    路由优化单任务接口。
    """

    task = {
        "Timestamp": Timestamp,
        "ConstellationId": ConstellationId,
        "TaskId": TaskId,
        "SrcSatId": SrcSatId,
        "DestSatId": DestSatId,
        "PacketSize": PacketSize,
        "StartTime": StartTime,
        "TaskPriority": TaskPriority,
    }

    # 显式传递状态给底层推理逻辑
    raw_result = route_planning_inference(
        topology_state=topology_state,
        node_state=node_state,
        task=task,
    )

    # 检查任务是否成功到达
    is_arrived = (
        raw_result.get("Status") == "Arrived"
        or raw_result.get("status") == "Arrived"
    )

    route_path = raw_result.get("RoutePath") or raw_result.get("route_path") or []
    if not is_arrived:
        route_path = []

    total_hop_count = raw_result.get(
        "TotalHopCount",
        raw_result.get("hop_count", max(len(route_path) - 1, 0)),
    )
    if not is_arrived:
        total_hop_count = 0

    path_total_cost = raw_result.get(
        "PathTotalCost",
        raw_result.get("total_path_cost", 0.0),
    )

    end_to_end_delay = raw_result.get(
        "EndToEndDelay",
        raw_result.get("end_to_end_delay_ms", 0.0),
    )

    isl_valid_rate = raw_result.get(
        "ISLValidRate",
        100.0 if len(route_path) > 1 else 0.0,
    )
    if not is_arrived:
        isl_valid_rate = 0.0

    start_time = raw_result.get("StartTime", StartTime)
    end_time = raw_result.get("EndTime", start_time)

    return {
        "TaskId": str(TaskId),
        "RoutePath": [int(x) for x in route_path],
        "TotalHopCount": int(total_hop_count),
        "PathTotalCost": float(path_total_cost),
        "EndToEndDelay": float(end_to_end_delay),
        "ISLValidRate": float(isl_valid_rate),
        "StartTime": start_time,
        "EndTime": end_time,
    }


def route_planning_batch_execution(
    topology_state: Any,      # [新增] 显式传入拓扑状态
    node_state: Any,          # [新增] 显式传入节点状态
    Timestamp: Any,
    ConstellationId: str,
    RouteTaskList: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    路由优化批量任务接口。
    """

    results: List[Dict[str, Any]] = []

    for task in RouteTaskList or []:
        results.append(
            route_planning_execution(
                topology_state=topology_state,
                node_state=node_state,
                Timestamp=Timestamp,
                ConstellationId=ConstellationId,
                TaskId=task.get("TaskId"),
                SrcSatId=task.get("SrcSatId"),
                DestSatId=task.get("DestSatId"),
                PacketSize=task.get("PacketSize"),
                StartTime=task.get("StartTime"),
                TaskPriority=task.get("TaskPriority", 0),
            )
        )

    return results


__all__ = [
    "route_planning_execution",
    "route_planning_batch_execution",
]