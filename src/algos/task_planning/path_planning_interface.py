# src/algos/task_planning/interface.py

from __future__ import annotations

from typing import Any, Dict

from src.simulation.data_model import SatelliteNodeState, TopologyState

from .config import DEFAULT_CONSTELLATION, SUPPORTED_CONSTELLATIONS
from .inference import task_planning_inference


REQUIRED_TASK_FIELDS = (
    "TaskId",
    "TaskType",
    "SourceSatId",
    "TargetSatId",
    "DemandGbps",
    "Duration",
    "TaskPriority",
)


def run_task_planning(
    topology_state: TopologyState,
    node_state: SatelliteNodeState | None,
    task_planning_params: Dict[str, Any],
    constellation: str | int = DEFAULT_CONSTELLATION,
) -> Dict[str, Any]:
    """
    任务规划模块统一入口。

    topology_state / node_state：
        始终由外部 SimulationEngine 传入当前状态表，不在这里切换拓扑。

    constellation：
        只用于选择加载哪个 actor 模型：
            3600 -> actor_only_3600.pt
            432  -> actor_only_432.pt
    """

    if task_planning_params is None:
        task_planning_params = {}

    constellation_key = str(constellation)

    missing_fields = [
        field for field in REQUIRED_TASK_FIELDS
        if field not in task_planning_params
    ]

    if missing_fields:
        return _fail_before_inference(
            task=task_planning_params,
            reason=f"missing required task fields: {missing_fields}",
            constellation=constellation_key,
        )

    if constellation_key not in SUPPORTED_CONSTELLATIONS:
        return _fail_before_inference(
            task=task_planning_params,
            reason=f"unsupported constellation: {constellation_key}",
            constellation=constellation_key,
        )

    if not isinstance(topology_state, TopologyState):
        return _fail_before_inference(
            task=task_planning_params,
            reason="topology_state must be TopologyState",
            constellation=constellation_key,
        )

    if node_state is not None and not isinstance(node_state, SatelliteNodeState):
        return _fail_before_inference(
            task=task_planning_params,
            reason="node_state must be SatelliteNodeState or None",
            constellation=constellation_key,
        )

    return task_planning_inference(
        topology_state=topology_state,
        node_state=node_state,
        task=task_planning_params,
        constellation=constellation_key,
    )


def _fail_before_inference(
    task: Dict[str, Any],
    reason: str,
    constellation: str | int | None = None,
) -> Dict[str, Any]:
    """进入 inference 前的统一失败返回。"""

    try:
        demand_gbps = float(task.get("DemandGbps", 0.0) or 0.0)
    except (TypeError, ValueError):
        demand_gbps = 0.0

    return {
        "accepted": False,
        "reason": reason,
        "constellation": None if constellation is None else str(constellation),
        "actor_model_path": None,

        "task_id": task.get("TaskId", "unknown_task"),
        "src": task.get("SourceSatId"),
        "dst": task.get("TargetSatId"),
        "demand_gbps": demand_gbps,

        "decision_total_ms": 0.0,

        "paths": [],
        "ratios": [],
        "allocations": [],

        "avg_delay_ms": None,
        "jitter_ms": None,
        "max_link_utilization_after": None,
        "overflow_amount": None,

        "capacity_status": "FAILED",
        "capacity_total_overflow": None,
        "capacity_max_util_after": None,
    }
