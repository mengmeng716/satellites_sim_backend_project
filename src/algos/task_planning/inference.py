# src/algos/task_planning/inference.py

from __future__ import annotations

from typing import Any, Dict

import time
import numpy as np
import torch

from src.simulation.data_model import SatelliteNodeState, TopologyState

from .config import (
    ACTOR_MODEL_PATHS,
    DEFAULT_CONSTELLATION,
    SUPPORTED_CONSTELLATIONS,
    NUM_CANDIDATE_PATHS,
)
from .src.graph_adapter import build_graph_from_simulation_state
from .src.path_generator import generate_candidate_paths, NoFeasiblePathError
from .src.state_encoder import encode_state
from .src.utils import (
    mask_and_normalize_action,
    build_allocations,
    add_allocation_times,
    compute_allocation_metrics,
    validate_allocations_capacity,
)


_ACTORS: dict[str, torch.jit.ScriptModule] = {}


def normalize_constellation(constellation: str | int | None) -> str:
    """统一 constellation 开关取值。"""
    key = DEFAULT_CONSTELLATION if constellation is None else str(constellation)

    if key not in SUPPORTED_CONSTELLATIONS:
        raise ValueError(
            f"unsupported constellation: {key}. "
            f"supported values: {list(SUPPORTED_CONSTELLATIONS)}"
        )

    return key


def get_actor(constellation: str | int | None = None) -> torch.jit.ScriptModule:
    """
    按 constellation 加载并缓存 actor-only 模型。

    constellation="3600" -> actor_only_3600.pt
    constellation="432"  -> actor_only_432.pt
    """
    key = normalize_constellation(constellation)

    if key not in _ACTORS:
        actor_path = ACTOR_MODEL_PATHS[key]

        if not actor_path.exists():
            raise FileNotFoundError(
                f"actor-only model for constellation={key} not found: {actor_path}. "
                f"Please put the exported TorchScript actor at this path."
            )

        actor = torch.jit.load(str(actor_path), map_location="cpu")
        actor.eval()
        _ACTORS[key] = actor

    return _ACTORS[key]


def actor_predict(actor: torch.jit.ScriptModule, obs: np.ndarray) -> np.ndarray:
    """执行 actor 前向推理，并将输出从 [-1, 1] 映射到环境动作空间 [0, 1]。"""

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        raw_action = actor(obs_tensor)

    if isinstance(raw_action, (tuple, list)):
        raw_action = raw_action[0]

    raw_action = raw_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
    action = (raw_action + 1.0) / 2.0

    return np.clip(action, 0.0, 1.0).astype(np.float32)


def task_planning_inference(
    topology_state: TopologyState,
    node_state: SatelliteNodeState | None,
    task: Dict[str, Any],
    constellation: str | int | None = None,
) -> Dict[str, Any]:
    """
    单业务流推理主流程。

    topology_state / node_state 始终来自 SimulationEngine 当前维护的状态表。
    constellation 只控制加载哪个 actor 模型：
        3600 -> 3600 星座训练模型
        432  -> 432 星座训练模型

    返回结果使用扁平结构，避免外部再从 metrics / capacity_check 多层嵌套中取值。
    """

    decision_t0 = time.perf_counter()

    try:
        constellation_key = normalize_constellation(constellation)
    except ValueError as exc:
        return _fail(
            task=task,
            reason=str(exc),
            constellation=str(constellation),
            decision_t0=decision_t0,
        )

    task_id = task.get("TaskId", "unknown_task")
    src = task.get("SourceSatId")
    dst = task.get("TargetSatId")

    try:
        demand_gbps = float(task.get("DemandGbps", 0.0))
    except (TypeError, ValueError):
        demand_gbps = 0.0

    if src is None or dst is None:
        return _fail(
            task=task,
            reason="missing SourceSatId or TargetSatId",
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            decision_t0=decision_t0,
        )

    if demand_gbps <= 0:
        return _fail(
            task=task,
            reason="invalid demand_gbps",
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            decision_t0=decision_t0,
        )

    src_str = str(src)
    dst_str = str(dst)

    # 1. 状态对象转算法图。
    G = build_graph_from_simulation_state(
        topology_state=topology_state,
        node_state=node_state,
    )

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return _fail(
            task=task,
            reason="empty topology graph",
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            decision_t0=decision_t0,
        )

    # 2. 生成候选路径。
    try:
        path_result = generate_candidate_paths(
            G=G,
            src=src,
            dst=dst,
            num_paths=NUM_CANDIDATE_PATHS,
        )
    except NoFeasiblePathError as exc:
        return _fail(
            task=task,
            reason=str(exc),
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            decision_t0=decision_t0,
        )

    paths = path_result["paths"]
    valid_mask = path_result["valid_mask"]

    if sum(valid_mask) <= 0:
        return _fail(
            task=task,
            reason="no valid candidate path",
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            decision_t0=decision_t0,
        )

    # 3. 模型推理与分流比例计算。
    obs = encode_state(
        G=G,
        candidate_paths=paths,
        valid_mask=valid_mask,
        demand_gbps=demand_gbps,
    )

    actor = get_actor(constellation_key)
    raw_action = actor_predict(actor, obs)

    ratios = mask_and_normalize_action(
        action=raw_action,
        valid_mask=valid_mask,
    )

    allocations = build_allocations(
        paths=paths,
        ratios=ratios,
        valid_mask=valid_mask,
        demand_gbps=demand_gbps,
    )

    if not allocations:
        return _fail(
            task=task,
            reason="empty allocations after action normalization",
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            decision_t0=decision_t0,
        )

    # 4. 补充每条路径占用时间。
    start_time = float(task.get("arrival_sim_time", 0.0) or 0.0)
    allocations = add_allocation_times(
        G=G,
        allocations=allocations,
        start_time=start_time,
    )

    # 5. 容量校验与指标计算。
    capacity_ok, capacity_reason, capacity_detail = validate_allocations_capacity(
        G=G,
        allocations=allocations,
    )

    network_metrics = compute_allocation_metrics(
        G=G,
        allocations=allocations,
    )

    decision_total_ms = (time.perf_counter() - decision_t0) * 1000.0

    if not capacity_ok:
        return _fail(
            task=task,
            reason=capacity_reason,
            constellation=constellation_key,
            demand_gbps=demand_gbps,
            network_metrics=network_metrics,
            capacity_detail=capacity_detail,
            decision_total_ms=decision_total_ms,
        )

    return {
        "accepted": True,
        "reason": "",
        "constellation": constellation_key,

        "task_id": task_id,
        "src": src_str,
        "dst": dst_str,
        "demand_gbps": demand_gbps,

        "decision_total_ms": decision_total_ms,

        "paths": [item["path"] for item in allocations],
        "ratios": [float(item["ratio"]) for item in allocations],
        "allocations": allocations,

        "avg_delay_ms": float(network_metrics.get("avg_delay_ms", 0.0)),
        "jitter_ms": float(network_metrics.get("jitter_ms", 0.0)),
        "max_link_utilization_after": float(
            network_metrics.get("max_link_utilization_after", 0.0)
        ),
        "overflow_amount": float(network_metrics.get("overflow_amount", 0.0)),

        "capacity_status": "OK",
        "capacity_total_overflow": float(capacity_detail.get("total_overflow", 0.0)),
        "capacity_max_util_after": float(capacity_detail.get("max_utilization_after", 0.0)),
    }


def _fail(
    task: Dict[str, Any],
    reason: str,
    constellation: str | int | None = None,
    demand_gbps: float | None = None,
    network_metrics: Dict[str, Any] | None = None,
    capacity_detail: Dict[str, Any] | None = None,
    decision_t0: float | None = None,
    decision_total_ms: float | None = None,
) -> Dict[str, Any]:
    """统一失败返回格式。"""

    network_metrics = network_metrics or {}
    capacity_detail = capacity_detail or {}

    if decision_total_ms is None:
        if decision_t0 is not None:
            decision_total_ms = (time.perf_counter() - decision_t0) * 1000.0
        else:
            decision_total_ms = 0.0

    if demand_gbps is None:
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
        "src": None if task.get("SourceSatId") is None else str(task.get("SourceSatId")),
        "dst": None if task.get("TargetSatId") is None else str(task.get("TargetSatId")),
        "demand_gbps": float(demand_gbps),

        "decision_total_ms": float(decision_total_ms),

        "paths": [],
        "ratios": [],
        "allocations": [],

        "avg_delay_ms": network_metrics.get("avg_delay_ms"),
        "jitter_ms": network_metrics.get("jitter_ms"),
        "max_link_utilization_after": network_metrics.get("max_link_utilization_after"),
        "overflow_amount": network_metrics.get("overflow_amount"),

        "capacity_status": "FAILED",
        "capacity_total_overflow": capacity_detail.get("total_overflow"),
        "capacity_max_util_after": capacity_detail.get("max_utilization_after"),
    }
