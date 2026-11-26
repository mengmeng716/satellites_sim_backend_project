"""Single engine-facing function for topology reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src import config as project_config

from .reconstruction_config import TopReconstructionConfig
from .src.adapters import (
    export_top_difference,
    extract_positions,
    normalize_topology,
    topology_to_project_shape,
)
from .src.business_demand_intensity import compute_business_demand_intensity
from .src.core_nodes import (
    build_business_demand_source_destination_pairs,
)
from .src.forwarding_delay import build_forwarding_delay_map
from .src.reconfiguration import run_reconfiguration_decision


def reconstruct_topology(
    topology_state: Any,
    node_state: Any = None,
    constellation: Any = None,
    predicted_links: Any = None,
    config: Optional[TopReconstructionConfig] = None,
    previous_topology_update: int = 0,
    task_list: Optional[list] = None,
    *,
    for_capacity_evaluation: bool = False,
    fixed_pairs=None,
    capacity_selector=None,
) -> dict:
    """拓扑重构对外入口。

    :param task_list: 用户业务需求列表（字段见 src/README.md「用户业务需求输入」）。
        当前版本只接收并透传，重构决策仍以链路预测给出的业务强度为准。
    """
    active_config = config or TopReconstructionConfig()
    initial_topology = normalize_topology(topology_state)
    topology_ids = sorted(initial_topology)
    predicted_constellation_id = (
        predicted_links.get("ConstellationId")
        if isinstance(predicted_links, dict)
        else getattr(predicted_links, "ConstellationId", None)
    )
    constellation_id = str(
        getattr(constellation, "constellationId", None)
        or predicted_constellation_id
        or project_config.SELECTED_CONSTELLATION_ID
    )
    if not constellation_id or constellation_id == "None":
        constellation_id = str(project_config.SELECTED_CONSTELLATION_ID)

    if constellation is None:
        num_planes = project_config.NUM_ORBIT_PLANES
        sats_per_plane = project_config.SATS_PER_PLANE
        orbit_height_km = project_config.ORBIT_HEIGHT_KM
    else:
        num_planes = int(getattr(constellation, "numOrbitPlane", project_config.NUM_ORBIT_PLANES))
        sats_per_plane = int(getattr(constellation, "numSatsInPlane", project_config.SATS_PER_PLANE))
        orbit_height_km = float(getattr(constellation, "orbitHeight", project_config.ORBIT_HEIGHT_KM))

    if topology_ids and sats_per_plane > 0:
        num_planes = max(num_planes, max(topology_ids) // sats_per_plane + 1)

    positions = extract_positions(node_state=node_state, constellation=constellation)
    business_intensity = compute_business_demand_intensity(
        predicted_links=predicted_links,
        initial_topology=initial_topology,
        horizon_steps=active_config.business_intensity_horizon_steps,
        decay_lambda=active_config.business_intensity_decay_lambda,
        max_link_capacity_gbps=project_config.MAX_LINK_SPEED_GBPS,
    )
    core_node_count = active_config.core_node_count(constellation_id)
    source_destination_pairs = fixed_pairs
    if source_destination_pairs is None:
        source_destination_pairs = build_business_demand_source_destination_pairs(
            node_state=node_state,
            satellite_business_intensity=business_intensity.satellite_intensity,
            core_node_count=core_node_count,
        )
    forwarding_params = active_config.forwarding_delay_params(constellation_id)
    forwarding_delay_ms_by_node = build_forwarding_delay_map(
        business_intensity.satellite_intensity,
        base_ms=forwarding_params.base_ms,
        kappa_ms=forwarding_params.kappa_ms,
        gamma=forwarding_params.gamma,
        max_ms=forwarding_params.max_ms,
        satellite_ids=topology_ids,
    )

    # 使用仿真时刻（TopologyState.timestamp，Unix ms），不用墙钟时间
    output_timestamp_ms = int(
        getattr(topology_state, "timestamp", 0)
        or datetime.now(timezone.utc).timestamp() * 1000
    )
    result = run_reconfiguration_decision(
        initial_topology=initial_topology,
        positions=positions,
        source_destination_pairs=source_destination_pairs,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        orbit_height_km=orbit_height_km,
        timestamp_ms=output_timestamp_ms,
        offset_window=active_config.offset_window,
        max_switch_rate=active_config.max_switch_rate,
        link_capacity_gbps=project_config.MAX_LINK_SPEED_GBPS,
        forwarding_delay_ms_by_node=forwarding_delay_ms_by_node,
        link_business_intensity=business_intensity.link_intensity,
        protected_link_intensity_threshold=active_config.protected_link_intensity_threshold,
        capacity_selector=capacity_selector,
    )

    average_hop_reduction_rate = result.average_hop_reduction_rate
    average_total_delay_reduction_rate = result.average_total_delay_reduction_rate
    min_average_reduction_rate = float(
        getattr(active_config, "min_average_reduction_rate", 0.10)
    )
    candidate_topology_update = int(
        average_hop_reduction_rate >= min_average_reduction_rate
        or average_total_delay_reduction_rate >= min_average_reduction_rate
    )
    previous_topology_update = int(previous_topology_update or 0)
    topology_update_cooldown = int(previous_topology_update == 1)
    if for_capacity_evaluation:
        topology_update = int(any(added or removed for added, removed in result.top_difference.values()))
    else:
        topology_update = 0 if topology_update_cooldown else candidate_topology_update

    output_topology = result.reconfigured_topology if topology_update else result.initial_topology
    top_difference = result.top_difference if topology_update else {}

    link_metrics = compute_link_metrics(output_topology)

    return {
        "ConstellationId": constellation_id,
        "Timestamp": result.timestamp_ms,
        "TopologyId": f"{constellation_id}_{result.timestamp_ms}",
        "TopologyUpdate": topology_update,
        "newTopology": topology_to_project_shape(output_topology),
        "TopQualities": {
            "NetworkLinkSwitchRate": round(result.network_link_switch_rate, 6),
            "OrbitLayerSwitchStdDev": round(result.orbit_layer_switch_std_dev, 6),
            "AverageHopsBefore": round(result.baseline_summary.average_hops, 6),
            "AverageHopsAfter": round(result.reconfigured_summary.average_hops, 6),
            "AverageHopReduction": round(result.average_hop_reduction, 6),
            "AverageHopReductionRate": round(average_hop_reduction_rate, 6),
            "AverageTotalDelayBeforeMs": round(
                result.baseline_summary.average_total_delay_ms,
                6,
            ),
            "AverageTotalDelayAfterMs": round(
                result.reconfigured_summary.average_total_delay_ms,
                6,
            ),
            "AverageTotalDelayReductionRate": round(
                average_total_delay_reduction_rate,
                6,
            ),
            "PairCount": len(source_destination_pairs),
            "ProtectedLinkCount": round(result.metadata.get("ProtectedLinkCount", 0.0), 6),
            "TopologyDecisionTime": round(result.metadata.get("TopologyDecisionTime", 0.0), 6),
            # ---- 表2-11 规定的四项 ----
            "AvgLinkDistanceScore": link_metrics["AvgLinkDistanceScore"],
            "AvgLinkAvailableDurationScore": link_metrics["AvgLinkAvailableDurationScore"],
            "TotalTransmissionCapacity": link_metrics["TotalTransmissionCapacity"],
            "AvgLinkPropagationDelay": link_metrics["AvgLinkPropagationDelay"],
        },
        "TopQualitiesNote": link_metrics["Note"],
        "TopDifference": export_top_difference(top_difference),
    }


def compute_link_metrics(topology: Any) -> dict:
    """按表2-11 统计链路级指标。

    * ``AvgLinkDistanceScore``：``1 − 距离/最大斜距``，截断到 [0,1]
    * ``AvgLinkAvailableDurationScore``：几何近似 ``1 − (距离/最大斜距)²``，
      距离越近可用时长越长；精确值需可见时长预测，此处为可解释的替代口径
    * ``TotalTransmissionCapacity``：去重后的额定容量之和（Gbps）
    * ``AvgLinkPropagationDelay``：链路传播时延均值（ms）

    链路距离由 ``update_topology_snapshot`` 在每个微周期刷新，
    正常运行时不会为 0；若整张拓扑的距离都为 0（例如初始化后尚未刷新），
    距离类指标返回 ``None`` 并在 ``Note`` 中说明，不给出虚假评分。
    """
    max_range = float(getattr(project_config, "MAX_SLANT_RANGE_KM", 2000.0)) or 2000.0

    distances = []
    delays = []
    seen = set()
    capacity_total = 0.0
    for source_id, links in (topology or {}).items():
        for target_id, attr in links:
            src_id, dst_id = int(source_id), int(target_id)
            distances.append(float(attr.get("LinkDistance", 0.0) or 0.0))
            delays.append(float(attr.get("LinkPropagationDelay", 0.0) or 0.0))
            key = (min(src_id, dst_id), max(src_id, dst_id))
            if key not in seen:
                seen.add(key)
                capacity_total += float(attr.get("LinkCapacity", 0.0) or 0.0)

    if not distances:
        return {
            "AvgLinkDistanceScore": None,
            "AvgLinkAvailableDurationScore": None,
            "TotalTransmissionCapacity": 0.0,
            "AvgLinkPropagationDelay": None,
            "Note": "无链路，指标不可计算",
        }

    if max(distances) <= 1e-9:
        return {
            "AvgLinkDistanceScore": None,
            "AvgLinkAvailableDurationScore": None,
            "TotalTransmissionCapacity": round(capacity_total, 6),
            "AvgLinkPropagationDelay": None,
            "Note": "链路距离尚未刷新（全为 0），距离类指标不可计算",
        }

    distance_scores = [max(0.0, min(1.0, 1.0 - d / max_range)) for d in distances]
    duration_scores = [max(0.0, min(1.0, 1.0 - (d / max_range) ** 2)) for d in distances]

    return {
        "AvgLinkDistanceScore": round(sum(distance_scores) / len(distance_scores), 6),
        "AvgLinkAvailableDurationScore": round(sum(duration_scores) / len(duration_scores), 6),
        "TotalTransmissionCapacity": round(capacity_total, 6),
        "AvgLinkPropagationDelay": round(sum(delays) / len(delays), 6),
        "Note": "AvgLinkAvailableDurationScore 为基于距离的几何近似（精确值需可见时长预测）",
    }
