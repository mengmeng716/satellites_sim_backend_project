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

) -> dict:
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

    output_timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
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
    topology_update = 0 if topology_update_cooldown else candidate_topology_update

    output_topology = result.reconfigured_topology if topology_update else result.initial_topology
    top_difference = result.top_difference if topology_update else {}

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
        },
        "TopDifference": export_top_difference(top_difference),
    }
