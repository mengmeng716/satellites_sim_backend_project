"""Single engine-facing function for topology reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from src import config as project_config

from .reconstruction_config import TopReconstructionConfig
from .src.adapters import (
    export_top_difference,
    extract_positions,
    normalize_topology,
    topology_to_project_shape,
)
from .src.core_nodes import (
    build_random_source_destination_pairs,
    get_core_satellite_ids,
    get_current_core_nodes,
)
from .src.reconfiguration import run_reconfiguration_decision


MIN_AVERAGE_HOP_REDUCTION_RATE = 0.10


def reconstruct_topology(
    topology_state: Any,
    node_state: Any = None,
    constellation: Any = None,
    predicted_links: Any = None,
    config: Optional[TopReconstructionConfig] = None,
) -> dict:
    _ = predicted_links

    active_config = config or TopReconstructionConfig()
    initial_topology = normalize_topology(topology_state)
    topology_ids = sorted(initial_topology)

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
    core_node_number = get_current_core_nodes(
        positions,
        orbit_height_km=orbit_height_km,
        min_elevation_deg=active_config.min_ground_elevation_deg,
    )
    core_ids = get_core_satellite_ids(core_node_number)
    if len(core_ids) < 2:
        source_destination_pairs = []
    else:
        source_destination_pairs = build_random_source_destination_pairs(
            core_ids,
            pair_count=active_config.pair_sample_count,
            seed=active_config.pair_sample_seed,
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
        processing_delay_ms=active_config.processing_delay_ms,
        link_capacity_gbps=project_config.MAX_LINK_SPEED_GBPS,
    )

    average_hop_reduction_rate = result.average_hop_reduction_rate
    topology_update = int(average_hop_reduction_rate >= MIN_AVERAGE_HOP_REDUCTION_RATE)
    output_topology = result.reconfigured_topology if topology_update else result.initial_topology
    top_difference = result.top_difference if topology_update else {}
    constellation_id = str(project_config.SELECTED_CONSTELLATION_ID)

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
            "AverageHopReductionRate": round(average_hop_reduction_rate, 6),
            "TopologyDecisionTime": round(result.metadata.get("TopologyDecisionTime", 0.0), 6),
        },
        "TopDifference": export_top_difference(top_difference),
    }
