"""Minimal data conversion helpers for topology reconstruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPEED_OF_LIGHT_KM_S = 299792.458

LinkAttributes = Dict[str, Any]
Topology = Dict[int, List[Tuple[int, LinkAttributes]]]
PositionMap = Dict[int, Tuple[float, float]]


def propagation_delay_ms(distance_km: float) -> float:
    return float(distance_km) / SPEED_OF_LIGHT_KM_S * 1000.0


def _load_link_model() -> Optional[type]:
    try:
        from src.simulation.data_model import LinksQualitiesValue

        return LinksQualitiesValue
    except Exception:
        data_model_path = Path(__file__).resolve().parents[3] / "simulation" / "data_model.py"
        spec = importlib.util.spec_from_file_location("_top_reconstruction_data_model", data_model_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "LinksQualitiesValue", None)


LinksQualitiesValue = _load_link_model()


def satellite_plane_slot(sat_id: int, sats_per_plane: int) -> Tuple[int, int]:
    return sat_id // sats_per_plane, sat_id % sats_per_plane


def plane_slot_to_sat_id(plane: int, slot: int, num_planes: int, sats_per_plane: int) -> int:
    return (plane % num_planes) * sats_per_plane + (slot % sats_per_plane)


def initial_inter_orbit_slot(
    plane: int,
    slot: int,
    direction: int,
    num_planes: int,
    sats_per_plane: int,
    inter_plane_offset: int = -1,
    seam_offset: int = 38,
) -> Tuple[int, int]:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    target_plane = (plane + direction) % num_planes
    if direction == 1:
        target_slot = (
            (slot - seam_offset) % sats_per_plane
            if plane == num_planes - 1 and target_plane == 0
            else (slot + inter_plane_offset) % sats_per_plane
        )
    else:
        target_slot = (
            (slot + seam_offset) % sats_per_plane
            if plane == 0 and target_plane == num_planes - 1
            else (slot - inter_plane_offset) % sats_per_plane
        )
    return target_plane, target_slot


def allowed_inter_orbit_targets(
    sat_id: int,
    direction: int,
    num_planes: int,
    sats_per_plane: int,
    offset_window: int = 1,
) -> List[int]:
    plane, slot = satellite_plane_slot(sat_id, sats_per_plane)
    target_plane, base_slot = initial_inter_orbit_slot(
        plane,
        slot,
        direction,
        num_planes,
        sats_per_plane,
    )
    return [
        plane_slot_to_sat_id(target_plane, base_slot + offset, num_planes, sats_per_plane)
        for offset in range(-offset_window, offset_window + 1)
    ]


def same_orbit(source_id: int, target_id: int, sats_per_plane: int) -> bool:
    return source_id // sats_per_plane == target_id // sats_per_plane


def _read_value(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    elif obj is not None:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canonical_link_attributes(
    attr: Any = None,
    distance_km: Optional[float] = None,
    capacity_gbps: Optional[float] = None,
    delay_ms: Optional[float] = None,
    link_type: str = "unknown",
) -> LinkAttributes:
    if isinstance(attr, (list, tuple)) and len(attr) >= 3:
        distance = _as_float(attr[0])
        capacity = _as_float(attr[1], 10.0)
        delay = _as_float(attr[2], propagation_delay_ms(distance))
    else:
        distance = _as_float(_read_value(attr, ("LinkDistance", "link_distance")), 0.0)
        capacity = _as_float(_read_value(attr, ("LinkCapacity", "link_capacity")), 10.0)
        delay = _as_float(
            _read_value(attr, ("LinkPropagationDelay", "link_propagation_delay")),
            propagation_delay_ms(distance),
        )

    if distance_km is not None:
        distance = float(distance_km)
    if capacity_gbps is not None:
        capacity = float(capacity_gbps)
    if delay_ms is not None:
        delay = float(delay_ms)

    return {
        "LinkDistance": max(0.0, distance),
        "LinkCapacity": max(0.0, capacity),
        "LeftCapacity": max(
            0.0,
            _as_float(_read_value(attr, ("LeftCapacity", "left_capacity")), capacity),
        ),
        "CurrentFlow": max(0.0, _as_float(_read_value(attr, ("CurrentFlow", "current_flow")), 0.0)),
        "LinkPropagationDelay": max(0.0, delay),
        "QueueDelay": max(0.0, _as_float(_read_value(attr, ("QueueDelay", "queue_delay")), 0.0)),
        "TransmissionDelay": max(
            0.0,
            _as_float(_read_value(attr, ("TransmissionDelay", "transmission_delay")), 0.0),
        ),
        "PacketLossRate": min(
            1.0,
            max(0.0, _as_float(_read_value(attr, ("PacketLossRate", "packet_loss_rate")), 0.0)),
        ),
        "HeatValue": min(1.0, max(0.0, _as_float(_read_value(attr, ("HeatValue", "heat_value")), 0.0))),
        "LinkType": str(_read_value(attr, ("LinkType", "link_type"), link_type)),
    }


def default_link_attributes(
    distance_km: float = 0.0,
    capacity_gbps: float = 10.0,
    link_type: str = "unknown",
) -> LinkAttributes:
    return canonical_link_attributes(
        distance_km=distance_km,
        capacity_gbps=capacity_gbps,
        delay_ms=propagation_delay_ms(distance_km),
        link_type=link_type,
    )


def iter_edges(topology: Topology) -> Iterable[Tuple[int, int, LinkAttributes]]:
    for source_id, links in topology.items():
        for target_id, attr in links:
            yield int(source_id), int(target_id), attr


def normalize_topology(topology_state_or_topology: Any) -> Topology:
    if hasattr(topology_state_or_topology, "new_topology"):
        raw_topology = topology_state_or_topology.new_topology
    elif isinstance(topology_state_or_topology, Mapping) and "newTopology" in topology_state_or_topology:
        raw_topology = topology_state_or_topology.get("newTopology", {})
    else:
        raw_topology = topology_state_or_topology or {}

    topology: Topology = {}
    for raw_source_id, raw_links in raw_topology.items():
        try:
            source_id = int(raw_source_id)
        except (TypeError, ValueError):
            continue

        iterable = raw_links.items() if isinstance(raw_links, Mapping) else raw_links or []
        topology[source_id] = []
        for item in iterable:
            try:
                target_id, raw_attr = item
                topology[source_id].append((int(target_id), canonical_link_attributes(raw_attr)))
            except (TypeError, ValueError):
                continue
    return topology


def extract_positions(node_state: Any = None, constellation: Any = None) -> PositionMap:
    positions: PositionMap = {}
    for sat in getattr(constellation, "satelliteList", []) or []:
        try:
            positions[int(getattr(sat, "satId"))] = (
                float(getattr(sat, "latitude")),
                float(getattr(sat, "longitude")),
            )
        except (TypeError, ValueError, AttributeError):
            continue

    raw_sat_list = {}
    if hasattr(node_state, "sat_list"):
        raw_sat_list = node_state.sat_list or {}
    elif isinstance(node_state, Mapping):
        raw_sat_list = node_state.get("SatList") or node_state.get("sat_list") or {}

    for raw_sat_id, sat_data in raw_sat_list.items():
        try:
            positions[int(raw_sat_id)] = (
                _as_float(_read_value(sat_data, ("Latitude", "latitude")), 0.0),
                _as_float(_read_value(sat_data, ("Longitude", "longitude")), 0.0),
            )
        except (TypeError, ValueError):
            continue
    return positions


def topology_to_project_shape(topology: Topology) -> Dict[str, Dict[str, Any]]:
    exported: Dict[str, Dict[str, Any]] = {}
    for source_id in sorted(topology):
        exported[str(source_id)] = {}
        for target_id, attr in sorted(topology[source_id], key=lambda item: item[0]):
            link = canonical_link_attributes(attr)
            exported[str(source_id)][str(target_id)] = LinksQualitiesValue(
                LinkDistance=link["LinkDistance"],
                LinkCapacity=link["LinkCapacity"],
                LeftCapacity=link["LeftCapacity"],
                CurrentFlow=link["CurrentFlow"],
                LinkPropagationDelay=link["LinkPropagationDelay"],
                QueueDelay=link["QueueDelay"],
                TransmissionDelay=link["TransmissionDelay"],
                PacketLossRate=link["PacketLossRate"],
                HeatValue=link["HeatValue"],
            ) if LinksQualitiesValue is not None else {
                key: value for key, value in link.items() if key != "LinkType"
            }
    return exported


def export_top_difference(
    top_difference: Mapping[int, Tuple[Sequence[int], Sequence[int]]],
) -> Dict[str, List[List[str]]]:
    return {
        str(source_id): [
            [str(target_id) for target_id in added],
            [str(target_id) for target_id in deleted],
        ]
        for source_id, (added, deleted) in sorted(top_difference.items())
    }
