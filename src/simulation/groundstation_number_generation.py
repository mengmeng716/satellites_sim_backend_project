"""Ground-station nearest-satellite assignment for node state.

The simulation keeps ``GroundStationNumber`` on each satellite node.  Every
ground station chooses its nearest visible satellite, and that satellite stores
all matching ground-station ids (1-based).
"""

from __future__ import annotations

from math import radians, sin
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src import config as simulation_config
from src.utils.gs_manager import GroundStationManager


PositionMap = Dict[int, Tuple[float, float]]


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


def _raw_sat_list(node_state: Any) -> Mapping[Any, Any]:
    if hasattr(node_state, "sat_list"):
        return node_state.sat_list or {}
    if isinstance(node_state, Mapping):
        return node_state.get("SatList") or node_state.get("sat_list") or {}
    return {}


def _satellite_positions(node_state: Any) -> PositionMap:
    positions: PositionMap = {}
    for raw_sat_id, sat_data in _raw_sat_list(node_state).items():
        try:
            sat_id = int(raw_sat_id)
            latitude = float(_read_value(sat_data, ("Latitude", "latitude")))
            longitude = float(_read_value(sat_data, ("Longitude", "longitude")))
        except (TypeError, ValueError):
            continue
        positions[sat_id] = (latitude, longitude)
    return positions


def _normalize_ground_station_numbers(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = (value,)

    numbers: List[int] = []
    for item in items:
        try:
            station_number = int(item)
        except (TypeError, ValueError):
            continue
        if station_number > 0:
            numbers.append(station_number)
    return numbers


def _write_ground_station_number(sat_data: Any, value: Any) -> None:
    numbers = _normalize_ground_station_numbers(value)
    if isinstance(sat_data, dict):
        if "GroundStationNumber" in sat_data:
            sat_data["GroundStationNumber"] = numbers
        elif "ground_station_number" in sat_data:
            sat_data["ground_station_number"] = numbers
        else:
            sat_data["GroundStationNumber"] = numbers
    else:
        setattr(sat_data, "ground_station_number", numbers)


def _satellite_ecef_km(
    latitude_deg: float,
    longitude_deg: float,
    orbit_height_km: float,
) -> np.ndarray:
    radius = float(simulation_config.EARTH_RADIUS_KM) + float(orbit_height_km)
    phi = radians(latitude_deg)
    lam = radians(longitude_deg)
    return np.array(
        [
            radius * np.cos(phi) * np.cos(lam),
            radius * np.cos(phi) * np.sin(lam),
            radius * np.sin(phi),
        ],
        dtype=float,
    )


def _is_visible_from_ground(
    ground_ecef_km: np.ndarray,
    satellite_ecef_km: np.ndarray,
    min_elevation_deg: float,
) -> bool:
    los = satellite_ecef_km - ground_ecef_km
    los_norm = float(np.linalg.norm(los))
    ground_norm = float(np.linalg.norm(ground_ecef_km))
    if los_norm <= 0.0 or ground_norm <= 0.0:
        return False

    zenith = ground_ecef_km / ground_norm
    sin_elevation = float(np.dot(los, zenith) / los_norm)
    return sin_elevation >= sin(radians(min_elevation_deg))


def calculate_ground_station_numbers(
    node_state: Any,
    orbit_height_km: float,
    min_elevation_deg: Optional[float] = None,
) -> Dict[int, List[int]]:
    """Return ``satellite_id -> [ground_station_number, ...]`` assignments."""
    positions = _satellite_positions(node_state)
    if not positions:
        return {}
    min_elevation_deg = float(
        simulation_config.min_elevation_deg
        if min_elevation_deg is None
        else min_elevation_deg
    )

    manager = GroundStationManager()
    ground_ecef = manager.get_ecef_coordinates()
    if ground_ecef is None or len(ground_ecef) == 0:
        return {}

    sat_ids_array = np.array(sorted(positions), dtype=int)
    sat_coords_array = np.array(
        [
            _satellite_ecef_km(
                positions[int(sat_id)][0],
                positions[int(sat_id)][1],
                orbit_height_km,
            )
            for sat_id in sat_ids_array
        ],
        dtype=float,
    )
    sat_tree = cKDTree(sat_coords_array)

    assignments: Dict[int, List[int]] = {}
    candidate_count = len(sat_ids_array)

    for station_number, station_ecef in enumerate(ground_ecef, start=1):
        station_vec = np.asarray(station_ecef, dtype=float)
        distances, indices = sat_tree.query(station_vec, k=candidate_count)
        if np.ndim(distances) == 0:
            distances = np.array([distances])
            indices = np.array([indices])

        nearest_sat_id: Optional[int] = None
        for _distance, index in zip(distances, indices):
            if index >= candidate_count:
                continue
            sat_vec = sat_coords_array[int(index)]
            if not _is_visible_from_ground(station_vec, sat_vec, min_elevation_deg):
                continue

            nearest_sat_id = int(sat_ids_array[int(index)])
            break

        if nearest_sat_id is None:
            continue

        assignments.setdefault(nearest_sat_id, []).append(station_number)

    return assignments


def update_ground_station_numbers(
    node_state: Any,
    orbit_height_km: float,
    min_elevation_deg: Optional[float] = None,
) -> int:
    """Update ``GroundStationNumber`` in-place and return matched ground-station count."""
    sat_list = _raw_sat_list(node_state)
    for sat_data in sat_list.values():
        _write_ground_station_number(sat_data, None)

    assignments = calculate_ground_station_numbers(
        node_state=node_state,
        orbit_height_km=orbit_height_km,
        min_elevation_deg=min_elevation_deg,
    )
    for sat_id, ground_station_numbers in assignments.items():
        sat_data = sat_list.get(str(sat_id), sat_list.get(sat_id))
        if sat_data is not None:
            _write_ground_station_number(sat_data, ground_station_numbers)

    return sum(
        len(ground_station_numbers)
        for ground_station_numbers in assignments.values()
    )
