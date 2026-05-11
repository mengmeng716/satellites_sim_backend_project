"""Core-node selection based on nearest visible satellites to ground stations."""

from __future__ import annotations

import random
from math import radians, sin
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .adapters import PositionMap
from src import config as simulation_config
from src.utils.gs_manager import GroundStationManager


Pair = Tuple[int, int]


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
    min_elevation_deg: float = 0.0,
) -> bool:
    """Return whether the satellite is above the local horizon.

    The horizon check uses the line-of-sight vector projected on the ground
    station zenith direction. ``min_elevation_deg=0`` means geometric horizon.
    """
    los = satellite_ecef_km - ground_ecef_km
    los_norm = float(np.linalg.norm(los))
    ground_norm = float(np.linalg.norm(ground_ecef_km))
    if los_norm <= 0.0 or ground_norm <= 0.0:
        return False

    zenith = ground_ecef_km / ground_norm
    sin_elevation = float(np.dot(los, zenith) / los_norm)
    return sin_elevation >= sin(radians(min_elevation_deg))


def get_current_core_nodes(
    positions: PositionMap,
    orbit_height_km: float,
    min_elevation_deg: float = 0.0,
) -> Tuple[List[int], ...]:
    """Return CoreNodeNumber: each satellite maps to covered ground-station ids.

    This mirrors ``Topology_Reconfiguration_Module.core_node.get_current_core_nodes``,
    but uses the current project's cached ground-station coordinates and filters
    out satellites below the ground-station horizon.
    """
    if not positions:
        return tuple()

    manager = GroundStationManager()
    ground_ecef = manager.get_ecef_coordinates()
    if ground_ecef is None or len(ground_ecef) == 0:
        return tuple()

    sat_vectors = {
        sat_id: _satellite_ecef_km(lat, lon, orbit_height_km)
        for sat_id, (lat, lon) in positions.items()
    }
    max_sat_id = max(sat_vectors) if sat_vectors else -1
    core_node_list: List[List[int]] = [[] for _ in range(max_sat_id + 1)]

    for station_index, station_ecef in enumerate(ground_ecef, start=1):
        nearest_sat_id: Optional[int] = None
        nearest_distance = float("inf")

        station_vec = np.asarray(station_ecef, dtype=float)
        for sat_id, sat_vec in sat_vectors.items():
            if not _is_visible_from_ground(station_vec, sat_vec, min_elevation_deg):
                continue
            distance = float(np.linalg.norm(sat_vec - station_vec))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_sat_id = sat_id

        if nearest_sat_id is not None:
            core_node_list[nearest_sat_id].append(station_index)

    return tuple(core_node_list)


def get_core_satellite_ids(core_node_number: Sequence[Sequence[int]]) -> List[int]:
    return [sat_id for sat_id, station_ids in enumerate(core_node_number) if station_ids]


def build_source_destination_pairs(
    core_satellite_ids: Iterable[int],
    ordered: bool = False,
) -> List[Pair]:
    core_ids = sorted(set(int(sat_id) for sat_id in core_satellite_ids))
    pairs: List[Pair] = []
    for index, source in enumerate(core_ids):
        for target in core_ids[index + 1 :]:
            if ordered:
                pairs.append((source, target))
                pairs.append((target, source))
            else:
                pairs.append((source, target))
    return pairs


def build_random_source_destination_pairs(
    core_satellite_ids: Iterable[int],
    pair_count: int = 100,
    seed: Optional[int] = None,
) -> List[Pair]:
    """Sample unordered source-destination pairs without reverse duplicates."""
    pairs = build_source_destination_pairs(core_satellite_ids, ordered=False)
    if pair_count <= 0 or len(pairs) <= pair_count:
        return pairs

    rng = random.Random(seed)
    return sorted(rng.sample(pairs, pair_count))


