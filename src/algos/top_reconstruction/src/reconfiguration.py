"""Dijkstra-based topology reconfiguration decision algorithm."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from statistics import pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .adapters import (
    LinkAttributes,
    PositionMap,
    Topology,
    allowed_inter_orbit_targets,
    canonical_link_attributes,
    default_link_attributes,
    iter_edges,
    propagation_delay_ms,
    same_orbit,
    satellite_plane_slot,
)
from .evaluation import (
    Graph,
    Pair,
    PathResult,
    RouteSummary,
    compute_pair_paths,
    summarize_paths,
    topology_to_graph,
)
from src.utils.link_distance import calculate_link_distance


@dataclass
class ReconfigurationResult:
    initial_topology: Topology
    candidate_graph: Graph
    reconfigured_topology: Topology
    top_difference: Dict[int, Tuple[List[int], List[int]]]
    baseline_paths: Dict[Pair, PathResult]
    reconfigured_paths: Dict[Pair, PathResult]
    baseline_summary: RouteSummary
    reconfigured_summary: RouteSummary
    timestamp_ms: int = 0
    network_link_switch_rate: float = 0.0
    orbit_layer_switch_std_dev: float = 0.0
    selected_inter_edge_count: int = 0
    considered_pair_count: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def average_hop_reduction(self) -> float:
        return self.baseline_summary.average_hops - self.reconfigured_summary.average_hops

    @property
    def average_hop_reduction_rate(self) -> float:
        if self.baseline_summary.average_hops <= 0.0:
            return 0.0
        return self.average_hop_reduction / self.baseline_summary.average_hops


def build_candidate_graph(
    initial_topology: Topology,
    positions: PositionMap,
    num_planes: int,
    sats_per_plane: int,
    orbit_height_km: float,
    offset_window: int = 1,
) -> Graph:
    """Build feasible candidate edges.

    Same-orbit adjacent links are fixed. Inter-orbit links may choose the
    initial nearest-neighbor target or the target's +/- ``offset_window`` in-plane
    slots, matching the reference module's candidate-edge idea.
    """
    graph: Graph = defaultdict(dict)

    for source_id, target_id, attr in iter_edges(initial_topology):
        if same_orbit(source_id, target_id, sats_per_plane):
            graph[source_id][target_id] = float(attr.get("LinkDistance", 0.0))

    if not positions:
        for source_id, target_id, attr in iter_edges(initial_topology):
            if not same_orbit(source_id, target_id, sats_per_plane):
                graph[source_id][target_id] = float(attr.get("LinkDistance", 0.0))
        return dict(graph)

    for source_id in sorted(positions):
        source_lat, source_lon = positions[source_id]
        for direction in (1, -1):
            for target_id in allowed_inter_orbit_targets(
                source_id,
                direction,
                num_planes,
                sats_per_plane,
                offset_window=offset_window,
            ):
                if target_id not in positions:
                    continue
                target_lat, target_lon = positions[target_id]
                distance = calculate_link_distance(
                    source_lat,
                    source_lon,
                    target_lat,
                    target_lon,
                    orbit_height_km,
                )
                existing = graph[source_id].get(target_id)
                if existing is None or distance < existing:
                    graph[source_id][target_id] = distance

    return dict(graph)


def inter_orbit_direction(
    source_id: int,
    target_id: int,
    num_planes: int,
    sats_per_plane: int,
) -> int:
    source_plane, _ = satellite_plane_slot(source_id, sats_per_plane)
    target_plane, _ = satellite_plane_slot(target_id, sats_per_plane)
    if target_plane == (source_plane + 1) % num_planes:
        return 1
    if target_plane == (source_plane - 1) % num_planes:
        return -1
    return 0


def count_candidate_inter_edges(
    candidate_paths: Mapping[Pair, PathResult],
    num_planes: int,
    sats_per_plane: int,
) -> Counter:
    counts: Counter = Counter()
    for result in candidate_paths.values():
        for source_id, target_id in zip(result.path, result.path[1:]):
            direction = inter_orbit_direction(source_id, target_id, num_planes, sats_per_plane)
            if direction:
                counts[(source_id, direction, target_id)] += 1
    return counts


def _initial_boundary_matching(
    initial_topology: Topology,
    plane: int,
    next_plane: int,
    sats_per_plane: int,
) -> Dict[int, int]:
    matching: Dict[int, int] = {}
    for source_id, links in initial_topology.items():
        source_plane, _ = satellite_plane_slot(source_id, sats_per_plane)
        if source_plane != plane:
            continue
        for target_id, _ in links:
            target_plane, _ = satellite_plane_slot(target_id, sats_per_plane)
            if target_plane == next_plane:
                matching[source_id] = target_id
                break
    return matching


def select_reconfigured_topology(
    initial_topology: Topology,
    candidate_graph: Graph,
    candidate_paths: Mapping[Pair, PathResult],
    num_planes: int,
    sats_per_plane: int,
    link_capacity_gbps: float = 10.0,
    max_switch_rate: Optional[float] = 0.10,
) -> Topology:
    all_sat_ids = set(initial_topology) | {target for links in initial_topology.values() for target, _ in links}
    selected: Topology = {sat_id: [] for sat_id in all_sat_ids}
    inter_edge_counts = count_candidate_inter_edges(candidate_paths, num_planes, sats_per_plane)

    def minimal_link_attr(attr: LinkAttributes, link_type: str) -> LinkAttributes:
        distance = float(attr.get("LinkDistance", 0.0))
        return canonical_link_attributes(
            attr,
            distance_km=distance,
            delay_ms=propagation_delay_ms(distance),
            link_type=link_type,
        )

    for source_id, initial_links in initial_topology.items():
        for target_id, attr in initial_links:
            if same_orbit(source_id, target_id, sats_per_plane):
                selected.setdefault(source_id, []).append((target_id, minimal_link_attr(attr, "intra")))

    initial_edge_set = {
        (source_id, target_id)
        for source_id, target_id, _ in iter_edges(initial_topology)
        if not same_orbit(source_id, target_id, sats_per_plane)
    }
    total_initial_links = sum(len(links) for links in initial_topology.values())
    if max_switch_rate is None:
        remaining_physical_budget: Optional[int] = None
    else:
        max_directed_changes = max(0, ceil(max_switch_rate * total_initial_links) - 1)
        remaining_physical_budget = max_directed_changes // 2

    edge_cost_cache: Dict[Tuple[int, int], float] = {}

    def edge_cost(source_id: int, target_id: int) -> float:
        cache_key = (source_id, target_id)
        if cache_key in edge_cost_cache:
            return edge_cost_cache[cache_key]

        distance = candidate_graph.get(source_id, {}).get(target_id, 0.0)
        score = (
            inter_edge_counts.get((source_id, 1, target_id), 0)
            + inter_edge_counts.get((target_id, -1, source_id), 0)
        )
        preserve_initial_penalty = (
            1_000_000.0
            if score == 0
            and (source_id, target_id) not in initial_edge_set
            and (target_id, source_id) not in initial_edge_set
            else 0.0
        )
        cost = -1_000_000.0 * score + preserve_initial_penalty + distance
        edge_cost_cache[cache_key] = cost
        return cost

    selected_matchings: Dict[int, Dict[int, int]] = {}
    swap_candidates: List[Tuple[float, int, int, int, int, int]] = []

    for plane in range(num_planes):
        next_plane = (plane + 1) % num_planes
        left_ids = [
            sat_id
            for sat_id in sorted(all_sat_ids)
            if satellite_plane_slot(sat_id, sats_per_plane)[0] == plane
        ]
        if not left_ids:
            selected_matchings[plane] = {}
            continue

        initial_matching = _initial_boundary_matching(
            initial_topology,
            plane,
            next_plane,
            sats_per_plane,
        )
        selected_matchings[plane] = dict(initial_matching)

        for index, source_id in enumerate(left_ids):
            next_source_id = left_ids[(index + 1) % len(left_ids)]
            if source_id not in initial_matching or next_source_id not in initial_matching:
                continue

            source_target = initial_matching[source_id]
            next_source_target = initial_matching[next_source_id]
            if (
                next_source_target not in candidate_graph.get(source_id, {})
                or source_target not in candidate_graph.get(next_source_id, {})
            ):
                continue

            current_cost = edge_cost(source_id, source_target) + edge_cost(
                next_source_id,
                next_source_target,
            )
            swapped_cost = edge_cost(source_id, next_source_target) + edge_cost(
                next_source_id,
                source_target,
            )
            benefit = current_cost - swapped_cost
            if benefit > 0:
                swap_candidates.append(
                    (
                        benefit,
                        plane,
                        source_id,
                        next_source_id,
                        source_target,
                        next_source_target,
                    )
                )

    used_sources_by_plane: Dict[int, Set[int]] = defaultdict(set)
    for _, plane, source_id, next_source_id, source_target, next_source_target in sorted(
        swap_candidates,
        reverse=True,
    ):
        if remaining_physical_budget is not None and remaining_physical_budget < 2:
            break
        if (
            source_id in used_sources_by_plane[plane]
            or next_source_id in used_sources_by_plane[plane]
        ):
            continue

        selected_matchings[plane][source_id] = next_source_target
        selected_matchings[plane][next_source_id] = source_target
        used_sources_by_plane[plane].add(source_id)
        used_sources_by_plane[plane].add(next_source_id)
        if remaining_physical_budget is not None:
            remaining_physical_budget -= 2

    for plane in range(num_planes):
        for source_id, selected_target in sorted(selected_matchings.get(plane, {}).items()):
            distance = candidate_graph.get(source_id, {}).get(selected_target, 0.0)
            delay = propagation_delay_ms(distance)
            forward_attr = default_link_attributes(
                distance_km=distance,
                capacity_gbps=link_capacity_gbps,
                link_type="inter",
            )
            forward_attr["LinkPropagationDelay"] = delay
            reverse_attr = deepcopy(forward_attr)
            selected.setdefault(source_id, []).append((selected_target, forward_attr))
            selected.setdefault(selected_target, []).append((source_id, reverse_attr))

    for source_id, links in list(selected.items()):
        deduped: Dict[int, LinkAttributes] = {}
        for target_id, attr in links:
            deduped[target_id] = attr
        selected[source_id] = [
            (target_id, deduped[target_id])
            for target_id in sorted(
                deduped,
                key=lambda target: (
                    0 if same_orbit(source_id, target, sats_per_plane) else 1,
                    target,
                ),
            )
        ]

    return selected


def compute_top_difference(
    initial_topology: Topology,
    reconfigured_topology: Topology,
) -> Dict[int, Tuple[List[int], List[int]]]:
    diff: Dict[int, Tuple[List[int], List[int]]] = {}
    all_sources = set(initial_topology) | set(reconfigured_topology)
    for source_id in all_sources:
        before = {target_id for target_id, _ in initial_topology.get(source_id, [])}
        after = {target_id for target_id, _ in reconfigured_topology.get(source_id, [])}
        diff[source_id] = (sorted(after - before), sorted(before - after))
    return diff


def compute_switch_metrics(
    top_difference: Mapping[int, Tuple[Sequence[int], Sequence[int]]],
    initial_topology: Topology,
    num_planes: int,
    sats_per_plane: int,
) -> Tuple[float, float, int]:
    total_initial_links = sum(len(links) for links in initial_topology.values())
    total_added = sum(len(added) for added, _ in top_difference.values())
    switch_rate = total_added / total_initial_links if total_initial_links else 0.0

    switches_by_plane = [0 for _ in range(num_planes)]
    for source_id, (added, deleted) in top_difference.items():
        plane, _ = satellite_plane_slot(source_id, sats_per_plane)
        if 0 <= plane < num_planes:
            switches_by_plane[plane] += max(len(added), len(deleted))

    std_dev = pstdev(switches_by_plane) if switches_by_plane else 0.0
    return switch_rate, std_dev, total_added


def run_reconfiguration_decision(
    initial_topology: Topology,
    positions: PositionMap,
    source_destination_pairs: Iterable[Pair],
    num_planes: int,
    sats_per_plane: int,
    orbit_height_km: float,
    timestamp_ms: int = 0,
    offset_window: int = 1,
    max_switch_rate: Optional[float] = 0.10,
    processing_delay_ms: float = 5.0,
    link_capacity_gbps: float = 10.0,
) -> ReconfigurationResult:
    pairs = [
        (int(source), int(target))
        for source, target in source_destination_pairs
        if int(source) != int(target)
    ]

    initial_graph = topology_to_graph(initial_topology)
    baseline_paths = compute_pair_paths(initial_graph, pairs, processing_delay_ms)
    baseline_summary = summarize_paths(baseline_paths, len(pairs))

    candidate_graph = build_candidate_graph(
        initial_topology,
        positions,
        num_planes,
        sats_per_plane,
        orbit_height_km,
        offset_window=offset_window,
    )
    candidate_paths = compute_pair_paths(candidate_graph, pairs, processing_delay_ms)

    decision_start = perf_counter()
    reconfigured_topology = select_reconfigured_topology(
        initial_topology,
        candidate_graph,
        candidate_paths,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        link_capacity_gbps=link_capacity_gbps,
        max_switch_rate=max_switch_rate,
    )
    topology_decision_time_ms = (perf_counter() - decision_start) * 1000.0
    reconfigured_graph = topology_to_graph(reconfigured_topology)
    reconfigured_paths = compute_pair_paths(reconfigured_graph, pairs, processing_delay_ms)
    reconfigured_summary = summarize_paths(reconfigured_paths, len(pairs))

    top_difference = compute_top_difference(initial_topology, reconfigured_topology)
    switch_rate, std_dev, selected_count = compute_switch_metrics(
        top_difference,
        initial_topology,
        num_planes,
        sats_per_plane,
    )

    return ReconfigurationResult(
        initial_topology=initial_topology,
        candidate_graph=candidate_graph,
        reconfigured_topology=reconfigured_topology,
        top_difference=top_difference,
        baseline_paths=baseline_paths,
        reconfigured_paths=reconfigured_paths,
        baseline_summary=baseline_summary,
        reconfigured_summary=reconfigured_summary,
        timestamp_ms=timestamp_ms,
        network_link_switch_rate=switch_rate,
        orbit_layer_switch_std_dev=std_dev,
        selected_inter_edge_count=selected_count,
        considered_pair_count=len(pairs),
        metadata={
            "OffsetWindow": float(offset_window),
            "MaxSwitchRate": float(max_switch_rate) if max_switch_rate is not None else -1.0,
            "ProcessingDelayMs": float(processing_delay_ms),
            "PairCount": float(len(pairs)),
            "TopologyDecisionTime": float(topology_decision_time_ms),
        },
    )
