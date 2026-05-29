"""Dijkstra-based topology reconfiguration decision algorithm."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .adapters import (
    LinkAttributes,
    PositionMap,
    Topology,
    canonical_link_attributes,
    iter_edges,
    plane_slot_to_sat_id,
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

from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def _physical_edge_key(source_id: int, target_id: int) -> Tuple[int, int]:
    return (
        min(int(source_id), int(target_id)),
        max(int(source_id), int(target_id)),
    )


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

    @property
    def average_total_delay_reduction_ms(self) -> float:
        return (
            self.baseline_summary.average_total_delay_ms
            - self.reconfigured_summary.average_total_delay_ms
        )

    @property
    def average_total_delay_reduction_rate(self) -> float:
        if self.baseline_summary.average_total_delay_ms <= 0.0:
            return 0.0
        return self.average_total_delay_reduction_ms / self.baseline_summary.average_total_delay_ms


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

    def add_position_edge(source_id: int, target_id: int) -> None:
        if source_id not in positions or target_id not in positions:
            return
        source_lat, source_lon = positions[source_id]
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

    for source_id, target_id, _ in iter_edges(initial_topology):
        if same_orbit(source_id, target_id, sats_per_plane):
            continue

        target_plane, target_slot = satellite_plane_slot(target_id, sats_per_plane)
        if not 0 <= target_plane < num_planes:
            continue

        for slot_offset in range(-offset_window, offset_window + 1):
            candidate_id = plane_slot_to_sat_id(
                target_plane,
                target_slot + slot_offset,
                num_planes,
                sats_per_plane,
            )
            add_position_edge(source_id, candidate_id)

    return dict(graph)


def _initial_boundary_matchings(
    initial_topology: Topology,
    num_planes: int,
    sats_per_plane: int,
) -> Dict[int, Dict[int, int]]:
    matchings: Dict[int, Dict[int, int]] = {plane: {} for plane in range(num_planes)}
    for source_id, links in initial_topology.items():
        source_plane, _ = satellite_plane_slot(source_id, sats_per_plane)
        if not 0 <= source_plane < num_planes:
            continue
        next_plane = (source_plane + 1) % num_planes
        for target_id, _ in links:
            target_plane, _ = satellite_plane_slot(target_id, sats_per_plane)
            if target_plane == next_plane:
                matchings[source_plane][source_id] = target_id
                break
    return matchings


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


_INF_ASSIGNMENT_COST = 1.0e30

_CANONICAL_LINK_ATTRIBUTE_KEYS = (
    "LinkDistance",
    "LinkCapacity",
    "LeftCapacity",
    "CurrentFlow",
    "LinkPropagationDelay",
    "QueueDelay",
    "TransmissionDelay",
    "PacketLossRate",
    "HeatValue",
    "LinkType",
)


def _copy_link_attributes_for_distance(
    attr: LinkAttributes,
    distance: float,
    link_type: str,
) -> LinkAttributes:
    """Copy a normalized link attr without re-running the generic adapter."""
    distance = max(0.0, float(distance))
    if all(key in attr for key in _CANONICAL_LINK_ATTRIBUTE_KEYS):
        copied = dict(attr)
        copied["LinkDistance"] = distance
        copied["LinkPropagationDelay"] = max(0.0, propagation_delay_ms(distance))
        return copied

    return canonical_link_attributes(
        attr,
        distance_km=distance,
        delay_ms=propagation_delay_ms(distance),
        link_type=link_type,
    )


def _new_link_attributes(
    distance: float,
    capacity_gbps: float,
    link_type: str,
) -> LinkAttributes:
    distance = max(0.0, float(distance))
    capacity = max(0.0, float(capacity_gbps))
    return {
        "LinkDistance": distance,
        "LinkCapacity": capacity,
        "LeftCapacity": capacity,
        "CurrentFlow": 0.0,
        "LinkPropagationDelay": propagation_delay_ms(distance),
        "QueueDelay": 0.0,
        "TransmissionDelay": 0.0,
        "PacketLossRate": 0.0,
        "HeatValue": 0.0,
        "LinkType": str(link_type),
    }

def _hungarian_min_cost_assignment(
    costs: Sequence[Sequence[float]],
) -> Optional[List[int]]:
    """Return row -> column assignment using SciPy's Hungarian solver."""
    if not costs:
        return []

    row_count = len(costs)
    column_count = len(costs[0])
    if row_count > column_count:
        return None
    if any(len(row) != column_count for row in costs):
        raise ValueError("all cost rows must have the same length")

    try:
        row_indices, column_indices = linear_sum_assignment(costs)
    except ValueError:
        return None
    assignment = [-1 for _ in range(row_count)]
    for row_index, column_index in zip(row_indices, column_indices):
        assignment[int(row_index)] = int(column_index)
    if any(column < 0 for column in assignment):
        return None
    return assignment


def _sparse_min_cost_assignment(
    row_count: int,
    column_count: int,
    row_indices: Sequence[int],
    column_indices: Sequence[int],
    costs: Sequence[float],
) -> Optional[List[int]]:
    if not row_indices:
        return None

    adjusted_costs = list(costs)
    min_cost = min(adjusted_costs)
    if min_cost <= 0.0:
        shift = -min_cost + 1.0
        adjusted_costs = [cost + shift for cost in adjusted_costs]

    cost_graph = csr_matrix(
        (adjusted_costs, (row_indices, column_indices)),
        shape=(row_count, column_count),
    )
    try:
        matched_rows, matched_columns = min_weight_full_bipartite_matching(cost_graph)
    except ValueError:
        return None

    assignment = [-1 for _ in range(row_count)]
    for row_index, column_index in zip(matched_rows, matched_columns):
        assignment[int(row_index)] = int(column_index)
    if any(column < 0 for column in assignment):
        return None
    return assignment


def select_reconfigured_topology(
    initial_topology: Topology,
    candidate_graph: Graph,
    candidate_paths: Mapping[Pair, PathResult],
    num_planes: int,
    sats_per_plane: int,
    link_capacity_gbps: float = 10.0,
    max_switch_rate: Optional[float] = 0.10,
    default_forwarding_delay_ms: float = 0.0,
    forwarding_delay_ms_by_node: Optional[Mapping[int, float]] = None,
    link_business_intensity: Optional[Mapping[Tuple[int, int], float]] = None,
    protected_link_intensity_threshold: Optional[float] = None,
) -> Topology:
    _ = (
        max_switch_rate,
        default_forwarding_delay_ms,
        forwarding_delay_ms_by_node,
    )
    all_sat_ids = set(initial_topology) | {target for links in initial_topology.values() for target, _ in links}
    selected: Topology = {sat_id: [] for sat_id in all_sat_ids}
    inter_edge_counts = count_candidate_inter_edges(
        candidate_paths,
        num_planes,
        sats_per_plane,
    )

    def minimal_link_attr(attr: LinkAttributes, link_type: str) -> LinkAttributes:
        distance = float(attr.get("LinkDistance", 0.0))
        return _copy_link_attributes_for_distance(attr, distance, link_type)


    for source_id, initial_links in initial_topology.items():
        for target_id, attr in initial_links:
            if same_orbit(source_id, target_id, sats_per_plane):
                selected.setdefault(source_id, []).append((target_id, minimal_link_attr(attr, "intra")))

    initial_edge_set = {
        (source_id, target_id)
        for source_id, target_id, _ in iter_edges(initial_topology)
        if not same_orbit(source_id, target_id, sats_per_plane)
    }
    business_intensity = link_business_intensity or {}
    protected_initial_edges = {
        _physical_edge_key(source_id, target_id)
        for source_id, target_id in initial_edge_set
        if protected_link_intensity_threshold is not None
        and max(
            float(business_intensity.get((source_id, target_id), 0.0)),
            float(business_intensity.get((target_id, source_id), 0.0)),
        )
        > float(protected_link_intensity_threshold)
    }

    def is_protected_initial_edge(source_id: int, target_id: int) -> bool:
        return _physical_edge_key(source_id, target_id) in protected_initial_edges

    def edge_cost(source_id: int, target_id: int) -> float:
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
        return cost

    selected_matchings: Dict[int, Dict[int, int]] = {}
    ids_by_plane: Dict[int, List[int]] = {plane: [] for plane in range(num_planes)}
    for sat_id in sorted(all_sat_ids):
        plane, _ = satellite_plane_slot(sat_id, sats_per_plane)
        if 0 <= plane < num_planes:
            ids_by_plane[plane].append(sat_id)
    initial_matchings_by_plane = _initial_boundary_matchings(
        initial_topology,
        num_planes,
        sats_per_plane,
    )

    for plane in range(num_planes):
        initial_matching = initial_matchings_by_plane.get(plane, {})
        if not initial_matching:
            selected_matchings[plane] = {}
            continue

        left_ids = sorted(initial_matching)
        next_plane = (plane + 1) % num_planes
        right_ids = sorted(ids_by_plane.get(next_plane, []))
        if len(left_ids) != len(right_ids):
            selected_matchings[plane] = dict(initial_matching)
            continue

        protected_source_targets = {
            source_id: target_id
            for source_id, target_id in initial_matching.items()
            if is_protected_initial_edge(source_id, target_id)
        }
        protected_target_owners = {
            target_id: source_id
            for source_id, target_id in protected_source_targets.items()
        }

        right_index_by_id = {
            target_id: index for index, target_id in enumerate(right_ids)
        }
        sparse_row_indices: List[int] = []
        sparse_column_indices: List[int] = []
        sparse_costs: List[float] = []
        for row_index, source_id in enumerate(left_ids):
            source_initial_target = initial_matching.get(source_id)
            if source_id in protected_source_targets:
                candidate_targets = [protected_source_targets[source_id]]
            else:
                candidate_targets = list(candidate_graph.get(source_id, {}))
                if source_initial_target is not None:
                    candidate_targets.append(source_initial_target)

            seen_targets: Set[int] = set()
            for target_id in candidate_targets:
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                column_index = right_index_by_id.get(target_id)
                if column_index is None:
                    continue
                target_plane, _ = satellite_plane_slot(target_id, sats_per_plane)
                allowed = target_plane == next_plane and (
                    target_id in candidate_graph.get(source_id, {})
                    or target_id == source_initial_target
                )
                if target_id in protected_target_owners:
                    allowed = source_id == protected_target_owners[target_id]
                if not allowed:
                    continue
                sparse_row_indices.append(row_index)
                sparse_column_indices.append(column_index)
                sparse_costs.append(edge_cost(source_id, target_id))

        assignment = _sparse_min_cost_assignment(
            len(left_ids),
            len(right_ids),
            sparse_row_indices,
            sparse_column_indices,
            sparse_costs,
        )

        if assignment is None:
            cost_matrix: List[List[float]] = []
            for source_id in left_ids:
                source_initial_target = initial_matching.get(source_id)
                row: List[float] = []
                for target_id in right_ids:
                    target_plane, _ = satellite_plane_slot(target_id, sats_per_plane)
                    allowed = target_plane == next_plane and (
                        target_id in candidate_graph.get(source_id, {})
                        or target_id == source_initial_target
                    )
                    if source_id in protected_source_targets:
                        allowed = target_id == protected_source_targets[source_id]
                    if target_id in protected_target_owners:
                        allowed = source_id == protected_target_owners[target_id]
                    row.append(
                        edge_cost(source_id, target_id)
                        if allowed
                        else _INF_ASSIGNMENT_COST
                    )
                cost_matrix.append(row)

            assignment = _hungarian_min_cost_assignment(cost_matrix)
            if assignment is None:
                selected_matchings[plane] = dict(initial_matching)
                continue

            assignment_is_valid = True
            for row_index, column_index in enumerate(assignment):
                if cost_matrix[row_index][column_index] >= _INF_ASSIGNMENT_COST / 2:
                    assignment_is_valid = False
                    break
            if not assignment_is_valid:
                selected_matchings[plane] = dict(initial_matching)
                continue

        plane_matching: Dict[int, int] = {}
        for row_index, column_index in enumerate(assignment):
            plane_matching[left_ids[row_index]] = right_ids[column_index]

        selected_matchings[plane] = plane_matching

    for plane in range(num_planes):
        for source_id, selected_target in sorted(selected_matchings.get(plane, {}).items()):
            distance = candidate_graph.get(source_id, {}).get(selected_target, 0.0)
            forward_attr = _new_link_attributes(distance, link_capacity_gbps, "inter")
            reverse_attr = dict(forward_attr)
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
    default_forwarding_delay_ms: float = 0.0,
    link_capacity_gbps: float = 10.0,
    forwarding_delay_ms_by_node: Optional[Mapping[int, float]] = None,
    link_business_intensity: Optional[Mapping[Tuple[int, int], float]] = None,
    protected_link_intensity_threshold: Optional[float] = None,
) -> ReconfigurationResult:
    pairs = [
        (int(source), int(target))
        for source, target in source_destination_pairs
        if int(source) != int(target)
    ]

    initial_graph = topology_to_graph(initial_topology)
    baseline_paths = compute_pair_paths(
        initial_graph,
        pairs,
        default_forwarding_delay_ms,
        forwarding_delay_ms_by_node,
    )
    baseline_summary = summarize_paths(baseline_paths, len(pairs))

    candidate_graph = build_candidate_graph(
        initial_topology,
        positions,
        num_planes,
        sats_per_plane,
        orbit_height_km,
        offset_window=offset_window,
    )
    candidate_paths = compute_pair_paths(
        candidate_graph,
        pairs,
        default_forwarding_delay_ms,
        forwarding_delay_ms_by_node,
    )
    decision_start = perf_counter()
    reconfigured_topology = select_reconfigured_topology(
        initial_topology,
        candidate_graph,
        candidate_paths,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        link_capacity_gbps=link_capacity_gbps,
        max_switch_rate=max_switch_rate,
        default_forwarding_delay_ms=default_forwarding_delay_ms,
        forwarding_delay_ms_by_node=forwarding_delay_ms_by_node,
        link_business_intensity=link_business_intensity,
        protected_link_intensity_threshold=protected_link_intensity_threshold,
    )
    topology_decision_time_ms = (perf_counter() - decision_start) * 1000.0
    reconfigured_graph = topology_to_graph(reconfigured_topology)
    reconfigured_paths = compute_pair_paths(
        reconfigured_graph,
        pairs,
        default_forwarding_delay_ms,
        forwarding_delay_ms_by_node,
    )
    reconfigured_summary = summarize_paths(reconfigured_paths, len(pairs))

    top_difference = compute_top_difference(initial_topology, reconfigured_topology)
    switch_rate, std_dev, selected_count = compute_switch_metrics(
        top_difference,
        initial_topology,
        num_planes,
        sats_per_plane,
    )
    protected_link_count = 0
    if link_business_intensity and protected_link_intensity_threshold is not None:
        protected_link_count = len(
            {
                _physical_edge_key(source_id, target_id)
                for source_id, target_id, _ in iter_edges(initial_topology)
                if not same_orbit(source_id, target_id, sats_per_plane)
                and max(
                    float(link_business_intensity.get((source_id, target_id), 0.0)),
                    float(link_business_intensity.get((target_id, source_id), 0.0)),
                )
                > float(protected_link_intensity_threshold)
            }
        )
    average_total_delay_reduction_ms = (
        baseline_summary.average_total_delay_ms
        - reconfigured_summary.average_total_delay_ms
    )
    average_total_delay_reduction_rate = (
        average_total_delay_reduction_ms / baseline_summary.average_total_delay_ms
        if baseline_summary.average_total_delay_ms > 0.0
        else 0.0
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
            "ProtectedLinkIntensityThreshold": (
                float(protected_link_intensity_threshold)
                if protected_link_intensity_threshold is not None
                else -1.0
            ),
            "ProtectedLinkCount": float(protected_link_count),
            "PairCount": float(len(pairs)),
            "AverageTotalDelayReductionMs": float(average_total_delay_reduction_ms),
            "AverageTotalDelayReductionRate": float(average_total_delay_reduction_rate),
            "TopologyDecisionTime": float(topology_decision_time_ms),
        },
    )
