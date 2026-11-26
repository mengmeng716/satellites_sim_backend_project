"""Capacity-first local search over legal adjacent-plane matchings."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.algos.capacity_evaluation.capacity import DEFAULT_SEARCH_PRECISION, MAX_LAMBDA

from .adapters import SPEED_OF_LIGHT_KM_S, Topology
from .reconfiguration import _initial_boundary_matchings, _new_link_attributes

# All widths count actual trials; short legal exchanges preserve terminal degree.
_SEARCH_ROUNDS = 8
_TRIALS_PER_ROUND = 120
_DELAY_TOLERANCE_MS = 1e-6
_SWAP_GROUP_SIZES = (3, 4)
_SAT_LINK_LOAD_PEAK_FRACTION = 0.90

Arc = tuple[int, int]


@dataclass
class _RouteProfile:
    cost: tuple[float, ...]
    utilization: dict[int, float]
    mean_delay: float
    p95_delay: float
    capacity_gbps: float


def _physical_key(source: int, target: int, node_count: int) -> int:
    return min(source, target) * node_count + max(source, target)


def _get_available_arcs(
    available_topology: Mapping[Any, Mapping[Any, Mapping[str, Any]]],
    node_count: int,
) -> dict[Arc, Mapping[str, Any]]:
    arcs = {}
    for raw_src, neighbors in available_topology.items():
        src = int(raw_src)
        for raw_dst, attrs in neighbors.items():
            dst = int(raw_dst)
            delay = float(attrs.get("LinkPropagationDelay", 0.0))
            rate = float(attrs.get("LinkCapacity", 0.0))
            if (0 <= src < node_count and 0 <= dst < node_count and src != dst
                    and isfinite(delay) and delay > 0.0 and isfinite(rate) and rate > 0.0):
                arcs[src, dst] = attrs
    return {arc: attrs for arc, attrs in arcs.items() if (arc[1], arc[0]) in arcs}


class _CapacityScorer:
    def __init__(self, node_count, traffic_matrix, shared_capacity):
        # Use the same offered Gbps for selection, the final gate, and the report.
        demands = []
        for row in traffic_matrix:
            src, dst, gbps = int(row["src"]), int(row["dst"]), float(row["demand_gbps"])
            if 0 <= src < node_count and 0 <= dst < node_count and src != dst and isfinite(gbps) and gbps > 0:
                demands.append((src, dst, gbps))
        self.node_count = node_count
        self.shared_capacity = shared_capacity
        self.sources = sorted({src for src, _, _ in demands})
        source_rows = {src: row for row, src in enumerate(self.sources)}
        self.origin = np.array([src for src, _, _ in demands], dtype=np.int32)
        self.target = np.array([dst for _, dst, _ in demands], dtype=np.int32)
        self.source_row = np.array([source_rows[src] for src, _, _ in demands], dtype=np.int32)
        self.demand_gbps = np.array([gbps for _, _, gbps in demands], dtype=float)
        self.total_demand_gbps = float(self.demand_gbps.sum())

    def score(self, delays_by_arc: Mapping[Arc, float]) -> _RouteProfile | None:
        if not self.sources or not delays_by_arc:
            return None
        arcs = np.array(list(delays_by_arc), dtype=np.int32)
        graph = csr_matrix(
            (list(delays_by_arc.values()), (arcs[:, 0], arcs[:, 1])),
            shape=(self.node_count, self.node_count),
        )
        distances, predecessors = dijkstra(graph, directed=True, indices=self.sources, return_predecessors=True)
        demand_delays = distances[self.source_row, self.target]
        if not np.isfinite(demand_delays).all():
            return None

        # Accumulate offered load on physical links shared by both directions.
        path_nodes = self.target.copy()
        path_links = []
        path_demands = []
        for _ in range(self.node_count):
            active = path_nodes != self.origin
            if not active.any():
                break
            after = path_nodes[active]
            before = predecessors[self.source_row[active], after]
            if (before < 0).any():
                return None
            path_links.append(np.minimum(after, before) * self.node_count + np.maximum(after, before))
            path_demands.append(self.demand_gbps[active])
            path_nodes[active] = before
        else:
            return None

        edges, index = np.unique(np.concatenate(path_links), return_inverse=True)
        loads = np.bincount(index, weights=np.concatenate(path_demands))
        utilization = {
            int(edge): float(load) / self.shared_capacity[int(edge)]
            for edge, load in zip(edges, loads)
        }
        # Compare multiple bottlenecks to improve ties before the peak goes down.
        cost = tuple(sorted(utilization.values(), reverse=True)[:50])
        return _RouteProfile(
            cost, utilization,
            float(demand_delays.mean()), float(np.percentile(demand_delays, 95)),
            self.total_demand_gbps * min(MAX_LAMBDA, 1.0 / cost[0]),
        )


def _forecast_delays_for_rotation(assignment, sources, targets, rotated_targets, eligible_delays):
    trial = assignment.copy()
    for source, old_target in zip(sources, targets):
        trial.pop((source, old_target), None)
        trial.pop((old_target, source), None)
    for source, new_target in zip(sources, rotated_targets):
        trial[source, new_target] = eligible_delays[source, new_target]
        trial[new_target, source] = eligible_delays[new_target, source]
    return trial


def _legal_rotations(
    matchings, original_matchings, protected_sources, used_groups, seats,
    eligible_delays, delays_by_arc, profile, node_count, changed_arcs, switch_budget,
):
    peak_cutoff = profile.cost[0] * _SAT_LINK_LOAD_PEAK_FRACTION
    hottest_edges = (edge for edge, load in profile.utilization.items() if load >= peak_cutoff)
    focus = {node for edge in hottest_edges for node in divmod(edge, node_count)}
    neighbors = {dst for src, dst in delays_by_arc if src in focus}
    second_neighbors = {dst for src, dst in delays_by_arc if src in neighbors}
    moves = []
    for plane, matching in matchings.items():
        original = original_matchings[plane]
        for first in matching:
            for size in _SWAP_GROUP_SIZES:
                if seats < size:
                    continue
                group = tuple(first // seats * seats + (first + i) % seats for i in range(size))
                if group in used_groups or any(src in protected_sources or src not in matching for src in group):
                    continue
                targets = tuple(matching[src] for src in group)
                importance = sum(3 * (node in focus) + 2 * (node in neighbors) + (node in second_neighbors)
                                 for node in group + targets)
                importance += 25.0 * sum(
                    profile.utilization.get(_physical_key(src, dst, node_count), 0.0)
                    for src, dst in zip(group, targets)
                )
                for rotation in (1, size - 1):
                    rotated = targets[rotation:] + targets[:rotation]
                    if any((src, dst) not in eligible_delays for src, dst in zip(group, rotated)):
                        continue
                    change = 2 * sum(
                        int(original[src] != new_dst) - int(original[src] != old_dst)
                        for src, old_dst, new_dst in zip(group, targets, rotated)
                    )
                    if changed_arcs + change <= switch_budget:
                        moves.append((-importance, group, targets, rotation, plane, change))
    return sorted(moves)[:_TRIALS_PER_ROUND]


def search_capacity_topology(
    initial_topology: Topology,
    available_topology: Mapping[Any, Mapping[Any, Mapping[str, Any]]],
    traffic_matrix: Sequence[Mapping[str, Any]],
    num_planes: int,
    sats_per_plane: int,
    *,
    max_switch_rate: float | None = 0.10,
    link_business_intensity: Mapping[Arc, float] | None = None,
    protected_link_intensity_threshold: float | None = None,
    minimum_capacity_gbps: float | None = None,
) -> Topology | None:
    """Propose a degree-preserving topology for the same fixed business matrix.

    Pass physical edge attributes computed for one geometry snapshot. The
    available topology includes the current links and mutually usable nearby
    candidate links. The caller verifies the winning topology with the normal
    capacity/performance evaluator before applying it.
    """
    node_count = num_planes * sats_per_plane
    if num_planes < 3 or sats_per_plane < 3:
        return None
    attrs_by_arc = _get_available_arcs(available_topology, node_count)
    eligible_delays = {arc: float(attrs["LinkPropagationDelay"]) for arc, attrs in attrs_by_arc.items()}
    shared_capacity = {
        _physical_key(src, dst, node_count): min(
            float(attrs["LinkCapacity"]), float(attrs_by_arc[dst, src]["LinkCapacity"])
        )
        for (src, dst), attrs in attrs_by_arc.items()
    }
    allocation = {
        (src, dst): eligible_delays[src, dst]
        for src, neighbors in initial_topology.items()
        for dst, _ in neighbors if (src, dst) in eligible_delays
    }
    scorer = _CapacityScorer(node_count, traffic_matrix, shared_capacity)
    baseline = scorer.score(allocation)
    if baseline is None:
        return None

    original_matchings = _initial_boundary_matchings(initial_topology, num_planes, sats_per_plane)
    matchings = {plane: assignment.copy() for plane, assignment in original_matchings.items()}
    business = link_business_intensity or {}
    protected_sources = {
        src for matching in original_matchings.values() for src, dst in matching.items()
        if protected_link_intensity_threshold is not None
        and max(business.get((src, dst), 0.0), business.get((dst, src), 0.0)) > protected_link_intensity_threshold
    }
    switch_budget = sum(len(neighbors) for neighbors in initial_topology.values())
    if max_switch_rate is not None:
        switch_budget = int(switch_budget * max(0.0, min(max_switch_rate, 1.0)))

    profile = baseline
    changed_arcs = 0
    used_groups = set()
    for _ in range(_SEARCH_ROUNDS):
        best = profile
        winner = None
        moves = _legal_rotations(
            matchings, original_matchings, protected_sources, used_groups, sats_per_plane,
            eligible_delays, allocation, profile, node_count, changed_arcs, switch_budget,
        )
        for _, group, targets, rotation, plane, change in moves:
            rotated = targets[rotation:] + targets[:rotation]
            trial = _forecast_delays_for_rotation(allocation, group, targets, rotated, eligible_delays)
            estimate = scorer.score(trial)
            if (
                estimate is not None and estimate.cost < best.cost
                and estimate.mean_delay <= baseline.mean_delay + _DELAY_TOLERANCE_MS
                and estimate.p95_delay <= baseline.p95_delay + _DELAY_TOLERANCE_MS
            ):
                best = estimate
                winner = (group, rotated, plane, change, trial)
        if winner is None:
            break
        group, rotated, plane, change, allocation = winner
        matchings[plane].update(zip(group, rotated))
        changed_arcs += change
        used_groups.add(group)
        profile = best
        # Leave enough margin for the reporting evaluator's capacity granularity.
        precision_margin = DEFAULT_SEARCH_PRECISION * scorer.total_demand_gbps
        if minimum_capacity_gbps is not None and profile.capacity_gbps > minimum_capacity_gbps + precision_margin:
            break

    if profile.capacity_gbps <= baseline.capacity_gbps or not changed_arcs:
        return None

    final = {src: dict(neighbors) for src, neighbors in initial_topology.items()}
    changes = [
        (src, original_matchings[plane][src], new_dst)
        for plane, matching in matchings.items() for src, new_dst in matching.items()
        if original_matchings[plane][src] != new_dst
    ]
    for src, old_dst, _ in changes:
        final[src].pop(old_dst, None)
        final[old_dst].pop(src, None)
    for src, _, dst in changes:
        for origin, target in ((src, dst), (dst, src)):
            attrs = attrs_by_arc[origin, target]
            delay = float(attrs["LinkPropagationDelay"])
            new_link = _new_link_attributes(
                delay * SPEED_OF_LIGHT_KM_S / 1000.0, float(attrs["LinkCapacity"]), "inter"
            )
            new_link.update(attrs)
            final[origin][target] = new_link
    return {src: sorted(neighbors.items()) for src, neighbors in final.items()}

