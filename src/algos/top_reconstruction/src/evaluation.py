"""Path evaluation helpers for topology reconstruction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .adapters import Topology, iter_edges, propagation_delay_ms


Graph = Dict[int, Dict[int, float]]
Pair = Tuple[int, int]


@dataclass
class PathResult:
    source: int
    target: int
    path: List[int]
    distance_km: float
    propagation_delay_ms: float
    total_delay_ms: float
    hops: int


@dataclass
class RouteSummary:
    pair_count: int = 0
    reachable_pair_count: int = 0
    average_hops: float = 0.0
    average_total_delay_ms: float = 0.0


def topology_to_graph(topology: Topology) -> Graph:
    graph: Graph = defaultdict(dict)
    for source_id, target_id, attr in iter_edges(topology):
        distance = float(attr.get("LinkDistance", 0.0))
        existing = graph[source_id].get(target_id)
        if existing is None or distance < existing:
            graph[source_id][target_id] = distance
    return dict(graph)


def _reconstruct_path(
    previous: Mapping[int, int],
    source: int,
    target: int,
) -> Optional[List[int]]:
    if source == target:
        return [source]
    if target not in previous:
        return None

    path = [target]
    node = target
    while node != source:
        node = previous.get(node)
        if node is None:
            return None
        path.append(node)
    path.reverse()
    return path


def evaluate_path(
    graph: Graph,
    source: int,
    target: int,
    path: Sequence[int],
    processing_delay_ms: float = 5.0,
) -> PathResult:
    distance_km = 0.0
    for left, right in zip(path, path[1:]):
        distance_km += graph[left][right]

    hops = max(0, len(path) - 1)
    prop_delay = propagation_delay_ms(distance_km)
    total_delay = prop_delay + processing_delay_ms * max(0, hops - 1)
    return PathResult(
        source=source,
        target=target,
        path=list(path),
        distance_km=distance_km,
        propagation_delay_ms=prop_delay,
        total_delay_ms=total_delay,
        hops=hops,
    )


def dijkstra_paths_from_source(
    graph: Graph,
    source: int,
    targets: Iterable[int],
    processing_delay_ms: float = 5.0,
) -> Dict[int, PathResult]:
    target_set = set(int(target) for target in targets)
    if not target_set:
        return {}

    distances: Dict[int, float] = {source: 0.0}
    previous: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, source)]
    settled_targets: Set[int] = set()

    while heap and settled_targets != target_set:
        current_cost, node = heappop(heap)
        if current_cost > distances.get(node, float("inf")):
            continue

        if node in target_set:
            settled_targets.add(node)

        for neighbor, distance_km in graph.get(node, {}).items():
            edge_cost = propagation_delay_ms(distance_km) + processing_delay_ms
            new_cost = current_cost + edge_cost
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                previous[neighbor] = node
                heappush(heap, (new_cost, neighbor))

    results: Dict[int, PathResult] = {}
    for target in target_set:
        path = _reconstruct_path(previous, source, target)
        if path is None:
            continue
        results[target] = evaluate_path(graph, source, target, path, processing_delay_ms)
    return results


def compute_pair_paths(
    graph: Graph,
    pairs: Iterable[Pair],
    processing_delay_ms: float = 5.0,
) -> Dict[Pair, PathResult]:
    targets_by_source: Dict[int, List[int]] = defaultdict(list)
    for source, target in pairs:
        targets_by_source[int(source)].append(int(target))

    paths: Dict[Pair, PathResult] = {}
    for source, targets in targets_by_source.items():
        source_paths = dijkstra_paths_from_source(graph, source, targets, processing_delay_ms)
        for target, result in source_paths.items():
            paths[(source, target)] = result
    return paths


def summarize_paths(paths: Mapping[Pair, PathResult], total_pair_count: int) -> RouteSummary:
    if not paths:
        return RouteSummary(pair_count=total_pair_count)
    return RouteSummary(
        pair_count=total_pair_count,
        reachable_pair_count=len(paths),
        average_hops=mean(path.hops for path in paths.values()),
        average_total_delay_ms=mean(path.total_delay_ms for path in paths.values()),
    )
