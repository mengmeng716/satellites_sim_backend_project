"""图构建与最短路。

链路属性兼容两种形式：

* ``LinksQualitiesValue`` 对象（仿真内使用，含 ``link_distance`` / ``link_propagation_delay``）
* ``dict``（离线/测试使用，键名同接口文档，如 ``LinkDistance``、``LinkPropagationDelay``）
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

INF = float("inf")


def _attr_value(attr: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(attr, Mapping):
            if name in attr and attr[name] is not None:
                return attr[name]
        elif attr is not None and hasattr(attr, name):
            value = getattr(attr, name)
            if value is not None:
                return value
    return default


def link_distance_km(attr: Any, default: float = 0.0) -> float:
    return float(_attr_value(attr, "link_distance", "LinkDistance", "distance_km",
                             default=default) or default)


def link_delay_ms(attr: Any, default: Optional[float] = None) -> Optional[float]:
    """链路传播时延（ms）。缺省由距离推算。"""
    value = _attr_value(attr, "link_propagation_delay", "LinkPropagationDelay",
                        "propagation_delay_ms")
    if value is not None:
        return float(value)
    distance = link_distance_km(attr, default=-1.0)
    if distance < 0:
        return default
    return distance / 299792.458 * 1000.0


def link_loss(attr: Any, default: float = 0.0) -> float:
    value = _attr_value(attr, "packet_loss_rate", "PacketLossRate", "loss")
    return float(value) if value is not None else float(default)


def link_ber(attr: Any, default: float = 1e-9) -> float:
    value = _attr_value(attr, "ber", "BER", "ber_post_fec", "MeanBER")
    return float(value) if value is not None else float(default)


def build_adjacency(
    topology: Mapping[Any, Mapping[Any, Any]],
) -> Dict[Any, Dict[Any, Any]]:
    """把 ``{src: {dst: attr}}`` 补成无向邻接表（缺失方向用已有属性补全）。"""
    adjacency: Dict[Any, Dict[Any, Any]] = {}
    for src, neighbours in (topology or {}).items():
        adjacency.setdefault(src, {})
        for dst, attr in (neighbours or {}).items():
            if src == dst:
                continue
            adjacency[src].setdefault(dst, attr)
            adjacency.setdefault(dst, {})
            if src not in adjacency[dst]:
                adjacency[dst][src] = attr
    return adjacency


def dijkstra(
    adjacency: Mapping[Any, Mapping[Any, Any]],
    source: Any,
    target: Any,
    weight: str = "delay",
) -> Tuple[Optional[list], float]:
    """堆优化 Dijkstra。

    :param weight: ``"delay"``（按传播时延）或 ``"hops"``
    :return: ``(path, cost)``；不可达返回 ``(None, inf)``
    """
    if source not in adjacency or target not in adjacency:
        return None, INF

    dist = {source: 0.0}
    prev: Dict[Any, Any] = {}
    heap = [(0.0, source)]
    visited = set()

    while heap:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            break
        for neighbour, attr in adjacency[node].items():
            if neighbour in visited:
                continue
            step = 1.0 if weight == "hops" else (link_delay_ms(attr, default=0.0) or 0.0)
            new_cost = cost + step
            if new_cost < dist.get(neighbour, INF):
                dist[neighbour] = new_cost
                prev[neighbour] = node
                heapq.heappush(heap, (new_cost, neighbour))

    if target not in dist:
        return None, INF

    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[target]


def _single_source(adjacency, source, targets, weight):
    """单源 Dijkstra，求到 ``targets`` 中全部节点的最短路（全部到达即停止）。"""
    dist = {source: 0.0}
    prev: Dict[Any, Any] = {}
    heap = [(0.0, source)]
    visited = set()
    remaining = set(targets)
    while heap and remaining:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        remaining.discard(node)
        for neighbour, attr in adjacency[node].items():
            if neighbour in visited:
                continue
            step = 1.0 if weight == "hops" else (link_delay_ms(attr, default=0.0) or 0.0)
            new_cost = cost + step
            if new_cost < dist.get(neighbour, INF):
                dist[neighbour] = new_cost
                prev[neighbour] = node
                heapq.heappush(heap, (new_cost, neighbour))
    routes = {}
    for target in targets:
        if target not in dist:
            routes[target] = (None, INF)
            continue
        path = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        path.reverse()
        routes[target] = (path, dist[target])
    return routes


def all_pairs_routes(
    adjacency: Mapping[Any, Mapping[Any, Any]],
    pairs: Iterable[Tuple[Any, Any]],
    weight: str = "delay",
) -> Dict[Tuple[Any, Any], Tuple[Optional[list], float]]:
    """批量最短路：同一源点只做一次单源 Dijkstra（供 λ 搜索复用）。"""
    targets_by_source: Dict[Any, set] = {}
    for src, dst in pairs:
        targets_by_source.setdefault(src, set()).add(dst)
    cache: Dict[Tuple[Any, Any], Tuple[Optional[list], float]] = {}
    for src, targets in targets_by_source.items():
        if src not in adjacency:
            for dst in targets:
                cache[(src, dst)] = (None, INF)
            continue
        reachable = {t for t in targets if t in adjacency}
        routes = _single_source(adjacency, src, reachable, weight) if reachable else {}
        for dst in targets:
            cache[(src, dst)] = routes.get(dst, (None, INF))
    return cache
