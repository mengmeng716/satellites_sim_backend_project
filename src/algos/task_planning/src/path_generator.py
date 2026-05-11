from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence

import networkx as nx


class NoFeasiblePathError(RuntimeError):
    pass


def generate_candidate_paths(
    G: nx.Graph,
    src: int | str,
    dst: int | str,
    num_paths: int = 3,
) -> Dict[str, Any]:
    """
    为一个业务流快速生成候选路径。

    当前优化版不再使用 nx.shortest_simple_paths 在线枚举补路，避免在大规模
    星座图上为了强行凑满 3 条路径而产生高尾延迟。

    路径槽含义：
        1. 最少跳路径：hop_weight 最短路
        2. 低重叠备用路径：对路径 1 已用边加大惩罚后，再找 hop_weight 最短路
        3. 拥塞/负载规避路径：cong_weight 最短路，并对已选路径边加惩罚

    输出固定 num_paths 个槽位。若不足 num_paths 条不同有效路径，则用 [] 补齐，
    valid_mask 对应为 0。后续分流只会在 valid_mask=1 的路径上归一化。
    """

    src = _to_node_id(src)
    dst = _to_node_id(dst)

    if src not in G or dst not in G:
        raise NoFeasiblePathError(f"src or dst not in graph: src={src}, dst={dst}")

    if src == dst:
        raise NoFeasiblePathError(f"src equals dst: {src}")

    raw_paths: List[List[int | str]] = []

    # 1. 最少跳路径
    p1 = _safe_shortest_path(G, src, dst, weight="hop_weight")
    _append_unique_path(raw_paths, p1)

    # 2. 备用低重叠路径：不复制/删图，直接对已选边加大惩罚
    p2 = _safe_shortest_path_with_penalty(
        G,
        src,
        dst,
        base_weight="hop_weight",
        penalized_paths=raw_paths,
        overlap_penalty=1_000_000.0,
    )
    _append_unique_path(raw_paths, p2)

    # 3. 拥塞/负载规避路径：拥塞权重 + 已选边惩罚
    p3 = _safe_shortest_path_with_penalty(
        G,
        src,
        dst,
        base_weight="cong_weight",
        penalized_paths=raw_paths,
        overlap_penalty=100.0,
    )
    _append_unique_path(raw_paths, p3)

    # 若仍不足 num_paths，不再调用 nx.shortest_simple_paths 强行枚举补齐。
    # 后续通过 valid_mask 屏蔽无效路径槽。
    if len(raw_paths) == 0:
        raise NoFeasiblePathError(f"no feasible path found: {src}->{dst}")

    paths: List[List[int | str]] = []
    valid_mask: List[int] = []

    for i in range(num_paths):
        if i < len(raw_paths):
            paths.append(raw_paths[i])
            valid_mask.append(1)
        else:
            paths.append([])
            valid_mask.append(0)

    path_metrics = [compute_path_metrics(G, path) if path else {} for path in paths]

    return {
        "paths": paths,
        "valid_mask": valid_mask,
        "path_metrics": path_metrics,
    }


def compute_path_metrics(G: nx.Graph, path: List[int | str]) -> Dict[str, float]:
    """
    计算单条路径的基本指标。
    """

    if not path or len(path) < 2:
        return {
            "hop_count": 0.0,
            "delay_ms": 0.0,
            "min_left_capacity": 0.0,
            "max_utilization": 0.0,
            "avg_utilization": 0.0,
        }

    delay_ms = 0.0
    left_caps = []
    utils = []

    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]

        delay_ms += float(data.get("delay_ms", 0.0))
        left_caps.append(float(data.get("left_capacity", 0.0)))
        utils.append(float(data.get("utilization", 0.0)))

    return {
        "hop_count": float(len(path) - 1),
        "delay_ms": float(delay_ms),
        "min_left_capacity": float(min(left_caps)) if left_caps else 0.0,
        "max_utilization": float(max(utils)) if utils else 0.0,
        "avg_utilization": float(sum(utils) / len(utils)) if utils else 0.0,
    }


def _safe_shortest_path(
    G: nx.Graph,
    src: int | str,
    dst: int | str,
    weight: str | Callable[[Any, Any, Dict[str, Any]], float],
) -> List[int | str] | None:
    try:
        return list(nx.shortest_path(G, src, dst, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _safe_shortest_path_with_penalty(
    G: nx.Graph,
    src: int | str,
    dst: int | str,
    base_weight: str,
    penalized_paths: Sequence[Sequence[int | str]],
    overlap_penalty: float,
) -> List[int | str] | None:
    penalized_edges = _path_edge_set(penalized_paths)

    def weight(u: Any, v: Any, data: Dict[str, Any]) -> float:
        base = float(data.get(base_weight, 1.0))
        e = _edge_key(u, v)
        if e in penalized_edges:
            return base + float(overlap_penalty)
        return base

    return _safe_shortest_path(G, src, dst, weight=weight)


def _append_unique_path(
    paths: List[List[int | str]],
    path: List[int | str] | None,
) -> None:
    if not path:
        return

    if len(path) < 2:
        return

    path_tuple = tuple(path)
    existing = {tuple(p) for p in paths}

    if path_tuple not in existing:
        paths.append(path)


def _path_edge_set(paths: Sequence[Sequence[int | str]]) -> set[tuple[Any, Any]]:
    edges: set[tuple[Any, Any]] = set()
    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            edges.add(_edge_key(u, v))
    return edges


def _edge_key(u: Any, v: Any) -> tuple[Any, Any]:
    # 当前图是无向图，统一边 key，便于判断重叠边。
    return tuple(sorted((_to_node_id(u), _to_node_id(v)), key=lambda x: str(x)))


def _to_node_id(x: Any) -> int | str:
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)
