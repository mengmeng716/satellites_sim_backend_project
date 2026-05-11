# src/algos/task_planning/src/utils.py

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import networkx as nx


def mask_and_normalize_action(
    action: np.ndarray,
    valid_mask: Sequence[int],
    eps: float = 1e-8,
) -> np.ndarray:
    """将 actor 输出动作转换成有效路径分流比例。"""

    a = np.asarray(action, dtype=np.float32).reshape(-1)
    mask = np.asarray(valid_mask, dtype=np.float32).reshape(-1)

    if a.shape[0] != mask.shape[0]:
        raise ValueError(f"action and valid_mask shape mismatch: {a.shape} vs {mask.shape}")

    a = np.maximum(a, 0.0)
    a = a * mask

    s = float(a.sum())

    if s <= eps:
        valid_count = int(mask.sum())
        if valid_count <= 0:
            return np.zeros_like(a, dtype=np.float32)
        return (mask / float(valid_count)).astype(np.float32)

    return (a / s).astype(np.float32)


def build_allocations(
    paths: Sequence[Sequence[int | str]],
    ratios: Sequence[float],
    valid_mask: Sequence[int],
    demand_gbps: float,
) -> List[Dict[str, Any]]:
    """根据路径和比例生成基础 allocations。"""

    allocations: List[Dict[str, Any]] = []

    for path, ratio, valid in zip(paths, ratios, valid_mask):
        if int(valid) == 0:
            continue

        ratio = float(ratio)
        if ratio <= 1e-8:
            continue

        allocations.append(
            {
                "path": [str(x) for x in path],
                "ratio": ratio,
                "bandwidth_gbps": float(demand_gbps) * ratio,
            }
        )

    return allocations


def add_allocation_times(
    G: nx.Graph,
    allocations: Sequence[Dict[str, Any]],
    start_time: float,
) -> List[Dict[str, Any]]:
    """
    为每条实际使用路径补充占用时间。

    start_time = 业务流到达时间
    end_time   = start_time + 当前路径传播时延
    """

    result: List[Dict[str, Any]] = []

    for alloc in allocations:
        item = dict(alloc)
        propagation_delay = compute_path_propagation_delay(G, item.get("path", []))

        item["start_time"] = float(start_time)
        item["end_time"] = float(start_time) + float(propagation_delay)

        result.append(item)

    return result


def compute_path_propagation_delay(
    G: nx.Graph,
    path: Sequence[int | str],
) -> float:
    """计算路径传播时延之和，只累加 link_propagation_delay。"""

    total_delay = 0.0

    for u, v in zip(path[:-1], path[1:]):
        u_id = _to_node_id(u)
        v_id = _to_node_id(v)

        if G.has_edge(u_id, v_id):
            data = G[u_id][v_id]
        elif not G.is_directed() and G.has_edge(v_id, u_id):
            data = G[v_id][u_id]
        else:
            continue

        total_delay += float(data.get("link_propagation_delay", 0.0))

    return float(total_delay)


def compute_path_delay_ms(
    G: nx.Graph,
    path: Sequence[int | str],
) -> float:
    """计算路径总时延，累加 delay_ms。"""

    total_delay = 0.0

    for u, v in zip(path[:-1], path[1:]):
        u_id = _to_node_id(u)
        v_id = _to_node_id(v)

        if G.has_edge(u_id, v_id):
            total_delay += float(G[u_id][v_id].get("delay_ms", 0.0))
        elif not G.is_directed() and G.has_edge(v_id, u_id):
            total_delay += float(G[v_id][u_id].get("delay_ms", 0.0))

    return float(total_delay)


def compute_allocation_metrics(
    G: nx.Graph,
    allocations: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """计算分流结果指标。"""

    if not allocations:
        return {
            "avg_delay_ms": 0.0,
            "jitter_ms": 0.0,
            "max_link_utilization_after": 0.0,
            "overflow_amount": 0.0,
        }

    delays = []
    ratios = []

    for alloc in allocations:
        path = alloc.get("path", [])
        ratio = float(alloc.get("ratio", 0.0))

        delays.append(compute_path_delay_ms(G, path))
        ratios.append(ratio)

    avg_delay = float(sum(d * r for d, r in zip(delays, ratios)))

    if len(delays) >= 2:
        jitter = float(np.std(np.asarray(delays, dtype=np.float32)))
    else:
        jitter = 0.0

    edge_added: Dict[tuple[Any, Any], float] = {}

    for alloc in allocations:
        path = [_to_node_id(x) for x in alloc.get("path", [])]
        bw = float(alloc.get("bandwidth_gbps", 0.0))

        for u, v in zip(path[:-1], path[1:]):
            edge_key = tuple(sorted((u, v)))
            edge_added[edge_key] = edge_added.get(edge_key, 0.0) + bw

    max_util_after = 0.0
    overflow_amount = 0.0

    for (u, v), added_bw in edge_added.items():
        if G.has_edge(u, v):
            data = G[u][v]
        elif not G.is_directed() and G.has_edge(v, u):
            data = G[v][u]
        else:
            continue

        capacity = max(float(data.get("capacity", 1.0)), 1e-12)
        load_before = float(data.get("load", 0.0))
        load_after = load_before + float(added_bw)

        max_util_after = max(max_util_after, load_after / capacity)
        overflow_amount += max(0.0, load_after - capacity)

    return {
        "avg_delay_ms": float(avg_delay),
        "jitter_ms": float(jitter),
        "max_link_utilization_after": float(max_util_after),
        "overflow_amount": float(overflow_amount),
    }


def validate_allocations_capacity(
    G: nx.Graph,
    allocations: Sequence[Dict[str, Any]],
    tol: float = 1e-8,
) -> tuple[bool, str, Dict[str, Any]]:
    """检查分流结果是否满足链路容量约束。"""

    edge_added: Dict[frozenset, Dict[str, Any]] = {}

    for alloc in allocations:
        path = [_to_node_id(x) for x in alloc.get("path", [])]
        bw = float(alloc.get("bandwidth_gbps", 0.0))

        if len(path) < 2:
            return False, "invalid path length", {"invalid_allocation": alloc}

        for u, v in zip(path[:-1], path[1:]):
            if G.has_edge(u, v):
                edge_data = G[u][v]
                edge_key = frozenset((u, v))
            elif not G.is_directed() and G.has_edge(v, u):
                edge_data = G[v][u]
                edge_key = frozenset((u, v))
            else:
                return False, f"edge not found: {u}->{v}", {
                    "missing_edge": [str(u), str(v)],
                    "path": [str(x) for x in path],
                }

            if edge_key not in edge_added:
                edge_added[edge_key] = {
                    "u": u,
                    "v": v,
                    "added_bw": 0.0,
                    "capacity": float(edge_data.get("capacity", 0.0)),
                    "load_before": float(edge_data.get("load", 0.0)),
                }

            edge_added[edge_key]["added_bw"] += bw

    max_util_after = 0.0
    total_overflow = 0.0
    edge_details = []

    for item in edge_added.values():
        capacity = max(float(item["capacity"]), 1e-12)
        load_before = float(item["load_before"])
        added_bw = float(item["added_bw"])
        load_after = load_before + added_bw
        util_after = load_after / capacity
        overflow = max(0.0, load_after - capacity)

        max_util_after = max(max_util_after, util_after)
        total_overflow += overflow

        detail = {
            "edge": [str(item["u"]), str(item["v"])],
            "capacity": capacity,
            "load_before": load_before,
            "added_bw": added_bw,
            "load_after": load_after,
            "util_after": util_after,
            "overflow": overflow,
        }

        edge_details.append(detail)

        if load_after > capacity + tol:
            return False, (
                f"capacity violation on edge {item['u']}->{item['v']}: "
                f"load_after={load_after:.6f}, capacity={capacity:.6f}"
            ), {
                "violation_edge": detail,
                "max_utilization_after": max_util_after,
                "total_overflow": total_overflow,
                "edge_details": edge_details,
            }

    return True, "ok", {
        "max_utilization_after": max_util_after,
        "total_overflow": total_overflow,
        "edge_details": edge_details,
    }


def _to_node_id(x: Any) -> int | str:
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)
