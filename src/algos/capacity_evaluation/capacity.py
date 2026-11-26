"""容量评估：业务倍数 λ*、瓶颈链路、容量增益。

口径（与接口文档表2-19/2-20 一致）：

* ``λ*``：在给定单链路额定容量下，全网业务可同时放大的倍数；
* ``Capacity``：可承载业务量 = Σ(需求 × λ*)，单位 Gbps；
* ``BottleneckLinks``：λ* 处残余容量耗尽的链路。

判定方式为「固定最短路 + 残余容量贪心」，再对 λ 二分搜索。
这是工程近似（精确解是并发多商品流 LP），但可复现、可解释、可批量。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .graph import all_pairs_routes, build_adjacency

#: λ 搜索的默认精度
DEFAULT_SEARCH_PRECISION = 0.01

#: λ 搜索上限（防止无限放大）
MAX_LAMBDA = 1000.0


def _normalize_demands(demands: Sequence[Mapping[str, Any]] | Mapping[Any, float]):
    """统一成 ``[(src, dst, gbps), ...]``。"""
    items: List[Tuple[Any, Any, float]] = []
    if isinstance(demands, Mapping):
        for key, value in demands.items():
            if isinstance(key, (list, tuple)) and len(key) == 2:
                items.append((key[0], key[1], float(value)))
        return items

    for entry in demands or []:
        src = entry.get("src", entry.get("SrcSatID", entry.get("source")))
        dst = entry.get("dst", entry.get("DstSatID", entry.get("target")))
        gbps = entry.get("demand_gbps", entry.get("DemandGbps", entry.get("gbps", 0.0)))
        if src is None or dst is None or float(gbps) <= 0:
            continue
        items.append((src, dst, float(gbps)))
    return items


def _path_bottleneck_capacity(path, adjacency, capacity_gbps: float) -> float:
    capacities = []
    for i in range(len(path) - 1):
        attr = adjacency[path[i]][path[i + 1]]
        capacities.append(_link_capacity(attr, capacity_gbps))
    return min(capacities) if capacities else float("inf")


def _link_capacity(attr: Any, default: float) -> float:
    for name in ("link_capacity", "LinkCapacity", "capacity_gbps"):
        if isinstance(attr, Mapping):
            if attr.get(name) is not None:
                return float(attr[name])
        elif attr is not None and getattr(attr, name, None) is not None:
            return float(getattr(attr, name))
    return float(default)


def feasible(
    routes: Mapping[Tuple[Any, Any], Tuple[Optional[list], float]],
    demands: Sequence[Tuple[Any, Any, float]],
    adjacency: Mapping[Any, Mapping[Any, Any]],
    capacity_gbps: float,
    lam: float,
) -> Tuple[bool, Dict[Tuple[Any, Any], float]]:
    """判断 λ 是否可行，返回 ``(可行, 各链路残余容量)``。"""
    residual: Dict[Tuple[Any, Any], float] = {}
    for node, neighbours in adjacency.items():
        for neighbour, attr in neighbours.items():
            key = tuple(sorted((str(node), str(neighbour))))
            if key not in residual:
                residual[key] = _link_capacity(attr, capacity_gbps)

    # 大象流优先，保证确定性
    ordered = sorted(demands, key=lambda item: item[2], reverse=True)
    for src, dst, gbps in ordered:
        path, _ = routes.get((src, dst), (None, float("inf")))
        if path is None:
            return False, residual
        need = gbps * lam
        for i in range(len(path) - 1):
            key = tuple(sorted((str(path[i]), str(path[i + 1]))))
            if residual.get(key, 0.0) + 1e-9 < need:
                return False, residual
        for i in range(len(path) - 1):
            key = tuple(sorted((str(path[i]), str(path[i + 1]))))
            residual[key] -= need
    return True, residual


def lambda_star(
    topology: Mapping[Any, Mapping[Any, Any]],
    demands: Sequence[Mapping[str, Any]] | Mapping[Any, float],
    capacity_gbps: float,
    precision: float = DEFAULT_SEARCH_PRECISION,
) -> Dict[str, Any]:
    """二分搜索最大可行业务倍数。

    :return: ``{"lambda_star", "capacity_gbps", "routable", "unroutable",
                "bottleneck_links", "iterations"}``
    """
    items = _normalize_demands(demands)
    adjacency = build_adjacency(topology)
    if not items or not adjacency:
        return {
            "lambda_star": 0.0, "capacity_gbps": 0.0, "routable": 0,
            "unroutable": len(items), "bottleneck_links": [], "iterations": 0,
        }

    pairs = [(src, dst) for src, dst, _ in items]
    routes = all_pairs_routes(adjacency, pairs)

    routable = [(s, d, g) for s, d, g in items if routes.get((s, d), (None, 0))[0] is not None]
    unroutable = len(items) - len(routable)
    if not routable:
        return {
            "lambda_star": 0.0, "capacity_gbps": 0.0, "routable": 0,
            "unroutable": len(items), "bottleneck_links": [], "iterations": 0,
        }

    # 上界：任一条需求的路径瓶颈决定的单流上限
    upper = MAX_LAMBDA
    for src, dst, gbps in routable:
        path, _ = routes[(src, dst)]
        bottleneck = _path_bottleneck_capacity(path, adjacency, capacity_gbps)
        upper = min(upper, bottleneck / gbps)
    upper = max(upper, 0.0)

    ok_upper, residual_upper = feasible(routes, routable, adjacency, capacity_gbps, upper)
    if ok_upper:
        return {
            "lambda_star": round(upper, 6),
            "capacity_gbps": round(sum(g for _, _, g in routable) * upper, 6),
            "routable": len(routable), "unroutable": unroutable,
            "bottleneck_links": [], "iterations": 0,
        }

    low, high = 0.0, upper
    iterations = 0
    residual_at_best: Dict[Tuple[Any, Any], float] = {}
    while high - low > precision:
        mid = (low + high) / 2.0
        ok, residual = feasible(routes, routable, adjacency, capacity_gbps, mid)
        iterations += 1
        if ok:
            low, residual_at_best = mid, residual
        else:
            high = mid

    # 用最终 λ 重算一次残余，识别瓶颈
    ok, residual = feasible(routes, routable, adjacency, capacity_gbps, low)
    if ok:
        residual_at_best = residual

    bottlenecks = sorted(
        f"{key[0]}->{key[1]}"
        for key, value in residual_at_best.items()
        if value <= 1e-6
    )

    return {
        "lambda_star": round(low, 6),
        "capacity_gbps": round(sum(g for _, _, g in routable) * low, 6),
        "routable": len(routable),
        "unroutable": unroutable,
        "bottleneck_links": bottlenecks,
        "iterations": iterations,
    }


def capacity_gain_pct(before_gbps: float, after_gbps: float) -> Optional[float]:
    """容量提升百分比；基准为 0 时返回 ``None``（不可计算而非无穷大）。"""
    if before_gbps is None or after_gbps is None:
        return None
    if before_gbps <= 1e-9:
        return None
    return round((after_gbps - before_gbps) / before_gbps * 100.0, 4)
