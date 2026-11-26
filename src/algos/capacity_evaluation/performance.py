"""性能评估：端到端时延、丢包、误码、有效吞吐。

口径：

* 端到端时延 = 沿途链路传播时延之和（ms）
* 丢包率 = 1 − Π(1 − 单链路丢包率)
* 误码率 = 1 − Π(1 − 单链路误码率)（同量级近似）
* 有效吞吐 = Σ 需求 × (1 − 丢包率)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .graph import all_pairs_routes, build_adjacency, link_ber, link_loss


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def evaluate_paths(
    topology: Mapping[Any, Mapping[Any, Any]],
    demands: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """逐条业务计算路径级性能指标并汇总。

    :return: ``{"MeanEndToEndDelay", "P95EndToEndDelay", "MeanPacketLossRate",
                "MeanBERPostFEC", "AggregateGoodput", "RoutedCount",
                "UnroutedCount", "Samples"}``
    """
    adjacency = build_adjacency(topology)
    pairs: List[Tuple[Any, Any]] = []
    normalized: List[Tuple[Any, Any, float]] = []
    for entry in demands or []:
        src = entry.get("src", entry.get("SrcSatID"))
        dst = entry.get("dst", entry.get("DstSatID"))
        gbps = float(entry.get("demand_gbps", entry.get("DemandGbps", 0.0)) or 0.0)
        if src is None or dst is None or gbps <= 0:
            continue
        normalized.append((src, dst, gbps))
        pairs.append((src, dst))

    routes = all_pairs_routes(adjacency, pairs)

    delays: List[float] = []
    losses: List[float] = []
    bers: List[float] = []
    goodput = 0.0
    samples: List[Dict[str, Any]] = []
    unrouted = 0

    for src, dst, gbps in normalized:
        path, delay = routes.get((src, dst), (None, float("inf")))
        if path is None:
            unrouted += 1
            continue

        keep = 1.0
        ber_keep = 1.0
        for i in range(len(path) - 1):
            attr = adjacency[path[i]][path[i + 1]]
            keep *= (1.0 - min(max(link_loss(attr), 0.0), 1.0))
            ber_keep *= (1.0 - min(max(link_ber(attr), 0.0), 1.0))

        loss = 1.0 - keep
        ber = 1.0 - ber_keep
        delays.append(float(delay))
        losses.append(loss)
        bers.append(ber)
        goodput += gbps * keep
        samples.append({
            "SrcSatID": str(src), "DstSatID": str(dst), "DemandGbps": gbps,
            "Path": [str(x) for x in path], "DelayMs": round(float(delay), 6),
            "PacketLossRate": round(loss, 9), "BER": round(ber, 12),
        })

    return {
        "MeanEndToEndDelay": round(float(np.mean(delays)), 6) if delays else None,
        "P95EndToEndDelay": round(_percentile(delays, 95), 6) if delays else None,
        "MeanPacketLossRate": round(float(np.mean(losses)), 9) if losses else None,
        "MeanBERPostFEC": round(float(np.mean(bers)), 12) if bers else None,
        "AggregateGoodput": round(goodput, 6),
        "RoutedCount": len(delays),
        "UnroutedCount": unrouted,
        "Samples": samples,
    }
