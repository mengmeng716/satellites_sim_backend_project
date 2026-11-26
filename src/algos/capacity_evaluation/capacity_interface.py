"""容量分析与效能评估 —— 对外唯一入口（表2-19 输入 / 表2-20 输出）。

用法::

    from src.algos.capacity_evaluation import evaluate

    result = evaluate(
        topology=topo,                       # {src: {dst: LinksQualitiesValue 或 dict}}
        traffic_matrix=[{"src": 0, "dst": 10, "demand_gbps": 8.0}],
        qos={"MaxDelay": 300.0, "MaxLoss": 0.05, "MaxBER": 1e-6},
        capacity_protocol={"LinkCapacity": 100.0, "SearchPrecision": 0.01},
        availability_samples=[{ "0->1": True, ... }, ...],
        planned_links=["0->1", ...],
        baseline_topology=topo_before,       # 可选：用于算容量提升
        timestamp_ms=1700000000000,
    )
"""

from __future__ import annotations

from operator import eq, ge, le
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .availability import measure_availability
from .capacity import capacity_gain_pct, lambda_star
from .performance import evaluate_paths

#: 技术要求中的验收目标
DEFAULT_TARGETS = {
    "AvailabilityMin": 0.999,
    "CapacityGainMinPct": 20.0,
}

DEFAULT_QOS = {"MaxDelay": None, "MaxLoss": None, "MaxBER": None}
DEFAULT_PROTOCOL = {"LinkCapacity": 100.0, "SearchPrecision": 0.01}
RECONSTRUCTION_TRIGGER_PCT = 20.0
RECONSTRUCTION_MIN_GAIN_PCT = 20.0


def reconstruction_decision(
    capacity_report,
    healthy_capacity,
    *,
    prediction_ready,
    cooling_down,
    periodic_due=False,
):
    """Monitor the capacity to serve the entire fixed demand matrix."""
    capacity = capacity_report["Capacity"]
    measured = capacity["CapacityAfter"]
    current = 0.0 if capacity["UnroutableDemands"] else measured
    reference = healthy_capacity
    if reference is None and current is not None and current > 0:
        reference = current

    drop_pct = None
    state = "no_reference"
    if reference is not None and current is not None:
        drop = max(reference - current, 0.0)
        drop_pct = round(drop * 100.0 / reference, 4)
        if drop * 100.0 <= RECONSTRUCTION_TRIGGER_PCT * reference:
            state = "healthy"
        elif cooling_down:
            state = "cooldown"
        elif not prediction_ready:
            state = "waiting_prediction"
        else:
            state = "candidate_unavailable"

    if periodic_due:
        state = "candidate_unavailable"

    return {
        "State": state,
        "TriggerThresholdPct": RECONSTRUCTION_TRIGGER_PCT,
        "HealthyCapacityGbps": reference,
        "CapacityDropPct": drop_pct,
        "CurrentCapacityGbps": current,
        "CandidateCapacityGbps": None,
        "CandidateGainPct": None,
        "RejectionReasons": [],
    }


def compare_reconstruction_candidate(decision, current_report, candidate_report):
    """Require >20% capacity gain and preserve reachability and aggregate QoS."""
    current_capacity = decision["CurrentCapacityGbps"]
    candidate_capacity = candidate_report["Capacity"]
    candidate_gbps = (
        0.0 if candidate_capacity["UnroutableDemands"] else candidate_capacity["CapacityAfter"]
    )
    reasons = []
    if candidate_capacity["UnroutableDemands"]:
        reasons.append("unroutable_demands")
    elif candidate_capacity["LambdaStar"] is not None and candidate_capacity["LambdaStar"] < 1.0:
        reasons.append("insufficient_demand_capacity")
    if candidate_gbps is None or current_capacity is None or candidate_gbps <= current_capacity:
        reasons.append("no_capacity_gain")
    elif current_capacity <= 0:
        reasons.append("capacity_gain_not_comparable")
    elif candidate_gbps <= current_capacity * (1.0 + RECONSTRUCTION_MIN_GAIN_PCT / 100.0):
        reasons.append("capacity_gain_below_target")

    if candidate_capacity.get("CapacityGainAcceptanceApplicable"):
        walker_gain = candidate_capacity.get("CapacityGainVsBaselinePct")
        if walker_gain is None or walker_gain <= RECONSTRUCTION_MIN_GAIN_PCT:
            reasons.append("walker_gain_below_target")

    before_perf = current_report["Performance"]
    after_perf = candidate_report["Performance"]
    if any(
        before_perf[metric] is not None
        and (after_perf[metric] is None or after_perf[metric] > before_perf[metric])
        for metric in ("MeanEndToEndDelay", "P95EndToEndDelay")
    ):
        reasons.append("delay_regression")
    if (
        before_perf["MeanPacketLossRate"] is not None
        and (
            after_perf["MeanPacketLossRate"] is None
            or after_perf["MeanPacketLossRate"] > before_perf["MeanPacketLossRate"]
        )
    ):
        reasons.append("loss_regression")
    if any(
        row["Result"] == "FAIL" and row["Metric"] in {"MeanEndToEndDelay", "MeanPacketLossRate", "MeanBERPostFEC"}
        for row in candidate_report["Verdict"]["Results"]
    ):
        reasons.append("qos_limit_exceeded")

    return {
        **decision,
        "State": "candidate_rejected" if reasons else "applied",
        "CandidateCapacityGbps": candidate_gbps,
        "CandidateGainPct": capacity_gain_pct(current_capacity, candidate_gbps),
        "RejectionReasons": reasons,
    }


def _merge(defaults: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(defaults)
    if override:
        merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _verdict(metric, target, actual, compare, label):
    if target is None:
        result = "N/A"
    elif actual is None:
        result = "NO_DATA"
    else:
        result = "PASS" if compare(actual, float(target)) else "FAIL"
    return {
        "Metric": metric,
        "Target": label.format(target) if target is not None else None,
        "Actual": actual,
        "Result": result,
    }


def evaluate(
    topology: Mapping[Any, Mapping[Any, Any]],
    traffic_matrix: Sequence[Mapping[str, Any]] | Mapping[Any, float],
    qos: Optional[Mapping[str, Any]] = None,
    capacity_protocol: Optional[Mapping[str, Any]] = None,
    availability_samples: Any = None,
    planned_links: Optional[Iterable[str]] = None,
    baseline_topology: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
    evaluation_window: Optional[Mapping[str, Any]] = None,
    frame_seconds: float = 5.0,
    targets: Optional[Mapping[str, Any]] = None,
    timestamp_ms: Optional[int] = None,
    *,
    walker_baseline_topology: Optional[Mapping[Any, Mapping[Any, Any]]] = None,
    walker_gain_applicable: bool = False,
) -> Dict[str, Any]:
    """此外可用同一业务矩阵比较重构前后拓扑与固定 Walker 连接关系。"""
    qos_cfg = _merge(DEFAULT_QOS, qos)
    protocol = _merge(DEFAULT_PROTOCOL, capacity_protocol)
    target_cfg = _merge(DEFAULT_TARGETS, targets)

    link_capacity = float(protocol["LinkCapacity"])
    precision = float(protocol["SearchPrecision"])

    # ---- 容量 ----
    after = lambda_star(topology, traffic_matrix, link_capacity, precision)
    if baseline_topology is not None and baseline_topology is not topology:
        before = lambda_star(baseline_topology, traffic_matrix, link_capacity, precision)
    else:
        before = after

    demand_count = after["routable"] + after["unroutable"]
    before_capacity = before["capacity_gbps"] if demand_count else None
    after_capacity = after["capacity_gbps"] if demand_count else None
    capacity = {
        "CapacityBefore": before_capacity,
        "CapacityAfter": after_capacity,
        "CapacityGainPct": capacity_gain_pct(
            before_capacity if not before["unroutable"] else None,
            after_capacity if not after["unroutable"] else None,
        ),
        "BottleneckLinks": after["bottleneck_links"],
        "LambdaStarBefore": before["lambda_star"] if demand_count else None,
        "LambdaStar": after["lambda_star"] if demand_count else None,
        "RoutableDemands": after["routable"],
        "UnroutableDemands": after["unroutable"],
    }
    gain_metric = "CapacityGainPct"
    gain_target = target_cfg["CapacityGainMinPct"]
    if walker_baseline_topology is not None:
        walker = lambda_star(walker_baseline_topology, traffic_matrix, link_capacity, precision)
        walker_capacity = walker["capacity_gbps"] if demand_count else None
        comparable_walker_capacity = walker_capacity if not walker["unroutable"] else None
        gain_metric = "CapacityGainVsBaselinePct"
        gain_target = gain_target if walker_gain_applicable else None
        capacity.update({
            "CapacityBaselineGbps": walker_capacity,
            "LambdaStarBaseline": walker["lambda_star"] if demand_count else None,
            "UnroutableDemandsBaseline": walker["unroutable"],
            "CapacityBeforeGainVsBaselinePct": capacity_gain_pct(
                comparable_walker_capacity, before_capacity if not before["unroutable"] else None
            ),
            "CapacityGainVsBaselinePct": capacity_gain_pct(
                comparable_walker_capacity, after_capacity if not after["unroutable"] else None
            ),
            "CapacityGainAcceptanceApplicable": walker_gain_applicable,
            "CapacityGainTargetPct": gain_target,
        })

    # ---- 性能 ----
    perf = evaluate_paths(topology, traffic_matrix)
    performance = {
        "MeanEndToEndDelay": perf["MeanEndToEndDelay"],
        "P95EndToEndDelay": perf["P95EndToEndDelay"],
        "MeanPacketLossRate": perf["MeanPacketLossRate"],
        "MeanBERPostFEC": perf["MeanBERPostFEC"],
        "AggregateGoodput": perf["AggregateGoodput"] if demand_count else None,
        "RoutedCount": perf["RoutedCount"],
        "UnroutedCount": perf["UnroutedCount"],
    }

    # ---- 可用率 ----
    availability = measure_availability(
        samples=availability_samples,
        planned_links=planned_links,
        frame_seconds=frame_seconds,
    )
    availability_out = {
        "Availability": availability["Availability"],
        "MinLinkAvailability": availability["MinLinkAvailability"],
        "DowntimeSeconds": availability["DowntimeSeconds"],
        "OutageEvents": availability["OutageEvents"],
    }

    # ---- 逐项判定 ----
    results = [
        _verdict("MeanEndToEndDelay", qos_cfg["MaxDelay"], performance["MeanEndToEndDelay"], le, "≤{} ms"),
        _verdict("MeanPacketLossRate", qos_cfg["MaxLoss"], performance["MeanPacketLossRate"], le, "≤{}"),
        _verdict("MeanBERPostFEC", qos_cfg["MaxBER"], performance["MeanBERPostFEC"], le, "≤{}"),
        _verdict("LinkAvailability", target_cfg["AvailabilityMin"], availability_out["Availability"], ge, "≥{}"),
        _verdict(gain_metric, gain_target, capacity[gain_metric], ge, "≥{}%"),
        _verdict("UnroutableDemands", 0, capacity["UnroutableDemands"] if demand_count else None, eq, "={}"),
    ]
    passed = sum(1 for item in results if item["Result"] == "PASS")
    assessed = sum(1 for item in results if item["Result"] in ("PASS", "FAIL"))

    return {
        "Capacity": capacity,
        "Availability": availability_out,
        "Performance": performance,
        "Verdict": {"Results": results, "Passed": passed, "Total": assessed},
        "Protocol": {"LinkCapacity": link_capacity, "SearchPrecision": precision,
                     "FrameSeconds": frame_seconds},
        "Window": dict(evaluation_window) if evaluation_window else None,
        "Timestamp": int(timestamp_ms) if timestamp_ms is not None else None,
        "Samples": perf["Samples"],
    }
