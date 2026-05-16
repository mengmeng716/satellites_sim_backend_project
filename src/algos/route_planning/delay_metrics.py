from __future__ import annotations

from typing import Any, Dict, Mapping


MIN_CAPACITY_GBPS = 1e-6
QUEUE_UTILIZATION_CAP = 0.99
PACKET_LOSS_START_UTILIZATION = 0.8
PACKET_LOSS_SLOPE = 0.5


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _get(qualities: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in qualities:
            return qualities[key]
    return default


def nominal_capacity_gbps(qualities: Mapping[str, Any]) -> float:
    return max(
        _to_float(_get(qualities, "MaxCapacity", "LinkCapacity", "Capacity"), 10.0),
        MIN_CAPACITY_GBPS,
    )


def available_capacity_gbps(qualities: Mapping[str, Any]) -> float:
    capacity_gbps = nominal_capacity_gbps(qualities)
    left_capacity = _to_float(
        _get(qualities, "LeftCapacity", "LinkAvailableGbps"),
        capacity_gbps,
    )

    if 0.0 <= left_capacity <= 1.0 and capacity_gbps > 1.0:
        return max(0.0, left_capacity * capacity_gbps)
    return max(0.0, left_capacity)


def remaining_capacity_ratio(qualities: Mapping[str, Any]) -> float:
    capacity_gbps = nominal_capacity_gbps(qualities)
    return max(0.0, min(1.0, available_capacity_gbps(qualities) / capacity_gbps))


def demand_gbps(packet_size_mbits: Any, duration_seconds: Any) -> float:
    packet_size = max(0.0, _to_float(packet_size_mbits, 0.0))
    duration = _to_float(duration_seconds, 0.0)
    if duration <= 0.0:
        return 0.0
    return packet_size / duration / 1000.0


def calculate_link_delay_metrics(
    qualities: Mapping[str, Any],
    packet_size_mbits: Any,
    duration_seconds: Any = 0.0,
) -> Dict[str, float]:
    capacity_gbps = nominal_capacity_gbps(qualities)
    available_gbps = available_capacity_gbps(qualities)
    current_flow_gbps = max(
        0.0,
        _to_float(_get(qualities, "CurrentFlow", "CumFlow", "HistoryFlow"), 0.0),
    )
    packet_size = max(0.0, _to_float(packet_size_mbits, 0.0))
    expected_flow_gbps = current_flow_gbps + demand_gbps(packet_size, duration_seconds)
    raw_utilization = expected_flow_gbps / capacity_gbps
    utilization = max(0.0, min(QUEUE_UTILIZATION_CAP, raw_utilization))

    transmission_delay_ms = (
        packet_size / max(available_gbps, MIN_CAPACITY_GBPS)
        if packet_size > 0.0
        else 0.0
    )
    queue_delay_ms = (
        transmission_delay_ms * utilization / (1.0 - utilization)
        if transmission_delay_ms > 0.0 and utilization > 0.0
        else 0.0
    )

    if available_gbps <= 0.0 or raw_utilization >= 1.0:
        packet_loss_rate = 1.0
    elif raw_utilization <= PACKET_LOSS_START_UTILIZATION:
        packet_loss_rate = 0.0
    else:
        packet_loss_rate = min(
            1.0,
            (raw_utilization - PACKET_LOSS_START_UTILIZATION) * PACKET_LOSS_SLOPE,
        )

    return {
        "TransmissionDelay": float(transmission_delay_ms),
        "QueueDelay": float(queue_delay_ms),
        "PacketLossRate": float(max(0.0, min(1.0, packet_loss_rate))),
        "AvailableCapacityGbps": float(available_gbps),
        "RemainingCapacityRatio": remaining_capacity_ratio(qualities),
        "Utilization": float(max(0.0, raw_utilization)),
    }
