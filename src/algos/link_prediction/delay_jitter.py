"""Estimate single-link propagation and queueing jitter from link records."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping
from math import fsum, isfinite, sqrt
from typing import Any

from src import config


_LinkSample = tuple[float, float, float | None]


def _duration_ms(seconds: float, name: str) -> int:
    milliseconds = seconds * 1000.0
    if not isfinite(milliseconds) or milliseconds < 1.0:
        raise ValueError(f"{name} must be finite and at least 0.001 seconds")
    return round(milliseconds)


def _link_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["SrcSatID"]), str(record["DstSatID"])


def _sample(record: Mapping[str, Any], flow_field: str) -> _LinkSample | None:
    try:
        flow = float(record[flow_field])
        rate = float(record["AchievableRate"])
        propagation = record.get("LinkPropagationDelay")
        if propagation is not None:
            propagation = float(propagation)
            if not isfinite(propagation) or propagation < 0.0:
                return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (isfinite(flow) and flow >= 0.0 and isfinite(rate) and rate > 0.0):
        return None
    return flow, rate, propagation


def _estimate_at(
    timestamp: int,
    timestamps: list[int],
    samples: Mapping[int, _LinkSample | None],
    step_ms: int,
    window_ms: int,
) -> float | None:
    window_start = timestamp - window_ms
    covered_until = timestamp
    packet_bits_ms = config.LINK_PACKET_BITS * 1e-6
    queue_moments = []
    last_index = bisect_right(timestamps, timestamp) - 1
    for index in range(last_index, -1, -1):
        sample_end = timestamps[index]
        if sample_end < covered_until:
            return None
        sample_start = max(window_start, sample_end - step_ms)
        sample = samples[sample_end]
        if sample is None:
            return None
        flow_gbps, rate_gbps, propagation_ms = sample
        if flow_gbps >= rate_gbps:
            return None

        # For fixed-size packets, traffic volume is proportional to packet count.
        if flow_gbps > 0.0 and covered_until > sample_start:
            service_time_ms = packet_bits_ms / rate_gbps
            load_ratio = flow_gbps / (rate_gbps - flow_gbps)
            mean_wait_ms = service_time_ms * load_ratio / 2.0
            variance_ms2 = (
                service_time_ms * service_time_ms * load_ratio / 3.0 + mean_wait_ms * mean_wait_ms
            )
            weight = flow_gbps * (covered_until - sample_start)
            queue_moments.append((weight, mean_wait_ms, variance_ms2, propagation_ms))
        covered_until = sample_start
        if covered_until == window_start:
            break
    else:
        return None

    if not queue_moments:
        return 0.0

    reference_propagation = queue_moments[0][3]
    if any((delay is None) != (reference_propagation is None) for _, _, _, delay in queue_moments):
        return None

    # Subtract a common propagation offset so fixed flight time adds no jitter.
    moments = [
        (weight, mean_wait + (delay - reference_propagation if delay is not None else 0.0), variance)
        for weight, mean_wait, variance, delay in queue_moments
    ]
    total_weight = fsum(weight for weight, _, _ in moments)
    mean_delay_ms = fsum((weight / total_weight) * mean for weight, mean, _ in moments)
    variance = fsum(
        (weight / total_weight) * (within_variance + (mean - mean_delay_ms) * (mean - mean_delay_ms))
        for weight, mean, within_variance in moments
    )
    jitter_ms = sqrt(variance)
    return jitter_ms if isfinite(jitter_ms) else None


def estimate_delay_jitter(
    records: Iterable[Mapping[str, Any]],
    step_seconds: float | None = None,
    *,
    history_records: Iterable[Mapping[str, Any]] = (),
    window_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Return one link-delay jitter estimate per Table 2-2 prediction row.

    Rows use SrcSatID, DstSatID, Timestamp (Unix milliseconds), AchievableRate
    (Gbps), and PredFlow (Gbps). Optional history rows use CurrentFlow instead
    of PredFlow. Each sample covers one step ending at its timestamp. The
    default step follows the simulation micro period. A prediction wins when
    history and prediction share a link and timestamp.

    DelayJitter is the standard deviation in milliseconds of propagation
    plus queue waiting time. Supply LinkPropagationDelay in milliseconds at
    each sample to include propagation. With no propagation data, use queue
    waiting time alone. The window weights each interval by carried traffic.
    A complete idle window returns zero. Results retain input order, timestamp
    and link IDs. Missing coverage, inconsistent propagation data, invalid
    measurements or saturated intervals return None.
    """
    step_ms = _duration_ms(
        config.MICRO_PERIOD_SECONDS if step_seconds is None else step_seconds,
        "step_seconds",
    )
    window_ms = _duration_ms(window_seconds, "window_seconds")
    predictions = [
        (_link_key(record), int(record["Timestamp"]), _sample(record, "PredFlow"))
        for record in records
    ]
    if not predictions:
        return []

    timelines: dict[tuple[str, str], dict[int, _LinkSample | None]] = {
        link: {} for link, _, _ in predictions
    }
    for record in history_records:
        link = _link_key(record)
        if link in timelines:
            timelines[link][int(record["Timestamp"])] = _sample(record, "CurrentFlow")
    for link, timestamp, sample in predictions:
        timelines[link][timestamp] = sample
    timestamps = {link: sorted(samples) for link, samples in timelines.items()}

    return [
        {
            "SrcSatID": link[0],
            "DstSatID": link[1],
            "Timestamp": timestamp,
            "DelayJitter": _estimate_at(timestamp, timestamps[link], timelines[link], step_ms, window_ms),
        }
        for link, timestamp, _ in predictions
    ]
