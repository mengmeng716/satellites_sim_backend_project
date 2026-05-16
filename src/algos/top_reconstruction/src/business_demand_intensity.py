"""Business demand intensity derived from link prediction output."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
from typing import Any, DefaultDict, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .adapters import Topology, canonical_link_attributes, iter_edges


LinkKey = Tuple[int, int]


@dataclass(frozen=True)
class BusinessDemandIntensity:
    link_intensity: Dict[LinkKey, float]
    satellite_intensity: Dict[int, float]
    raw_satellite_intensity: Dict[int, float]
    weights: Tuple[float, ...]


def _read_value(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    elif obj is not None:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def exponential_decay_weights(horizon_steps: int, decay_lambda: float) -> Tuple[float, ...]:
    steps = max(0, int(horizon_steps))
    if steps <= 0:
        return tuple()
    raw_weights = [exp(-float(decay_lambda) * k) for k in range(1, steps + 1)]
    total = sum(raw_weights)
    if total <= 0.0:
        return tuple(1.0 / steps for _ in range(steps))
    return tuple(weight / total for weight in raw_weights)


def _predicted_heat_value(link_state: Any) -> Optional[float]:
    heat_value = _as_float(
        _read_value(
            link_state,
            (
                "HeatValue",
                "heat_value",
            ),
        )
    )
    if heat_value is not None:
        return _clamp01(heat_value)
    return None


def _iter_step_links(step: Any) -> Iterable[Tuple[int, int, Any]]:
    topology = _read_value(step, ("Topology", "topology", "LinksPredTopology", "links"))
    if topology is None:
        topology = step
    if not isinstance(topology, Mapping):
        return

    for raw_source_id, raw_links in topology.items():
        try:
            source_id = int(raw_source_id)
        except (TypeError, ValueError):
            continue

        iterable = raw_links.items() if isinstance(raw_links, Mapping) else raw_links or []
        for item in iterable:
            try:
                raw_target_id, link_state = item
                target_id = int(raw_target_id)
            except (TypeError, ValueError):
                continue
            yield source_id, target_id, link_state


def _fallback_link_intensity(
    initial_topology: Optional[Topology],
    max_link_capacity_gbps: float,
) -> Dict[LinkKey, float]:
    link_intensity: Dict[LinkKey, float] = {}
    if not initial_topology:
        return link_intensity

    for source_id, target_id, attr in iter_edges(initial_topology):
        link_attr = canonical_link_attributes(attr)
        heat_value = _as_float(link_attr.get("HeatValue"))
        if heat_value is not None and heat_value > 0.0:
            link_intensity[(source_id, target_id)] = _clamp01(heat_value)
            continue

        capacity = max(
            _as_float(link_attr.get("LinkCapacity"), max_link_capacity_gbps) or max_link_capacity_gbps,
            1e-9,
        )
        current_flow = _as_float(link_attr.get("CurrentFlow"), 0.0) or 0.0
        link_intensity[(source_id, target_id)] = _clamp01(current_flow / capacity)

    return link_intensity


def _all_topology_satellites(initial_topology: Optional[Topology]) -> Iterable[int]:
    if not initial_topology:
        return tuple()
    sat_ids = set(initial_topology)
    for links in initial_topology.values():
        for target_id, _ in links:
            sat_ids.add(int(target_id))
    return sat_ids


def compute_business_demand_intensity(
    predicted_links: Any,
    initial_topology: Optional[Topology] = None,
    horizon_steps: int = 30,
    decay_lambda: float = 0.15,
    max_link_capacity_gbps: float = 10.0,
) -> BusinessDemandIntensity:
    """Compute normalized link and satellite demand intensity.

    Link intensity follows the EWMA model from the reconstruction design:
    predicted normalized flow is weighted by exponentially decaying future-step
    weights. Satellite intensity is the incident-link sum normalized by the
    largest satellite sum in the current reconstruction frame.
    """
    raw_steps = _read_value(predicted_links, ("LinksPredTopology", "links_pred_topology"))
    steps = list(raw_steps or [])[: max(0, int(horizon_steps))]
    weights = exponential_decay_weights(len(steps), decay_lambda)

    link_intensity: DefaultDict[LinkKey, float] = defaultdict(float)
    for step_index, step in enumerate(steps):
        if step_index >= len(weights):
            break
        weight = weights[step_index]
        for source_id, target_id, link_state in _iter_step_links(step):
            heat_value = _predicted_heat_value(link_state)
            if heat_value is None:
                continue
            link_intensity[(source_id, target_id)] += weight * heat_value

    if not link_intensity:
        link_intensity.update(
            _fallback_link_intensity(initial_topology, max_link_capacity_gbps)
        )

    clamped_link_intensity = {
        link: _clamp01(value)
        for link, value in link_intensity.items()
    }

    raw_satellite_intensity: DefaultDict[int, float] = defaultdict(float)
    for sat_id in _all_topology_satellites(initial_topology):
        raw_satellite_intensity[int(sat_id)] += 0.0
    for (source_id, target_id), value in clamped_link_intensity.items():
        raw_satellite_intensity[source_id] += value
        raw_satellite_intensity[target_id] += value

    max_satellite_intensity = max(raw_satellite_intensity.values(), default=0.0)
    if max_satellite_intensity > 0.0:
        satellite_intensity = {
            sat_id: _clamp01(value / max_satellite_intensity)
            for sat_id, value in raw_satellite_intensity.items()
        }
    else:
        satellite_intensity = {
            sat_id: 0.0
            for sat_id in raw_satellite_intensity
        }

    return BusinessDemandIntensity(
        link_intensity=clamped_link_intensity,
        satellite_intensity=satellite_intensity,
        raw_satellite_intensity=dict(raw_satellite_intensity),
        weights=weights,
    )
