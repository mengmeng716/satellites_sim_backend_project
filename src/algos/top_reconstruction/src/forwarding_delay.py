"""Node forwarding delay model for topology reconstruction."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def forwarding_delay_ms(
    intensity: float,
    base_ms: float,
    kappa_ms: float,
    gamma: float,
    max_ms: float,
) -> float:
    """Return L_f = L_base + kappa * I / (1 - gamma * I)."""
    demand_intensity = _clamp01(intensity)
    denominator = max(1.0 - float(gamma) * demand_intensity, 1e-9)
    delay = float(base_ms) + float(kappa_ms) * (demand_intensity / denominator)
    return max(0.0, min(float(max_ms), delay))


def build_forwarding_delay_map(
    satellite_business_intensity: Mapping[int, float],
    base_ms: float,
    kappa_ms: float,
    gamma: float,
    max_ms: float,
    satellite_ids: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    delays: Dict[int, float] = {}
    if satellite_ids is not None:
        for sat_id in satellite_ids:
            delays[int(sat_id)] = forwarding_delay_ms(
                satellite_business_intensity.get(int(sat_id), 0.0),
                base_ms,
                kappa_ms,
                gamma,
                max_ms,
            )

    for sat_id, intensity in satellite_business_intensity.items():
        delays[int(sat_id)] = forwarding_delay_ms(
            intensity,
            base_ms,
            kappa_ms,
            gamma,
            max_ms,
        )
    return delays
