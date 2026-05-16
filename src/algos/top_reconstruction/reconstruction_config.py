"""Topology reconstruction algorithm parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class ForwardingDelayParams:
    base_ms: float
    kappa_ms: float
    gamma: float
    max_ms: float


@dataclass(frozen=True)
class TopReconstructionConfig:
    offset_window: int = 1
    max_switch_rate: Optional[float] = 0.10
    min_average_reduction_rate: float = 0.10
    business_intensity_horizon_steps: int = 6
    business_intensity_decay_lambda: float = 0.15
    protected_link_intensity_threshold: float = 0.5
    core_node_count_by_constellation: Mapping[str, int] = field(
        default_factory=lambda: {
            "3600": 20,
            "432": 15,
            "default": 10,
        }
    )
    forwarding_delay_base_ms_by_constellation: Mapping[str, float] = field(
        default_factory=lambda: {
            "3600": 0.5,
            "432": 0.6,
            "default": 0.5,
        }
    )
    forwarding_delay_kappa_ms_by_constellation: Mapping[str, float] = field(
        default_factory=lambda: {
            "3600": 1.0,
            "432": 1.0,
            "default": 1.0,
        }
    )
    forwarding_delay_gamma_by_constellation: Mapping[str, float] = field(
        default_factory=lambda: {
            "3600": 0.95,
            "432": 0.95,
            "default": 0.95,
        }
    )
    forwarding_delay_max_ms_by_constellation: Mapping[str, float] = field(
        default_factory=lambda: {
            "3600": 20.0,
            "432": 20.0,
            "default": 20.0,
        }
    )

    @staticmethod
    def _constellation_value(
        values: Mapping[str, float],
        constellation_id: str,
    ) -> float:
        key = str(constellation_id)
        if key in values:
            return float(values[key])
        return float(values["default"])

    def core_node_count(self, constellation_id: str) -> int:
        key = str(constellation_id)
        if key in self.core_node_count_by_constellation:
            return int(self.core_node_count_by_constellation[key])
        return int(self.core_node_count_by_constellation["default"])

    def forwarding_delay_params(self, constellation_id: str) -> ForwardingDelayParams:
        return ForwardingDelayParams(
            base_ms=self._constellation_value(
                self.forwarding_delay_base_ms_by_constellation,
                constellation_id,
            ),
            kappa_ms=self._constellation_value(
                self.forwarding_delay_kappa_ms_by_constellation,
                constellation_id,
            ),
            gamma=self._constellation_value(
                self.forwarding_delay_gamma_by_constellation,
                constellation_id,
            ),
            max_ms=self._constellation_value(
                self.forwarding_delay_max_ms_by_constellation,
                constellation_id,
            ),
        )
