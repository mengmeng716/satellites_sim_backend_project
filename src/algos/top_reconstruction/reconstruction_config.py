"""Topology reconstruction algorithm parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TopReconstructionConfig:
    offset_window: int = 1
    max_switch_rate: Optional[float] = 0.10
    processing_delay_ms: float = 5.0
    pair_sample_count: int = 100
    pair_sample_seed: int = 20260428
    min_ground_elevation_deg: float = 10.0
