"""采样计划、遥测存储、采样调度、覆盖率统计与参数健康度。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

from .metrics import PARAM_SPECS

#: 判定“迟到”的时间容差（s）
GAP_TOLERANCE_S = 1e-6

#: 参数健康度统计的回看窗口（s）
HEALTH_WINDOW_S = 60.0


@dataclass
class SamplingPlan:
    """各参数的计划采样间隔（s）。"""

    intervals: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "SamplingPlan":
        return cls({name: spec.interval_s for name, spec in PARAM_SPECS.items()})

    def interval_of(self, name: str) -> float:
        return float(self.intervals[name])

    def is_critical(self, name: str) -> bool:
        return bool(PARAM_SPECS[name].critical)


@dataclass
class _Series:
    entity_ids: List[str]
    unit: str
    entity_type: str
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    latest: Optional[np.ndarray] = None
    latest_frame: str = ""
    missing_count: int = 0


class TelemetryStore:
    """按参数保存采样时刻与最新一次的逐实体取值。"""

    def __init__(self) -> None:
        self._series: Dict[str, _Series] = {}

    def register(self, name: str, entity_ids: Sequence[str], unit: str, entity_type: str) -> None:
        """登记或更新一个参数的实体清单（拓扑变化时重新登记）。"""
        ids = [str(x) for x in entity_ids]
        series = self._series.get(name)
        if series is None:
            self._series[name] = _Series(ids, unit, entity_type)
        else:
            series.entity_ids = ids
            series.unit = unit
            series.entity_type = entity_type

    def update(self, name: str, values: Sequence[Any], sim_time_s: float, frame_id: str,
               interval_s: float) -> None:
        """写入一次采样。数值型取值中的非有限值计为缺失。"""
        series = self._series[name]
        array = np.asarray(values)
        if array.dtype.kind in "fc":
            series.missing_count += int(np.count_nonzero(~np.isfinite(array)))
        series.timestamps.append(float(sim_time_s))
        series.latest = array
        series.latest_frame = str(frame_id)

    def names(self) -> List[str]:
        return list(self._series)

    def timestamps(self, name: str) -> List[float]:
        return list(self._series[name].timestamps)

    def latest(self, name: str) -> Optional[np.ndarray]:
        return self._series[name].latest

    def entity_ids(self, name: str) -> List[str]:
        return list(self._series[name].entity_ids)

    def unit(self, name: str) -> str:
        return self._series[name].unit

    def entity_type(self, name: str) -> str:
        return self._series[name].entity_type

    def missing_count(self, name: str) -> int:
        return self._series[name].missing_count


class SamplingScheduler:
    """按计划间隔判定本时刻需要采样的参数。"""

    def __init__(self, plan: SamplingPlan) -> None:
        self.plan = plan
        self._last: Dict[str, float] = {}

    def due(self, names: Iterable[str], sim_time_s: float) -> List[str]:
        out = []
        for name in names:
            last = self._last.get(name)
            if last is None or sim_time_s - last >= self.plan.interval_of(name) - GAP_TOLERANCE_S:
                out.append(name)
        return out

    def mark_sampled(self, names: Iterable[str], sim_time_s: float) -> None:
        for name in names:
            self._last[name] = float(sim_time_s)


class CoverageTracker:
    """实体覆盖率：本窗口内被采样的实体 / 登记的实体。"""

    def __init__(self) -> None:
        self._registry: Dict[str, Set[str]] = {}
        self._sampled: Dict[str, Set[str]] = {}
        self._window_start: Optional[float] = None

    def replace_registry(self, entity_type: str, entity_ids: Set[str]) -> None:
        self._registry[entity_type] = set(str(x) for x in entity_ids)
        self._sampled.setdefault(entity_type, set())

    def begin_window(self, sim_time_s: float) -> None:
        self._window_start = float(sim_time_s)
        self._sampled = {entity_type: set() for entity_type in self._registry}

    def mark_sampled(self, entity_type: str, entity_ids: Set[str]) -> None:
        self._sampled.setdefault(entity_type, set()).update(str(x) for x in entity_ids)

    def report(self) -> Dict[str, Any]:
        entity_coverage = {}
        all_full = True
        for entity_type, registry in self._registry.items():
            total = len(registry)
            sampled = len(self._sampled.get(entity_type, set()) & registry)
            ratio = (sampled / total) if total else None
            if total and sampled < total:
                all_full = False
            entity_coverage[entity_type] = {"total": total, "sampled": sampled, "coverage_ratio": ratio}
        return {"window_start": self._window_start, "entity_coverage": entity_coverage,
                "all_full_coverage": all_full}

    def end_window(self, sim_time_s: float) -> None:
        self._sampled = {entity_type: set() for entity_type in self._registry}
        self._window_start = None


def compute_param_health(store: TelemetryStore, plan: SamplingPlan, sim_time_s: float,
                         window_s: float = HEALTH_WINDOW_S) -> Dict[str, Any]:
    """统计回看窗口内各参数的实际最大采样间隔、迟到次数与缺失值数。"""
    params: Dict[str, Dict[str, Any]] = {}
    for name in store.names():
        planned = plan.interval_of(name)
        times = [t for t in store.timestamps(name) if t >= sim_time_s - window_s]
        gaps = np.diff(times) if len(times) > 1 else np.empty(0)
        params[name] = {
            "critical": plan.is_critical(name),
            "planned_interval_s": planned,
            "max_gap_s": float(gaps.max()) if gaps.size else None,
            "late_count": int(np.count_nonzero(gaps > planned + GAP_TOLERANCE_S)),
            "missing_count": store.missing_count(name),
        }
    return {"sim_time_s": float(sim_time_s), "window_s": float(window_s), "params": params}
