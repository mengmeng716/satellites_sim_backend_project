"""跨模块公共契约（大规模项目）。

本模块只定义**数据结构与常量**，不依赖仿真引擎、不依赖 torch，任何模块都可以安全导入。
目的是把“观测—预测—策略—评价”四个阶段的交换格式固定下来，避免各模块各自解释字段单位。

约定
----
* 时间：``sim_time_s`` 为场景逻辑秒；``observed_at_utc`` 为采样发生的 UTC 时间（ISO8601，带 Z）。
* 单位：一律显式写在字段名或 ``unit`` 里，禁止“猜单位”。
  - 功率：dBm；损耗/增益：dB；SNR：dB；BER：无量纲比值；
  - 速率/容量/流量：Gbps；时延：ms；抖动：ms；距离：km；速度：km/s。
* 标识：``entity_type`` ∈ {node, link, topology, business}；``entity_id`` 规则见下。

实体 ID 规则
------------
* 节点：``"3600"``（全局整数 ID 的字符串形式，与业务项目一致）
* 有向链路：``"3600->3601"``
* 拓扑：``"topology"``
* 业务：``"biz-{n}"``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "2.0"

ENTITY_NODE = "node"
ENTITY_LINK = "link"
ENTITY_TOPOLOGY = "topology"
ENTITY_BUSINESS = "business"

QUALITY_VALID = "VALID"
QUALITY_MISSING = "MISSING"
QUALITY_STALE = "STALE"
QUALITY_INVALID = "INVALID"

SOURCE_MEASURED = "measured"            # 真实设备实测（本项目暂不产生）
SOURCE_SYNTHETIC = "synthetic"          # 生成式神经网络输出
SOURCE_PHYSICS = "physics_simulated"    # 物理模型正向计算
SOURCE_EVENT = "simulated_event"        # 可追踪的仿真事件/计数器
SOURCE_DERIVED = "derived"              # 明确公式派生
SOURCE_ESTIMATED = "estimated"          # 模型估计

RESULT_PASS = "PASS"
RESULT_FAIL = "FAIL"
RESULT_NOT_MEASURABLE = "NOT_MEASURABLE"

#: 技术要求编号（固定 18 项，禁止重排）
REQUIREMENT_IDS: List[str] = [
    "F1.1-1", "F1.1-2", "F1.1-3", "F1.1-4", "F1.1-5",
    "F1.2-1", "F1.2-2", "F1.2-3",
    "P2.1-1", "P2.1-2", "P2.1-3", "P2.1-4", "P2.1-5",
    "P2.2-1", "P2.2-2", "P2.2-3", "P2.2-4", "P2.2-5",
]


def utc_iso(moment: Optional[datetime] = None) -> str:
    """返回带 ``Z`` 的 UTC ISO8601 字符串。"""
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def link_entity_id(src_id: Any, dst_id: Any) -> str:
    return f"{src_id}->{dst_id}"


@dataclass
class Observation:
    """单条观测记录（对应技术要求“逐指标数据来源”要求）。"""

    entity_type: str
    entity_id: str
    metric: str
    value: Optional[float]
    unit: str
    source_kind: str
    method_id: str
    observed_at_utc: str
    sim_time_s: float
    quality: str = QUALITY_VALID
    missing_reason: str = ""
    model_version: str = ""
    window_ms: int = 0
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "source_kind": self.source_kind,
            "method_id": self.method_id,
            "observed_at_utc": self.observed_at_utc,
            "sim_time_s": self.sim_time_s,
            "quality": self.quality,
            "missing_reason": self.missing_reason,
            "model_version": self.model_version,
            "window_ms": self.window_ms,
            "sequence": self.sequence,
        }


@dataclass
class BusinessFlowRecord:
    """业务流资源账记录（背景/业务/探测分账的基础）。"""

    business_id: str
    business_type: str
    src_node: int
    dst_node: int
    demand_gbps: float
    delivered_gbps: float = 0.0
    path: List[int] = field(default_factory=list)
    qos_class: str = "BEST_EFFORT"
    start_sim_s: float = 0.0
    end_sim_s: Optional[float] = None
    drop_gbps: float = 0.0
    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    status: str = "PENDING"      # PENDING / ACTIVE / COMPLETED / FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_id": self.business_id,
            "business_type": self.business_type,
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "demand_gbps": self.demand_gbps,
            "delivered_gbps": self.delivered_gbps,
            "path": list(self.path),
            "qos_class": self.qos_class,
            "start_sim_s": self.start_sim_s,
            "end_sim_s": self.end_sim_s,
            "drop_gbps": self.drop_gbps,
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "status": self.status,
        }


@dataclass
class StrategyRecord:
    """拓扑重构策略（控制平面输出）。"""

    strategy_id: str
    trigger_event_id: str
    snapshot_version: int
    topology_version: int
    add_links: List[Dict[str, Any]] = field(default_factory=list)
    remove_links: List[Dict[str, Any]] = field(default_factory=list)
    keep_links: List[Dict[str, Any]] = field(default_factory=list)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    constraints_checked: Dict[str, bool] = field(default_factory=dict)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    feasible: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trigger_event_id": self.trigger_event_id,
            "snapshot_version": self.snapshot_version,
            "topology_version": self.topology_version,
            "add_links": self.add_links,
            "remove_links": self.remove_links,
            "keep_links": self.keep_links,
            "target_metrics": self.target_metrics,
            "constraints_checked": self.constraints_checked,
            "stage_timings_ms": self.stage_timings_ms,
            "total_ms": self.total_ms,
            "feasible": self.feasible,
            "reason": self.reason,
        }


@dataclass
class RequirementResult:
    """逐项验收结论。"""

    requirement_id: str
    result: str = RESULT_NOT_MEASURABLE
    measured_value: Optional[float] = None
    target: Optional[float] = None
    unit: str = ""
    metric: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "result": self.result,
            "measured_value": self.measured_value,
            "target": self.target,
            "unit": self.unit,
            "metric": self.metric,
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }
