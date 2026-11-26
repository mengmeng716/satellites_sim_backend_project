"""感知参数定义：表2-1 中每个参数的单位、分组、实体类型与采样等级。

采样等级按技术要求 2.1(4)：

* 关键参数（Critical）：采样间隔 ≤ 1 s
* 非关键参数：采样间隔 ≤ 10 s

关键参数取物理层/网络层中随链路几何与负载快速变化、且直接决定链路是否可用的量：
传输损耗、信噪比、误码率、丢包率、时延。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

CRITICAL_INTERVAL_S = 1.0
NON_CRITICAL_INTERVAL_S = 10.0


@dataclass(frozen=True)
class ParamSpec:
    name: str
    unit: str
    group: str          # Environment / Physical / Network / Business
    entity_type: str    # node / link
    critical: bool
    source: str         # 仿真中的取值来源

    @property
    def interval_s(self) -> float:
        return CRITICAL_INTERVAL_S if self.critical else NON_CRITICAL_INTERVAL_S


_SPECS = [
    ParamSpec("AmbientTemperature", "K", "Environment", "node", False, "NodeAttribute.ambient_temperature"),
    ParamSpec("TerminalTemperature", "K", "Environment", "node", False, "mean(NodeAttribute.terminal_temperature_list)"),
    ParamSpec("PathLoss", "dB", "Physical", "link", True, "evaluate_link_physics_batch.path_loss"),
    ParamSpec("SNR", "dB", "Physical", "link", True, "evaluate_link_physics_batch.snr"),
    ParamSpec("PropagateDelay", "ms", "Physical", "link", False, "evaluate_link_physics_batch.link_propagation_delay"),
    ParamSpec("BER", "-", "Physical", "link", True, "evaluate_link_physics_batch.ber（纠错后）"),
    ParamSpec("LaserConfigRef", "-", "Physical", "link", False, "LinksQualitiesValue.laser_type"),
    ParamSpec("PacketLossRate", "-", "Network", "link", True, "被动/主动测量（真值 evaluate_link_physics_batch.packet_loss_rate）"),
    ParamSpec("Delay", "ms", "Network", "link", True, "被动/主动测量（真值 传播+排队+发送）"),
    ParamSpec("DelayJitter", "ms", "Network", "link", False, "被动测量"),
    ParamSpec("Throughput", "Gbps", "Network", "link", False, "被动测量（真值 LinksQualitiesValue.throughput）"),
    ParamSpec("HistoryFlow", "Gbps", "Business", "node", False, "NodeAttribute.flow"),
    ParamSpec("RemainingEnergy", "-", "Business", "node", False, "NodeAttribute.energy_ratio"),
    ParamSpec("SatelliteLoad", "-", "Business", "node", False, "NodeAttribute.node_load_ratio"),
]

#: 参数名 → 定义
PARAM_SPECS: Dict[str, ParamSpec] = {spec.name: spec for spec in _SPECS}


def critical_params() -> List[str]:
    """关键参数名列表（按定义顺序）。"""
    return [spec.name for spec in _SPECS if spec.critical]


def non_critical_params() -> List[str]:
    return [spec.name for spec in _SPECS if not spec.critical]
