"""多层级参数字典（大规模项目新增）。

技术要求要求感知“物理层参数”和“网络层参数”，并区分关键/非关键参数的采样间隔。
本模块把**全部参数**登记为唯一权威清单：名称、单位、所属层级、是否关键、
采样间隔上限、数据来源类型。任何模块引用参数时必须查这张表，禁止在别处另起名字或猜单位。

层级（layer）
-------------
* ``environment`` 环境层：温度、振动、PAT 跟瞄、太阳辐照
* ``physical``    物理层：波长、带宽、收发光功率、损耗、SNR、BER
* ``network``     网络层：丢包率、时延、抖动、吞吐、队列占用
* ``node``        节点资源：CPU/处理负载、缓冲占用、能量
* ``business``    业务层：需求、实际送达、拥塞

关键参数（critical=True）采样间隔必须 ≤1 s；非关键 ≤10 s。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

LAYER_ENVIRONMENT = "environment"
LAYER_PHYSICAL = "physical"
LAYER_NETWORK = "network"
LAYER_NODE = "node"
LAYER_BUSINESS = "business"

#: 关键参数采样间隔上限（秒）——技术要求 P0019
CRITICAL_INTERVAL_S = 1.0
#: 非关键参数采样间隔上限（秒）
NON_CRITICAL_INTERVAL_S = 10.0


@dataclass(frozen=True)
class ParamSpec:
    name: str
    unit: str
    layer: str
    critical: bool
    entity_type: str          # node / link / business
    source_kind: str          # physics_simulated / simulated_event / derived / synthetic / estimated
    description: str

    @property
    def max_interval_s(self) -> float:
        return CRITICAL_INTERVAL_S if self.critical else NON_CRITICAL_INTERVAL_S


#: 权威参数字典。顺序即对外文档顺序。
PARAM_SPECS: Dict[str, ParamSpec] = {
    # ---------------- 环境层 ----------------
    "ambient_temperature_k": ParamSpec(
        "ambient_temperature_k", "K", LAYER_ENVIRONMENT, False, "node",
        "physics_simulated", "星体环境温度（日照/阴影交替引起的周期性变化）",
    ),
    "terminal_temperature_k": ParamSpec(
        "terminal_temperature_k", "K", LAYER_ENVIRONMENT, False, "node",
        "physics_simulated", "激光终端温度（受负载与散热影响）",
    ),
    "vibration_rms_g": ParamSpec(
        "vibration_rms_g", "g", LAYER_ENVIRONMENT, False, "node",
        "physics_simulated", "星体振动加速度均方根",
    ),
    "pointing_error_urad": ParamSpec(
        "pointing_error_urad", "urad", LAYER_ENVIRONMENT, True, "node",
        "physics_simulated", "PAT 指向误差（跟瞄残差）",
    ),
    "pat_state": ParamSpec(
        "pat_state", "enum", LAYER_ENVIRONMENT, True, "node",
        "physics_simulated", "PAT 状态：LOCKED / ACQUIRING / LOST",
    ),
    "sunlit": ParamSpec(
        "sunlit", "bool", LAYER_ENVIRONMENT, False, "node",
        "physics_simulated", "是否处于日照区",
    ),
    # ---------------- 物理层 ----------------
    "wavelength_nm": ParamSpec(
        "wavelength_nm", "nm", LAYER_PHYSICAL, False, "link",
        "physics_simulated", "激光器波长（配置量，非高频测量）",
    ),
    "modulation_bandwidth_hz": ParamSpec(
        "modulation_bandwidth_hz", "Hz", LAYER_PHYSICAL, False, "link",
        "physics_simulated", "调制带宽",
    ),
    "tx_power_dbm": ParamSpec(
        "tx_power_dbm", "dBm", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "发送光功率",
    ),
    "path_loss_db": ParamSpec(
        "path_loss_db", "dB", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "链路传输损耗（自由空间 + 指向 + 大气）",
    ),
    "rx_power_dbm": ParamSpec(
        "rx_power_dbm", "dBm", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "接收光功率",
    ),
    "snr_db": ParamSpec(
        "snr_db", "dB", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "接收信噪比",
    ),
    "ber_pre_fec": ParamSpec(
        "ber_pre_fec", "ratio", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "纠错前误码率",
    ),
    "ber_post_fec": ParamSpec(
        "ber_post_fec", "ratio", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "纠错后误码率",
    ),
    "link_distance_km": ParamSpec(
        "link_distance_km", "km", LAYER_PHYSICAL, True, "link",
        "physics_simulated", "链路空间距离",
    ),
    "link_capacity_gbps": ParamSpec(
        "link_capacity_gbps", "Gbps", LAYER_PHYSICAL, False, "link",
        "physics_simulated", "链路额定容量",
    ),
    "achievable_rate_gbps": ParamSpec(
        "achievable_rate_gbps", "Gbps", LAYER_PHYSICAL, True, "link",
        "derived", "由 SNR/BER 与调制编码推导的可实现速率",
    ),
    # ---------------- 网络层 ----------------
    "packet_loss_rate": ParamSpec(
        "packet_loss_rate", "ratio", LAYER_NETWORK, True, "link",
        "simulated_event", "丢包率（按数据包事件统计）",
    ),
    "link_delay_ms": ParamSpec(
        "link_delay_ms", "ms", LAYER_NETWORK, True, "link",
        "simulated_event", "链路时延（传播 + 排队 + 传输）",
    ),
    "delay_jitter_ms": ParamSpec(
        "delay_jitter_ms", "ms", LAYER_NETWORK, True, "link",
        "simulated_event", "时延抖动（同流相邻包对 IPDV）",
    ),
    "throughput_gbps": ParamSpec(
        "throughput_gbps", "Gbps", LAYER_NETWORK, True, "link",
        "simulated_event", "实际有效吞吐",
    ),
    "queue_occupancy_ratio": ParamSpec(
        "queue_occupancy_ratio", "ratio", LAYER_NETWORK, False, "link",
        "simulated_event", "队列占用比例",
    ),
    "link_availability": ParamSpec(
        "link_availability", "ratio", LAYER_NETWORK, True, "link",
        "derived", "链路可用性（按可用链路时间/计划链路时间统计）",
    ),
    # ---------------- 节点资源 ----------------
    "node_load_gbps": ParamSpec(
        "node_load_gbps", "Gbps", LAYER_NODE, True, "node",
        "simulated_event", "节点承载业务负载",
    ),
    "node_load_ratio": ParamSpec(
        "node_load_ratio", "ratio", LAYER_NODE, True, "node",
        "derived", "节点负载率（相对最大处理能力）",
    ),
    "buffer_occupancy_ratio": ParamSpec(
        "buffer_occupancy_ratio", "ratio", LAYER_NODE, False, "node",
        "simulated_event", "节点缓冲占用比例",
    ),
    "processing_load_ratio": ParamSpec(
        "processing_load_ratio", "ratio", LAYER_NODE, False, "node",
        "derived", "处理负载比例",
    ),
    "energy_ratio": ParamSpec(
        "energy_ratio", "ratio", LAYER_NODE, False, "node",
        "physics_simulated", "剩余电量比例",
    ),
    "speed_km_s": ParamSpec(
        "speed_km_s", "km/s", LAYER_NODE, True, "node",
        "physics_simulated", "惯性系移动速度",
    ),
    # ---------------- 业务层 ----------------
    "business_demand_gbps": ParamSpec(
        "business_demand_gbps", "Gbps", LAYER_BUSINESS, True, "business",
        "simulated_event", "业务需求量",
    ),
    "business_delivered_gbps": ParamSpec(
        "business_delivered_gbps", "Gbps", LAYER_BUSINESS, True, "business",
        "simulated_event", "业务实际送达有效吞吐",
    ),
    "business_drop_gbps": ParamSpec(
        "business_drop_gbps", "Gbps", LAYER_BUSINESS, True, "business",
        "simulated_event", "业务丢弃量",
    ),
    "business_delay_ms": ParamSpec(
        "business_delay_ms", "ms", LAYER_BUSINESS, True, "business",
        "simulated_event", "业务端到端时延",
    ),
    "business_jitter_ms": ParamSpec(
        "business_jitter_ms", "ms", LAYER_BUSINESS, True, "business",
        "simulated_event", "业务时延抖动",
    ),
    "business_congestion": ParamSpec(
        "business_congestion", "ratio", LAYER_BUSINESS, True, "business",
        "derived", "业务拥塞程度",
    ),
}


def spec(name: str) -> ParamSpec:
    try:
        return PARAM_SPECS[name]
    except KeyError as exc:  # pragma: no cover - 编程错误
        raise KeyError(f"未在参数字典中登记: {name}") from exc


def critical_params() -> List[str]:
    return [name for name, item in PARAM_SPECS.items() if item.critical]


def non_critical_params() -> List[str]:
    return [name for name, item in PARAM_SPECS.items() if not item.critical]


def params_by_layer(layer: str) -> List[str]:
    return [name for name, item in PARAM_SPECS.items() if item.layer == layer]


def sampling_plan() -> Dict[str, Dict[str, object]]:
    """返回权威采样计划：每个参数的实际采样间隔与上限。"""
    plan: Dict[str, Dict[str, object]] = {}
    for name, item in PARAM_SPECS.items():
        plan[name] = {
            "layer": item.layer,
            "unit": item.unit,
            "critical": item.critical,
            "entity_type": item.entity_type,
            "source_kind": item.source_kind,
            "max_interval_s": item.max_interval_s,
            "configured_interval_s": item.max_interval_s,
        }
    return plan


#: 跨层关联模型必须覆盖的数据类型（技术要求 P0018：不少于 6 种）。
#: 这里给出 12 种，同时覆盖环境层、物理层、网络层，避免只用同一时延的 mean/p95 凑数。
CROSS_LAYER_DATASET_PARAMS: List[str] = [
    "ambient_temperature_k",      # 环境
    "vibration_rms_g",            # 环境
    "pointing_error_urad",        # 环境
    "tx_power_dbm",               # 物理
    "path_loss_db",               # 物理
    "rx_power_dbm",               # 物理
    "snr_db",                     # 物理
    "ber_pre_fec",                # 物理
    "achievable_rate_gbps",       # 物理->派生
    "link_delay_ms",              # 网络
    "packet_loss_rate",           # 网络
    "delay_jitter_ms",            # 网络
]
