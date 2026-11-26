from dataclasses import dataclass, field
from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict, field_validator

# --- 卫星节点状态定义 ---
class NodeAttribute(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    latitude: float = Field(0.0, alias="Latitude")          
    longitude: float = Field(0.0, alias="Longitude")        
    flow: float = Field(0.0, alias="Flow")                  
    energy_ratio: float = Field(1.0, alias="EnergyRatio")   
    congestion: float = Field(0.0, alias="Congestion")      
    heat_flow: float = Field(0.0, alias="HeatFlow")
    ambient_temperature: float = Field(293.15, alias="AmbientTemperature")
    terminal_temperature_list: List[float] = Field(default_factory=list, alias="TerminalTemperatureList")
    node_load_ratio: float = Field(0.0, alias="NodeLoadRatio")
    load_fluctuation: float = Field(0.0, alias="LoadFluctuation")
    ground_station_number: List[int] = Field(default_factory=list, alias="GroundStationNumber")

    @field_validator("ground_station_number", mode="before")
    @classmethod
    def _normalize_ground_station_number(cls, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @field_validator("terminal_temperature_list", mode="before")
    @classmethod
    def _normalize_terminal_temperature_list(cls, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [float(item) for item in value]
        return [float(value)]

@dataclass
class SatelliteNodeState:
    constellation_id: str
    timestamp: int = 0             # Unix 时间戳 (ms)
    sat_list: Dict[str, NodeAttribute] = field(default_factory=dict)

# --- 拓扑链路状态定义 ---
class LaserConfig(BaseModel):
    """链路激光器配置：仿真启动时加载，运行期保持不变。"""
    model_config = ConfigDict(populate_by_name=True)

    wavelength: float = Field(1550.0, alias="Wavelength")
    modulation_bandwidth: float = Field(1.25e11, alias="ModulationBandwidth")
    tx_power_nominal: float = Field(30.0, alias="TxPowerNominal")
    rx_sensitivity: float = Field(-33.0, alias="RxSensitivity")
    rx_power_operating_range: List[float] = Field(
        default_factory=lambda: [-33.0, -10.0],
        alias="RxPowerOperatingRange",
    )
    optics_efficiency: float = Field(0.8, alias="OpticsEfficiency")
    pointing_loss_max: float = Field(3.0, alias="PointingLossMax")

    @field_validator("rx_power_operating_range", mode="before")
    @classmethod
    def _normalize_rx_power_operating_range(cls, value):
        if value is None:
            return [-33.0, -10.0]
        if isinstance(value, (list, tuple, set)):
            values = [float(item) for item in value]
            if len(values) >= 2:
                return values[:2]
            if len(values) == 1:
                return [values[0], values[0]]
            return [-33.0, -10.0]
        scalar = float(value)
        return [scalar, scalar]

class LinksQualitiesValue(BaseModel):
    """
    使用 Pydantic BaseModel 进行别名映射。
    底层拓扑字典使用 alias 的驼峰命名 (如 LinkCapacity),
    上层代码可以直接使用下划线属性 (如 link_capacity)。
    """
    # 允许通过代码侧的下划线名称，或是字典侧的驼峰别名进行读写赋值
    model_config = ConfigDict(populate_by_name=True)

    link_distance: float = Field(0.0, alias="LinkDistance")              # 链路空间距离 (km)
    max_capacity: float = Field(100.0, alias="MaxCapacity")               # 链路额定容量 (Gbps)
    left_capacity: float = Field(1.0, alias="LeftCapacity")              # 链路剩余容量比例 [0, 1]
    cum_flow: float = Field(0.0, alias="CumFlow")                        # 链路累计流量 (GB)
    path_loss: float = Field(0.0, alias="PathLoss")                      # 路径损耗 (dB)
    snr: float = Field(0.0, alias="SNR")                                 # 信噪比 (dB)
    link_propagation_delay: float = Field(0.0, alias="LinkPropagationDelay") # 链路传播时延 (ms)
    delay: float = Field(0.0, alias="Delay")                             # 总时延 (ms)
    delay_jitter: float = Field(0.0, alias="DelayJitter")                # 时延抖动 (ms)
    ber: float = Field(0.0, alias="BER")                                 # 误码率 [0, 0.5]
    packet_loss_rate: float = Field(0.0, alias="PacketLossRate")         # 丢包率 [0, 1]
    throughput: float = Field(0.0, alias="Throughput")                   # 吞吐量 (Gbps)
    heat_value: float = Field(0.0, alias="HeatValue")                    # 链路流量热力值 [0, 1]
    achievable_rate: float = Field(0.0, alias="AchievableRate")          # 可实现速率 (Gbps)
    load_fluctuation: float = Field(0.0, alias="LoadFluctuation")        # 负载波动 (Gbps)
    laser_type: str = Field("STD", alias="LaserType")
    laser_config: LaserConfig = Field(default_factory=LaserConfig, alias="LaserConfig")

    # 向后兼容字段（仍保留当前引擎与前端逻辑依赖）
    link_capacity: float = Field(100.0, alias="LinkCapacity")             # 兼容旧容量字段 (Gbps)
    current_flow: float = Field(0.0, alias="CurrentFlow")                # 兼容旧流量字段 (Gbps)
    queue_delay: float = Field(0.0, alias="QueueDelay")                  # 兼容旧排队时延 (ms)
    transmission_delay: float = Field(0.0, alias="TransmissionDelay")    # 兼容旧传输时延 (ms)

@dataclass
class TopologyState:
    constellation_id: str
    timestamp: int = 0                # Unix 时间戳 (ms)
    # newTopology: 字典的 key 是卫星 ID，value 是包含 [目标卫星ID, 链路质量属性] 的字典/列表
    new_topology: Dict[str, Dict[str, LinksQualitiesValue]] = field(default_factory=dict)

# --- 仿真全局共享上下文 ---

@dataclass
class SimulationContext:
    node_state: SatelliteNodeState
    topo_state: TopologyState
    current_time_s: int = 0
