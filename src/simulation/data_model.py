from dataclasses import dataclass, field
from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict

# --- 卫星节点状态定义 ---
class NodeAttribute(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    latitude: float = Field(0.0, alias="Latitude")          
    longitude: float = Field(0.0, alias="Longitude")        
    flow: float = Field(0.0, alias="Flow")                  
    energy_ratio: float = Field(1.0, alias="EnergyRatio")   
    congestion: float = Field(0.0, alias="Congestion")      
    heat_flow: float = Field(0.0, alias="HeatFlow")  

@dataclass
class SatelliteNodeState:
    constellation_id: str
    timestamp: int = 0             # Unix 时间戳 (ms)
    sat_list: Dict[str, NodeAttribute] = field(default_factory=dict)

# --- 拓扑链路状态定义 ---
class LinksQualitiesValue(BaseModel):
    """
    使用 Pydantic BaseModel 进行别名映射。
    底层拓扑字典使用 alias 的驼峰命名 (如 LinkCapacity),
    上层代码可以直接使用下划线属性 (如 link_capacity)。
    """
    # 允许通过代码侧的下划线名称，或是字典侧的驼峰别名进行读写赋值
    model_config = ConfigDict(populate_by_name=True)

    link_distance: float = Field(0.0, alias="LinkDistance")              # 链路空间距离 (km)
    link_capacity: float = Field(10.0, alias="LinkCapacity")             # 链路额定容量 (Gbps) - [修正名称对齐]
    left_capacity: float = Field(10.0, alias="LeftCapacity")             # 链路剩余容量 (Gbps)
    current_flow: float = Field(0.0, alias="CurrentFlow")                # 链路当前流量负载 - [修正名称对齐]
    link_propagation_delay: float = Field(0.0, alias="LinkPropagationDelay") # 链路传播时延 (ms)
    queue_delay: float = Field(0.0, alias="QueueDelay")                  # 链路排队时延 (ms)
    transmission_delay: float = Field(0.0, alias="TransmissionDelay")    # 链路传输时延 (ms)
    packet_loss_rate: float = Field(0.0, alias="PacketLossRate")         # 链路丢包率 [0, 1]
    heat_value: float = Field(0.0, alias="HeatValue")                    # 链路流量热力值 [0, 1]

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