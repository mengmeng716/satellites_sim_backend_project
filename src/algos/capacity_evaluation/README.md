# 容量分析与效能评估

接口位置：`src/algos/capacity_evaluation/capacity_interface.py`

对应接口文档：**表2-19 输入 / 表2-20 输出**

## 引擎调用

每个执行拓扑重构的宏周期调用一次 `evaluate`，结果保存在 `SimulationEngine.latest_capacity_evaluation`，并以 `capacity_evaluation` 消息推送。

| 输入 | 引擎取值 |
|---|---|
| `topology` | 重构后的仿真拓扑（未重构时为当前拓扑） |
| `baseline_topology` | 重构前的仿真拓扑；未重构时不传（CapacityBefore = CapacityAfter） |
| `traffic_matrix` | 拓扑重构所用的核心节点源-目的对，每对 1 Gbps |
| `capacity_protocol.LinkCapacity` | `MAX_LINK_SPEED_GBPS`（链路属性缺失时的缺省值；λ* 按每条链路的 `LinkCapacity`，即仿真可实现速率搜索） |
| `availability_samples` | 感知监测最近一个宏周期的逐秒链路通断帧 |
| `planned_links` | 当前拓扑全部有向链路 `src->dst` |
| `frame_seconds` | 1.0 |
| 链路时延 / 丢包 / 误码 | 仿真链路属性 `LinkPropagationDelay` / `PacketLossRate` / `BER`（纠错后） |

## 输入（表2-19）

| 参数 | 类型 | 含义 |
|---|---|---|
| `topology` | `{src: {dst: attr}}` | 拓扑与链路属性（距离/时延/丢包/误码/容量） |
| `traffic_matrix` | `[{src, dst, demand_gbps}]` | 业务矩阵 |
| `qos.MaxDelay` | Float, >0 ms | 最大可容忍端到端时延 |
| `qos.MaxLoss` | Float, [0,1] | 最大可容忍丢包率 |
| `qos.MaxBER` | Float, [0,1] | 最大可容忍误码率 |
| `capacity_protocol.LinkCapacity` | Float, >0 Gbps | 单链路额定容量 |
| `capacity_protocol.SearchPrecision` | Float, >0 | 承载倍数搜索精度 |
| `availability_samples` | 帧列表或按链路的通断序列 | 可用率统计样本 |
| `planned_links` | `["src->dst", ...]` | 计划服务链路（可用率分母） |
| `baseline_topology` | 同 `topology` | 可选，用于计算容量提升 |
| `evaluation_window` | `{StartTime, EndTime}` | 评估窗口（仅记录） |

## 输出（表2-20）

| 分组 | 字段 | 类型 | 值域 |
|---|---|---|---|
| Capacity | CapacityBefore | Float | ≥0 Gbps |
| | CapacityAfter | Float | ≥0 Gbps |
| | CapacityGainPct | Float | — |
| | BottleneckLinks | List<String> | — |
| Availability | Availability | Float | [0,1] |
| | MinLinkAvailability | Float | [0,1] |
| | DowntimeSeconds | Float | ≥0 s |
| | OutageEvents | Int | ≥0 |
| Performance | MeanEndToEndDelay | Float | ≥0 ms |
| | P95EndToEndDelay | Float | ≥0 ms |
| | MeanPacketLossRate | Float | [0,1] |
| | MeanBERPostFEC | Float | [0,1] |
| | AggregateGoodput | Float | ≥0 Gbps |
| Verdict | Results | List | `{Metric, Target, Actual, Result}` |

## 算法

1. **λ\*（业务倍数）**：固定最短路的残余容量贪心 + 二分搜索；
   可承载业务量 = Σ(需求 × λ\*)。
2. **瓶颈链路**：λ\* 处残余容量耗尽的链路。
3. **性能**：端到端时延 = 沿途传播时延之和；
   丢包率 = 1 − Π(1 − 单链路丢包率)；有效吞吐 = Σ 需求 × (1 − 丢包率)。
4. **可用率**：单链路可用率 = 通帧/总帧；全网为计划链路的算术平均；
   不可用时长 = 总时长 × (1 − 可用率)；中断次数 = 通→断跳变数之和。
5. **判定**：时延/丢包/误码对 QoS 阈值，可用率对 99.9%，容量提升对 20%，不可达业务数对 0。

## 函数

| 函数 | 说明 |
|---|---|
| `evaluate(...)` | 主入口，返回表2-20 结构 |
| `lambda_star(topology, demands, capacity_gbps, precision)` | 只算业务倍数 |
| `evaluate_paths(topology, demands)` | 只算路径级性能 |
| `measure_availability(samples, planned_links, frame_seconds)` | 只算可用率 |

## 单位口径

* `Capacity*`：Gbps（不归一化）
* `MeanEndToEndDelay` / `P95EndToEndDelay`：ms
* `AggregateGoodput`：Gbps
* `DowntimeSeconds`：s
