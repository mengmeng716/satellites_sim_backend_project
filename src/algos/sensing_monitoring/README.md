# 感知监测

接口位置：`src/algos/sensing_monitoring/interface.py`

对应接口文档：**表2-1 感知参数表**、**表3-2 状态感知与监测查询接口**

## 数据来源

| 层级 | 来源 |
|---|---|
| 物理层 | 仿真链路物理 `evaluate_link_physics_batch`（链路距离 + 激光器预设 + 链路流量） |
| 网络层 | 被动统计（passive-flow-stats）与主动探测（active-probe）对仿真真值的有限样本测量 |
| 环境层 / 业务层 | `SatelliteNodeState.sat_list` |

## 采样时序

引擎每个微周期调用一次 `run_sensing_cycle`。

| 等级 | 参数 | 间隔 |
|---|---|---|
| 关键 | PathLoss、SNR、BER、PacketLossRate、Delay | 1 s |
| 非关键 | AmbientTemperature、TerminalTemperature、PropagateDelay、LaserConfigRef、DelayJitter、Throughput、HistoryFlow、RemainingEnergy、SatelliteLoad | 10 s |

关键参数在相邻两个微周期之间逐秒采样：链路距离按两帧线性插值，流量取该秒生效周期的值，每秒调用一次链路物理计算，样本按实际采样时刻入库。

## 网络层两种感知方法

| 方法 | MethodId | 适用链路 | 测量模型 |
|---|---|---|---|
| 被动统计 | `passive-flow-stats` | 承载业务的链路 | 报文数 = 流量 × 窗口 / 帧长；丢包 ~ 二项分布；时延 = 传播 + 发送 + M/M/1 排队（指数分布） |
| 主动探测 | `active-probe` | 全部链路 | 10 包/秒/链路；丢包 ~ 二项分布；单向时延 = 往返时延 / 2 |

链路网络层取值：有业务时取被动统计，空闲时取主动探测。每周期轮转选取 16 条承载业务的链路，两种方法同时测量并做 3σ 交叉核对。

## 覆盖实体

| EntityType | 实体 | 判定为已采样 |
|---|---|---|
| `node` | 全部卫星 | 非关键参数已有有效采样 |
| `link` | 拓扑中全部有向链路 | 本窗口内关键参数全部为有限值 |
| `topology` | 当前拓扑快照（1 个） | 本窗口内至少一条链路已采样 |
| `service` | 当前承载业务流量的链路 | 同 `link` |

## 函数

| 函数 | 输入 | 输出 |
|---|---|---|
| `SensingMonitor()` | — | 跨周期状态：采样计划、遥测库、调度器、覆盖统计、上一帧链路快照、逐秒链路通断帧 |
| `run_sensing_cycle(monitor, topology_state, node_state, sim_time_s, seed=None)` | 感知状态、仿真拓扑、仿真节点、仿真时刻（s）、噪声种子（缺省取仿真时刻） | 见下表 |
| `collect_parameters(values)` | `{参数名: 取值}` | 表2-1 四组结构 |
| `query_status(coverage_report, health_report, checks)` | 覆盖报告、参数健康报告、交叉核对结果 | 表3-2 三段结构 |
| `SensingMonitor.availability_samples(since_s=None)` | 起始时刻（s） | `[{"src->dst": bool}, ...]`，逐秒链路通断（Rx ≥ 灵敏度为通），保留最近 300 s |

## `run_sensing_cycle` 输出

| 字段 | 类型 | 值域 | 含义 |
|---|---|---|---|
| Timestamp | Float | ≥0 s | 仿真时刻 |
| Parameters.{Environment,Physical,Network,Business}.{参数}.Unit | String | — | 单位 |
| Parameters.….EntityType | String | node / link | 实体类型 |
| Parameters.….Count | Int | ≥0 | 实体数 |
| Parameters.….Mean / Min / Max | Float | 随参数 | 最近一次采样的全网统计（数值型参数） |
| Parameters.Physical.LaserConfigRef.Distribution | Map<String,Int> | — | 各激光器类型的链路数 |
| Coverage.EntityCoverage[].EntityType | String | node / link / topology / service | 实体类型 |
| Coverage.EntityCoverage[].Total | Int | ≥0 | 登记实体数 |
| Coverage.EntityCoverage[].Sampled | Int | ≥0 | 本窗口已采样实体数 |
| Coverage.EntityCoverage[].CoverageRatio | Float | [0,1] | 覆盖率 |
| Coverage.AllFullCoverage | Bool | — | 全部实体类型覆盖率为 1 |
| Sampling[].ParamName | String | 表2-1 参数名 | 参数 |
| Sampling[].Critical | Bool | — | 是否关键参数 |
| Sampling[].PlannedInterval | Float | 1 / 10 s | 计划间隔 |
| Sampling[].ActualMaxGap | Float | ≥0 s | 最近 60 s 内实际最大间隔 |
| Sampling[].LateCount | Int | ≥0 | 最近 60 s 内超过计划间隔的次数 |
| Sampling[].MissingCount | Int | ≥0 | 累计缺失值个数 |
| SensingMethod[].MethodId | String | passive-flow-stats / active-probe | 感知方法 |
| SensingMethod[].CrossCheckConsistent | Bool | — | 本周期交叉核对全部一致 |
| SamplingImpact.Param | String | SNR | 分析参数 |
| SamplingImpact.ReferenceSamples | Int | ≥0 | 1 s 参照序列长度（链路集合不变的最近一段，≤120） |
| SamplingImpact.Results[].Interval | Float | 1 / 2 / 5 / 10 s | 候选采样间隔 |
| SamplingImpact.Results[].RMSE / MaxAbsError | Float | ≥0 dB | 零阶保持重建误差（全部链路） |
| SamplingImpact.Results[].TelemetryKbps | Float | ≥0 kbit/s | 遥测开销 = 链路数 × 关键参数数 × 8 B × 8 / 间隔 |

## 参数单位

| 参数 | 单位 | 参数 | 单位 |
|---|---|---|---|
| AmbientTemperature / TerminalTemperature | K | PacketLossRate | 无量纲 |
| PathLoss | dB（Tx − Rx） | Delay / DelayJitter | ms |
| SNR | dB | Throughput / HistoryFlow | Gbps |
| PropagateDelay | ms | RemainingEnergy / SatelliteLoad | [0,1] |
| BER | 纠错后，无量纲 | LaserConfigRef | STD / ROBUST / FAST |
