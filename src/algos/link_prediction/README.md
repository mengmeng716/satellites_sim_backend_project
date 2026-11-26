# 链路预测与激光链路质量分析

## 引擎调用

| 宏周期步骤 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 1 | `LinkPredictor.predict(current_utc, input_window, link_index, step_seconds)` | `(链路数, 30, 11)` 历史特征窗口、链路索引 | `LinksPredTopology`：6 步，每步每链路 `Capacity` / `Survival` / `HeatValue` / `LinkAvailability` |
| 2 | `adapt_scfe_output(scfe_output, link_states_from_topology(topology_state))` | 步骤 1 输出、仿真当前链路状态 | 表2-2 记录（6 步 × 链路数） |

流量归一化：输入特征 `flow / MAX_LINK_SPEED_GBPS`，输出 `pred × MAX_LINK_SPEED_GBPS`（100 Gbps）。

## 表2-2 单条链路状态预测输出

| 字段 | 类型 | 值域 | 含义 | 来源 |
|---|---|---|---|---|
| Timestamp | Long | [0,+∞) | 预测时间戳（ms） | 模型步时刻 |
| SrcSatID | String | 0~3599 | 源卫星 ID | 拓扑 |
| DstSatID | String | 0~3599 | 目的卫星 ID | 拓扑 |
| SignalIntensity | Float | [-80,-9] dBm | 接收光功率 = TxPowerNominal − PathLoss | 仿真链路物理 |
| Delay | Float | [0,0.3] s | 综合时延 = 传播 + 排队 + 发送 | 仿真链路物理（预测流量） |
| AchievableRate | Float | [0,100] Gbps | 链路可实现速率 | 仿真链路物理 |
| DelayJitter | Float | ≥0 ms | 本步与上一步综合时延之差的绝对值（第 1 步相对仿真当前时延） | 相邻步 |
| LoadFluctuation | Float | [-100,100] Gbps | 本步与上一步预测流量之差（第 1 步相对仿真当前流量） | 相邻步 |
| Capacity | Float | [0,100] Gbps | 剩余容量 | 模型 |
| Survival | Float | (0,1] | 连续生存度 | 模型 |
| HeatValue | Float | [0,1] | 流量热力值 | 模型 |
| PredFlow | Float | [0,100] Gbps | 预测业务流量 = 额定容量 − 剩余容量 | 模型 |

SignalIntensity 与 Delay 为物理计算值，不做裁剪。

## 表3-4 LinkQuality 段

| 字段 | 类型 | 值域 | 含义 |
|---|---|---|---|
| EntityId | String | src->dst | 链路标识 |
| QualityScore | Float | [0,1] | 0.5×速率分 + 0.3×生存度 + 0.2×时延分 |
| AchievableRateScore | Float | [0,1] | AchievableRate / 100 |
| DelayScore | Float | [0,1] | 1 − Delay / 0.01 s |

## 函数

| 函数 | 输入 | 输出 |
|---|---|---|
| `adapt_scfe_output(scfe_output, link_states, previous_flow_gbps=None, max_capacity_gbps=None)` | SCFE 输出；`{(src, dst): LinksQualitiesValue}`；第 1 步负载波动基准（缺省取链路当前流量）；额定容量（缺省 `MAX_LINK_SPEED_GBPS`） | 表2-2 记录列表 |
| `link_states_from_topology(topology_state)` | 仿真 `TopologyState` | `{(src, dst): LinksQualitiesValue}` |
| `predict_link_state(src_id, dst_id, distance_km, capacity_gbps=100.0, flow_gbps=0.0, prev_flow_gbps=None, survival=1.0, heat_value=0.0, timestamp_ms=0, delay_jitter_ms=0.0, laser_config=None)` | 单条链路几何与状态；`laser_config` 缺省按链路方向取仿真激光器预设 | 表2-2 的 12 个字段 |
| `analyze_link_quality(states)` | 表2-2 记录列表 | 表3-4 LinkQuality |
| `load_predictor(model_dir=None, device="auto", max_batch=65536)` | 模型目录、设备 | 多目标预测器对象 |
| `predict_with_model(predictor, history_window, entity_ids, horizon=5, ...)` | 预测器、历史窗口、实体 ID、步数 | 多步预测结果 |

`load_predictor` / `predict_with_model` 使用 `models/predictor/` 下的 `multitarget_lstm_attn.pt` 及特征契约、标准化参数、训练元数据；历史窗口长度 48 步，`history_window` 形状为 `(N, 48, feature_dim)`。
