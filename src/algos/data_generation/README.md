# 多层关联数据模拟参数生成

## 表3-3 生成式多层数据模拟输出接口

| 字段 | 类型 | 值域 | 含义 |
|---|---|---|---|
| DataSetId | String | — | 数据集标识 |
| Provenance.SourceKind | String | synthetic | 数据来源类型 |
| Provenance.ModelVersion | String | — | 生成模型版本 |
| Provenance.PhysicsModuleHash | String | — | 物理模块指纹 |
| EnvironmentSeries.AmbientTemperature | List<Float> | [233,313] K | 环境温度序列 |
| EnvironmentSeries.TerminalTemperature | List<Float> | [233,353] K | 终端温度序列 |
| PhysicalSeries.Wavelength | List<Float> | [1520,1580] nm | 激光波长序列 |
| PhysicalSeries.ModulationBandwidth | List<Float> | >0 Hz | 调制带宽序列 |
| PhysicalSeries.TxPower | List<Float> | [0,40] dBm | 发送光功率序列 |
| PhysicalSeries.PathLoss | List<Float> | [0,300] dB | 链路传输损耗序列 |
| PhysicalSeries.RxPower | List<Float> | [-80,-9] dBm | 接收光功率序列 |
| PhysicalSeries.SNR | List<Float> | [-20,40] dB | 接收信噪比序列 |
| PhysicalSeries.PropagateDelay | List<Float> | [0,0.3] s | 链路传播时延序列 |
| PhysicalSeries.BER | List<Float> | (0,0.5) | 误码率序列（纠错前） |
| NetworkSeries.PacketLossRate | List<Float> | [0,1] | 丢包率序列 |
| NetworkSeries.Delay | List<Float> | ≥0 ms | 总时延序列 |
| NetworkSeries.DelayJitter | List<Float> | ≥0 ms | 时延抖动序列 |
| NetworkSeries.Throughput | List<Float> | ≥0 Gbps | 吞吐量序列 |
| Timestamp | List<Long> | ≥0 ms | 采样时间戳 |

## 表3-4 跨层数据关联与质量综合评估输出接口

| 字段 | 类型 | 值域 | 含义 |
|---|---|---|---|
| ModelType | Enum | layered_regression | 关联模型类型 |
| ParamCoverage.ParamCount | Int | ≥6 | 参与建模的数据类型数 |
| ParamCoverage.LayerCount | Int | ≥3 | 覆盖层级数 |
| Correlation.ParamPairs | List<Struct> | — | 参数对及相关系数、最佳滞后 |
| Ablation.LayerR2Drop | List<Float> | — | 逐层剔除后 R² 下降量 |
| LinkQuality.EntityId | String | src->dst | 链路标识 |
| LinkQuality.QualityScore | Float | [0,1] | 链路质量综合评分 |
| LinkQuality.AchievableRateScore | Float | [0,1] | 速率分项得分 |
| LinkQuality.DelayScore | Float | [0,1] | 时延分项得分 |
| FitMetrics.R2Test | Float | — | 测试集 R² |
| FitMetrics.RMSETest | Float | — | 测试集 RMSE |

## 函数

| 函数 | 输入 | 输出 |
|---|---|---|
| `simulate_layered_series(n_entities=32, steps=64, seed=0)` | 实体数、步数、种子 | 表3-3 数据集 |
| `fit_association_model(payload)` | 表3-3 数据集 | 表3-4 结果 |
| `generate_with_model(spec, model_dir=None, device=None)` | 生成场景、模型目录、设备 | 模型生成的数据集（params / conditions / provenance） |

`generate_with_model` 使用训练好的 CVAE，权重随包提供于 `models/generator/`（`cvae_multilayer.pt` 及特征契约、标准化参数、训练元数据）。
