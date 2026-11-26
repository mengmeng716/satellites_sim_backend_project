# 任务书条款到前端证据映射矩阵

## 1. 任务书条款原文（摘要）
支持使用生成式神经网络模拟生成环境参数、物理层参数、网络层参数多层级数据，模拟真实环境中的动态变化，覆盖不同时间段和场景；支持基于生成的模拟数据构建各层数据间关联模型，具备从不同层级综合评估激光链路质量能力。

## 2. 可验收映射表

| 条款点 | 判定问题 | 前端证据位 | 后端字段/接口 | 通过标准 |
|---|---|---|---|---|
| 生成式神经网络 | 是否由生成模型输出 | 生成任务页-ProvenanceCard | provenance.generator_version, generator_weight_sha256, source_kind=synthetic | 可见模型版本和权重哈希 |
| 环境参数覆盖 | 是否包含温度/振动/PAT 等 | 多层总览-环境层图组 | ambient_temperature_k, terminal_temperature_k, vibration_rms_g, pointing_error_urad, pat_state, sunlit | 指标可选可绘可导出 |
| 物理层参数覆盖 | 是否包含波长/带宽/功率/损耗/SNR/BER | 多层总览-物理层图组 | wavelength_nm, modulation_bandwidth_hz, tx_power_dbm, path_loss_db, rx_power_dbm, snr_db, ber_pre_fec | 指标可选可绘可导出 |
| 网络层参数覆盖 | 是否包含丢包/时延/抖动/吞吐 | 多层总览-网络层图组 | packet_loss_rate, link_delay_ms, delay_jitter_ms, throughput_gbps | 指标可选可绘可导出 |
| 动态变化能力 | 是否支持时间维回放 | 多层总览-TimeCursor + Playback | timestamp, n_steps, dt_s | 可按时间滑动查看变化 |
| 场景覆盖能力 | 是否可配置时段与场景 | 生成任务页-场景配置区 | layers, hours, distance_range_km, latitude_range_deg, sunlit | 可配置并体现在结果摘要 |
| 层间关联建模 | 是否提供跨层相关与拟合结果 | 关联分析页-Heatmap/StageFit | fit_association_model 输出 | 可展示相关系数与 R2 |
| 综合质量评估 | 是否提供链路质量综合评分 | 质量评估页-Leaderboard/Radar | LinkQuality.QualityScore, AchievableRateScore, DelayScore | 可查看 TopN 与分项 |
| 可审计性 | 是否可追溯和复现实验 | 验收报告页-ProvenanceTrace | seed, feature_contract_id, physics_module_hash | 可导出报告复查 |

## 3. 验收报告最小字段清单
1. datasetId
2. sourceKind
3. generatorVersion
4. generatorWeightSha256
5. featureContractId
6. nEntities
7. nSteps
8. 参数覆盖统计（按层）
9. 关联模型结果（ParamCoverage, Correlation, Ablation, FitMetrics）
10. 链路质量结果（QualityScore 及分项）
11. 校验 checks（阈值、比较符、结果）
12. 导出时间与操作人

## 4. 前端验收流程建议
1. 新建生成任务（mode=cvae）。
2. 生成完成后查看 Provenance 证据。
3. 在多层总览核对三层参数均可视化。
4. 在关联分析页核对相关/贡献/拟合指标。
5. 在质量评估页核对综合质量分与分项分。
6. 在验收报告页导出 JSON/Markdown 归档。

## 5. 风险与规避
1. 风险: source_kind 非 synthetic 仍被当作生成式结果展示。
   规避: 页面显著展示 source_kind，并在 fallback 情况打红色提示。
2. 风险: 参数单位误读。
   规避: 每个图例强制展示 unit。
3. 风险: 关联指标无阈值口径。
   规避: 报告里固定展示阈值与比较符。
