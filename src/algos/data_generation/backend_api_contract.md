# 多层关联数据模拟后端接口契约草案

## 1. 设计原则
1. 不破坏现有实时仿真事件。
2. 新能力独立命名空间：/api/v1/data-generation。
3. 所有响应带 datasetId 和 provenance。
4. 支持同步小任务与异步大任务。

## 2. API 列表

### 2.1 创建生成任务
POST /api/v1/data-generation/datasets

请求示例：
```json
{
  "mode": "cvae",
  "spec": {
    "n_samples": 64,
    "n_steps": 180,
    "dt_s": 1.0,
    "layers": [0, 1],
    "hours": [0, 6, 12, 18],
    "seed": 20260101,
    "temperature": 1.0,
    "prior": "standard",
    "prior_clip": 3.0,
    "distance_range_km": [1000, 1400],
    "latitude_range_deg": [-60, 60]
  }
}
```

响应示例：
```json
{
  "jobId": "job-20261123-001",
  "datasetId": "gen-20261123-001",
  "status": "queued"
}
```

### 2.2 查询任务状态
GET /api/v1/data-generation/jobs/{jobId}

响应字段：
- status: queued | running | ready | failed
- progress: 0~100
- message
- datasetId

### 2.3 获取数据集摘要
GET /api/v1/data-generation/datasets/{datasetId}/summary

响应字段：
- datasetId
- sourceKind
- shapes: n_entities, n_steps, n_params
- spec
- provenance

### 2.4 获取多层时序
GET /api/v1/data-generation/datasets/{datasetId}/series

查询参数：
- layer: environment | physical | network | all
- entityIds: csv
- startStep, endStep
- metrics: csv

响应示例：
```json
{
  "datasetId": "gen-20261123-001",
  "timestamp": [1700000000000, 1700000001000],
  "environmentSeries": {
    "ambient_temperature_k": [[285.1, 285.3]],
    "vibration_rms_g": [[0.031, 0.034]]
  },
  "physicalSeries": {
    "tx_power_dbm": [[22.8, 22.7]],
    "snr_db": [[14.2, 14.1]]
  },
  "networkSeries": {
    "packet_loss_rate": [[0.002, 0.003]],
    "link_delay_ms": [[8.9, 9.1]]
  },
  "units": {
    "ambient_temperature_k": "K",
    "vibration_rms_g": "g",
    "tx_power_dbm": "dBm",
    "snr_db": "dB",
    "packet_loss_rate": "ratio",
    "link_delay_ms": "ms"
  },
  "sourceKind": "synthetic"
}
```

### 2.5 执行跨层关联建模
POST /api/v1/data-generation/datasets/{datasetId}/association

请求体可选：
- window
- stride
- qualityWeights

响应字段（对应 fit_association_model）：
- modelType
- paramCoverage
- correlation.paramPairs
- ablation.layerR2Drop
- linkQuality[]
- fitMetrics

### 2.6 执行物理合理性校验
POST /api/v1/data-generation/datasets/{datasetId}/validation

响应字段（对应 validate_physical_bounds）：
- status
- checks[]
- violation_rate
- correlations
- thresholds
- provenance

### 2.7 导出验收证据
GET /api/v1/data-generation/datasets/{datasetId}/export?format=json|markdown|csv

## 3. WebSocket 事件
命名建议：
- data_generation_job_status
- data_generation_dataset_ready
- data_generation_validation_ready

事件示例：
```json
{
  "event": "data_generation_job_status",
  "data": {
    "jobId": "job-20261123-001",
    "status": "running",
    "progress": 64,
    "datasetId": "gen-20261123-001"
  }
}
```

## 4. 与现有代码映射
- simulate_layered_series: src/algos/data_generation/interface.py
- generate_with_model: src/algos/data_generation/interface.py
- fit_association_model: src/algos/data_generation/interface.py
- validate_physical_bounds: src/algos/data_generation/generate.py
- 参数权威字典: src/algos/data_generation/metrics.py

## 5. 错误码建议
- DG4001: spec 参数非法
- DG4002: mode 不支持
- DG4041: dataset 不存在
- DG4091: 任务状态冲突
- DG5001: 模型文件缺失
- DG5002: 校验执行失败

## 6. 最小实现建议
1. 先实现 datasets create + jobs status + summary。
2. 再实现 series + association + validation。
3. 最后接入 export 与 WebSocket 推送。
