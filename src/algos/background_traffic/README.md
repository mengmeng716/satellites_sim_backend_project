# 背景流量生成

接口位置：`src/algos/background_traffic/background_traffic_interface.py`

## 职责

按「地面人口网格 × 时段因子 → 可见卫星上传流量 → 星间中继 → 链路流量」
生成每个微周期的背景业务负载，为链路预测与拓扑重构提供输入。

与「多层关联数据模拟参数生成」是两个独立算法：

| | 背景流量生成 | 参数生成 |
|---|---|---|
| 产物 | 业务流量（Gbps） | 参数序列（温度/损耗/SNR/误码等） |
| 运行方式 | 在线，每 5 s 一次 | 离线，一次性批量 |
| 输入 | 人口网格、地面站、拓扑 | 物理链路公式 / 生成模型 |

## 输入

| 参数 | 含义 |
|---|---|
| `topology_state` | 当前拓扑，含 `new_topology`（`{src: {dst: LinksQualitiesValue}}`） |
| `node_state` | 节点状态，含 `sat_list`（每颗卫星的经纬度） |
| `timestamp` | 当前 UTC 时间，用于计算当地时间与时段因子 |

数据文件：

| 文件 | 用途 |
|---|---|
| `data/Internet_area.xlsx` | 全球流量网格（行×列，单位为相对权重） |
| `data/GroundStation.xls` | 地面站经纬度，用于中继落点 |

## 输出

`background_traffic_generation(...)` 返回四元组：

```python
(new_topology, new_node, [sat_flows, link_flows], utc_str)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `sat_flows` | `[[sat_id, upload_gbps, total_gbps], ...]` | 每颗卫星的上传流量与节点总流量（Gbps） |
| `link_flows` | `[[src_id, dst_id, gbps], ...]` | 星间链路上的中继流量（Gbps） |
| `utc_str` | `"YYYY-MM-DD HH:MM:SS"` | 本周期时间戳 |

副作用：把 `total_gbps` 写回 `node_state.sat_list[sat_id].flow`，
把链路流量写回 `topology_state.new_topology[src][dst].current_flow`。

## 算法

1. **网格加载**：读取人口网格，按 12×24 行列表计算各单元格中心经纬度；
   缺文件时退化为全零网格并打警告，不中断仿真。
2. **时段因子**：按当地时间，06–10 时线性上升到 1.0，10–22 时维持 1.0，
   22–24 时线性下降到 0.05，其余时段 0.05。
3. **流量分配**：每个活跃网格按 `1/斜距` 权重分配给可视卫星（斜距 ≤ `MAX_SLANT_RANGE_KM`）。
4. **中继路由**：以卫星 + 地面站建有向图，按上传流量从大到小逐条 Dijkstra，
   边权为拥塞感知距离（负载 >50%/80%/99% 容量时放大 2/10/10000 倍）。

## 函数

| 函数 | 说明 |
|---|---|
| `background_traffic_generation(topology_state, node_state, timestamp)` | 主入口，生成并回写 |
| `background_traffic_snapshot(topology_state, node_state, timestamp)` | 只读摘要（不改状态），用于离线检查 |

## 兼容层

为不改动骨架调用方式，以下两个文件保留为转发：

* `src/simulation/background_traffic_generation.py`
* `src/simulation/background_routing.py`
