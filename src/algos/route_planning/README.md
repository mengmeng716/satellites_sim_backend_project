# 🌐 动态拓扑驱动的高效任务规划与路由优化 (GRLR 在线服务版)

本项目为“业务驱动的巨星座激光链路按需智能重构优化验证平台”的核心模块之一（第三部分）。

针对巨型低轨星座拓扑高速演化、链路质量剧烈波动的痛点，本模块采用 **图神经网络与强化学习路由 (GRLR - Graph Neural Network + Reinforcement Learning Routing)** 技术。经过最新架构重构，本模块现已全面转向**在线服务化（Online Serving）**，支持通过标准化服务接口对巨型星座（如 3600 颗卫星规模）进行毫秒级的智能逐跳转发推断与多路径分流。

## ✨ 核心特性与架构升级

- **微服务化设计 (Service-Oriented)**：新增 `route_planning_service.py` 与上下文管理器，彻底解耦算法底层与业务调用层，方便以 RPC 或 HTTP API 形式无缝接入外部巨星座仿真平台。
- **高性能在线推断 (Online Inference)**：核心引擎升级为 `grlr_online_router.py`，剥离了训练阶段的冗余环境组件，专注于对实时拓扑与流量状态的快速张量构建与 Actor 贪心决策，保障极低延迟。
- **多规模星座无缝切换**：预置 `3600_qx4`（3600星规模）与 `432_delta`（432星规模）等多套预训练模型权重，业务层可根据仿真场景动态加载对应权重体系。
- **统一的数据契约拦截**：通过 `interface.py` 统管所有上下游数据结构交互，内置严格的合法性校验、默认值补全与脏数据清洗机制，守护核心推断层的稳定性。

------

## 📂 目录结构

```Plaintext
.
├── route_planning_context.py     # 负责保存总项目当前 data_model 状态读取函数
├── routing_config.py             # 全局业务路径与环境常量配置
├── routing_interface.py          # 数据接口定义层 (验证与解析输入拓扑/输出决策)
├── inference.py              	  # 本地端到端推理样例与测试启动入口
├── models-weights/               # 预训练模型权重存放区
│   ├── 3600_qx4/                 # 3600颗卫星规模模型 checkpoint
│   └── 432_delta/                # 432颗卫星规模模型 checkpoint
└── src/                          # 算法底层源码
    ├── grlr_model.py             # 图神经网络与强化学习(Actor-Critic)结构实现
    ├── grlr_model_config.py      # 模型网络结构参数与超参数定义
    ├── grlr_feature_builder.py   # 在线特征转换器: 将字典数据转化为图节点/边张量
    └── grlr_online_router.py     # 在线核心路由器: 整合特征并执行毫秒级推断

```

------

## 🚀 快速开始

### 1. 环境依赖安装

请确保您的 Python 环境为 3.8+，并安装以下核心依赖：

```bash
pip install torch numpy skyfield
```

### 2. 启动路由服务

如果您需要将本模块作为独立服务接入大型仿真平台，可以直接启动服务主入口，系统将初始化上下文并挂载指定规模的 GRLR 预训练模型：

Bash

```
python route_planning_service.py
```

### 3. 本地推理与联调测试

如果您希望直接输入构建好的拓扑数据与节点数据，快速验证推断结果与性能指标，可使用推理脚本：

Bash

```
python route_planning/inference.py
```

------

## 💻 核心逻辑流转说明

在执行一次在线路由规划时，系统内部的数据流转遵循以下标准化管线：

1. **请求接入**：`route_planning_service.py` 接收到包含 `SrcSatId`、`DestSatId` 以及重构后新拓扑的大字典。
2. **契约校验**：数据送入 `interface.py` 封装为标准化 Dataclass，自动剔除无效链路或非法经纬度。
3. **特征构建**：`src/grlr_feature_builder.py` 提取当前卫星及候选邻居的属性（拥塞度、剩余电量、链路剩余容量、时延等），归一化后拼接为 PyTorch 友好的图特征张量。
4. **模型推断**：`src/grlr_online_router.py` 调用加载在内存中的 `grlr_model.py` 权重，通过 GAT 层评估网络状态，由 Actor 网络直接输出下一跳的最优决策概率。
5. **结果封装**：上下文将其打包为符合输出规范的 `RouteDecision` 返回给仿真平台的链路传输层执行。

------

> **架构亮点**：通过将底层的 `src` 与外部的 `service/interface` 严格分层，本模块既保留了深度学习模型对复杂拓扑的高维表征能力，又具备了商业级软件的健壮性与可扩展性。
