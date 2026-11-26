"""算法三：链路预测与激光链路质量分析 —— 对外接口。

严格对应接口文档：
  * 表2-2 单条链路状态预测模块输入输出（12 个输出字段）
  * 表3-4 LinkQuality 段（EntityId / QualityScore / AchievableRateScore / DelayScore）

两种用法：

1. **模型路径**：:func:`adapt_scfe_output` 把 SCFE 模型的 ``LinksPredTopology``
   适配成表2-2 的单链路记录；
2. **解析路径**：:func:`predict_link_state` 用仿真链路物理直接给出一条链路的状态。

物理量来源：仿真唯一的链路物理实现
:func:`src.simulation.topology_link_state.evaluate_link_physics_batch`，输入为仿真链路的
当前距离、激光器配置与预测流量。本模块不另设链路预算参数，也不裁剪物理值。

单位口径（与接口文档一致）：

* ``Capacity``：剩余容量，**Gbps**（沿用 SCFE 输出，不做归一化）
* ``PredFlow``：预测业务流量，**Gbps**
* ``Delay``：综合时延（传播 + 排队 + 发送），**秒**
* ``SignalIntensity``：接收光功率 dBm；``AchievableRate``：Gbps；``DelayJitter``：ms
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

#: 表2-2 规定的 12 个输出字段
OUTPUT_FIELDS = ["Timestamp", "SrcSatID", "DstSatID", "SignalIntensity", "Delay",
                 "AchievableRate", "DelayJitter", "LoadFluctuation", "Capacity",
                 "Survival", "HeatValue", "PredFlow"]


def _physics_batch(distances_km, laser_configs, flows_gbps):
    from src.simulation.topology_link_state import evaluate_link_physics_batch

    return evaluate_link_physics_batch(distances_km, laser_configs, flows_gbps)


def _default_laser_config(src_id: str, dst_id: str):
    """按链路方向取仿真的激光器预设。"""
    from src import config
    from src.simulation.data_model import LaserConfig
    from src.simulation.topology_link_state import laser_type_for_link

    return LaserConfig.model_validate(dict(config.LASER_CONFIG_PRESETS[laser_type_for_link(src_id, dst_id)]))


def _record(timestamp_ms, src_id, dst_id, physics, i, laser_config, delay_jitter_ms,
            load_fluctuation, capacity_gbps, survival, heat_value, pred_flow):
    delay_ms = float(physics["link_propagation_delay"][i] + physics["queue_delay"][i]
                     + physics["transmission_delay"][i])
    return {
        "Timestamp": int(timestamp_ms),
        "SrcSatID": str(src_id),
        "DstSatID": str(dst_id),
        "SignalIntensity": round(float(laser_config.tx_power_nominal - physics["path_loss"][i]), 4),
        "Delay": round(delay_ms / 1e3, 9),
        "AchievableRate": round(float(physics["achievable_rate"][i]), 3),
        "DelayJitter": round(float(delay_jitter_ms), 6),
        "LoadFluctuation": round(float(load_fluctuation), 4),
        "Capacity": round(float(capacity_gbps), 6),
        "Survival": round(float(np.clip(survival, 0.0, 1.0)), 6),
        "HeatValue": round(float(np.clip(heat_value, 0.0, 1.0)), 6),
        "PredFlow": round(float(pred_flow), 6),
    }


def predict_link_state(
    src_id: str,
    dst_id: str,
    distance_km: float,
    capacity_gbps: float = 100.0,
    flow_gbps: float = 0.0,
    prev_flow_gbps: Optional[float] = None,
    survival: float = 1.0,
    heat_value: float = 0.0,
    timestamp_ms: int = 0,
    delay_jitter_ms: float = 0.0,
    laser_config: Any = None,
) -> Dict[str, Any]:
    """按表2-2 输出一条链路的状态。

    SignalIntensity / Delay / AchievableRate 由仿真链路物理计算（距离、激光器配置、流量）；
    ``laser_config`` 缺省按链路方向取仿真激光器预设。
    """
    cfg = laser_config if laser_config is not None else _default_laser_config(src_id, dst_id)
    physics = _physics_batch([distance_km], [cfg], [flow_gbps])
    load_fluctuation = 0.0 if prev_flow_gbps is None else float(flow_gbps - prev_flow_gbps)
    return _record(timestamp_ms, src_id, dst_id, physics, 0, cfg, delay_jitter_ms,
                   load_fluctuation, capacity_gbps, survival, heat_value, flow_gbps)


def analyze_link_quality(states: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """按表3-4 的 LinkQuality 段输出链路质量评分。

    质量分 = 0.5×速率分 + 0.3×(1−丢包代理) + 0.2×时延分，各项均在 [0,1]。
    """
    out = []
    for st in states:
        rate_score = float(np.clip(st.get("AchievableRate", 0.0) / 100.0, 0.0, 1.0))
        delay_score = float(np.clip(1.0 - st.get("Delay", 0.0) / 0.01, 0.0, 1.0))
        surviv = float(np.clip(st.get("Survival", 0.0), 0.0, 1.0))
        quality = 0.5 * rate_score + 0.3 * surviv + 0.2 * delay_score
        out.append({
            "EntityId": f"{st.get('SrcSatID')}->{st.get('DstSatID')}",
            "QualityScore": round(float(np.clip(quality, 0.0, 1.0)), 6),
            "AchievableRateScore": round(rate_score, 6),
            "DelayScore": round(delay_score, 6),
        })
    return out


def load_predictor(model_dir=None, device="auto", max_batch=65536):
    """加载训练好的多目标预测模型（LSTM/时空网络）。

    :param model_dir: 模型目录，需包含权重与 feature_contract / scaler
    :param device: "auto" / "cpu" / "cuda"
    """
    from .predictor import MultiTargetPredictor

    return MultiTargetPredictor(model_dir=model_dir, device=device, max_batch=max_batch)


def predict_with_model(predictor, history_window, entity_ids, horizon=5,
                       entity_types=None, sim_time_s=None):
    """用已加载的模型做批量多步预测。

    :param predictor: ``load_predictor`` 的返回值
    :param history_window: ``(N, window, feat_dim)`` 观测历史
    :param entity_ids: 实体标识列表
    :param horizon: 预测步数
    :return: 模型输出的预测字典
    """
    return predictor.predict(history_window, entity_ids, horizon=horizon,
                             entity_types=entity_types, sim_time_s=sim_time_s)


# ---------------------------------------------------------------------------
# SCFE 输出 → 表2-2 单链路记录
# ---------------------------------------------------------------------------

def _iso_to_ms(value: Any) -> int:
    """ISO8601 字符串转毫秒时间戳；失败返回 0。"""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        from datetime import datetime

        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return 0


def link_states_from_topology(topology_state: Any) -> Dict[tuple, Any]:
    """把仿真 ``TopologyState`` 展开成 ``{(src, dst): LinksQualitiesValue}``。"""
    topology = getattr(topology_state, "new_topology", topology_state) or {}
    return {(str(src), str(dst)): attr for src, links in topology.items() for dst, attr in links.items()}


def adapt_scfe_output(
    scfe_output: Mapping[str, Any],
    link_states: Mapping[Any, Any],
    previous_flow_gbps: Optional[Mapping[Any, float]] = None,
    max_capacity_gbps: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """把 SCFE 的 ``LinksPredTopology`` 适配成表2-2 的单链路记录。

    :param scfe_output: ``LinkPredictor.predict(...)`` 的返回值
    :param link_states: ``{(src, dst): LinksQualitiesValue}``，仿真当前链路状态
        （见 :func:`link_states_from_topology`），提供距离、激光器配置与当前流量/时延
    :param previous_flow_gbps: ``{(src, dst): 流量 Gbps}``，第 1 步负载波动的基准；缺省取链路当前流量
    :param max_capacity_gbps: 链路额定容量，缺省取 ``config.MAX_LINK_SPEED_GBPS``
    :return: 表2-2 记录列表，每条含 12 个字段

    字段来源：

    * 模型给出：``Capacity``（剩余容量，Gbps）、``Survival``、``HeatValue``
    * 仿真物理（当前几何 + 激光器配置 + 预测流量）：``SignalIntensity``、``Delay``、``AchievableRate``
    * 由相邻步计算：``PredFlow`` = 额定容量 − 剩余容量；``LoadFluctuation`` / ``DelayJitter``
      为本步相对上一步（第 1 步相对仿真当前值）的变化
    """
    steps = (scfe_output or {}).get("LinksPredTopology") or []
    if not steps:
        return []

    if max_capacity_gbps is None:
        from src import config

        max_capacity_gbps = float(getattr(config, "MAX_LINK_SPEED_GBPS", 100.0))
    max_cap = float(max_capacity_gbps)
    prev_flow_map = previous_flow_gbps or {}

    last_flow: Dict[tuple, float] = {}
    last_delay_ms: Dict[tuple, float] = {}
    records: list[Dict[str, Any]] = []
    for step in steps:
        timestamp_ms = _iso_to_ms(step.get("Timestamp"))
        entries = []
        for src_id, neighbours in (step.get("Topology") or {}).items():
            for item in neighbours:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    dst_id, state = item[0], item[1] or {}
                elif isinstance(item, Mapping):
                    dst_id, state = item.get("DstSatID") or item.get("dst"), item
                else:
                    continue
                key = (str(src_id), str(dst_id))
                attr = link_states.get(key)
                if attr is None:
                    raise KeyError(f"仿真拓扑中缺少链路: {key}")
                entries.append((key, attr, state))
        if not entries:
            continue

        remaining = np.array([float(state.get("Capacity", max_cap)) for _, _, state in entries])
        pred_flow = np.maximum(max_cap - remaining, 0.0)
        physics = _physics_batch([a.link_distance for _, a, _ in entries],
                                 [a.laser_config for _, a, _ in entries], pred_flow)
        delay_ms = physics["link_propagation_delay"] + physics["queue_delay"] + physics["transmission_delay"]

        for i, (key, attr, state) in enumerate(entries):
            base_flow = last_flow.get(key, prev_flow_map.get(key, float(attr.current_flow)))
            base_delay = last_delay_ms.get(key, float(attr.delay))
            records.append(_record(
                timestamp_ms, key[0], key[1], physics, i, attr.laser_config,
                delay_jitter_ms=abs(float(delay_ms[i]) - base_delay),
                load_fluctuation=float(pred_flow[i]) - base_flow,
                capacity_gbps=remaining[i],
                survival=float(state.get("Survival", 1.0)),
                heat_value=float(state.get("HeatValue", 0.0)),
                pred_flow=pred_flow[i],
            ))
            last_flow[key] = float(pred_flow[i])
            last_delay_ms[key] = float(delay_ms[i])
    return records
