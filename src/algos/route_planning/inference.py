"""
route_planning 在线推理适配层。

本文件负责：
1. 从总项目 data_model 维护的 topology_state / node_state 中提取 newTopology 和 SatList；
2. 将任务字段整理为 GRLR 可用的单任务 dict；
3. 调用 src/grlr_online_router.py 中的 DistributedRouter；
4. 返回内部推理结果，最终由 route_planning_interface.py 展开为接口文档要求的输出字段。

本文件不训练模型，不构建拓扑，不直接实现 GAT 网络结构。
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .routing_config import (
    DEFAULT_CONSTELLATION_ID,
    DEFAULT_DURATION,
    DEFAULT_PACKET_SIZE,
    DEVICE,
    MAX_ROUTE_HOPS,
    resolve_model_key,
    resolve_model_profile,
    validate_route_planning_config,
)
from .delay_metrics import calculate_link_delay_metrics


_ROUTER_CACHE: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# 通用对象读取工具
# -----------------------------------------------------------------------------


def _as_mapping(obj: Any) -> Dict[str, Any]:
    """
    将各种类型的对象（Pydantic 模型、dataclass、普通对象等）统一转换为字典，提供灵活的数据读取能力。
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        data: Dict[str, Any] = {}
        try:
            data.update(obj.model_dump(by_alias=True))
        except Exception:
            pass
        try:
            data.update(obj.model_dump())
        except Exception:
            pass
        return data
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    data = _as_mapping(obj)
    for key in keys:
        if key in data:
            return data[key]
    for key in keys:
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def _to_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} 不能为空")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是整数，当前值为: {value!r}") from exc


def _to_float(value: Any, default: float = 0.0, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if min_value is not None:
        result = max(result, float(min_value))
    if max_value is not None:
        result = min(result, float(max_value))
    return result


def _to_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "available", "valid"}:
            return True
        if text in {"false", "0", "no", "n", "unavailable", "invalid", "none", "null", ""}:
            return False
        return default
    return bool(value)


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return _dt.datetime.now(_dt.timezone.utc).isoformat()
    if isinstance(value, _dt.datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        ts = float(value) / 1000.0 if float(value) > 1e12 else float(value)
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
    text = str(value).strip()
    try:
        dt = _dt.datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc).isoformat()
    except Exception:
        return text


# -----------------------------------------------------------------------------
# GRLR 懒加载
# -----------------------------------------------------------------------------


def get_router(constellation_id: Any = None):
    """
    获取指定 ConstellationId 对应的 GRLR 路由器。

    ConstellationId 为 432 时加载 models-weights/432_delta/checkpoint_final.pt；
    ConstellationId 为 3600 时加载 models-weights/3600_qx4/checkpoint_final.pt。
    每个模型只在第一次被使用时加载一次，后续任务复用缓存。
    """
    model_key = resolve_model_key(constellation_id)
    if model_key in _ROUTER_CACHE:
        return _ROUTER_CACHE[model_key]

    validate_route_planning_config(model_key)
    profile = resolve_model_profile(model_key)

    try:
        from .src.grlr_model_config import ConstellationConfig
        from .src.grlr_online_router import DistributedRouter
    except ModuleNotFoundError as exc:
        missing = exc.name or "未知依赖"
        raise ModuleNotFoundError(
            f"加载 GRLR 推理模块失败，缺少依赖: {missing}。请检查 route_planning/src/ 目录和 requirements.txt。"
        ) from exc

    const_config = ConstellationConfig(
        n_orbits=profile.n_orbits,
        n_sats_per_orbit=profile.n_sats_per_orbit,
    )

    _ROUTER_CACHE[model_key] = DistributedRouter(
        model_path=str(profile.model_path),
        device=DEVICE,
        const_config=const_config,
        use_checkpoint_const_config=False,
    )
    return _ROUTER_CACHE[model_key]


# -----------------------------------------------------------------------------
# data_model -> GRLR 纯 dict 输入
# -----------------------------------------------------------------------------


def _extract_node_map(node_state: Any) -> Dict[Any, Any]:
    sat_list = _get(node_state, "SatList", "sat_list", "satList", default=None)
    if sat_list is None and isinstance(node_state, Mapping):
        sat_list = node_state
    return dict(sat_list or {})


def _convert_node_state(node_state: Any) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {}
    for sat_ref, attr in _extract_node_map(node_state).items():
        sat_id = _to_int(str(sat_ref).replace("Sat", ""), "sat_id")
        result[sat_id] = {
            "Latitude": _to_float(_get(attr, "Latitude", "latitude"), default=0.0, min_value=-90.0, max_value=90.0),
            "Longitude": _to_float(_get(attr, "Longitude", "longitude"), default=0.0, min_value=-180.0, max_value=180.0),
            "Flow": _to_float(_get(attr, "Flow", "flow", "CurrentLoad"), default=0.0, min_value=0.0),
            "EnergyRatio": _to_float(_get(attr, "EnergyRatio", "energy_ratio", "RemainingEnergy"), default=1.0, min_value=0.0, max_value=1.0),
            "Congestion": _to_float(_get(attr, "Congestion", "congestion"), default=0.0, min_value=0.0, max_value=1.0),
            "HeatFlow": _to_float(_get(attr, "HeatFlow", "heat_flow", "HeatValue"), default=0.0, min_value=0.0, max_value=1.0),
        }
    return result


def _extract_topology_map(topology_state: Any) -> Dict[Any, Any]:
    new_topology = _get(topology_state, "newTopology", "new_topology", "topology", "Topology", default=None)
    if new_topology is None and isinstance(topology_state, Mapping):
        new_topology = topology_state
    return dict(new_topology or {})


def _convert_link_attr(attr: Any) -> Dict[str, Any]:
    max_capacity = _to_float(_get(attr, "MaxCapacity", "LinkCapacity", "Capacity"), default=10.0, min_value=1e-6)
    left_capacity = _to_float(_get(attr, "LeftCapacity", "LinkAvailableGbps"), default=max_capacity, min_value=0.0)
    cum_flow = _to_float(_get(attr, "CumFlow", "CurrentFlow", "HistoryFlow"), default=0.0, min_value=0.0)
    heat_value = _to_float(_get(attr, "HeatValue", "TrafficHeat"), default=cum_flow, min_value=0.0)
    return {
        "LinkDistance": _to_float(_get(attr, "LinkDistance", "SlantRange"), default=0.0, min_value=0.0),
        "MaxCapacity": max_capacity,
        "LeftCapacity": left_capacity,
        "CumFlow": cum_flow,
        "CurrentFlow": cum_flow,
        "LinkPropagationDelay": _to_float(_get(attr, "LinkPropagationDelay", "PropagateDelay"), default=0.0, min_value=0.0),
        "QueueDelay": _to_float(_get(attr, "QueueDelay"), default=0.0, min_value=0.0),
        "TransmissionDelay": _to_float(_get(attr, "TransmissionDelay"), default=0.0, min_value=0.0),
        "PacketLossRate": _to_float(_get(attr, "PacketLossRate"), default=0.0, min_value=0.0, max_value=1.0),
        "HeatValue": heat_value,
        "TrafficHeat": heat_value,
        "is_valid": _to_bool(_get(attr, "is_valid", "Availability", default=True)),
    }


def _iter_link_entries(links: Any) -> Iterable[Tuple[Any, Any]]:
    if isinstance(links, Mapping):
        yield from links.items()
        return
    if isinstance(links, list):
        for item in links:
            if isinstance(item, Mapping):
                target_ref = _get(item, "TargetSatID", "TargetSatId", "targetSatId", "target_sat_id", default=None)
                qualities = _get(item, "LinksQualitiesValue", "LinkQualitiesValue", "qualities", default=None) or item
                if target_ref is not None:
                    yield target_ref, qualities
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                yield item[0], item[1]


def _convert_topology_state(topology_state: Any) -> Dict[int, list[list[Any]]]:
    result: Dict[int, list[list[Any]]] = {}
    for source_ref, links in _extract_topology_map(topology_state).items():
        source_id = _to_int(str(source_ref).replace("Sat", ""), "source_sat_id")
        result.setdefault(source_id, [])
        for target_ref, attr in _iter_link_entries(links):
            target_id = _to_int(str(target_ref).replace("Sat", ""), "target_sat_id")
            result[source_id].append([target_id, _convert_link_attr(attr)])
    return result


def _extract_timestamp(topology_state: Any, node_state: Any, task: Any) -> str:
    timestamp = (
        _get(task, "Timestamp", default=None)
        or _get(topology_state, "Timestamp", "timestamp", default=None)
        or _get(node_state, "Timestamp", "timestamp", default=None)
        or _get(task, "StartTime", "ArrivalTime", default=None)
    )
    return _normalize_timestamp(timestamp)


def _extract_constellation_id(topology_state: Any, node_state: Any, task: Any) -> str:
    return str(
        _get(task, "ConstellationId", default=None)
        or _get(topology_state, "ConstellationId", "constellation_id", default=None)
        or _get(node_state, "ConstellationId", "constellation_id", default=None)
        or DEFAULT_CONSTELLATION_ID
    )


def _build_task_dict(task: Any, timestamp_iso: str, constellation_id: str) -> Dict[str, Any]:
    src_sat_id = _get(task, "SrcSatId", "SrcSatID", "SourceSatId", "source_sat_id", default=None)
    dest_sat_id = _get(task, "DestSatId", "DestSatID", "DstSatId", "DestinationSatId", "dest_sat_id", default=None)
    if src_sat_id is None or dest_sat_id is None:
        raise ValueError("路由任务缺少 SrcSatId 或 DestSatId。")

    return {
        "Timestamp": timestamp_iso,
        "ConstellationId": constellation_id,
        "TaskId": str(_get(task, "TaskId", "task_id", "id", default="UNKNOWN_TASK")),
        "SrcSatId": _to_int(src_sat_id, "SrcSatId"),
        "DestSatId": _to_int(dest_sat_id, "DestSatId"),
        "PacketSize": _to_float(
            _get(task, "PacketSize", "packet_size", "DataSizeMB", "FlowSize", default=DEFAULT_PACKET_SIZE),
            default=DEFAULT_PACKET_SIZE,
            min_value=1e-9,
        ),
        "StartTime": _normalize_timestamp(_get(task, "StartTime", "ArrivalTime", default=timestamp_iso)),
        "TaskPriority": _to_int(_get(task, "TaskPriority", "Priority", default=0), "TaskPriority"),
        "Duration": _to_float(
            _get(task, "Duration", "duration", default=DEFAULT_DURATION),
            default=DEFAULT_DURATION,
            min_value=0.0,
        ),
    }


def build_grlr_input(topology_state: Any, node_state: Any, task: Any) -> Dict[str, Any]:
    """构造便于测试的 GRLR 输入 dict。"""
    timestamp_iso = _extract_timestamp(topology_state, node_state, task)
    constellation_id = _extract_constellation_id(topology_state, node_state, task)
    return {
        "timestamp": timestamp_iso,
        "constellation_id": constellation_id,
        "topology": _convert_topology_state(topology_state),
        "node_state": _convert_node_state(node_state),
        "task": _build_task_dict(task, timestamp_iso, constellation_id),
    }


def route_planning_inference(topology_state: Any, node_state: Any, task: Any) -> Dict[str, Any]:
    """总项目路由优化推理入口。"""

    grlr_input = build_grlr_input(topology_state, node_state, task)
    model_key = resolve_model_key(grlr_input["constellation_id"])
    model_profile = resolve_model_profile(model_key)
    router = get_router(model_key)

    # 记录推理开始时间
    start_time = time.time()
    result = router.route(
        topology=grlr_input["topology"],
        node_state=grlr_input["node_state"],
        task=grlr_input["task"],
        max_hops=MAX_ROUTE_HOPS,
    )

    # 获取模型生成的路径和拓扑字典
    route_path = result.get("RoutePath") or result.get("route_path") or []
    topology = grlr_input["topology"]

    # 获取任务大小(Mbits) 和 需求带宽(Gbps)
    packet_size_mbits = float(grlr_input["task"].get("PacketSize", DEFAULT_PACKET_SIZE))
    duration = float(grlr_input["task"].get("Duration", DEFAULT_DURATION))

    link_propagation_delays = []
    link_queue_delays = []
    link_transmission_delays = []
    link_packet_losses = []

    if len(route_path) > 1:
        for i in range(len(route_path) - 1):
            u, v = int(route_path[i]), int(route_path[i + 1])
            link_attr = None

            # 查找 u -> v 的链路属性
            for neighbor_id, attr in topology.get(u, []):
                if neighbor_id == v:
                    link_attr = attr
                    break

            if link_attr:
                metrics = calculate_link_delay_metrics(link_attr, packet_size_mbits, duration)
                link_propagation_delays.append(float(link_attr.get("LinkPropagationDelay", 0.0)))
                link_transmission_delays.append(metrics["TransmissionDelay"])
                link_queue_delays.append(metrics["QueueDelay"])
                link_packet_losses.append(metrics["PacketLossRate"])
                # 预期流量 = 链路当前背景流量 + 本次任务预期分配的流量

                # 1. 传输时延 (ms): 数据量(Mbits) / 链路容量(Gbps) 刚好等于 ms

                # 2. 排队时延 (ms): 基于 M/M/1 队列模型近似

                # 3. 单跳丢包率: 负载超过 80% 时开始丢包
            else:
                # 兜底：未找到链路属性时填0
                link_propagation_delays.append(0.0)
                link_transmission_delays.append(0.0)
                link_queue_delays.append(0.0)
                link_packet_losses.append(0.0)

    # 添加推理时间信息
    inference_time = time.time() - start_time
    result["InferenceTimeSeconds"] = inference_time
    # 将计算结果写入 result 字典（现为逐跳链路的列表）
    result["PathQueueDelay"] = link_queue_delays
    result["PathTransmissionDelay"] = link_transmission_delays
    result["PathPacketLossRate"] = link_packet_losses
    result["PathPropagationDelay"] = link_propagation_delays

    if len(route_path) > 1:
        end_to_end_delay = (
            sum(link_propagation_delays)
            + sum(link_queue_delays)
            + sum(link_transmission_delays)
        )
        result["EndToEndDelay"] = float(end_to_end_delay)
        result["end_to_end_delay_ms"] = float(end_to_end_delay)
        result["EndTime"] = _estimate_end_time(
            result.get("StartTime", grlr_input["task"].get("StartTime")),
            end_to_end_delay,
        )

    result["ModelKey"] = model_key
    result["ConstellationSize"] = model_profile.constellation_size
    result["ModelWeightPath"] = str(model_profile.model_path)
    result["flow_size"] = float(grlr_input["task"].get("PacketSize", DEFAULT_PACKET_SIZE))
    result["duration"] = duration
    result["timestamp"] = grlr_input["timestamp"]
    result["src_sat_id"] = grlr_input["task"]["SrcSatId"]
    result["dest_sat_id"] = grlr_input["task"]["DestSatId"]
    return result


def _estimate_end_time(start_time: Any, delay_ms: float) -> Any:
    try:
        dt = _dt.datetime.fromisoformat(str(start_time))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return (dt + _dt.timedelta(milliseconds=float(delay_ms))).astimezone(_dt.timezone.utc).isoformat()
    except Exception:
        return start_time


def reset_router_cache() -> None:
    """清空已加载的模型缓存。测试或切换部署配置时可调用。"""
    _ROUTER_CACHE.clear()
