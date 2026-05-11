"""
GRLR 在线推理路由器。

本文件是 src/ 中唯一负责“路径推理流程”的文件。
它不读取 TLE，不生成星座拓扑，不训练模型。

职责：
1. 加载 checkpoint_final.pt 中已经训练好的 GAT + RL 模型参数；
2. 接收总项目 data_model 已经维护好的 newTopology 和 SatList；
3. 在已有拓扑上逐跳选择下一跳；
4. 输出一条从源卫星到目的卫星的 RoutePath 以及展开指标。
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

# 兼容历史 checkpoint 中保存的 grlr.config.* 配置对象。
# 文件已重命名为 grlr_model_config.py，但 torch.load 反序列化时仍可能查找 grlr.config。
from . import grlr_model_config as _grlr_model_config
import sys as _sys
import types as _types

# checkpoint 由原训练工程保存，反序列化时可能仍查找 grlr.config。
# 当前目录已改名为 route_planning/src，这里只保留兼容别名，不恢复旧目录结构。
_grlr_compat_pkg = _sys.modules.setdefault("grlr", _types.ModuleType("grlr"))
setattr(_grlr_compat_pkg, "config", _grlr_model_config)
_sys.modules.setdefault("grlr.config", _grlr_model_config)

from .grlr_model_config import ConstellationConfig, ModelConfig, RoutingConfig, resolve_checkpoint_path
from .grlr_feature_builder import FeatureBuilder
from .grlr_model import GRLRAgent


class DistributedRouter:
    """GRLR 在线推理路由器。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        const_config: Optional[ConstellationConfig] = None,
        use_checkpoint_const_config: bool = True,
    ):
        self.device = torch.device(device)
        checkpoint = self._load_checkpoint(model_path)

        if const_config is not None:
            self.const_config = const_config
        elif use_checkpoint_const_config:
            self.const_config = self._load_config(checkpoint, "const_config", ConstellationConfig)
        else:
            self.const_config = ConstellationConfig()

        self.model_config = self._load_config(checkpoint, "model_config", ModelConfig)
        self.routing_config = self._load_config(checkpoint, "routing_config", RoutingConfig)

        self.model = GRLRAgent(self.model_config).to(self.device)
        state_dict = self._extract_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError("checkpoint 中未找到模型参数 model_state_dict/state_dict。")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.feature_builder = FeatureBuilder(self.device)
        self._topology: Dict[int, List[List[Any]]] = {}
        self._nodes: Dict[int, Dict[str, float]] = {}
        self._shortest_hop_cache: Dict[int, Dict[int, int]] = {}

    # ------------------------------------------------------------------
    # checkpoint / config
    # ------------------------------------------------------------------

    def _load_checkpoint(self, model_path: str):
        resolved_path = resolve_checkpoint_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"GRLR checkpoint 不存在: {resolved_path}")
        return torch.load(str(resolved_path), map_location=self.device, weights_only=False)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if checkpoint is None:
            return None
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, Mapping):
                    return value
            if any(str(k).startswith(("feature_extractor", "actor", "critic")) for k in checkpoint.keys()):
                return checkpoint
        return None

    @staticmethod
    def _load_config(checkpoint, key: str, config_cls):
        if not isinstance(checkpoint, dict) or key not in checkpoint:
            return config_cls()
        value = checkpoint[key]
        if isinstance(value, config_cls):
            return value
        if isinstance(value, Mapping):
            fields = getattr(config_cls, "__dataclass_fields__", {})
            return config_cls(**{k: v for k, v in value.items() if k in fields})
        if hasattr(value, "__dict__"):
            fields = getattr(config_cls, "__dataclass_fields__", {})
            data = {k: getattr(value, k) for k in fields.keys() if hasattr(value, k)}
            return config_cls(**data)
        return config_cls()

    # ------------------------------------------------------------------
    # 输入解析
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sat_id(sat_ref: Any) -> Optional[int]:
        if isinstance(sat_ref, str) and sat_ref.startswith("Sat"):
            sat_ref = sat_ref[3:]
        try:
            return int(sat_ref)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sat_keys(sat_id: int):
        return sat_id, str(sat_id), f"Sat{sat_id}"

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            try:
                return dict(value.model_dump(by_alias=True))
            except Exception:
                try:
                    return dict(value.model_dump())
                except Exception:
                    return {}
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        return {}

    @staticmethod
    def _get_attr(value: Any, *keys: str, default: Any = None) -> Any:
        data = DistributedRouter._as_dict(value)
        for key in keys:
            if key in data:
                return data[key]
        for key in keys:
            if hasattr(value, key):
                return getattr(value, key)
        return default

    def _normalize_topology(self, topology: Mapping[Any, Any] | None) -> Dict[int, List[List[Any]]]:
        result: Dict[int, List[List[Any]]] = {}
        for source_ref, links in dict(topology or {}).items():
            source_id = self._normalize_sat_id(source_ref)
            if source_id is None:
                continue
            result.setdefault(source_id, [])
            for target_ref, attr in self._iter_link_entries(links):
                target_id = self._normalize_sat_id(target_ref)
                if target_id is None:
                    continue
                result[source_id].append([target_id, self._normalize_link_attr(attr)])
        return result

    def _normalize_nodes(self, nodes: Mapping[Any, Any] | None) -> Dict[int, Dict[str, float]]:
        result: Dict[int, Dict[str, float]] = {}
        for sat_ref, attr in dict(nodes or {}).items():
            sat_id = self._normalize_sat_id(sat_ref)
            if sat_id is None:
                continue
            result[sat_id] = self._normalize_node_attr(attr)
        return result

    def _iter_link_entries(self, links: Any) -> Iterable[Tuple[Any, Any]]:
        if isinstance(links, Mapping):
            yield from links.items()
            return
        if isinstance(links, list):
            for item in links:
                if isinstance(item, Mapping):
                    target_ref = self._get_attr(
                        item,
                        "TargetSatID",
                        "TargetSatId",
                        "targetSatId",
                        "target_sat_id",
                        default=None,
                    )
                    qualities = (
                        self._get_attr(item, "LinksQualitiesValue", "LinkQualitiesValue", "qualities", default=None)
                        or item
                    )
                    if target_ref is not None:
                        yield target_ref, qualities
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    yield item[0], item[1]

    def _normalize_node_attr(self, attr: Any) -> Dict[str, float]:
        return {
            "Latitude": self._to_float(self._get_attr(attr, "Latitude", "latitude"), 0.0, -90.0, 90.0),
            "Longitude": self._to_float(self._get_attr(attr, "Longitude", "longitude"), 0.0, -180.0, 180.0),
            "Flow": self._to_float(self._get_attr(attr, "Flow", "flow", "CurrentLoad"), 0.0, 0.0, None),
            "EnergyRatio": self._to_float(self._get_attr(attr, "EnergyRatio", "energy_ratio", "RemainingEnergy"), 1.0, 0.0, 1.0),
            "Congestion": self._to_float(self._get_attr(attr, "Congestion", "congestion"), 0.0, 0.0, 1.0),
            "HeatFlow": self._to_float(self._get_attr(attr, "HeatFlow", "heat_flow", "HeatValue"), 0.0, 0.0, 1.0),
        }

    def _normalize_link_attr(self, attr: Any) -> Dict[str, Any]:
        max_capacity = self._to_float(
            self._get_attr(attr, "MaxCapacity", "max_capacity", "LinkCapacity", "link_capacity", "Capacity"),
            10.0,
            1e-6,
            None,
        )
        left_capacity = self._to_float(
            self._get_attr(attr, "LeftCapacity", "left_capacity", "LinkAvailableGbps"),
            max_capacity,
            0.0,
            None,
        )
        cum_flow = self._to_float(
            self._get_attr(attr, "CumFlow", "cum_flow", "CurrentFlow", "current_flow", "HistoryFlow"),
            0.0,
            0.0,
            None,
        )
        heat = self._to_float(
            self._get_attr(attr, "HeatValue", "heat_value", "TrafficHeat", "traffic_heat"),
            cum_flow,
            0.0,
            None,
        )
        return {
            "LinkDistance": self._to_float(self._get_attr(attr, "LinkDistance", "link_distance", "SlantRange"), 0.0, 0.0, None),
            "MaxCapacity": max_capacity,
            "LeftCapacity": left_capacity,
            "CumFlow": cum_flow,
            "CurrentFlow": cum_flow,
            "LinkPropagationDelay": self._to_float(self._get_attr(attr, "LinkPropagationDelay", "link_propagation_delay", "PropagateDelay"), 0.0, 0.0, None),
            "QueueDelay": self._to_float(self._get_attr(attr, "QueueDelay", "queue_delay"), 0.0, 0.0, None),
            "TransmissionDelay": self._to_float(self._get_attr(attr, "TransmissionDelay", "transmission_delay"), 0.0, 0.0, None),
            "PacketLossRate": self._to_float(self._get_attr(attr, "PacketLossRate", "packet_loss_rate"), 0.0, 0.0, 1.0),
            "HeatValue": heat,
            "TrafficHeat": heat,
            "is_valid": self._to_bool(self._get_attr(attr, "is_valid", "Availability", default=True)),
        }

    @staticmethod
    def _to_float(value: Any, default: float, min_value: Optional[float], max_value: Optional[float]) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = float(default)
        if min_value is not None:
            result = max(result, min_value)
        if max_value is not None:
            result = min(result, max_value)
        return result

    @staticmethod
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

    # ------------------------------------------------------------------
    # 已有拓扑查询。原 topology_view.py 的职责已合并到本文件。
    # ------------------------------------------------------------------

    def update_external_state(self, topology: Mapping[Any, Any], node_state: Mapping[Any, Any] | None = None) -> None:
        self._topology = self._normalize_topology(topology)
        self._nodes = self._normalize_nodes(node_state)
        self._shortest_hop_cache.clear()
        for source_id, links in self._topology.items():
            self._nodes.setdefault(source_id, self._default_node())
            for target_id, _ in links:
                self._nodes.setdefault(int(target_id), self._default_node())

    def _default_node(self) -> Dict[str, float]:
        return {
            "Latitude": 0.0,
            "Longitude": 0.0,
            "Flow": 0.0,
            "EnergyRatio": 1.0,
            "Congestion": 0.0,
            "HeatFlow": 0.0,
            "DistToDest": 0.0,
        }

    def _get_node(self, sat_id: int, dest_id: Optional[int] = None) -> Dict[str, float]:
        node = dict(self._nodes.get(int(sat_id), self._default_node()))
        if dest_id is not None:
            hops = self._shortest_hops_to_dest(int(sat_id), int(dest_id))
            node["DistToDest"] = 20000.0 if hops is None else min(hops, 20) * 1000.0
        return node

    def _get_neighbors(self, sat_id: int) -> List[Tuple[int, Dict[str, Any]]]:
        return [(int(dst), dict(attr or {})) for dst, attr in self._topology.get(int(sat_id), [])]

    def _get_link_attr(self, src_id: int, dst_id: int) -> Optional[Dict[str, Any]]:
        for candidate_id, attr in self._get_neighbors(int(src_id)):
            if candidate_id == int(dst_id):
                return attr
        return None

    def _is_known_satellite(self, sat_id: int) -> bool:
        sat_id = int(sat_id)
        return sat_id in self._nodes or sat_id in self._topology

    def _is_link_available(self, src_id: int, dst_id: int, packet_size: float = 0.0) -> bool:
        attr = self._get_link_attr(src_id, dst_id)
        if attr is None:
            return False
        if not self._to_bool(attr.get("is_valid", True)):
            return False
        if float(attr.get("PacketLossRate", 0.0)) >= 1.0:
            return False
        return self._available_capacity(attr) > max(0.0, float(packet_size)) * 0.0

    def _shortest_hop_map_to(self, dest_id: int) -> Dict[int, int]:
        dest_id = int(dest_id)
        if dest_id in self._shortest_hop_cache:
            return self._shortest_hop_cache[dest_id]

        reverse_adj: Dict[int, List[int]] = {}
        for source_id, links in self._topology.items():
            for target_id, attr in links:
                if self._to_bool(attr.get("is_valid", True)) and float(attr.get("PacketLossRate", 0.0)) < 1.0:
                    reverse_adj.setdefault(int(target_id), []).append(int(source_id))

        hop_map = {dest_id: 0}
        queue = [dest_id]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            next_hops = hop_map[node] + 1
            for prev in reverse_adj.get(node, []):
                if prev not in hop_map:
                    hop_map[prev] = next_hops
                    queue.append(prev)

        self._shortest_hop_cache[dest_id] = hop_map
        return hop_map

    def _shortest_hops_to_dest(self, node_id: int, dest_id: int) -> Optional[int]:
        return self._shortest_hop_map_to(dest_id).get(int(node_id))

    # ------------------------------------------------------------------
    # 代价与特征
    # ------------------------------------------------------------------

    @staticmethod
    def _hop_delay_ms(qualities: Mapping[str, Any]) -> float:
        return (
            float(qualities.get("LinkPropagationDelay", 0.0))
            + float(qualities.get("QueueDelay", 0.0))
            + float(qualities.get("TransmissionDelay", 0.0))
        )

    @staticmethod
    def _available_capacity(qualities: Mapping[str, Any]) -> float:
        max_capacity = max(float(qualities.get("MaxCapacity", qualities.get("LinkCapacity", 0.0))), 1e-6)
        left_capacity = float(qualities.get("LeftCapacity", qualities.get("LinkAvailableGbps", max_capacity)))
        if 0.0 <= left_capacity <= 1.0 and max_capacity > 1.0:
            return left_capacity * max_capacity
        return left_capacity

    def _hop_overflow(self, packet_size: float, qualities: Mapping[str, Any]) -> float:
        return max(0.0, float(packet_size) - self._available_capacity(qualities))

    def _hop_cost(self, next_sat: int, qualities: Mapping[str, Any]) -> float:
        next_state = self._get_node(next_sat)
        packet_loss_rate = min(max(float(qualities.get("PacketLossRate", 0.0)), 0.0), 1.0)
        max_capacity = max(float(qualities.get("MaxCapacity", 1.0)), 1e-6)
        capacity_penalty = max(0.0, 1.0 - (self._available_capacity(qualities) / max_capacity))
        congestion = float(next_state.get("Congestion", 0.0))
        return (
            self.routing_config.delay_weight * self._hop_delay_ms(qualities)
            + self.routing_config.packet_loss_weight * packet_loss_rate
            + self.routing_config.congestion_weight * congestion
            + self.routing_config.capacity_penalty_weight * capacity_penalty
            + self.routing_config.hop_weight
        )

    def _sorted_candidates(self, current_sat: int, dest_sat: int, visited_nodes: List[int], packet_size: float) -> List[Tuple[int, Dict[str, Any]]]:
        candidates: List[Tuple[int, Dict[str, Any]]] = []
        visited = set(int(x) for x in visited_nodes)
        for dst_id, attr in self._get_neighbors(current_sat):
            if not self._is_link_available(current_sat, dst_id, packet_size):
                continue
            if dst_id in visited and dst_id != dest_sat:
                continue
            candidates.append((dst_id, attr))

        if not candidates:
            for dst_id, attr in self._get_neighbors(current_sat):
                if self._is_link_available(current_sat, dst_id, packet_size):
                    candidates.append((dst_id, attr))

        def key(item: Tuple[int, Dict[str, Any]]):
            dst_id, attr = item
            hops = self._shortest_hops_to_dest(dst_id, dest_sat)
            return (9999 if hops is None else hops, self._hop_cost(dst_id, attr))

        candidates.sort(key=key)
        return candidates[: self.feature_builder.N_ACTIONS]

    def _build_batch_item(
        self,
        current_sat: int,
        dest_sat: int,
        candidates: List[Tuple[int, Dict[str, Any]]],
    ):
        current_node = self._get_node(current_sat, dest_sat)
        dest_node = self._get_node(dest_sat, dest_sat)
        candidate_nodes = [self._get_node(dst_id, dest_sat) for dst_id, _ in candidates]
        edge_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for action_idx, (candidate_id, attr) in enumerate(candidates, start=1):
            edge_info[(0, action_idx)] = dict(attr, is_valid=True)
            direct_to_dest = self._get_link_attr(candidate_id, dest_sat)
            if direct_to_dest is not None:
                edge_info[(action_idx, 5)] = dict(direct_to_dest, is_valid=True)
            else:
                hops = self._shortest_hops_to_dest(candidate_id, dest_sat)
                edge_info[(action_idx, 5)] = {
                    "LinkPropagationDelay": 0.0 if hops is None else float(hops),
                    "QueueDelay": 0.0,
                    "TransmissionDelay": 0.0,
                    "MaxCapacity": attr.get("MaxCapacity", 1.0),
                    "LeftCapacity": attr.get("LeftCapacity", attr.get("MaxCapacity", 1.0)),
                    "HeatValue": attr.get("HeatValue", 0.0),
                    "PacketLossRate": 1.0 if hops is None else 0.0,
                    "is_valid": hops is not None,
                }

        return current_node, candidate_nodes, dest_node, edge_info

    def _choose_next_hop(
        self,
        current_sat: int,
        dest_sat: int,
        visited_nodes: List[int],
        packet_size: float,
    ) -> Tuple[Optional[int], float]:
        start = time.perf_counter()
        candidates = self._sorted_candidates(current_sat, dest_sat, visited_nodes, packet_size)
        if not candidates:
            return None, round((time.perf_counter() - start) * 1000, 3)

        batch_item = self._build_batch_item(current_sat, dest_sat, candidates)
        node_feats, edge_index, edge_attr, action_mask = self.feature_builder.build_batch([batch_item])

        selected_idx = 0
        try:
            with torch.no_grad():
                actions, _, _ = self.model.get_action(
                    node_feats,
                    edge_index,
                    edge_attr,
                    action_mask=action_mask,
                    deterministic=True,
                )
            candidate_idx = int(actions[0].item())
            if 0 <= candidate_idx < len(candidates) and bool(action_mask[0, candidate_idx].item()):
                selected_idx = candidate_idx
        except Exception:
            # 模型推理异常时，不中断系统流程，回退到候选列表第一个低代价邻居。
            selected_idx = 0

        next_hop = int(candidates[selected_idx][0])
        return next_hop, round((time.perf_counter() - start) * 1000, 3)

    # ------------------------------------------------------------------
    # 对外推理入口
    # ------------------------------------------------------------------

    def route(
        self,
        topology: Mapping[Any, Any],
        node_state: Mapping[Any, Any],
        task: Mapping[str, Any],
        max_hops: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        在总项目已有拓扑上执行单任务路由推理。

        Parameters
        ----------
        topology:
            总项目当前 newTopology。
        node_state:
            总项目当前 SatList。
        task:
            已展开的路由任务字段。
        max_hops:
            最大搜索跳数。默认取 RoutingConfig.max_search_hops。
        """
        self.update_external_state(topology, node_state)

        task_id = str(task.get("TaskId", "UNKNOWN_TASK"))
        src_sat_id = int(task["SrcSatId"])
        dest_sat_id = int(task["DestSatId"])
        packet_size = float(task.get("PacketSize", 1.0))
        start_time = self._normalize_time(task.get("StartTime") or task.get("Timestamp"))
        max_hops = int(max_hops or self.routing_config.max_search_hops)

        route_path = [src_sat_id]
        current_sat = src_sat_id
        total_cost = 0.0
        total_delay_ms = 0.0
        decision_time_ms = 0.0
        delivery_probability = 1.0
        attempted_hops = 0
        valid_hops = 0
        max_overflow = 0.0
        status = "Dropped"

        if not self._is_known_satellite(src_sat_id) or not self._is_known_satellite(dest_sat_id):
            status = "InvalidTask"
        elif src_sat_id == dest_sat_id:
            status = "Arrived"
        else:
            for _ in range(max_hops):
                next_hop, one_decision_ms = self._choose_next_hop(
                    current_sat=current_sat,
                    dest_sat=dest_sat_id,
                    visited_nodes=route_path,
                    packet_size=packet_size,
                )
                decision_time_ms += one_decision_ms
                if next_hop is None or next_hop == current_sat:
                    status = "Dropped"
                    break

                attempted_hops += 1
                link_attr = self._get_link_attr(current_sat, next_hop)
                if link_attr is None or not self._is_link_available(current_sat, next_hop, packet_size):
                    status = "Dropped"
                    break

                valid_hops += 1
                route_path.append(next_hop)
                total_delay_ms += self._hop_delay_ms(link_attr)
                total_cost += self._hop_cost(next_hop, link_attr)
                delivery_probability *= max(0.0, 1.0 - float(link_attr.get("PacketLossRate", 0.0)))
                max_overflow = max(max_overflow, self._hop_overflow(packet_size, link_attr))
                current_sat = next_hop

                if current_sat == dest_sat_id:
                    status = "Arrived"
                    break

                if len(route_path) != len(set(route_path)):
                    status = "LoopDetected"
                    break

        output_route_path = route_path if status == "Arrived" else []
        isl_valid_rate = 100.0 * valid_hops / attempted_hops if attempted_hops > 0 else (100.0 if status == "Arrived" else 0.0)
        end_time = self._estimate_end_time(start_time, total_delay_ms)

        return {
            "TaskId": task_id,
            "Status": status,
            "status": status,
            "RoutePath": output_route_path,
            "route_path": output_route_path,
            "TotalHopCount": max(len(output_route_path) - 1, 0),
            "hop_count": max(len(output_route_path) - 1, 0),
            "PathTotalCost": float(total_cost),
            "total_path_cost": float(total_cost),
            "EndToEndDelay": float(total_delay_ms),
            "end_to_end_delay_ms": float(total_delay_ms),
            "ISLValidRate": float(isl_valid_rate),
            "StartTime": start_time,
            "EndTime": end_time,
            "Overflow": float(max_overflow),
            "delivery_probability": float(delivery_probability),
            "decision_time_ms": float(decision_time_ms),
            "next_hop": output_route_path[1] if len(output_route_path) > 1 else None,
        }

    @staticmethod
    def _normalize_time(value: Any) -> str:
        if value is None:
            return _dt.datetime.now(_dt.timezone.utc).isoformat()
        if isinstance(value, _dt.datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
            return dt.astimezone(_dt.timezone.utc).isoformat()
        if isinstance(value, (int, float)):
            ts = float(value) / 1000.0 if float(value) > 1e12 else float(value)
            return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
        text = str(value)
        try:
            dt = _dt.datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return dt.astimezone(_dt.timezone.utc).isoformat()
        except Exception:
            return text

    @staticmethod
    def _estimate_end_time(start_time: str, delay_ms: float) -> str:
        try:
            dt = _dt.datetime.fromisoformat(str(start_time))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return (dt + _dt.timedelta(milliseconds=float(delay_ms))).astimezone(_dt.timezone.utc).isoformat()
        except Exception:
            return start_time
