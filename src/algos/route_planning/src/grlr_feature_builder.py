"""
GRLR 特征构造器。

本文件不生成星座拓扑，只把总项目已经给出的 newTopology / SatList
转换为 GAT + Actor 推理所需的 node_features、edge_features、edge_index 和 action_mask。
"""

import torch
from typing import List, Tuple


class FeatureBuilder:
    N_NODES = 6
    N_ACTIONS = 4
    N_EDGES = 8
    NODE_FEAT_DIM = 6
    EDGE_FEAT_DIM = 5

    EDGE_INDEX = torch.tensor([
        [0, 0, 0, 0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5, 5, 5, 5]
    ], dtype=torch.long)

    EDGES = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 5), (2, 5), (3, 5), (4, 5)
    ]

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.edge_index = self.EDGE_INDEX.to(device)
        self._node_feat_buffer = None
        self._edge_feat_buffer = None
        self._action_mask_buffer = None
        self._current_batch_size = 0

    def preallocate_buffers(self, batch_size: int):
        if batch_size > self._current_batch_size:
            self._node_feat_buffer = torch.zeros(
                batch_size,
                self.N_NODES,
                self.NODE_FEAT_DIM,
                device=self.device
            )
            self._edge_feat_buffer = torch.zeros(
                batch_size,
                self.N_EDGES,
                self.EDGE_FEAT_DIM,
                device=self.device
            )
            self._action_mask_buffer = torch.zeros(
                batch_size,
                self.N_ACTIONS,
                dtype=torch.bool,
                device=self.device
            )
            self._current_batch_size = batch_size

    def _parse_node_features(self, attr: dict) -> list:
        return [
            float(attr.get("Latitude", 0.0)) / 90.0,
            float(attr.get("Longitude", 0.0)) / 180.0,
            float(attr.get("Flow", 0.0)) / 10.0,
            float(attr.get("EnergyRatio", 1.0)),
            float(attr.get("Congestion", 0.0)),
            float(attr.get("DistToDest", 0.0)) / 20000.0
        ]

    def _normalize_heat_value(self, heat_value: float) -> float:
        heat_value = float(heat_value)
        if abs(heat_value) <= 1.0:
            return heat_value
        return heat_value / 10.0

    def build_batch(self, batch_data: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(batch_data)
        self.preallocate_buffers(batch_size)

        node_feats = self._node_feat_buffer[:batch_size]
        edge_feats = self._edge_feat_buffer[:batch_size]
        action_masks = self._action_mask_buffer[:batch_size]

        node_feats.zero_()
        edge_feats.zero_()
        action_masks.zero_()

        for b, (current, candidates, dest, edge_info) in enumerate(batch_data):
            node_feats[b, 0] = torch.tensor(self._parse_node_features(current), device=self.device)

            for i, cand in enumerate(candidates[:self.N_ACTIONS]):
                node_feats[b, i + 1] = torch.tensor(self._parse_node_features(cand), device=self.device)

            node_feats[b, self.N_NODES - 1] = torch.tensor(
                self._parse_node_features(dest),
                device=self.device
            )

            for idx, (s, d) in enumerate(self.EDGES):
                info = edge_info.get((s, d), {})
                propagation_delay = float(
                    info.get("PropagationDelay", info.get("LinkPropagationDelay", info.get("LinkDistance", 0.0)))
                )
                queue_delay = float(info.get("QueueDelay", 0.0))
                remaining_capacity_ratio = info.get("RemainingCapacityRatio")
                if remaining_capacity_ratio is None:
                    max_capacity = max(float(info.get("MaxCapacity", 0.0)), 1e-6)
                    left_capacity = float(info.get("LeftCapacity", 0.0))
                    remaining_capacity_ratio = left_capacity / max_capacity
                heat_value = info.get(
                    "HeatValue",
                    info.get("TrafficHeat", info.get("CurrentFlow", info.get("CumFlow", 0.0))),
                )
                edge_feats[b, idx] = torch.tensor([
                    propagation_delay / 100.0,
                    queue_delay / 100.0,
                    float(remaining_capacity_ratio),
                    self._normalize_heat_value(float(heat_value)),
                    float(info.get("PacketLossRate", 1.0)),
                ], device=self.device)

                if s == 0 and 1 <= d <= self.N_ACTIONS:
                    action_masks[b, d - 1] = bool(info.get("is_valid", False))

        # Clone outputs so future builds do not mutate already-returned observations.
        return node_feats.clone(), self.edge_index, edge_feats.clone(), action_masks.clone()


# 兼容旧代码命名。
GraphBuilder = FeatureBuilder
