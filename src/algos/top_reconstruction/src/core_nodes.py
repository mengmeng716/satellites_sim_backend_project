"""Core-node selection from node-state GroundStationNumber."""

from __future__ import annotations

import random
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.utils.source_destination_constraint import check_src_dst_distance_constraint


Pair = Tuple[int, int]


def _read_value(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    elif obj is not None:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def _raw_sat_list(node_state: Any) -> Mapping[Any, Any]:
    if hasattr(node_state, "sat_list"):
        return node_state.sat_list or {}
    if isinstance(node_state, Mapping):
        return node_state.get("SatList") or node_state.get("sat_list") or {}
    return {}


def _ground_station_numbers(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = (value,)

    station_ids: List[int] = []
    for item in items:
        try:
            station_id = int(item)
        except (TypeError, ValueError):
            continue
        if station_id > 0:
            station_ids.append(station_id)
    return station_ids


def get_current_core_nodes(node_state: Any) -> Tuple[List[int], ...]:
    """Return CoreNodeNumber by reading node-state GroundStationNumber.

    GroundStationNumber is refreshed by the simulation engine before each
    topology reconstruction, so this function must not recalculate nearest
    ground-station satellites.
    """
    sat_list = _raw_sat_list(node_state)
    if not sat_list:
        return tuple()

    valid_sat_ids: List[int] = []
    for raw_sat_id in sat_list:
        try:
            valid_sat_ids.append(int(raw_sat_id))
        except (TypeError, ValueError):
            continue

    if not valid_sat_ids:
        return tuple()

    core_node_list: List[List[int]] = [[] for _ in range(max(valid_sat_ids) + 1)]
    for raw_sat_id, sat_data in sat_list.items():
        try:
            sat_id = int(raw_sat_id)
        except (TypeError, ValueError):
            continue

        station_number = _read_value(
            sat_data,
            ("GroundStationNumber", "ground_station_number"),
        )
        core_node_list[sat_id].extend(_ground_station_numbers(station_number))

    return tuple(core_node_list)


def get_core_satellite_ids(core_node_number: Sequence[Sequence[int]]) -> List[int]:
    return [sat_id for sat_id, station_ids in enumerate(core_node_number) if station_ids]


def build_source_destination_pairs(
    core_satellite_ids: Iterable[int],
    ordered: bool = False,
) -> List[Pair]:
    core_ids = sorted(set(int(sat_id) for sat_id in core_satellite_ids))
    pairs: List[Pair] = []
    for index, source in enumerate(core_ids):
        for target in core_ids[index + 1 :]:
            if ordered:
                pairs.append((source, target))
                pairs.append((target, source))
            else:
                pairs.append((source, target))
    return pairs


def get_top_business_demand_core_satellite_ids(
    satellite_business_intensity: Mapping[int, float],
    core_node_count: int,
    candidate_satellite_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """Return satellites with the highest normalized business demand intensity."""
    ranked_satellites: List[Tuple[int, float]] = []
    if candidate_satellite_ids is None:
        for raw_sat_id, intensity in satellite_business_intensity.items():
            try:
                ranked_satellites.append((int(raw_sat_id), float(intensity)))
            except (TypeError, ValueError):
                continue
    else:
        candidate_ids = set()
        for raw_sat_id in candidate_satellite_ids:
            try:
                candidate_ids.add(int(raw_sat_id))
            except (TypeError, ValueError):
                continue
        for sat_id in candidate_ids:
            ranked_satellites.append(
                (sat_id, float(satellite_business_intensity.get(sat_id, 0.0)))
            )

    ranked_satellites.sort(key=lambda item: (-item[1], item[0]))
    return [sat_id for sat_id, _ in ranked_satellites[: max(0, int(core_node_count))]]


def build_business_demand_source_destination_pairs(
    node_state: Any,
    satellite_business_intensity: Mapping[int, float],
    core_node_count: int,
    ordered: bool = False,
) -> List[Pair]:
    """Build all constrained pairs from top business-demand core satellites."""
    ground_station_core_ids = get_core_satellite_ids(get_current_core_nodes(node_state))
    core_ids = get_top_business_demand_core_satellite_ids(
        satellite_business_intensity,
        core_node_count,
        candidate_satellite_ids=ground_station_core_ids,
    )
    pairs = build_source_destination_pairs(core_ids, ordered=ordered)
    return sorted(
        (source_id, target_id)
        for source_id, target_id in pairs
        if check_src_dst_distance_constraint(node_state, source_id, target_id)
    )


def build_random_source_destination_pairs(
    node_state: Any,
    core_satellite_ids: Iterable[int],
    pair_count: int = 100,
    seed: Optional[int] = None,
) -> List[Pair]:
    """Sample unordered source-destination pairs with distance constraint.
    
    生成的卫星对会经过经纬度差异约束检查，避免选择过近的卫星作为源-目的对。
    
    :param node_state: 节点状态对象，用于获取卫星经纬度信息
    :param core_satellite_ids: 核心卫星 ID 列表
    :param pair_count: 需要生成的配对数量
    :param seed: 随机种子（用于可复现性）
    :return: 满足距离约束的 (源ID, 目的ID) 对列表
    """
    # 首先构建所有可能的候选对
    all_pairs = build_source_destination_pairs(core_satellite_ids, ordered=False)
    
    # 过滤出满足距离约束的配对
    valid_pairs = []
    for src_id, dst_id in all_pairs:
        if check_src_dst_distance_constraint(node_state, src_id, dst_id):
            valid_pairs.append((src_id, dst_id))
    
    # 如果有效配对数量不足，返回全部（并给出警告）
    if not valid_pairs:
        print(f"[警告] 没有满足经纬度差异约束的卫星对，返回空列表")
        return []
    
    if len(valid_pairs) <= pair_count:
        print(f"[提示] 满足约束的候选对数量 ({len(valid_pairs)}) 少于请求数量 ({pair_count})，返回全部")
        return sorted(valid_pairs)
    
    # 随机采样指定数量的配对
    rng = random.Random(seed)
    selected_pairs = rng.sample(valid_pairs, pair_count)
    
    return sorted(selected_pairs)
