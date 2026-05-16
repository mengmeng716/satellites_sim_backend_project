"""
源-目的卫星对约束模块

提供基于经纬度差异的卫星对筛选功能，避免选择经纬度过近的卫星作为业务流的源和目的节点，
使可视化效果更加合理。

注意：本模块仅使用经纬度差异进行判断，不涉及距离计算或轨道高度参数。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from src import config


def _read_satellite_position(
    node_state: Any,
    sat_id: int
) -> Optional[Tuple[float, float]]:
    """
    从 node_state 中读取指定卫星的经纬度
    
    :param node_state: 节点状态对象（SatelliteNodeState 或字典）
    :param sat_id: 卫星 ID
    :return: (latitude, longitude) 或 None（如果卫星不存在）
    """
    sat_list = {}
    
    # 尝试从不同格式获取 sat_list
    if hasattr(node_state, "sat_list"):
        sat_list = node_state.sat_list or {}
    elif isinstance(node_state, Mapping):
        sat_list = node_state.get("SatList") or node_state.get("sat_list") or {}
    
    if not sat_list:
        return None
    
    sat_key = str(sat_id)
    if sat_key in sat_list:
        sat_data = sat_list[sat_key]
    elif sat_id in sat_list:
        sat_data = sat_list[sat_id]
    else:
        return None
    
    # 读取经纬度（支持 Pydantic 模型或字典）
    lat = None
    lon = None
    
    if hasattr(sat_data, "latitude"):
        lat = sat_data.latitude
        lon = sat_data.longitude
    elif isinstance(sat_data, Mapping):
        lat = sat_data.get("Latitude") if "Latitude" in sat_data else sat_data.get("latitude")
        lon = sat_data.get("Longitude") if "Longitude" in sat_data else sat_data.get("longitude")
    
    if lat is None or lon is None:
        return None
    
    return float(lat), float(lon)


def check_src_dst_distance_constraint(
    node_state: Any,
    src_sat_id: int,
    dst_sat_id: int,
    min_lat_diff: Optional[float] = None,
    min_lon_diff: Optional[float] = None
) -> bool:
    """
    检查两个卫星是否满足源-目的对的经纬度差异约束
    
    :param node_state: 节点状态对象
    :param src_sat_id: 源卫星 ID
    :param dst_sat_id: 目的卫星 ID
    :param min_lat_diff: 最小纬度差阈值（度），默认使用 config.MIN_SRC_DST_LAT_DIFF_DEG
    :param min_lon_diff: 最小经度差阈值（度），默认使用 config.MIN_SRC_DST_LON_DIFF_DEG
    :return: True 如果满足约束（两颗卫星足够远），False 否则
    
    使用示例：
        if check_src_dst_distance_constraint(node_state, src_id, dst_id):
            # 这两颗卫星可以作为源-目的对
            pass
    """
    # 使用配置中的默认值
    if min_lat_diff is None:
        min_lat_diff = config.MIN_SRC_DST_LAT_DIFF_DEG
    if min_lon_diff is None:
        min_lon_diff = config.MIN_SRC_DST_LON_DIFF_DEG
    
    # 读取两颗卫星的经纬度
    src_pos = _read_satellite_position(node_state, src_sat_id)
    dst_pos = _read_satellite_position(node_state, dst_sat_id)
    
    # 如果任一卫星位置信息缺失，认为不满足约束
    if src_pos is None or dst_pos is None:
        return False
    
    src_lat, src_lon = src_pos
    dst_lat, dst_lon = dst_pos
    
    # 计算纬度差（绝对值）
    lat_diff = abs(src_lat - dst_lat)
    
    # 计算经度差（考虑 360 度循环，取最短弧长）
    lon_diff_raw = abs(src_lon - dst_lon)
    lon_diff = min(lon_diff_raw, 360.0 - lon_diff_raw)
    
    # 判断是否同时满足纬度和经度的最小差异要求
    return lat_diff >= min_lat_diff or lon_diff >= min_lon_diff
