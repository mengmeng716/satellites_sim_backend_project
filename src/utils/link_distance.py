import numpy as np
from typing import Tuple
# 若需直接对接 SatelliteInfo，可导入（可选）
# from consetellation import SatelliteInfo


# ===================== WGS84 椭球参数（标准常量）=====================
WGS84_A = 6378137.0          # 长半轴（米）
WGS84_F = 1 / 298.257223563  # 扁率
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # 第一偏心率平方


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> Tuple[float, float, float]:
    """
    【内部函数】大地坐标（经纬度+高度）→ ECEF直角坐标
    :param lat_deg: 纬度（度）
    :param lon_deg: 经度（度）
    :param height_m: 高度（米，相对于WGS84椭球面）
    :return: (X, Y, Z) 单位：米
    """
    # 角度转弧度
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # 卯酉圈半径 N
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)

    # 计算 ECEF 坐标
    X = (N + height_m) * cos_lat * cos_lon
    Y = (N + height_m) * cos_lat * sin_lon
    Z = (N * (1 - WGS84_E2) + height_m) * sin_lat

    return X, Y, Z


def calculate_link_distance(
    sat1_lat: float, sat1_lon: float,
    sat2_lat: float, sat2_lon: float,
    orbit_height_km: float
) -> float:
    """
    【核心函数】计算两颗卫星的欧氏链路距离
    :param sat1_lat: 卫星1纬度（度）
    :param sat1_lon: 卫星1经度（度）
    :param sat2_lat: 卫星2纬度（度）
    :param sat2_lon: 卫星2经度（度）
    :param orbit_height_km: 卫星轨道高度（公里，从 CommunicationConstellation 获取）
    :return: 链路距离（公里）
    """
    # 轨道高度：公里 → 米
    orbit_height_m = orbit_height_km * 1000.0

    # 两颗卫星分别转 ECEF 坐标
    x1, y1, z1 = _geodetic_to_ecef(sat1_lat, sat1_lon, orbit_height_m)
    x2, y2, z2 = _geodetic_to_ecef(sat2_lat, sat2_lon, orbit_height_m)

    # 三维欧氏距离计算
    distance_m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    # 距离：米 → 公里（返回）
    return distance_m / 1000.0


def calculate_sat_link_distance(
    sat1,  # Type: SatelliteInfo（为避免循环依赖，暂不写死类型）
    sat2,  # Type: SatelliteInfo
    orbit_height_km: float
) -> float:
    """
    【便捷函数】直接传入 SatelliteInfo 对象计算链路距离
    :param sat1: 卫星1对象（来自 constellation.py 的 SatelliteInfo）
    :param sat2: 卫星2对象（来自 constellation.py 的 SatelliteInfo）
    :param orbit_height_km: 卫星轨道高度（公里）
    :return: 链路距离（公里）
    """
    return calculate_link_distance(
        sat1.latitude, sat1.longitude,
        sat2.latitude, sat2.longitude,
        orbit_height_km
    )