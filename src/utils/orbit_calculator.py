"""
功能：空间物理与轨道计算模块
1. calculate_satellite_coordinates: 封装 SGP4 轨道模型，基于 TLE 和 UTC 时间戳计算卫星实时经纬度。
2. calculate_slant_range_vectorized: 提供高性能向量化的空间斜距计算，用于快速判断卫星与地面网格的可见性。
"""

import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime
from src import config


def calculate_slant_range_vectorized(lat_ground, lon_ground, lat_sats, lon_sats):
    """计算地面网格到卫星阵列的向量化斜距"""
    phi1, lam1 = np.radians(lat_ground), np.radians(lon_ground)
    phi2, lam2 = np.radians(lat_sats), np.radians(lon_sats)

    x1 = config.EARTH_RADIUS_KM * np.cos(phi1) * np.cos(lam1)
    y1 = config.EARTH_RADIUS_KM * np.cos(phi1) * np.sin(lam1)
    z1 = config.EARTH_RADIUS_KM * np.sin(phi1)

    r_total = config.EARTH_RADIUS_KM + config.ORBIT_HEIGHT_KM
    x2 = r_total * np.cos(phi2) * np.cos(lam2)
    y2 = r_total * np.cos(phi2) * np.sin(lam2)
    z2 = r_total * np.sin(phi2)

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_satellite_coordinates(tle1: str, tle2: str, current_utc: datetime):
    """基于 TLE 和 UTC 时间戳计算单颗卫星的真实经纬度"""
    satellite = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(current_utc.year, current_utc.month, current_utc.day,
                  current_utc.hour, current_utc.minute, current_utc.second)

    e, pos, _ = satellite.sgp4(jd, fr)
    if e != 0:
        return 0.0, 0.0, False

    t_gmst = (jd + fr - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd + fr - 2451545.0) + 0.000387933 * t_gmst ** 2 - (
                t_gmst ** 3 / 38710000.0)
    gmst_rad = np.radians(gmst % 360)

    sat_vec = np.array(pos, dtype=np.float32)
    r = np.sqrt(np.dot(sat_vec, sat_vec))
    lat = np.degrees(np.arcsin(sat_vec[2] / r))
    lon = np.degrees(np.arctan2(sat_vec[1], sat_vec[0])) - np.degrees(gmst_rad)
    lon = (lon + 180) % 360 - 180

    return round(float(lat), 6), round(float(lon), 6), True