"""
地理计算工具
仅处理与经纬度、球面距离相关的计算
"""
import numpy as np
from config import *


def generate_global_latlon_grid() -> tuple[np.ndarray, np.ndarray]:
    """
    生成全球经纬度网格中心点（流量计算专用）
    :return: 展平后的纬度数组、经度数组
    """
    # 生成网格边界
    lat_grid = np.linspace(90, -90, TRAFFIC_GRID_ROWS + 1)
    lon_grid = np.linspace(-180, 180, TRAFFIC_GRID_COLS + 1)

    # 计算网格中心点
    lat_centers = (lat_grid[:-1] + lat_grid[1:]) / 2
    lon_centers = (lon_grid[:-1] + lon_grid[1:]) / 2

    # 生成网格矩阵并展平
    lon_mat, lat_mat = np.meshgrid(lon_centers, lat_centers)
    return lat_mat.flatten(), lon_mat.flatten()


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的球面距离（Haversine公式）
    :param lat1: 点1纬度
    :param lon1: 点1经度
    :param lat2: 点2纬度
    :param lon2: 点2经度
    :return: 距离（km）
    """
    # 角度转弧度
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c