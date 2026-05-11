"""
时间计算工具
仅处理与时间相关的计算
"""
from config import *


def utc_to_local_hour(utc_hour: float, longitude: float) -> float:
    """
    将UTC小时转换为指定经度的本地小时
    :param utc_hour: UTC时间的小时数（0-24）
    :param longitude: 地理经度
    :return: 本地小时数（0-24）
    """
    return (utc_hour + longitude / 15) % 24


def calculate_traffic_time_factor(local_hour: float) -> float:
    """
    计算流量时间因子（昼夜波动系数）
    :param local_hour: 本地小时数（0-24）
    :return: 时间因子，用于流量计算
    """
    if DAYTIME_START_HOUR <= local_hour < DAYTIME_PEAK_START_HOUR:
        return MIN_TIME_FACTOR + (local_hour - DAYTIME_START_HOUR) / 4
    elif DAYTIME_PEAK_START_HOUR <= local_hour < DAYTIME_PEAK_END_HOUR:
        return MAX_TIME_FACTOR
    elif DAYTIME_PEAK_END_HOUR <= local_hour < DAYTIME_END_HOUR:
        return MAX_TIME_FACTOR - (local_hour - DAYTIME_PEAK_END_HOUR) / 2
    else:
        return MIN_TIME_FACTOR