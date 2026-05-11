from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from sgp4.api import Satrec, jday
from datetime import datetime
import numpy as np


class ABAttribute(Enum):
    J = 0
    M = 1


@dataclass
class SatelliteInfo:
    # ===================== 接口标准字段（完全按你的表） =====================
    satId: str
    tle1: str
    tle2: str
    orbitPlaneNum: int
    numInPlane: int
    ABAttribute: ABAttribute = ABAttribute.J
    storage: float = 0.0
    compPower: float = 0.0
    hasSatsLink: bool = True
    numLinks: int = 4
    maxLinkSpeed: float = 10.0
    linkStability: float = 0.95

    # =====================节点状态字段 =====================
    current_time: Optional[datetime] = None
    latitude: float = 0.0  # 纬度
    longitude: float = 0.0  # 经度
    flow: int = 0
    energy_ratio: float = 1.0
    heat_flow: float = 0.0

    # 内部仅临时使用，不存储
    _satrec: Optional[Satrec] = None

    def __post_init__(self):
        # 校验
        if self.orbitPlaneNum <= 0:
            raise ValueError
        if self.numInPlane <= 0:
            raise ValueError
        self._satrec = Satrec.twoline2rv(self.tle1, self.tle2)

    # ==================核心：直接更新经纬度===================
    def update_position(self, target_time: datetime):
        # 时间转换
        yr, mo, dy, hr, mi = (
            target_time.year, target_time.month, target_time.day,
            target_time.hour, target_time.minute
        )
        # 【修改】保留微秒级精度
        sc = target_time.second + target_time.microsecond / 1000000.0
        jd, fr = jday(yr, mo, dy, hr, mi, sc)

        # 轨道计算
        e, pos, vel = self._satrec.sgp4(jd, fr)
        if e != 0:
            self.latitude = 0.0
            self.longitude = 0.0
            self.current_time = target_time
            return

        # 直接转经纬度，不保存XYZ
        x, y, z = pos
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x)) - (jd + fr - 2451545.0) * 360.9856473668683
        lon = lon % 360
        if lon > 180:
            lon -= 360

        # 只保存经纬度
        self.current_time = target_time
        self.latitude = round(lat, 6)
        self.longitude = round(lon, 6)


@dataclass
class CommunicationConstellation:
    sceneId: str
    constellationId: str
    orbitHeight: float
    orbitInclination: float
    numOrbitPlane: int
    numSatsInPlane: int
    phaseFactor: int
    satelliteList: List[SatelliteInfo] = field(default_factory=list)

    def __post_init__(self):
        if self.orbitHeight <= 0:
            raise ValueError
        if self.numOrbitPlane <= 0:
            raise ValueError

    @property
    def total_satellites(self):
        return self.numOrbitPlane * self.numSatsInPlane

    def get_satellite_by_id(self, sat_id):
        for sat in self.satelliteList:
            if sat.satId == sat_id:
                return sat
        return None

    def update_constellation_time(self, t):
        for sat in self.satelliteList:
            sat.update_position(t)


@dataclass
class DoubleShellConstellation:
    constellation_name: str = "QG_Constellation_4032"
    shell1: Optional[CommunicationConstellation] = None
    shell2: Optional[CommunicationConstellation] = None

    @property
    def total_satellites(self):
        count = 0
        if self.shell1:
            count += len(self.shell1.satelliteList)
        if self.shell2:
            count += len(self.shell2.satelliteList)
        return count

    def update_global_time(self, t):
        if self.shell1:
            self.shell1.update_constellation_time(t)
        if self.shell2:
            self.shell2.update_constellation_time(t)

# ===================== TLE 读取（适配你的文件 + ID 0-3599） =====================
def load_satellites_from_tle(file_path, sats_per_plane):
    sats = []
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if (line.startswith("1") or line.startswith("2")) and len(line) >= 60:
                lines.append(line)

    for i in range(0, len(lines) - 1, 2):
        l1 = lines[i]
        l2 = lines[i + 1]
        if l1.startswith("1") and l2.startswith("2"):
            # 卫星 ID = 0,1,2,...,3599 纯整数
            sat_id = len(sats)
            orbit_plane = len(sats) // sats_per_plane + 1
            num_in_plane = len(sats) % sats_per_plane + 1

            sat = SatelliteInfo(
                satId=sat_id,
                tle1=l1,
                tle2=l2,
                orbitPlaneNum=orbit_plane,
                numInPlane=num_in_plane
            )
            sats.append(sat)
    return sats


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 只加载 3600 颗卫星（60×60）
    tle_file = "Com_qx4_60p60_TLE(1).txt"
    sats = load_satellites_from_tle(tle_file, 60)

    # 仅创建 shell1（3600 颗）
    shell = CommunicationConstellation("SCENE1", "SHELL1", 500, 58, 60, 60, 1, sats)
    constellation = DoubleShellConstellation(shell1=shell, shell2=None)

    # 数量显示
    total = len(sats)
    print(f"total:{total}")

    import time
    from datetime import timedelta

    # 2 秒 → 2 个文件
    for step in range(2):
        now = datetime.utcnow()
        constellation.update_global_time(now)

        # 输出文件：无中文、无空格、紧凑格式
        filename = f"pos_{step}.txt"
        with open(filename, "w") as f:
            for sat in constellation.shell1.satelliteList:
                # 星下点当地时间 = UTC + 经度/15
                lon_hour = sat.longitude / 15.0
                local_time = sat.current_time + timedelta(hours=lon_hour)
                local_str = local_time.strftime("%Y%m%d%H%M%S")

                # 输出格式：id,lat,lon,local_time
                f.write(f"{sat.satId},{sat.latitude},{sat.longitude},{local_str}\n")

        print(f"saved:{filename}")
        time.sleep(1)

    # 测试输出
    test = constellation.shell1.get_satellite_by_id(0)
    lon_hour = test.longitude / 15.0
    local_time = test.current_time + timedelta(hours=lon_hour)
    local_str = local_time.strftime("%Y%m%d%H%M%S")
    print(f"{test.satId},{test.latitude},{test.longitude},{local_str}")
    print("all_done")