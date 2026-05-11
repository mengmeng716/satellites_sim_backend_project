import numpy as np
import pandas as pd
import src.config as config

class GroundStationManager:
    _instance = None
    _is_initialized = False

    # 缓存数据
    gs_lats = None
    gs_lons = None
    gs_ecef = None
    num_gs = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GroundStationManager, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """仅在系统启动时（或者首次被使用时）被调用一次"""
        if self._is_initialized:
            return

        try:
            df_gs = pd.read_excel(config.GROUND_STATION_FILE)
            df_gs.columns = [str(c).strip().lower() for c in df_gs.columns]

            if 'latitude' in df_gs.columns and 'longitude' in df_gs.columns:
                self.gs_lats = df_gs['latitude'].values.astype(float)
                self.gs_lons = df_gs['longitude'].values.astype(float)
            else:
                self.gs_lats = df_gs.iloc[:, 0].values.astype(float)
                self.gs_lons = df_gs.iloc[:, 1].values.astype(float)

            # 提前一口气全部转为弧度和 3D ECEF 坐标（这最消耗算力，一次算好，终身受益）
            phi, lam = np.radians(self.gs_lats), np.radians(self.gs_lons)
            
            x = config.EARTH_RADIUS_KM * np.cos(phi) * np.cos(lam)
            y = config.EARTH_RADIUS_KM * np.cos(phi) * np.sin(lam)
            z = config.EARTH_RADIUS_KM * np.sin(phi)

            self.gs_ecef = np.column_stack((x, y, z))
            self.num_gs = len(self.gs_lats)
            self._is_initialized = True
            
            print(f"[Engine] 全局缓存: 成功且仅一次加载了 {self.num_gs} 个地面基站。")

        except Exception as e:
            print(f"[Warning] 全局地面站缓存加载失败: {e}")
            self.gs_lats, self.gs_lons = np.array([]), np.array([])
            self.gs_ecef = np.empty((0, 3))
            self.num_gs = 0
            self._is_initialized = True

    def get_ecef_coordinates(self):
        self.initialize()
        return self.gs_ecef
        
    def get_lat_lon(self):
        self.initialize()
        return self.gs_lats, self.gs_lons

# 就在文件最后一行加
gs_manager = GroundStationManager()