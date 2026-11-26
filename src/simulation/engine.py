import time
import threading
import datetime
import math
import logging
from contextlib import contextmanager
from collections import deque
from pprint import pformat
import numpy as np


# [可以直接放在文件顶部引入，告别循环依赖]
from src.simulation.topology_link_state import (
    evaluate_link_physics_batch,
    init_walker_topology,
    laser_type_for_link,
    update_topology_snapshot,
)
from src.simulation.constellation import CommunicationConstellation, load_satellites_from_tle
from src.utils.orbit_calculator import calculate_satellite_coordinates
from src.simulation.background_traffic_generation import background_traffic_generation
from src.simulation.groundstation_number_generation import update_ground_station_numbers
from src.algos.link_prediction.prediction_interface import LinkPredictor
from src.algos.top_reconstruction.reconstruction_interface import reconstruct_topology
from src.algos.top_reconstruction.reconstruction_config import TopReconstructionConfig
from src.algos.top_reconstruction.src.adapters import normalize_topology
from src.algos.top_reconstruction.src.business_demand_intensity import compute_business_demand_intensity
from src.algos.top_reconstruction.src.core_nodes import build_business_demand_source_destination_pairs
from src.algos.link_prediction.interface import adapt_scfe_output, link_states_from_topology
from src.algos.sensing_monitoring import SensingMonitor, run_sensing_cycle
from src.algos.capacity_evaluation import evaluate as evaluate_capacity, lambda_star

import src.config as config

from src.simulation.data_model import (
    LaserConfig,
    LinksQualitiesValue,
    NodeAttribute,
    SatelliteNodeState,
    TopologyState,
)

logger = logging.getLogger(__name__)

CAPACITY_V2_EMIT_LEGACY_ONLY = bool(getattr(config, "CAPACITY_V2_EMIT_LEGACY_ONLY", True))

class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    @contextmanager
    def read_lock(self):
        self._read_ready.acquire()
        self._readers += 1
        self._read_ready.release()
        try:
            yield
        finally:
            self._read_ready.acquire()
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify_all()
            self._read_ready.release()

    @contextmanager
    def write_lock(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
        try:
            yield
        finally:
            self._read_ready.release()


class SimulationEngine:
    def __init__(self, constellation_id, duration_seconds, micro_period_seconds=None):
        self.constellation_id = constellation_id
        self.duration = duration_seconds
        self.current_time = 0

        if micro_period_seconds not in (None, ""):
            try:
                micro_period_seconds = int(micro_period_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError("micro_period_seconds must be one of 1, 2, 3, 4, 5") from exc
            if micro_period_seconds not in (1, 2, 3, 4, 5):
                raise ValueError("micro_period_seconds must be one of 1, 2, 3, 4, 5")
            config.MICRO_PERIOD_SECONDS = micro_period_seconds

        # 初始化后端回调接口为空，防止报错
        self.backend_callback = None

        # 将核心状态设为实例属性，并在仿真生命周期内维护
        self.topology_state: TopologyState | None = None
        self.node_state: SatelliteNodeState | None = None

        # 星座对象，在 initialize_constellation 中赋值，供位置更新使用
        self.constellation = None

        # 记录仿真的起始真实时间 (UTC datetime)，用于与任务的 ArrivalTime 比对
        self.sim_start_utctime = None

        # 使用读写锁包裹状态
        self.state_rw_lock = ReadWriteLock()

        # 线程同步事件 (逻辑时间驱动)
        self._stop_event = threading.Event()
        self._macro_event = threading.Event()
        self._micro_event = threading.Event()
        self._macro_done = threading.Event()
        self._micro_done = threading.Event()

        # [新增] 声明线程实例，防止后续判断时报 AttributeError
        self._macro_thread = None
        self._micro_thread = None

        # 30 micro-cycle frames: 5s * 30 = 150s macro-cycle history.
        self._prediction_buffer = deque(maxlen=30)

        # [新增] 初始化链路预测模块 - 传入星座ID而非文件路径
        self.link_predictor = LinkPredictor(constellation_id)
        self._last_energy_update_timestamp_ms = None
        self._last_topology_reconstruction_result = None
        self._ground_station_min_elevation_deg = float(config.min_elevation_deg)
        self.task_list = []

        # 感知监测（算法一）、表2-2 链路状态记录、容量与效能评估结果
        self.sensing_monitor = SensingMonitor()
        self.latest_sensing_status = None
        self.latest_prediction_result = None
        self.latest_link_state_records = []
        self.latest_capacity_evaluation = None

        self._active_prediction_accuracy = None
        self._pending_prediction_accuracy = None
        self._last_node_flow_by_sat = {}
        self._link_history_by_key = {}
        self._cum_flow_gb_by_key = {}
        self._last_cum_flow_time_by_key = {}
        self._laser_default_type = str(getattr(config, "LASER_DEFAULT_TYPE", "STD"))
        self._laser_profile_presets = dict(getattr(config, "LASER_CONFIG_PRESETS", {}) or {})
        
        # [新增] 缓存上一个周期的状态，用于计算增量(Delta)给前端
        self._last_notified_links = {}  # { "src_dst": attr_hash }
        self._last_topo_time = 0.0
        self._last_plan_time = 0.0
        self._last_pred_acc = 98.5
        self._micro_log_interval_seconds = 30
        self._last_micro_log_bucket = -1


    def initialize_constellation(self, timestamp, constellation_id, tle_file_path, task_list):
        """
        初始化阶段：加载 TLE、构建星座对象、生成拓扑、产生首次背景流量。
        :param timestamp:        仿真起始 UTC 时间 (datetime)
        :param constellation_id: 星座 ID (str)
        :param tle_file_path:    TLE 文件路径 (str)
        :param task_list:        保留参数（当前版本不执行路径/路由任务）
        :return: topology_state, node_state, []
        """
        logger.info(f"[{timestamp}] 初始化卫星星座: {constellation_id}, 加载TLE文件: {tle_file_path}")

        # 0. 初始化系统仿真时间和基准UTC时间
        self.current_time = 0
        self.sim_start_utctime = timestamp

        # 1. 初始化卫星节点状态 - 使用 SatelliteNodeState dataclass
        unix_timestamp_ms = self._timestamp_ms(timestamp)
        self._last_energy_update_timestamp_ms = unix_timestamp_ms
        node_state = SatelliteNodeState(
            constellation_id=constellation_id,
            timestamp=unix_timestamp_ms,
            sat_list={}
        )

        # TLE 解析：读取经纬度并填入 SatList
        with open(tle_file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip().startswith("1 ") or l.strip().startswith("2 ")]

        for i in range(0, len(lines) - 1, 2):
            sat_id = str(i // 2)
            lat, lon, _ = calculate_satellite_coordinates(lines[i], lines[i + 1], self.sim_start_utctime)
            # 实例化 NodeAttribute 模型对象
            node_state.sat_list[sat_id] = NodeAttribute(
                latitude=lat,
                longitude=lon,
                flow=0.0,
                energy_ratio=1.0,
                congestion=0.0,
                heat_flow=0.0,
                ambient_temperature=293.15,
                terminal_temperature_list=[293.15],
                node_load_ratio=0.0,
                load_fluctuation=0.0,
            )

        update_ground_station_numbers(
            node_state,
            orbit_height_km=config.ORBIT_HEIGHT_KM,
            min_elevation_deg=self._ground_station_min_elevation_deg,
        )

        # 2. 构建 constellation 对象（用同一份 TLE 文件）
        sats = load_satellites_from_tle(tle_file_path, config.SATS_PER_PLANE)
        self.constellation = CommunicationConstellation(
            sceneId="SCENE1",
            constellationId=constellation_id,
            orbitHeight=config.ORBIT_HEIGHT_KM,
            orbitInclination=config.ORBIT_INCLINATION,
            numOrbitPlane=config.NUM_ORBIT_PLANES,
            numSatsInPlane=config.SATS_PER_PLANE,
            phaseFactor=config.PHASE_FACTOR,
            satelliteList=sats
        )

        # 3. 用 TopologyState dataclass 初始化拓扑，并按 T=0 位置刷新链路距离
        #    （否则首次背景流量与 T=0 预测帧使用的链路距离全为 0）
        topology_state = TopologyState(
            constellation_id=constellation_id,
            timestamp=unix_timestamp_ms,
            new_topology=init_walker_topology(self.constellation)
        )
        self.constellation.update_constellation_time(timestamp)
        update_topology_snapshot(self.constellation, timestamp, topology=topology_state.new_topology)

        # 将初始化数据写入引擎全局状态
        with self.state_rw_lock.write_lock():
            self.topology_state = topology_state
            self.node_state = node_state
            self._last_node_flow_by_sat = {
                sid: float(attr.flow)
                for sid, attr in node_state.sat_list.items()
            }

        # 4. 产生一次初始的背景流量
        try:
            self.background_traffic_update()
            logger.info(f"[{self.current_time}s] 初始化阶段: 首次背景流量生成完成")
        except Exception as e:
            logger.error(f"[{self.current_time}s] 初始化阶段: 背景流量调用异常 - {e}")

        self.task_list = list(task_list or [])
        if self.task_list:
            logger.info(
                f"[{self.current_time}s] 初始化阶段收到 {len(self.task_list)} 个用户业务需求，"
                "已传入拓扑重构接口（当前版本重构决策尚不使用）"
            )

        return self.topology_state, self.node_state, []

    def get_current_states(self):
        """安全地读取当前的核心状态"""
        with self.state_rw_lock.read_lock():
            return self.topology_state, self.node_state

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _coerce_float(value, default=0.0, min_value=None, max_value=None):
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = float(default)
        if min_value is not None:
            result = max(float(min_value), result)
        if max_value is not None:
            result = min(float(max_value), result)
        return result

    @staticmethod
    def _timestamp_ms(value):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return int(value.timestamp() * 1000)

    @staticmethod
    def _sat_id_sort_key(value):
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    def _current_utc(self):
        if self.sim_start_utctime is None:
            return datetime.datetime.now(datetime.timezone.utc)
        return self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)

    @staticmethod
    def _to_utc_datetime(value):
        if isinstance(value, datetime.datetime):
            dt = value
        elif isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.datetime.fromisoformat(text)
        else:
            raise TypeError(f"unsupported datetime value: {type(value).__name__}")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)

    def _should_log_interval(self, interval_seconds, bucket_attr):
        interval = max(int(interval_seconds), 1)
        bucket = int(self.current_time // interval)
        last_bucket = getattr(self, bucket_attr, -1)
        if bucket != last_bucket:
            setattr(self, bucket_attr, bucket)
            return True
        return False

    def _should_log_micro_summary(self):
        return self._should_log_interval(
            self._micro_log_interval_seconds,
            "_last_micro_log_bucket",
        )

    def _sun_unit_vector_ecef(self, current_utc):
        if current_utc.tzinfo is not None:
            current_utc = current_utc.astimezone(datetime.timezone.utc).replace(tzinfo=None)

        year = current_utc.year
        month = current_utc.month
        day = current_utc.day
        hour = (
            current_utc.hour
            + current_utc.minute / 60.0
            + (current_utc.second + current_utc.microsecond / 1_000_000.0) / 3600.0
        )
        if month <= 2:
            year -= 1
            month += 12

        a = year // 100
        b = 2 - a + a // 4
        jd = (
            int(365.25 * (year + 4716))
            + int(30.6001 * (month + 1))
            + day
            + b
            - 1524.5
            + hour / 24.0
        )
        n = jd - 2451545.0
        mean_long = math.radians((280.460 + 0.9856474 * n) % 360.0)
        mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
        ecliptic_long = (
            mean_long
            + math.radians(1.915) * math.sin(mean_anomaly)
            + math.radians(0.020) * math.sin(2.0 * mean_anomaly)
        )
        obliquity = math.radians(23.439 - 0.0000004 * n)
        right_ascension = math.atan2(
            math.cos(obliquity) * math.sin(ecliptic_long),
            math.cos(ecliptic_long),
        )
        declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_long))
        gmst = math.radians(
            (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0
        )
        hour_angle = right_ascension - gmst
        return np.array(
            [
                math.cos(declination) * math.cos(hour_angle),
                math.cos(declination) * math.sin(hour_angle),
                math.sin(declination),
            ],
            dtype=np.float64,
        )

    def _is_satellite_sunlit(self, latitude, longitude, sun_unit):
        orbit_radius = config.EARTH_RADIUS_KM + config.ORBIT_HEIGHT_KM
        lat_rad = math.radians(float(latitude))
        lon_rad = math.radians(float(longitude))
        sat_vec = np.array(
            [
                orbit_radius * math.cos(lat_rad) * math.cos(lon_rad),
                orbit_radius * math.cos(lat_rad) * math.sin(lon_rad),
                orbit_radius * math.sin(lat_rad),
            ],
            dtype=np.float64,
        )
        projection = float(np.dot(sat_vec, sun_unit))
        if projection >= 0.0:
            return True
        perpendicular_sq = float(np.dot(sat_vec, sat_vec) - projection * projection)
        return perpendicular_sq >= config.EARTH_RADIUS_KM * config.EARTH_RADIUS_KM

    def _sync_timestamps_locked(self, current_utc=None):
        current_utc = current_utc or self._current_utc()
        timestamp_ms = self._timestamp_ms(current_utc)
        if self.topology_state is not None:
            self.topology_state.timestamp = timestamp_ms
        if self.node_state is not None:
            self.node_state.timestamp = timestamp_ms
        return timestamp_ms

    def _sync_node_load_metrics_locked(self):
        if self.node_state is None:
            return
        capacity = max(float(getattr(config, "MAX_NODE_FLOW_GBPS", 18.0)), 1e-9)
        for sat_id, sat_attr in self.node_state.sat_list.items():
            current_flow = max(0.0, float(sat_attr.flow))
            prev_flow = self._last_node_flow_by_sat.get(sat_id, current_flow)
            sat_attr.flow = current_flow
            utilization = current_flow / capacity
            sat_attr.congestion = self._clamp01(utilization)
            sat_attr.heat_flow = self._clamp01(utilization)
            sat_attr.node_load_ratio = self._clamp01(utilization)
            sat_attr.load_fluctuation = current_flow - prev_flow
            self._last_node_flow_by_sat[sat_id] = current_flow

    def _sync_node_energy_locked(self, current_utc):
        if self.node_state is None:
            return
        timestamp_ms = self._timestamp_ms(current_utc)
        if self._last_energy_update_timestamp_ms is None:
            self._last_energy_update_timestamp_ms = timestamp_ms
            return

        elapsed_seconds = max(0.0, (timestamp_ms - self._last_energy_update_timestamp_ms) / 1000.0)
        self._last_energy_update_timestamp_ms = timestamp_ms
        if elapsed_seconds <= 0.0:
            return

        discharge_rate = float(getattr(config, "DISCHARGE_RATE_1HZ", 0.00022))
        charge_rate = float(getattr(config, "CHARGE_RATE_1HZ", 2.0 * discharge_rate))
        sun_unit = self._sun_unit_vector_ecef(current_utc)

        for sat_attr in self.node_state.sat_list.values():
            delta = -discharge_rate * elapsed_seconds
            if self._is_satellite_sunlit(sat_attr.latitude, sat_attr.longitude, sun_unit):
                delta += charge_rate * elapsed_seconds
            sat_attr.energy_ratio = self._clamp01(float(sat_attr.energy_ratio) + delta)

    def _laser_config_for_type(self, laser_type):
        """返回该激光器类型的 LaserConfig（同类型共享一个只读实例）。"""
        cache = self.__dict__.setdefault("_laser_config_cache", {})
        if laser_type not in cache:
            preset = self._laser_profile_presets.get(laser_type)
            if preset is None:
                preset = self._laser_profile_presets.get(self._laser_default_type, {})
            cache[laser_type] = LaserConfig.model_validate(dict(preset))
        return cache[laser_type]

    def _sync_topology_load_metrics_locked(self):
        """全网链路状态同步：激光器预设 + 当前几何 + 当前流量 → 物理层/网络层量。

        物理量统一由 :func:`evaluate_link_physics_batch` 计算；激光器类型按链路方向
        确定，拓扑重构新建的链路同样拿到正确的预设。
        """
        if self.topology_state is None:
            return
        keys, attrs = [], []
        for src_id, links in self.topology_state.new_topology.items():
            for dst_id, link_attr in links.items():
                laser_type = laser_type_for_link(src_id, dst_id)
                link_attr.laser_type = laser_type
                link_attr.laser_config = self._laser_config_for_type(laser_type)
                link_attr.current_flow = max(0.0, float(link_attr.current_flow))
                keys.append(f"{src_id}_{dst_id}")
                attrs.append(link_attr)
        if not attrs:
            return

        physics = evaluate_link_physics_batch(
            [a.link_distance for a in attrs],
            [a.laser_config for a in attrs],
            [a.current_flow for a in attrs],
        )
        rated_capacity = max(float(getattr(config, "MAX_LINK_SPEED_GBPS", 100.0)), 1e-9)
        current_sim_time = float(self.current_time)
        delay_total = (
            physics["link_propagation_delay"] + physics["queue_delay"] + physics["transmission_delay"]
        )

        for i, (link_key, link_attr) in enumerate(zip(keys, attrs)):
            current_flow = link_attr.current_flow
            usable_capacity = float(physics["achievable_rate"][i])
            link_attr.max_capacity = rated_capacity
            link_attr.link_capacity = usable_capacity
            link_attr.achievable_rate = usable_capacity

            utilization = current_flow / usable_capacity if usable_capacity > 0.0 else 1.0
            link_attr.left_capacity = self._clamp01(1.0 - utilization)
            link_attr.heat_value = self._clamp01(utilization)
            plr = float(physics["packet_loss_rate"][i])
            link_attr.throughput = min(current_flow, usable_capacity) * (1.0 - plr)

            link_attr.path_loss = float(physics["path_loss"][i])
            link_attr.snr = float(physics["snr"][i])
            link_attr.ber = float(physics["ber"][i])
            link_attr.packet_loss_rate = plr
            link_attr.link_propagation_delay = float(physics["link_propagation_delay"][i])
            link_attr.transmission_delay = float(physics["transmission_delay"][i])
            link_attr.queue_delay = float(physics["queue_delay"][i])
            link_attr.delay = float(delay_total[i])

            # 累计流量：Gbps × s → Gb，/8 → GB
            prev_cum_time = self._last_cum_flow_time_by_key.get(link_key, current_sim_time)
            delta_seconds = max(0.0, current_sim_time - prev_cum_time)
            previous_cum_gb = self._cum_flow_gb_by_key.get(link_key, max(0.0, float(link_attr.cum_flow)))
            cum_flow_gb = previous_cum_gb + current_flow * delta_seconds / 8.0
            link_attr.cum_flow = cum_flow_gb
            self._cum_flow_gb_by_key[link_key] = cum_flow_gb
            self._last_cum_flow_time_by_key[link_key] = current_sim_time

            # 负载波动与时延抖动：相对上一个仿真时刻（同一时刻内多次同步不推进基准）
            history = self._link_history_by_key.get(link_key)
            # history = [上一时刻流量, 上一时刻时延, 当前时刻, 当前流量, 当前时延]
            if history is None or history[2] != current_sim_time:
                previous = (history[3], history[4]) if history is not None else (current_flow, link_attr.delay)
                history = [previous[0], previous[1], current_sim_time, current_flow, link_attr.delay]
            else:
                history[3], history[4] = current_flow, link_attr.delay
            self._link_history_by_key[link_key] = history
            link_attr.load_fluctuation = current_flow - history[0]
            link_attr.delay_jitter = abs(link_attr.delay - history[1])

    def _refresh_ground_station_numbers_locked(self):
        if self.node_state is None:
            return 0
        orbit_height_km = float(getattr(self.constellation, "orbitHeight", config.ORBIT_HEIGHT_KM))
        return update_ground_station_numbers(
            self.node_state,
            orbit_height_km=orbit_height_km,
            min_elevation_deg=self._ground_station_min_elevation_deg,
        )

    def _refresh_ground_station_numbers(self):
        current_utc = self._current_utc()
        with self.state_rw_lock.write_lock():
            matched_ground_station_count = self._refresh_ground_station_numbers_locked()
            self._sync_timestamps_locked(current_utc)
        return matched_ground_station_count
        #     assigned_count = self._refresh_ground_station_numbers_locked()
        #     self._sync_timestamps_locked(current_utc)
        # return assigned_count

    def notify_backend(self, msg_type, data):
        """统一的后端数据回传接口"""
        if self.backend_callback:
            try:
                # 针对巨型数据（如预测矩阵等），向前端仅发送摘要或阶段性通知防止 websocket [Errno 104]
                
                # [新增] 递归将 Pydantic 模型转化为 dict 以支持 JSON 序列化
                def _recursive_dict(obj):
                    if hasattr(obj, "dict"):
                        return _recursive_dict(obj.dict(by_alias=True))
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return _recursive_dict(obj.tolist())
                    elif isinstance(obj, (datetime.datetime, datetime.date)):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: _recursive_dict(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple, set)):
                        return [_recursive_dict(v) for v in obj]
                    return obj
                    
                data = _recursive_dict(data)

                if msg_type == "link_prediction_result":
                    # 方案二：后端定向推送（聚焦强业务/热点链路），防止 websocket [Errno 104]
                    hot_links_topology = []

                    def _to_float_or_none(value):
                        try:
                            f = float(value)
                        except (TypeError, ValueError):
                            return None
                        if not math.isfinite(f):
                            return None
                        return f

                    def _iter_topology_links(topology_dict):
                        if not isinstance(topology_dict, dict):
                            return
                        for src_sat_str, raw_links in topology_dict.items():
                            if isinstance(raw_links, dict):
                                for dst_sat_str, metrics in raw_links.items():
                                    yield src_sat_str, dst_sat_str, metrics
                                continue
                            for dst_item in (raw_links or []):
                                if not isinstance(dst_item, (list, tuple)) or len(dst_item) < 2:
                                    continue
                                yield src_sat_str, dst_item[0], dst_item[1]

                    # 我们过滤出 HeatValue > 0的（或者有实际业务流量趋势的）链路，
                    # 这样可以从14400条直接降至产生拥塞预警的几十条，极大减轻前端压力
                    if isinstance(data, dict) and "LinksPredTopology" in data:
                        for step_data in data["LinksPredTopology"]:
                            load_links_for_step = []
                            step_baseline_precision = _to_float_or_none(step_data.get("BaselinePrecision"))
                            if step_baseline_precision is None:
                                step_baseline_precision = 0.0
                            step_scfe_precision = _to_float_or_none(step_data.get("SCFEPrecision"))
                            if step_scfe_precision is None:
                                step_scfe_precision = 0.0

                            for src_sat_str, dst_sat_str, metrics in _iter_topology_links(step_data.get("Topology", {})):
                                if not isinstance(metrics, dict):
                                    continue
                                link_item = {
                                    "src": src_sat_str,
                                    "dst": dst_sat_str,
                                    "metrics": {
                                        "HeatValue": metrics.get("HeatValue", 0.0),
                                        "QueueDelay": metrics.get("QueueDelay", 0.0),
                                        "LinkAvailability": metrics.get("LinkAvailability", 1.0),
                                        "Survival": metrics.get("Survival", 1.0),
                                        "Capacity": metrics.get("Capacity", 10.0),
                                        "PredFlow": metrics.get("PredFlow", 0.0),
                                        "AchievableRate": metrics.get("AchievableRate", 0.0),
                                        "DelayJitter": metrics.get("DelayJitter", 0.0),
                                    }
                                }
                                load_links_for_step.append(link_item)
                            selected_links = []
                            mode = "load"
                            load_links_for_step.sort(
                                key=lambda x: (
                                    float(x["metrics"].get("PredFlow", 0.0) or 0.0),
                                    float(x["metrics"].get("HeatValue", 0.0) or 0.0),
                                    float(x["metrics"].get("QueueDelay", 0.0) or 0.0),
                                ),
                                reverse=True
                            )
                            if load_links_for_step:
                                # 真正按预测负载排序后发送 Top50，前端取 Top5 做“最大负载链路前5”。
                                selected_links = load_links_for_step[:50]

                            if selected_links:
                                hot_links_topology.append({
                                    "Timestamp": step_data.get("Timestamp"),
                                    "HotLinks": selected_links,
                                    "Mode": mode,
                                    "BaselinePrecision": step_baseline_precision,
                                    "SCFEPrecision": step_scfe_precision,
                                })
                    
                    import json
                    # 将预测步长分为每次5个步长的块进行分批发送，既不截断也不撑爆 WebSocket
                    chunk_size = 5
                    if not hot_links_topology:
                        hot_prediction_payload = {
                            "ConstellationId": data.get("ConstellationId") if isinstance(data, dict) else 0,
                            "Inference_time": data.get("Inference_time") if isinstance(data, dict) else 0,
                            "LinksPredTopology": [],
                            "msg": "No prediction links available",
                            "chunk_index": 1,
                            "total_chunks": 1
                        }
                        self.backend_callback(msg_type, hot_prediction_payload)
                    else:
                        total_chunks = (len(hot_links_topology) + chunk_size - 1) // chunk_size
                        for i in range(0, len(hot_links_topology), chunk_size):
                            chunk = hot_links_topology[i:i + chunk_size]
                            chunk_idx = (i // chunk_size) + 1
                            
                            hot_prediction_payload = {
                                "ConstellationId": data.get("ConstellationId") if isinstance(data, dict) else 0,
                                "Inference_time": data.get("Inference_time") if isinstance(data, dict) else 0,
                                "LinksPredTopology": chunk,
                                "msg": f"Filtered hot links part {chunk_idx}/{total_chunks}",
                                "chunk_index": chunk_idx,
                                "total_chunks": total_chunks
                            }
                            
                            payload_size = len(json.dumps(hot_prediction_payload))
                            logger.debug(f"[Engine] HotLinks payload size (chunk {chunk_idx}/{total_chunks}): {payload_size} bytes")
                            if payload_size > 500000:
                                logger.warning(f"[Engine] Warning: HotLinks payload chunk {chunk_idx} is extremely large, might cause WS Connection reset!")
                            self.backend_callback(msg_type, hot_prediction_payload)
                else:
                    self.backend_callback(msg_type, data)
            except Exception as e:
                logger.error(f"[Engine] 回调后端失败: {e}")

    # """
    # 根据zz的拓扑状态更新之后接入
    # """

    def update_topology_state(self, new_topology: TopologyState):
        """拓扑状态更新入口，写入时自动更新 Timestamp 为当前帧 Unix ms"""
        with self.state_rw_lock.write_lock():
            self.topology_state = new_topology
            current_utc = self._current_utc()
            self._sync_topology_load_metrics_locked()
            self._sync_timestamps_locked(current_utc)
            
            # --- 构造批量数据推入 Celery 落库字典 ---
            try:
                from simulation_api.db_services import save_realtime_link_states
                c_id = self.topology_state.constellation_id
                t_ms = self.topology_state.timestamp
                
                # 【新增】周期性(如每 15 仿真秒)强制清空一次哈希缓存下发全量，解决前端中途重连后链路丢失的问题
                if getattr(self, "_last_full_sync_tick", -1) != (self.current_time // 15):
                    self._last_notified_links.clear()
                    self._last_full_sync_tick = self.current_time // 15

                link_list = []
                delta_links = []
                for src_id, links in self.topology_state.new_topology.items():
                    for dst_id, attr in links.items():
                        if attr.link_distance > 0:  # 排除未建链的脏数据
                            link_dict = {
                                "src_sat": src_id,
                                "dst_sat": dst_id,
                                "srcSat": src_id,
                                "dstSat": dst_id,
                                "laser_type": attr.laser_type,
                                "laser_config": attr.laser_config.dict(by_alias=True),
                                "distance": attr.link_distance,
                                "link_distance": attr.link_distance,
                                "current_flow": attr.current_flow,
                                "left_capacity": attr.left_capacity,
                                "link_capacity": attr.link_capacity,
                                "max_capacity": attr.max_capacity,
                                "cum_flow": attr.cum_flow,
                                "path_loss": attr.path_loss,
                                "snr": attr.snr,
                                "link_propagation_delay": attr.link_propagation_delay,
                                "queue_delay": attr.queue_delay,
                                "transmission_delay": attr.transmission_delay,
                                "delay": attr.delay,
                                "delay_jitter": attr.delay_jitter,
                                "ber": attr.ber,
                                "packet_loss_rate": attr.packet_loss_rate,
                                "throughput": attr.throughput,
                                "heat_value": attr.heat_value,
                                "achievable_rate": attr.achievable_rate,
                                "load_fluctuation": attr.load_fluctuation,
                                # CamelCase aliases for direct UI consumption.
                                "LaserConfig": attr.laser_config.dict(by_alias=True),
                                "LaserType": attr.laser_type,
                                "LinkDistance": attr.link_distance,
                                "MaxCapacity": attr.max_capacity,
                                "CurrentFlow": attr.current_flow,
                                "CumFlow": attr.cum_flow,
                                "LeftCapacity": attr.left_capacity,
                                "LinkCapacity": attr.link_capacity,
                                "PathLoss": attr.path_loss,
                                "SNR": attr.snr,
                                "LinkPropagationDelay": attr.link_propagation_delay,
                                "QueueDelay": attr.queue_delay,
                                "TransmissionDelay": attr.transmission_delay,
                                "Delay": attr.delay,
                                "DelayJitter": attr.delay_jitter,
                                "BER": attr.ber,
                                "PacketLossRate": attr.packet_loss_rate,
                                "Throughput": attr.throughput,
                                "HeatValue": attr.heat_value,
                                "AchievableRate": attr.achievable_rate,
                                "LoadFluctuation": attr.load_fluctuation,
                            }
                            link_list.append(link_dict)
                            
                            # 计算增量供前端更新
                            link_key = f"{src_id}_{dst_id}"
                            hash_val = hash(
                                f"{attr.current_flow}_{attr.left_capacity}_{attr.link_capacity}_{attr.max_capacity}_"
                                f"{attr.link_distance}_{attr.path_loss}_{attr.snr}_{attr.link_propagation_delay}_"
                                f"{attr.queue_delay}_{attr.transmission_delay}_{attr.delay}_{attr.delay_jitter}_"
                                f"{attr.ber}_{attr.packet_loss_rate}_{attr.throughput}_{attr.heat_value}_"
                                f"{attr.achievable_rate}_{attr.load_fluctuation}_{attr.cum_flow}"
                            )
                            if self._last_notified_links.get(link_key) != hash_val:
                                delta_links.append(link_dict)
                                self._last_notified_links[link_key] = hash_val

                if link_list:
                    # 将更新时刻的 **链路增量** 状态通过 WebSocket 实时推送前端，防止数据过大
                    if delta_links:
                        # 【新增】基于前端要求，针对链路增量进行分片限制（最大500条/片），防止 WebSocket 超限断开
                        chunk_size = 500
                        total_chunks = (len(delta_links) + chunk_size - 1) // chunk_size
                        for i in range(0, len(delta_links), chunk_size):
                            chunk = delta_links[i:i + chunk_size]
                            chunk_idx = (i // chunk_size) + 1
                            self.notify_backend("topology_state_update", {
                                "constellation_id": c_id,
                                "timestamp": t_ms,
                                "links": chunk,
                                "chunk_index": chunk_idx,
                                "total_chunks": total_chunks
                            })
                    
                    # 数据库始终落库完整数据
                    save_realtime_link_states.delay(c_id, t_ms, link_list)
            except Exception as e:
                logger.error(f"[Engine] Async topology save failed: {e}")

        # print(f"[{self.current_time}s] 拓扑状态已更新")

    def update_node_state(self, new_node_data: SatelliteNodeState):
        """节点状态更新入口，写入时自动更新 Timestamp 为当前帧 Unix ms"""
        with self.state_rw_lock.write_lock():
            if self.node_state is None:
                self.node_state = new_node_data
            else:
                # 更新 sat_list 内容
                self.node_state.sat_list = new_node_data.sat_list
            current_utc = self._current_utc()
            self._sync_node_load_metrics_locked()
            self._sync_timestamps_locked(current_utc)
            
            # --- 构造批量数据推入 Celery 落库字典 ---
            try:
                from simulation_api.db_services import save_realtime_node_states
                c_id = self.node_state.constellation_id
                t_ms = self.node_state.timestamp
                
                node_list = []
                for sid, sdata in self.node_state.sat_list.items():
                    node_dict = {
                        "sat_id": sid,
                        "latitude": sdata.latitude,
                        "longitude": sdata.longitude,
                        "flow": sdata.flow,
                        "energy_ratio": sdata.energy_ratio,
                        "congestion": sdata.congestion,
                        "heat_flow": sdata.heat_flow,
                        "ambient_temperature": sdata.ambient_temperature,
                        "terminal_temperature_list": sdata.terminal_temperature_list,
                        "node_load_ratio": sdata.node_load_ratio,
                        "load_fluctuation": sdata.load_fluctuation,
                        "ground_station_number": sdata.ground_station_number
                    }
                    node_list.append(node_dict)

                if node_list:
                    # 3600 节点以内采用全量推送，前端通过 store 全量覆盖，避免增量错位
                    self.notify_backend("node_state_update", {
                        "constellation_id": c_id,
                        "timestamp": t_ms,
                        "nodes": node_list
                    })
                    
                    # 数据库始终落库完整数据
                    save_realtime_node_states.delay(c_id, t_ms, node_list)
            except Exception as e:
                logger.error(f"[Engine] Async node state save failed: {e}")

        # print(f"[{self.current_time}s] 节点状态已更新")

    @staticmethod
    def _printable_metric(value, digits=6):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [SimulationEngine._printable_metric(item, digits) for item in value]
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _format_path(path, max_nodes=12):
        nodes = [str(item) for item in (path or [])]
        if not nodes:
            return "[]"
        if len(nodes) <= max_nodes:
            return " -> ".join(f"#{node}" for node in nodes)

        head_count = max_nodes // 2
        tail_count = max_nodes - head_count
        trimmed_nodes = nodes[:head_count] + ["..."] + nodes[-tail_count:]
        return " -> ".join(f"#{node}" if node != "..." else "..." for node in trimmed_nodes)

    def _print_topology_reconstruction_result(self, result_dict):
        if not isinstance(result_dict, dict):
            return

        constellation_id = result_dict.get("ConstellationId")
        timestamp = result_dict.get("Timestamp")
        topology_id = result_dict.get("TopologyId")
        topology_update = result_dict.get("TopologyUpdate")
        top_qualities = result_dict.get("TopQualities", {})

        logger.info(f"[{self.current_time}s] [TopReconstruction] ConstellationId={constellation_id}, Timestamp={timestamp}, TopologyId={topology_id}, TopologyUpdate={topology_update}")
        logger.debug(f"[{self.current_time}s] [TopReconstruction] TopQualities=")
        logger.debug(pformat(top_qualities, width=120, compact=True))



    def _print_snapshot(self, label="微周期", prev_time=None):
        """
        物理量快照输出。每次调用打印当前帧的关键数值：
        - 卫星节点状态：总流量、活跃卫星数、TOP3高负载卫星（含经纬度）
        - 链路状态：有效链路数、最短/最长/均值距离、时延样本
        - 链路流量：TOP3高流量链路
        返回当前时间戳供下一帧计算帧间隔。
        """
        topo, node = self.get_current_states()
        if node is None or topo is None:
            return time.time()

        # ── 卫星流量统计（来自 node_state.sat_list）──────────────
        sat_list = node.sat_list
        total_sats = len(sat_list)

        # 使用 Pydantic BaseModel 对象的下划线属性访问
        sat_data = [(sid,
                     sdata.latitude,
                     sdata.longitude,
                     sdata.flow,
                     sdata.energy_ratio,
                     sdata.congestion,
                     sdata.heat_flow,
                     sdata.ground_station_number)
                    for sid, sdata in sat_list.items()]
        active = [(sid, lat, lon, f, er, cg, hf, gsn) for sid, lat, lon, f, er, cg, hf, gsn in sat_data if f > 0]
        total_flow = sum(f for _, _, _, f, _, _, _, _ in sat_data)
        top3_sats = sorted(active, key=lambda x: x[3], reverse=True)[:3]

        now = time.time()
        interval_str = f"  Δt={now - prev_time:.2f}s" if prev_time is not None else ""
        logger.debug(f"\n[T={self.current_time}s] {label}{interval_str}")
        logger.debug(f"  卫星  总流量={total_flow:.1f}Gbps  活跃={len(active)}/{total_sats}")
        rank_labels = ["第一活跃卫星", "第二活跃卫星", "第三活跃卫星"]
        
        first_active_sat_id = None
        for i, (sid, lat, lon, f, er, cg, hf, gsn) in enumerate(top3_sats):
            logger.debug(f"        {rank_labels[i]}: Sat#{sid:<5} 纬度={lat:+6.1f}°  经度={lon:+7.1f}°  流量={f:.3f}Gbps")
            
            # 记录第一活跃卫星ID，用于后续打印详细状态
            if i == 0:
                first_active_sat_id = sid
        
        # 打印第一活跃卫星的详细节点状态表
        if first_active_sat_id and first_active_sat_id in sat_list:
            first_sat_attr = sat_list[first_active_sat_id]
            ground_station_numbers = first_sat_attr.ground_station_number
            logger.debug(f"\n        [第一活跃卫星 #{first_active_sat_id} 详细状态]")
            logger.debug(f"          Latitude:              {first_sat_attr.latitude:+.6f}°")
            logger.debug(f"          Longitude:             {first_sat_attr.longitude:+.6f}°")
            logger.debug(f"          Flow:                  {first_sat_attr.flow:.6f} Gbps")
            logger.debug(f"          EnergyRatio:           {first_sat_attr.energy_ratio:.6f}")
            logger.debug(f"          Congestion:            {first_sat_attr.congestion:.6f}")
            logger.debug(f"          HeatFlow:              {first_sat_attr.heat_flow:.6f}")
            logger.debug(f"          GroundStationNumber:   {first_sat_attr.ground_station_number if first_sat_attr.ground_station_number is not None else 'None'}")

        # ── 链路距离统计（来自 topology_state.new_topology）────────────────
        new_topology = topo.new_topology
        all_links, flow_links = [], []
        for u_id, links in new_topology.items():
            for target_id, attr in links.items():  # 字典套 LinksQualitiesValue 对象
                # 使用对象属性访问
                dist = attr.link_distance
                delay = attr.link_propagation_delay
                flow = attr.current_flow
                if dist > 0:
                    all_links.append((u_id, target_id, dist, delay))
                if flow > 0:
                    flow_links.append((u_id, target_id, flow, attr))

        if all_links:
            min_l = min(all_links, key=lambda x: x[2])
            max_l = max(all_links, key=lambda x: x[2])
            avg_d = sum(x[2] for x in all_links) / len(all_links)
            logger.debug(f"  链路  有效={len(all_links)}条  "
                  f"最短链路={min_l[2]:.1f}km(#{min_l[0]}→#{min_l[1]})  "
                  f"最长链路={max_l[2]:.1f}km  均值={avg_d:.1f}km  "
                  f"时延样本={all_links[0][3]:.3f}ms")
        else:
            logger.debug(f"  链路  LinkDistance 全为 0，位置同步未生效")

        # ── TOP3 链路流量（来自 topology_state["newTopology"] CurrentFlow）────
        if flow_links:
            top3_lf = sorted(flow_links, key=lambda x: x[2], reverse=True)[:3]
            lf_str = "  ".join(f"#{u}→#{v}({f:.3f}Gbps)" for u, v, f, _ in top3_lf)
            logger.debug(f"  链路流量  {lf_str}")
            
            # 打印第一链路流量的详细拓扑链路状态表
            if top3_lf:
                first_link_u, first_link_v, first_link_flow, first_link_attr = top3_lf[0]
                logger.debug(f"\n        [第一链路流量 #{first_link_u}→#{first_link_v} 详细状态]")
                logger.debug(f"          LinkDistance:            {first_link_attr.link_distance:.6f} km")
                logger.debug(f"          LinkCapacity:            {first_link_attr.link_capacity:.6f} Gbps")
                logger.debug(f"          LeftCapacity:            {first_link_attr.left_capacity:.6f}")
                logger.debug(f"          CurrentFlow:             {first_link_attr.current_flow:.6f} Gbps")
                logger.debug(f"          LinkPropagationDelay:    {first_link_attr.link_propagation_delay:.6f} ms")
                logger.debug(f"          QueueDelay:              {first_link_attr.queue_delay:.6f} ms")
                logger.debug(f"          TransmissionDelay:       {first_link_attr.transmission_delay:.6f} ms")
                logger.debug(f"          PacketLossRate:          {first_link_attr.packet_loss_rate:.8f}")
                logger.debug(f"          HeatValue:               {first_link_attr.heat_value:.6f}")

        return now

    def _print_macro_snapshot(self):
        """
        宏周期完成后打印链路物理量摘要（距离、时延）。
        详细的卫星流量信息在紧随其后的微周期 _print_snapshot 中输出。
        """
        topo, _ = self.get_current_states()
        if topo is None:
            return
        new_topology = topo.new_topology
        all_links = [
            (u, t, a.link_distance, a.link_propagation_delay)
            for u, links in new_topology.items()
            for t, a in links.items()
            if a.link_distance > 0
        ]
        if all_links:
            min_l = min(all_links, key=lambda x: x[2])
            max_l = max(all_links, key=lambda x: x[2])
            logger.info(f"  拓扑重构完成 | 链路={len(all_links)}条 | "
                  f"最短={min_l[2]:.1f}km(时延={min_l[3]:.3f}ms)  "
                  f"最长={max_l[2]:.1f}km(时延={max_l[3]:.3f}ms)")
        else:
            logger.info(f"  拓扑重构完成 | 链路距离尚未更新")
            
        # ==========================================
        # [新增]：每次宏周期完成重构后，触发前端 KPI 关键指标大屏的异步计算与 WebSocket 推送
        # ==========================================
        try:
            from simulation_api.db_services import push_constellation_kpi_dashboard
            push_constellation_kpi_dashboard.delay()
        except Exception as e:
            logger.error(f"[{self.current_time}s] KPI 大屏看板推送任务触发异常: {e}")

    def sensing_update(self):
        """感知监测：读取当前仿真状态，输出表2-1 参数与表3-2 覆盖/采样/方法状态。"""
        topology_state, node_state = self.get_current_states()
        status = run_sensing_cycle(self.sensing_monitor, topology_state, node_state, self.current_time)
        self.latest_sensing_status = status
        self.notify_backend("sensing_status", status)
        return status

    def link_state_records_update(self, predicted_links):
        """把 SCFE 预测适配为表2-2 单链路记录（物理量取自仿真链路状态）。"""
        topology_state, _ = self.get_current_states()
        records = adapt_scfe_output(predicted_links, link_states_from_topology(topology_state))
        self.latest_link_state_records = records
        return records

    def capacity_evaluation_update(self, baseline_topology_state, predicted_links, window_start_s):
        """容量分析与效能评估（表2-19/2-20）：重构前后拓扑对比 + 逐秒链路可用率。

        业务矩阵：用户业务需求接入前，取拓扑重构所用的核心节点源-目的对，每对 1 Gbps；
        λ* 按链路实际可实现速率（仿真 LinkCapacity）搜索。
        """
        topology_state, node_state = self.get_current_states()
        recon_config = TopReconstructionConfig()
        intensity = compute_business_demand_intensity(
            predicted_links=predicted_links,
            initial_topology=normalize_topology(baseline_topology_state),
            horizon_steps=recon_config.business_intensity_horizon_steps,
            decay_lambda=recon_config.business_intensity_decay_lambda,
            max_link_capacity_gbps=config.MAX_LINK_SPEED_GBPS,
        )
        pairs = build_business_demand_source_destination_pairs(
            node_state=node_state,
            satellite_business_intensity=intensity.satellite_intensity,
            core_node_count=recon_config.core_node_count(self.constellation_id),
        )
        traffic_matrix = [{"src": str(a), "dst": str(b), "demand_gbps": 1.0} for a, b in pairs]
        topology = topology_state.new_topology
        baseline = baseline_topology_state.new_topology
        # baseline = init_walker_topology(self.constellation)
        update_topology_snapshot(self.constellation, self._current_utc(), topology=baseline)
        result = evaluate_capacity(
            topology=topology,
            traffic_matrix=traffic_matrix,
            capacity_protocol={"LinkCapacity": config.MAX_LINK_SPEED_GBPS, "SearchPrecision": 0.01},
            availability_samples=self.sensing_monitor.availability_samples(since_s=window_start_s),
            planned_links=[f"{src}->{dst}" for src, links in topology.items() for dst in links],
            baseline_topology=None if baseline is topology else baseline,
            # baseline_topology= baseline,
            evaluation_window={"StartTime": window_start_s, "EndTime": self.current_time},
            frame_seconds=1.0,
            timestamp_ms=self._timestamp_ms(self._current_utc()),
        )

        # 始终发布 v2 通道，供新前端按需渐进接入（默认前端仍消费 legacy 通道）。
        self.notify_backend("capacity_evaluation_v2", result)
        # 无论是否触发 legacy 前端推送，都更新 latest 缓存，
        # 以保证 /api/simulation/capacity/latest 可返回最近一次容量评估结果。
        self.latest_capacity_evaluation = result

        gain_pct_raw = ((result.get("Capacity") or {}).get("CapacityGainPct"))
        try:
            gain_pct = float(gain_pct_raw) if gain_pct_raw is not None else None
        except (TypeError, ValueError):
            gain_pct = None

        if gain_pct is not None and gain_pct <= 20.0:
            logger.info(
                f"[{self.current_time}s] [CapacityEvaluation] low gain but still emit frontend event: "
                f"GainPct={gain_pct_raw}"
            )
        self.notify_backend("capacity_evaluation", result)
        return result

    def background_traffic_update(self):
        """
        step1: 获取当前状态
        step2: 调用背景流量算法
        step3: 更新拓扑和节点状态
        返回: background_traffic_obj, ret_timestamp
        """
        topology_state, node_state = self.get_current_states()
        current_timestamp = self._current_utc()

        new_top, new_node, background_traffic_obj, ret_timestamp = background_traffic_generation(
            topology_state, node_state, current_timestamp
        )
        self.update_topology_state(new_top)
        self.update_node_state(new_node)
        return background_traffic_obj, ret_timestamp

    def _update_satellite_positions(self):
        """
        根据当前逻辑时间更新所有卫星经纬度及链路距离。
        直接把 local_topo 传给 update_topology_snapshot，在原地更新链路属性，
        不依赖全局 newTopology 变量，避免初始化时全局变量为空的问题。
        """
        current_utc = self._current_utc()

        with self.state_rw_lock.write_lock():
            if self.constellation is not None:
                self.constellation.update_constellation_time(current_utc)

            # 1. 获取本地拓扑，直接传给 update_topology_snapshot 在原地更新链路距离
            local_topo = self.topology_state.new_topology
            update_topology_snapshot(self.constellation, current_utc, topology=local_topo)

            # 2. 同步卫星经纬度回 node_state - 使用下划线属性访问
            sat_map = {sat.satId: sat for sat in self.constellation.satelliteList}
            for sat_id_int, sat in sat_map.items():
                sid = str(sat_id_int)
                if sid in self.node_state.sat_list:
                    self.node_state.sat_list[sid].latitude = sat.latitude
                    self.node_state.sat_list[sid].longitude = sat.longitude

                    # self.node_state.sat_list[sid].flow      = sat.flow
                    # self.node_state.sat_list[sid].energy_ratio = sat.energy_ratio
                    # self.node_state.sat_list[sid].congestion = sat.congestion
                    # self.node_state.sat_list[sid].heat_flow = sat.heat_flow

            self._sync_node_energy_locked(current_utc)
            self._sync_node_load_metrics_locked()
            self._sync_topology_load_metrics_locked()
            self._sync_timestamps_locked(current_utc)

    def _build_prediction_frame(self):
        """
        Build one link-level feature frame for SCFE.
        Returns:
            {
                "features": np.ndarray, shape=(link_num, 11), dtype=float32,
                "link_index": list[tuple[str, str]]
            }
        """
        topo, node = self.get_current_states()
        if topo is None or node is None:
            return {
                "features": np.empty((0, 11), dtype=np.float32),
                "link_index": [],
                "link_availability": {},
            }

        new_topo = topo.new_topology or {}
        sat_list = node.sat_list or {}

        current_utc = self._current_utc()
        minute_of_day = (
            current_utc.hour * 60
            + current_utc.minute
            + current_utc.second / 60.0
            + current_utc.microsecond / 60_000_000.0
        )
        time_feature = minute_of_day / 1440.0
        num_planes = max(int(getattr(config, "NUM_ORBIT_PLANES", 1)), 1)
        sats_per_plane = max(int(getattr(config, "SATS_PER_PLANE", 1)), 1)
        max_link_speed = max(float(getattr(config, "MAX_LINK_SPEED_GBPS", 10.0)), 1e-9)

        frame_rows = []
        link_index = []
        actual_availability_by_link = {}
        actual_flow_by_link = {}

        for src_id in sorted(new_topo.keys(), key=self._sat_id_sort_key):
            src_sat = sat_list.get(str(src_id))
            if src_sat is None:
                continue

            src_int = int(src_id)
            src_plane = (src_int // sats_per_plane) / num_planes
            src_seat = (src_int % sats_per_plane) / sats_per_plane
            src_lat = float(src_sat.latitude)
            src_lon = float(src_sat.longitude)

            links = new_topo.get(src_id, {})
            for dst_id in sorted(links.keys(), key=self._sat_id_sort_key):
                dst_sat = sat_list.get(str(dst_id))
                if dst_sat is None:
                    continue

                dst_int = int(dst_id)
                dst_plane = (dst_int // sats_per_plane) / num_planes
                dst_seat = (dst_int % sats_per_plane) / sats_per_plane
                dst_lat = float(dst_sat.latitude)
                dst_lon = float(dst_sat.longitude)
                link_attr = links[dst_id]
                current_flow = self._coerce_float(
                    getattr(link_attr, "current_flow", 0.0),
                    default=0.0,
                    min_value=0.0,
                )
                flow_feature = min(current_flow / max_link_speed, 1.0)

                # 与 LinkPredictor 的 LinkAvailability 同一口径：
                # 0.5·剩余容量比例(相对额定容量) + 0.5·链路生存度(物理速率/额定速率)
                rated = max(float(getattr(link_attr, "max_capacity", max_link_speed)), 1e-9)
                capacity_norm = self._clamp01((rated - current_flow) / rated)
                survival = self._clamp01(float(getattr(link_attr, "achievable_rate", 0.0)) / rated)
                actual_availability = self._clamp01(0.5 * capacity_norm + 0.5 * survival)


                frame_rows.append([
                    src_plane,
                    src_seat,
                    src_lat,
                    src_lon,
                    time_feature,
                    dst_plane,
                    dst_seat,
                    dst_lat,
                    dst_lon,
                    time_feature,
                    flow_feature,
                ])
                link_index.append((str(src_id), str(dst_id)))
                actual_availability_by_link[(str(src_id), str(dst_id))] = actual_availability
                actual_flow_by_link[(str(src_id), str(dst_id))] = current_flow


        features = (
            np.asarray(frame_rows, dtype=np.float32)
            if frame_rows
            else np.empty((0, 11), dtype=np.float32)
        )

        self._evaluate_prediction_availability_frame(
            current_utc,
            actual_availability_by_link,
            actual_flow_by_link,
        )

        return {
            "features": features,
            "link_index": link_index,
            "link_availability": actual_availability_by_link,

        }
    
    def _link_availability_threshold(self):
        return self._clamp01(
            getattr(config, "LINK_AVAILABILITY_THRESHOLD", 0.8)
        )

    def _prediction_timestamp_key(self, value):
        if isinstance(value, datetime.datetime):
            dt = value
        else:
            dt = self._to_utc_datetime(value)
        return int(round(dt.timestamp() * 1000))

    def _flow_precision_tolerance_gbps(self):
        predictor = getattr(self, "link_predictor", None)
        table = getattr(predictor, "FLOW_TOLERANCE_GBPS", None)
        if isinstance(table, dict):
            raw = table.get(str(self.constellation_id))
            try:
                val = float(raw)
                if math.isfinite(val) and val >= 0.0:
                    return val
            except (TypeError, ValueError):
                pass
        return 0.0

    def _extract_predicted_flow_by_timestamp(self, predicted_links):
        if not isinstance(predicted_links, dict):
            return {}, {}

        debug = predicted_links.get("_debug_info") or {}
        forecasts = debug.get("flow_forecasts_gbps") or {}
        link_index = forecasts.get("link_index") or []
        baseline = forecasts.get("baseline")
        scfe = forecasts.get("scfe")
        timestamps = predicted_links.get("Timestamp") or []

        try:
            baseline_arr = np.asarray(baseline, dtype=np.float64)
            scfe_arr = np.asarray(scfe, dtype=np.float64)
        except Exception:
            return {}, {}

        if baseline_arr.ndim != 2 or scfe_arr.ndim != 2:
            return {}, {}
        if baseline_arr.shape != scfe_arr.shape:
            return {}, {}
        if baseline_arr.shape[0] != len(link_index):
            return {}, {}
        if baseline_arr.shape[1] != len(timestamps):
            return {}, {}

        baseline_by_ts = {}
        scfe_by_ts = {}
        for step, ts in enumerate(timestamps):
            ts_key = self._prediction_timestamp_key(ts)
            baseline_map = {}
            scfe_map = {}
            for row, link in enumerate(link_index):
                try:
                    src_id, dst_id = link
                except (TypeError, ValueError):
                    continue
                key = (str(src_id), str(dst_id))
                b = baseline_arr[row, step]
                s = scfe_arr[row, step]
                if math.isfinite(float(b)):
                    baseline_map[key] = float(b)
                if math.isfinite(float(s)):
                    scfe_map[key] = float(s)
            if baseline_map:
                baseline_by_ts[ts_key] = baseline_map
            if scfe_map:
                scfe_by_ts[ts_key] = scfe_map
        return baseline_by_ts, scfe_by_ts

    def _extract_predicted_availability_by_timestamp(self, predicted_links):
        if not isinstance(predicted_links, dict):
            return {}

        predicted_by_timestamp = {}
        for step in predicted_links.get("LinksPredTopology", []) or []:
            if not isinstance(step, dict):
                continue
            timestamp = step.get("Timestamp")
            if timestamp is None:
                continue

            topology = step.get("Topology") or {}
            if not isinstance(topology, dict):
                continue

            link_map = {}
            for src_id, raw_links in topology.items():
                iterable = raw_links.items() if isinstance(raw_links, dict) else (raw_links or [])
                for item in iterable:
                    try:
                        dst_id, link_state = item
                    except (TypeError, ValueError):
                        continue

                    if isinstance(link_state, dict):
                        raw_availability = link_state.get("LinkAvailability")
                    else:
                        raw_availability = getattr(
                            link_state,
                            "LinkAvailability",
                            getattr(link_state, "link_availability", None),
                        )
                    link_map[(str(src_id), str(dst_id))] = self._coerce_float(
                        raw_availability,
                        default=0.0,
                        min_value=0.0,
                        max_value=1.0,
                    )

            if link_map:
                predicted_by_timestamp[self._prediction_timestamp_key(timestamp)] = link_map

        return predicted_by_timestamp

    def _start_prediction_accuracy_if_applicable(self, predicted_links, reconstruction_result):
        self._active_prediction_accuracy = None
        if not predicted_links:
            return

        topology_update = 0
        if isinstance(reconstruction_result, dict):
            topology_update = int(reconstruction_result.get("TopologyUpdate", 0) or 0)

        if topology_update != 0:
            logger.info(
                f"[{self.current_time}s] [LinkPredictionAccuracy] TopologyUpdate=1, "
                "skip availability accuracy for this macro window."
            )
            return

        predicted_by_timestamp = self._extract_predicted_availability_by_timestamp(predicted_links)
        baseline_flow_by_ts, scfe_flow_by_ts = self._extract_predicted_flow_by_timestamp(predicted_links)
        expected_timestamps = set(predicted_by_timestamp.keys()) | set(baseline_flow_by_ts.keys()) | set(scfe_flow_by_ts.keys())
        if not expected_timestamps:
            logger.info(
                f"[{self.current_time}s] [LinkPredictionAccuracy] No predicted "
                "samples to evaluate."
            )
            return

        threshold = self._link_availability_threshold()
        tolerance = self._flow_precision_tolerance_gbps()
        self._active_prediction_accuracy = {
            "trigger_time": self.current_time,
            "window_start": self.current_time,
            "window_end": self.current_time + int(getattr(config, "MACRO_PERIOD_SECONDS", 30)),
            "threshold": threshold,
            "tolerance_gbps": tolerance,
            "predicted_by_timestamp": predicted_by_timestamp,
            "baseline_flow_by_ts": baseline_flow_by_ts,
            "scfe_flow_by_ts": scfe_flow_by_ts,
            "expected_timestamps": expected_timestamps,
            "evaluated_timestamps": set(),
            "frame_results": [],
            "correct": 0,
            "total": 0,
            "baseline_correct": 0,
            "baseline_total": 0,
            "scfe_correct": 0,
            "scfe_total": 0,
        }
        logger.info(
            f"[{self.current_time}s] [LinkPredictionAccuracy] Start evaluating "
            f"{len(expected_timestamps)} predicted frames with "
            f"threshold={threshold:.3f}, tolerance={tolerance:.3f}Gbps."
        )

    def _evaluate_prediction_availability_frame(self, timestamp, actual_availability_by_link, actual_flow_by_link=None):
        active = self._active_prediction_accuracy
        if not active:
            return

        timestamp_key = self._prediction_timestamp_key(timestamp)
        predicted_map = active["predicted_by_timestamp"].get(timestamp_key)
        if predicted_map is None or timestamp_key in active["evaluated_timestamps"]:
            return

        threshold = active["threshold"]
        correct = 0
        total = 0
        for link_key, predicted_availability in predicted_map.items():
            actual_availability = actual_availability_by_link.get(link_key)
            if actual_availability is None:
                continue
            predicted_label = 1 if predicted_availability >= threshold else 0
            actual_label = 1 if actual_availability >= threshold else 0
            correct += int(predicted_label == actual_label)
            total += 1

        active["correct"] += correct
        active["total"] += total

        baseline_map = active["baseline_flow_by_ts"].get(timestamp_key) or {}
        scfe_map = active["scfe_flow_by_ts"].get(timestamp_key) or {}
        tolerance = float(active.get("tolerance_gbps", 0.0))
        baseline_correct = 0
        baseline_total = 0
        scfe_correct = 0
        scfe_total = 0
        actual_flow_by_link = actual_flow_by_link or {}

        for link_key, predicted_flow in baseline_map.items():
            actual_flow = actual_flow_by_link.get(link_key)
            if actual_flow is None:
                continue
            baseline_total += 1
            baseline_correct += int(abs(predicted_flow - actual_flow) <= tolerance)

        for link_key, predicted_flow in scfe_map.items():
            actual_flow = actual_flow_by_link.get(link_key)
            if actual_flow is None:
                continue
            scfe_total += 1
            scfe_correct += int(abs(predicted_flow - actual_flow) <= tolerance)

        active["baseline_correct"] += baseline_correct
        active["baseline_total"] += baseline_total
        active["scfe_correct"] += scfe_correct
        active["scfe_total"] += scfe_total

        active["evaluated_timestamps"].add(timestamp_key)
        active["frame_results"].append({
            "Timestamp": timestamp.isoformat(),
            "Correct": correct,
            "Total": total,
            "BaselineCorrect": baseline_correct,
            "BaselineTotal": baseline_total,
            "SCFECorrect": scfe_correct,
            "SCFETotal": scfe_total,
        })

        if active["expected_timestamps"].issubset(active["evaluated_timestamps"]):
            self._finalize_active_prediction_accuracy()

    def _finalize_active_prediction_accuracy(self, force=False):
        active = self._active_prediction_accuracy
        if not active:
            return
        if (
            not force
            and not active["expected_timestamps"].issubset(active["evaluated_timestamps"])
        ):
            return

        total = active["total"]
        accuracy = (active["correct"] / total) if total > 0 else None
        baseline_total = active["baseline_total"]
        scfe_total = active["scfe_total"]
        baseline_precision = (active["baseline_correct"] / baseline_total) if baseline_total > 0 else None
        scfe_precision = (active["scfe_correct"] / scfe_total) if scfe_total > 0 else None
        self._pending_prediction_accuracy = {
            "TriggerTime": active["trigger_time"],
            "WindowStart": active["window_start"],
            "WindowEnd": active["window_end"],
            "Threshold": active["threshold"],
            "ToleranceGbps": active.get("tolerance_gbps", 0.0),
            "Accuracy": scfe_precision if scfe_precision is not None else accuracy,
            "AvailabilityAccuracy": accuracy,
            "BaselinePrecision": baseline_precision,
            "SCFEPrecision": scfe_precision,
            "BaselineCorrect": active["baseline_correct"],
            "BaselineTotal": baseline_total,
            "SCFECorrect": active["scfe_correct"],
            "SCFETotal": scfe_total,
            "Correct": active["correct"],
            "Total": total,
            "EvaluatedFrames": len(active["evaluated_timestamps"]),
            "ExpectedFrames": len(active["expected_timestamps"]),
            "FrameResults": active["frame_results"],
        }
        self._active_prediction_accuracy = None

    def _print_pending_prediction_accuracy(self):
        summary = self._pending_prediction_accuracy
        if not summary:
            return

        accuracy = summary.get("Accuracy")
        baseline_precision = summary.get("BaselinePrecision")
        scfe_precision = summary.get("SCFEPrecision")
        accuracy_text = "N/A" if accuracy is None else f"{accuracy * 100:.2f}%"
        baseline_text = "N/A" if baseline_precision is None else f"{baseline_precision * 100:.2f}%"
        scfe_text = "N/A" if scfe_precision is None else f"{scfe_precision * 100:.2f}%"
        logger.info(
            f"[{self.current_time}s] [LinkPredictionAccuracy] Previous macro "
            f"T={summary['TriggerTime']}s, window=[{summary['WindowStart']}s,"
            f"{summary['WindowEnd']}s), threshold={summary['Threshold']:.3f}, "
            f"tolerance={summary.get('ToleranceGbps', 0.0):.3f}Gbps, "
            f"accuracy={accuracy_text}, baseline={baseline_text}, scfe={scfe_text}, "
            f"correct={summary['Correct']}/{summary['Total']}, "
            f"frames={summary['EvaluatedFrames']}/{summary['ExpectedFrames']}"
        )
        self.notify_backend("link_prediction_accuracy", summary)
        
        # 将预测精度异步落库
        try:
            from simulation_api.db_services import save_prediction_accuracy
            save_prediction_accuracy.delay(self.constellation_id, summary)
        except Exception as e:
            logger.error(f"[{self.current_time}s] 预测精度数据异步落库异常: {e}")
            
        self._pending_prediction_accuracy = None



    def _prediction_window_to_model_input(self):
        """
        Convert the rolling micro-cycle frame buffer into SCFE input:
            (30, link_num, 11) -> (link_num, 30, 11)
        """
        frames = list(self._prediction_buffer)
        expected_len = getattr(self.link_predictor, "input_len", 30)
        if len(frames) != expected_len:
            raise ValueError(
                f"prediction buffer length must be {expected_len}, got {len(frames)}"
            )

        link_index = list(frames[0].get("link_index", []))
        if not link_index:
            raise ValueError("prediction link_index cannot be empty")

        feature_frames = []
        for idx, frame in enumerate(frames):
            frame_link_index = list(frame.get("link_index", []))
            if frame_link_index != link_index:
                raise ValueError(
                    f"prediction link_index changed inside the 30-step window at frame {idx}"
                )
            features = frame.get("features")
            if not isinstance(features, np.ndarray):
                raise TypeError(f"prediction frame {idx} features must be a numpy.ndarray")
            if features.shape != (len(link_index), 11):
                raise ValueError(
                    f"prediction frame {idx} shape must be ({len(link_index)}, 11), got {features.shape}"
                )
            feature_frames.append(features)

        raw_window = np.stack(feature_frames, axis=0)
        input_window = np.transpose(raw_window, (1, 0, 2)).astype(np.float32, copy=False)
        return input_window, link_index

    def _async_rolling_prediction(self, input_window, trigger_time, link_index=None):
        """后台异步调用大模型进行链路预测"""
        logger.info(f"[{trigger_time}s] [Async] 后台预测线程开始执行推理, 窗口 shape={input_window.shape}...")
        t_start = time.time()

        # 调用大模型
        prediction = self.link_prediction_execution(input_window, link_index)

        # 将最新结果挂载到引擎实例
        with self.state_rw_lock.write_lock():
            self.latest_prediction_result = prediction

        t_cost = time.time() - t_start
        logger.info(f"[{trigger_time}s] [Async] 后台预测完成并成功挂载，耗时 {t_cost * 1000:.2f} ms")
        
        # ==========================================
        # [新增]：推送链路预测结果至前端，并调用 Celery 异步落库
        # ==========================================
        if prediction:
            self.notify_backend("link_prediction_result", prediction)
            try:
                from simulation_api.db_services import save_prediction_results
                save_prediction_results.delay(prediction)
            except Exception as e:
                logger.error(f"[{trigger_time}s] 链路预测结果异步落库异常: {e}")



    def link_prediction_execution(self, input_window=None, link_index=None):
        """调用封装好的预测接口，输入为 (link_num, 30, 11) 的 Numpy 数组"""
        if input_window is None or link_index is None:
            input_window, link_index = self._prediction_window_to_model_input()
        current_utc = self._current_utc()
        current_flow_rows = []
        with self.state_rw_lock.read_lock():
            topo = self.topology_state
            topology_map = (topo.new_topology or {}) if topo else {}
            for src_id, dst_id in link_index:
                src_key = str(src_id)
                dst_key = str(dst_id)
                link_attr = (topology_map.get(src_key) or {}).get(dst_key)
                if link_attr is None:
                    continue
                flow_value = getattr(link_attr, "current_flow", None)
                if flow_value is None:
                    continue
                try:
                    flow_value = float(flow_value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(flow_value):
                    continue
                current_flow_rows.append((current_utc, src_key, dst_key, max(0.0, flow_value)))

        return self.link_predictor.predict(
            current_utc,
            input_window,
            link_index,
            step_seconds=config.MICRO_PERIOD_SECONDS,
            current_flow_rows=current_flow_rows,
        )

    def topology_reconstruction_execution(self, predicted_links=None):
        """
        执行拓扑重构。
        注意：只负责计算并返回新拓扑数据，不在内部调用 update_topology_state()，
        状态写入统一由 _macro_loop 执行，避免重复写入。
        """
        #  填充真实拓扑重构算法
        with self.state_rw_lock.read_lock():
            current_topo = self.topology_state
            current_node = self.node_state
        previous_topology_update = 0
        if isinstance(self._last_topology_reconstruction_result, dict):
            previous_topology_update = int(
                self._last_topology_reconstruction_result.get("TopologyUpdate", 0) or 0
            )

        result_dict = reconstruct_topology(
            current_topo,
            node_state=current_node,
            constellation=self.constellation,
            predicted_links=predicted_links,
            previous_topology_update=previous_topology_update,
            task_list=self.task_list,
        )
        self._last_topology_reconstruction_result = result_dict
        
        # 通知前端拓扑重构指标（向前端 websocket 推送 TopQualities）的逻辑
        if "TopQualities" in result_dict:
            self.notify_backend("TOPOLOGY_METRICS", {"TopQualities": result_dict["TopQualities"]})

        # 【核心修复】将字典结果转换为 TopologyState 对象，并将内部的普通字典转换为 LinksQualitiesValue 对象
        
        raw_new_topology = result_dict.get("newTopology", {})
        
        # 遍历嵌套字典，将每个链路属性字典转换为 LinksQualitiesValue 对象
        converted_topology = {}
        for src_sat_id, neighbors in raw_new_topology.items():
            converted_neighbors = {}
            for dst_sat_id, link_attr_dict in neighbors.items():
                if isinstance(link_attr_dict, dict):
                    # 将普通字典转换为 LinksQualitiesValue 对象
                    converted_neighbors[dst_sat_id] = LinksQualitiesValue(**link_attr_dict)
                else:
                    # 如果已经是 LinksQualitiesValue 对象，直接使用
                    converted_neighbors[dst_sat_id] = link_attr_dict
            converted_topology[src_sat_id] = converted_neighbors
        
        new_topology_state = TopologyState(
            constellation_id=result_dict.get("ConstellationId", self.constellation_id),
            timestamp=result_dict.get("Timestamp", self._timestamp_ms(self._current_utc())),
            new_topology=converted_topology
        )
        
        return new_topology_state


    def _macro_loop(self):
        """宏周期线程：链路预测 -> 地面站数量更新 -> 拓扑重构"""
        while not self._stop_event.is_set():
            self._macro_event.wait()      # 等待主线程触发
            self._macro_event.clear()     # 唤醒后立即清除触发信号

            if self._stop_event.is_set():
                self._macro_done.set()    # 退出前放开主线程，防死锁
                break

            try:
                logger.info(f"\n{'━' * 54}")
                logger.info(f"  [T={self.current_time}s] 宏周期触发")
                logger.info(f"{'━' * 54}")

                self._finalize_active_prediction_accuracy(force=True)
                self._print_pending_prediction_accuracy()

                # 1. 提取完整的 30 个微周期历史张量，执行链路预测
                predicted_links = None
                expected_len = getattr(self.link_predictor, "input_len", 30)
                if len(self._prediction_buffer) == expected_len:
                    try:
                        input_window, link_index = self._prediction_window_to_model_input()
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            f"[{self.current_time}s] [Macro] 预测窗口无效，跳过链路预测和拓扑重构: {e}"
                        )
                        self._prediction_buffer.clear()
                    else:
                        logger.info(
                            f"[{self.current_time}s] [Macro] 提取完整历史窗口 shape={input_window.shape}, "
                            f"links={len(link_index)}，执行大模型链路预测...")
                    
                        t_start = time.time()
                        predicted_links = self.link_prediction_execution(input_window, link_index)

                        t_cost = (time.time() - t_start) * 1000
                        horizon = predicted_links.get("PredictionHorizon", 0) if isinstance(predicted_links, dict) else 0
                        logger.info(
                            f"[{self.current_time}s] [Macro] 链路预测完成，horizon={horizon}，"
                            f"耗时 {t_cost:.2f} ms"
                        )
                        self.latest_prediction_result = predicted_links
                        self.notify_backend("link_prediction_result", predicted_links)

                        t_start = time.time()
                        records = self.link_state_records_update(predicted_links)
                        logger.info(
                            f"[{self.current_time}s] [Macro] 表2-2 链路状态记录 {len(records)} 条，"
                            f"耗时 {(time.time() - t_start) * 1000:.2f} ms"
                        )
                else:

                    logger.info(
                        f"[{self.current_time}s] [Macro] 预测窗口未满 "
                        f"{len(self._prediction_buffer)}/{expected_len}，"
                        "跳过链路预测和拓扑重构"
                    )

                matched_ground_station_count = self._refresh_ground_station_numbers()
                logger.info(
                    f"[{self.current_time}s] [Macro] GroundStationNumber refreshed: "
                    f"{matched_ground_station_count} ground stations"
                )
                     
                # 2. 拓扑重构（强依赖第 1 步的输出）
                if predicted_links is None:
                    continue

                baseline_topology_state, _ = self.get_current_states()
                t_start_topo = time.time()
                new_topo_data = self.topology_reconstruction_execution(predicted_links)
                self._last_topo_time = (time.time() - t_start_topo) * 1000
                topology_update = int(
                    (self._last_topology_reconstruction_result or {}).get("TopologyUpdate", 0) or 0)
                self.update_topology_state(new_topo_data if topology_update else baseline_topology_state)

                t_start = time.time()
                capacity = self.capacity_evaluation_update(
                    baseline_topology_state,
                    predicted_links,
                    window_start_s=self.current_time - int(getattr(config, "MACRO_PERIOD_SECONDS", 30)),
                )
                logger.info(
                    f"[{self.current_time}s] [CapacityEvaluation] "
                    f"CapacityBefore={capacity['Capacity']['CapacityBefore']} "
                    f"CapacityAfter={capacity['Capacity']['CapacityAfter']} "
                    f"GainPct={capacity['Capacity']['CapacityGainPct']} "
                    f"Availability={capacity['Availability']['Availability']} "
                    f"Passed={capacity['Verdict']['Passed']}/{capacity['Verdict']['Total']} "
                    f"耗时 {(time.time() - t_start) * 1000:.2f} ms"
                )
                
                # ==========================================
                # [新增]：推送拓扑重构结果至前端，并调用 Celery 异步落库
                # 注意：这里传 _last_topology_reconstruction_result 字典，因为它包含 TopDifference/TopQualities 等关键指标
                # ==========================================
                # ========== 发送关键指标 =============
                try:
                    # 唯一数据源：统一使用 db_services 中的 IQR 统计口径
                    from simulation_api.db_services import build_key_metrics_payload
                    key_metrics_data = build_key_metrics_payload()
                except Exception as e:
                    logger.warning(f"DB query failed: {e}")
                    key_metrics_data = {
                        'gw': {'topoTime': '--', 'planTime': '--', 'predAccuracy': '--'},
                        'delta': {'topoTime': '--', 'planTime': '--', 'predAccuracy': '--'}
                    }
                self.notify_backend('key_metrics', key_metrics_data)

                reconstruction_dict = self._last_topology_reconstruction_result
                if reconstruction_dict:
                    self.notify_backend("topology_reconstruction_result", reconstruction_dict)
                    try:
                        def _recursive_dict(obj):
                            if hasattr(obj, "dict"):
                                return _recursive_dict(obj.dict(by_alias=True))
                            elif isinstance(obj, dict):
                                return {k: _recursive_dict(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [_recursive_dict(v) for v in obj]
                            return obj
                                
                        safe_dict = _recursive_dict(reconstruction_dict)  # 化解嵌套的复杂对象为基石 dict

                        from simulation_api.db_services import save_reconstruction_results
                        save_reconstruction_results.delay(safe_dict)
                    except Exception as e:
                        logger.error(f"[{self.current_time}s] 拓扑重构结果异步落库异常: {e}")

                self._print_macro_snapshot()
                self._print_topology_reconstruction_result(self._last_topology_reconstruction_result)
                self._start_prediction_accuracy_if_applicable(
                    predicted_links,
                    self._last_topology_reconstruction_result,
                )
                topology_update = 0
                if isinstance(self._last_topology_reconstruction_result, dict):
                    topology_update = int(
                        self._last_topology_reconstruction_result.get("TopologyUpdate", 0) or 0
                    )
                if topology_update != 0:
                    self._prediction_buffer.clear()
                    logger.info(
                        f"[{self.current_time}s] [Macro] TopologyUpdate=1，"
                        "清空预测窗口，进入新拓扑 150s 冷启动"
                    )


            except Exception as e:
                logger.error(f"[_macro_loop] 宏周期执行期间发生异常: {e}")
            finally:
                # 【核心修复】：无论正常还是异常，必须通知主线程完成，解除阻塞
                self._macro_done.set()

    def _micro_loop(self):
        """
        微周期线程：每个微周期执行一次，更新位置与背景流量并刷新预测窗口。
        已按需求移除微周期内的路径规划、路由需求生成与路由优化计算。
        输出时机由调用方（run_simulation / run_extended_simulation）控制，
        本线程只负责计算，不负责打印。
        """
        while not self._stop_event.is_set():
            self._micro_event.wait()
            self._micro_event.clear()

            if self._stop_event.is_set():
                self._micro_done.set()
                break

            try:
                # 0. 每帧先更新卫星位置（同时把链路距离同步回 topology_state）
                self._update_satellite_positions()

                # 1. 背景流量生成（最新坐标）
                bg_flow_data, ret_timestamp = self.background_traffic_update()
                self.notify_backend("background_traffic_result", bg_flow_data)

                # ================= [新增] 预测窗口更新与触发 =================
                current_frame = self._build_prediction_frame()
                self._prediction_buffer.append(current_frame)

                # 2. 感知监测
                self.sensing_update()

                if self._should_log_micro_summary():
                    logger.info(
                        f"[{self.current_time}s] 微周期计算完成 | 路径规划/路由需求/路由优化已禁用"
                    )

            except Exception as e:
                logger.error(f"[_micro_loop] 微周期执行期间发生异常: {e}")
            finally:
                self._micro_done.set()


    def stop(self):
        """安全停止仿真引擎。"""
        logger.info(f"[{self.current_time}s] [Engine] 接收到终止指令，正在关闭...")
        self._stop_event.set()
        self._macro_event.set()
        self._micro_event.set()

    def run_simulation(self, timestamp, constellation_id, tle_file_path, task_list):
        logger.info("====== 仿真开始: 初始化阶段 ======")

        real_start_time = time.time()

        # 第0秒：初始化，内部完成拓扑构建和第0秒背景流量
        self.initialize_constellation(timestamp, constellation_id, tle_file_path, task_list)

        prediction_input_steps = int(getattr(self.link_predictor, "input_len", 30))
        warmup_seconds = prediction_input_steps * int(getattr(config, "MICRO_PERIOD_SECONDS", 5))

        # 初始化完成后，先强行抓取 T=0 的状态推入窗口！
        initial_frame = self._build_prediction_frame()
        self._prediction_buffer.append(initial_frame)
        self.sensing_update()
        logger.info(
            f"[0s] [Warmup] 装入 T=0 初始特征帧 "
            f"{len(self._prediction_buffer)}/{prediction_input_steps}"
        )


        # 初始化完成后推进到第一个微周期起点，第0秒已处理完毕
        self.current_time = config.MICRO_PERIOD_SECONDS

        init_cost = time.time() - real_start_time
        logger.info(f"初始化实质耗时: {init_cost:.2f} 秒")

        # 第一阶段：按微周期步长预热，直到第一个宏周期触发前
        logger.info(
            "====== 第一阶段: 空转预热阶段 "
            f"({config.MICRO_PERIOD_SECONDS}s ~ "
            f"{warmup_seconds - config.MICRO_PERIOD_SECONDS}s, "
            f"step={config.MICRO_PERIOD_SECONDS}s) ======"
        )
        _warmup_prev_time = None
        while self.current_time < warmup_seconds:


            # 1. 更新卫星位置（基于当前逻辑时间推算真实UTC）
            self._update_satellite_positions()

            # 2. 更新背景流量
            bg_flow_data, ts = self.background_traffic_update()
            self.notify_backend("background_traffic_result", bg_flow_data)

            # ================= [新增：填入数据，不触发预测] =================
            current_frame = self._build_prediction_frame()
            self._prediction_buffer.append(current_frame)
            self.sensing_update()
            # 加一句打印，让你在控制台清楚看到预热帧正在平稳装载数据
            # 加上 T=0 时装入的那一帧，这里打印的进度会从 2/30 一直走到 30/30
            logger.debug(
                f"[{self.current_time}s] [Warmup] 数据生成，用于预测: "
                f"{len(self._prediction_buffer)}/{prediction_input_steps}"
            )

            # ================================================================

            # 3. 每5秒打印一次7点诊断
            if self.current_time % config.MICRO_PERIOD_SECONDS == 0:
                _warmup_prev_time = self._print_snapshot("预热", _warmup_prev_time)

            # 4. 动态时延补偿
            target_time = real_start_time + self.current_time
            sleep_duration = target_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            self.current_time += config.MICRO_PERIOD_SECONDS

        # 第二阶段：完整宏/微周期循环
        logger.info("====== 初始化结束，进入第二阶段: 完整周期循环 ======")

        # macro_thread = threading.Thread(target=self._macro_loop, daemon=True)
        # micro_thread = threading.Thread(target=self._micro_loop, daemon=True)
        # macro_thread.start()
        # micro_thread.start()

        # 只在线程不存在或已死亡时启动，使用实例属性保存
        if self._macro_thread is None or not self._macro_thread.is_alive():
            self._macro_thread = threading.Thread(target=self._macro_loop, daemon=True)
            self._micro_thread = threading.Thread(target=self._micro_loop, daemon=True)
            self._macro_thread.start()
            self._micro_thread.start()

        _main_prev_time = None
        is_interrupted = False
        try:
            while self.current_time <= self.duration:

                # 宏周期优先：拓扑重构完成后微周期才能拿到新拓扑
                if self.current_time % config.MACRO_PERIOD_SECONDS == 0:
                    self._macro_done.clear()
                    self._macro_event.set()
                    self._macro_done.wait()

                # 微周期：位置更新 + 背景流量 + 预测窗口刷新（不做路径/路由计算）
                self._micro_done.clear()
                self._micro_event.set()
                self._micro_done.wait()

                # 每5秒输出一次物理量快照
                if self.current_time % config.MICRO_PERIOD_SECONDS == 0:
                    _main_prev_time = self._print_snapshot("微周期", _main_prev_time)
                    #TODO 向前端推送当前状态快照数据，供可视化展示使用。

                # 动态时延补偿
                target_time = real_start_time + self.current_time
                sleep_duration = target_time - time.time()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                self.current_time += config.MICRO_PERIOD_SECONDS
        except KeyboardInterrupt:
            is_interrupted = True
            logger.warning("\n[Engine] 收到中断信号，正在终止仿真...")
        finally:
            if is_interrupted:
                self._stop_event.set()
                # 唤醒可能正在等待指令的工作线程，令其进入退出逻辑
                self._micro_event.set()
                self._macro_event.set()
                
                if self._macro_thread and self._macro_thread.is_alive():
                    self._macro_thread.join(timeout=2.0)
                if self._micro_thread and self._micro_thread.is_alive():
                    self._micro_thread.join(timeout=2.0)
                logger.info("仿真系统退出。")
            else:
                logger.info("\n====== 基础仿真已完成，底层驱动线程保持待命 ======")


    def run_extended_simulation(self):
        """
        24小时扩展仿真：在60秒主仿真结束后接续运行。
        每小时（3600秒）为一个大周期，每小时输出一次物理量快照。
        卫星位置和背景流量按微周期更新，宏/微周期线程继续运行。
        时延补偿已注释，以最快速度跑完（几分钟内完成24小时仿真）。
        """
        logger.info("\n====== 开始24小时扩展仿真 ======")
    
        # 从当前 current_time 接续（60s 主仿真结束时 current_time = 61）
        extended_start = self.current_time
        extended_end = extended_start + 24 * 3600  # 再跑 24 小时
    
        _prev_snapshot_time = None

        try:
            while self.current_time <= extended_end:
                if self.current_time % config.MACRO_PERIOD_SECONDS == 0:
                    self._macro_done.clear()
                    self._macro_event.set()
                    self._macro_done.wait()

                self._micro_done.clear()
                self._micro_event.set()
                self._micro_done.wait()

                elapsed = self.current_time - extended_start
                if elapsed > 0 and elapsed % 3600 == 0:
                    hour = elapsed // 3600
                    current_utc = self._current_utc()
                    logger.info(f"\n{'═' * 54}")
                    logger.info(f"  第 {hour:2d} 小时  UTC={current_utc.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"{'═' * 54}")
                    _prev_snapshot_time = self._print_snapshot("小时快照", _prev_snapshot_time)

                self.current_time += config.MICRO_PERIOD_SECONDS
        except KeyboardInterrupt:
            logger.warning("\n[Engine] 24小时扩展仿真被手动中断...")
        finally:
            logger.info("\n====== 扩展仿真结束，清理所有底层资源 ======")
            self._stop_event.set()
            self._micro_event.set()
            self._macro_event.set()
            
            if self._macro_thread and self._macro_thread.is_alive():
                self._macro_thread.join(timeout=2.0)
            if self._micro_thread and self._micro_thread.is_alive():
                self._micro_thread.join(timeout=2.0)
            logger.info("系统完全退出。")
