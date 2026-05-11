import time
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections import deque
import numpy as np
import os


# [可以直接放在文件顶部引入，告别循环依赖]
from src.simulation.topology_link_state import init_walker_topology, update_topology_snapshot
from src.simulation.constellation import CommunicationConstellation, load_satellites_from_tle
from src.utils.orbit_calculator import calculate_satellite_coordinates
from src.simulation.background_traffic_generation import background_traffic_generation
from src.simulation.routing_demand_generation import routing_demand_generation
from src.algos.link_prediction.prediction_interface import LinkPredictor
from src.algos.top_reconstruction.reconstruction_interface import reconstruct_topology
from src.algos.task_planning.path_planning_interface import run_task_planning
from src.algos.route_planning.routing_interface import route_planning_batch_execution

import src.config as config

from src.simulation.data_model import NodeAttribute, SatelliteNodeState, TopologyState, LinksQualitiesValue

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
    def __init__(self, constellation_id, duration_seconds):
        self.constellation_id = constellation_id
        self.duration = duration_seconds
        self.current_time = 0

        # 初始化后端回调接口为空，防止报错
        self.backend_callback = None

        # 将核心状态设为实例属性，并在仿真生命周期内维护
        self.topology_state: TopologyState | None = None
        self.node_state: SatelliteNodeState | None = None

        # 星座对象，在 initialize_constellation 中赋值，供位置更新使用
        self.constellation = None

        # 任务规划队列及锁
        self.task_queue = []
        self.task_queue_lock = threading.Lock()

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

        # 专门用于处理耗时任务规划的线程池（避免阻塞微周期主循环）
        self.task_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TaskPlanner")

        # [新增] 30秒固定长度滑动窗口
        self._prediction_buffer = deque(maxlen=30)

        # [新增] 初始化链路预测模块 - 传入星座ID而非文件路径
        self.link_predictor = LinkPredictor(constellation_id)


    def initialize_constellation(self, timestamp, constellation_id, tle_file_path, task_list):
        """
        初始化阶段：加载 TLE、构建星座对象、生成拓扑、产生首次背景流量、装载任务队列。
        :param timestamp:        仿真起始 UTC 时间 (datetime)
        :param constellation_id: 星座 ID (str)
        :param tle_file_path:    TLE 文件路径 (str)
        :param task_list:        首批任务列表 (List[dict])
        :return: topology_state, node_state, task_queue
        """
        print(f"[{timestamp}] 初始化卫星星座: {constellation_id}, 加载TLE文件: {tle_file_path}")

        # 0. 初始化系统仿真时间和基准UTC时间
        self.current_time = 0
        self.sim_start_utctime = timestamp

        # 1. 初始化卫星节点状态 - 使用 SatelliteNodeState dataclass
        unix_timestamp_ms = int(timestamp.timestamp() * 1000)
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
                heat_flow=0.0
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

        # 3. 用 TopologyState dataclass 初始化拓扑
        topology_state = TopologyState(
            constellation_id=constellation_id,
            timestamp=unix_timestamp_ms,
            new_topology=init_walker_topology(self.constellation)
        )

        # 将初始化数据写入引擎全局状态
        with self.state_rw_lock.write_lock():
            self.topology_state = topology_state
            self.node_state = node_state

        # 4. 产生一次初始的背景流量
        try:
            self.background_traffic_update()
            print(f"[{self.current_time}s] 初始化阶段: 首次背景流量生成完成")
        except Exception as e:
            print(f"[{self.current_time}s] 初始化阶段: 背景流量调用异常 - {e}")

        # 5. 将首批任务装入 task_queue
        if task_list:
            with self.task_queue_lock:
                for task in task_list:
                    self.task_queue.append(task)
                self.task_queue.sort(
                    key=lambda x: (-x.get("TaskPriority", 0), x.get("ArrivalTime", 0))
                )
            print(f"[{self.current_time}s] 初始化阶段: 首批 {len(task_list)} 个任务已加入队列")

            # Deleted:# 记录 T=0 的初始特征帧
            # Deleted:initial_frame = self._build_prediction_frame()
            # Deleted:self._prediction_buffer.append(initial_frame)
            # Deleted:print(
            # Deleted:f"[{self.current_time}s] 初始化阶段: T=0 初始特征帧插入: {len(self._prediction_buffer)}/30")

        return self.topology_state, self.node_state, self.task_queue

    def get_current_states(self):
        """安全地读取当前的核心状态"""
        with self.state_rw_lock.read_lock():
            return self.topology_state, self.node_state

    def notify_backend(self, msg_type, data):
        """统一的后端数据回传接口"""
        if self.backend_callback:
            try:
                self.backend_callback(msg_type, data)
            except Exception as e:
                print(f"[Engine] 回调后端失败: {e}")

    def receive_backend_task_planning(self, task_planning_params):
        """监听并处理后端发来的实时任务规划请求"""
        task_list = task_planning_params.get("TaskList", [])
        print(f"[{self.current_time}s] [Backend] 收到任务规划请求, 数量: {len(task_list)}")
        with self.task_queue_lock:
            for task in task_list:
                self.task_queue.append(task)
            self.task_queue.sort(
                key=lambda x: (-x.get("TaskPriority", 0), x.get("ArrivalTime", 0))
            )

    def _process_pending_tasks(self):
        """检查并执行到达时间的任务"""
        tasks_to_execute = []
        with self.task_queue_lock:
            remaining_tasks = []
            for task in self.task_queue:
                if "arrival_sim_time" in task:
                    arrival_sim_time = task["arrival_sim_time"]
                else:
                    arrival_utc = task.get("ArrivalTime")
                    if arrival_utc is None:
                        arrival_sim_time = 0
                    elif isinstance(arrival_utc, datetime.datetime) and self.sim_start_utctime is not None:
                        delta = (arrival_utc - self.sim_start_utctime).total_seconds()
                        arrival_sim_time = max(0, delta)
                    else:
                        print(f"[{self.current_time}s] [警告] 任务 {task.get('TaskId')} 的 ArrivalTime "
                              f"格式无法解析 ({type(arrival_utc).__name__})，将立即执行。")
                        arrival_sim_time = 0
                    task["arrival_sim_time"] = arrival_sim_time

                if self.current_time >= arrival_sim_time:
                    tasks_to_execute.append(task)
                else:
                    remaining_tasks.append(task)
            self.task_queue = remaining_tasks

        if tasks_to_execute:
            # 定义一个内部函数供线程池后台执行
            def _async_run_planning(planning_task):
                try:
                    # 如果是框架自带的流量撤销任务，直接走状态更新接口还原流量，跳过规划算法
                    if planning_task.get("TaskType") == "TeardownFlow":
                        self.update_topology_link_flow_state(
                            route_path=planning_task.get("route_path"),
                            timestamp=None, 
                            flow_size=planning_task.get("flow_size"),  # 此处传过来已经是负数
                            duration=0  # 置0防止无限递归产生下一次清理任务
                        )
                        return
                    
                    # 每次执行单条任务时获取最新的一瞥状态，而不是成百上千任务共用最开始的老状态
                    t_topo, t_node = self.get_current_states()
                    res = self.path_planning_execution(t_topo, t_node, planning_task)
                    self.notify_backend("path_planning_result", {
                        "task_id": planning_task.get("TaskId"),
                        "result": res
                    })
                except Exception as e:
                    print(f"[_process_pending_tasks] 任务 {planning_task.get('TaskId')} 执行异常: {e}")

            for task in tasks_to_execute:
                # 瞬间投递，不阻塞微周期
                self.task_executor.submit(_async_run_planning, task)

# """
# 根据zz的拓扑状态更新之后接入
# """

    def update_topology_state(self, new_topology: TopologyState):
        """拓扑状态更新入口，写入时自动更新 Timestamp 为当前帧 Unix ms"""
        with self.state_rw_lock.write_lock():
            self.topology_state = new_topology
            current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
            self.topology_state.timestamp = int(current_utc.timestamp() * 1000)
        # print(f"[{self.current_time}s] 拓扑状态已更新")

    def update_topology_link_flow_state(self, route_path, timestamp, flow_size, duration):
        """
        根据路由优化和任务规划执行返回结果，更新拓扑链路属性和节点负载属性。
        如果 duration > 0，会自动创建一个未来的流量释放任务挂入队列。
        链路属性：LinkDistance, LinkCapacity, CurrentFlow, LinkPropagationDelay,
                  QueueDelay, TransmissionDelay, PacketLossRate, HeatValue
        节点属性：Flow, Congestion
        
        """
        if not route_path or not flow_size:
            return

        with self.state_rw_lock.write_lock():
            # 1. 更新节点流量 (Node Flow) - 使用下划线属性访问
            for sat_id in route_path:
                sid = str(sat_id)
                if sid in self.node_state.sat_list:
                    self.node_state.sat_list[sid].flow += flow_size
                    # 避免浮点数精度问题变为负数
                    if self.node_state.sat_list[sid].flow < 0:
                        self.node_state.sat_list[sid].flow = 0.0

            # 2. 更新链路流量 (Edge Flow) - 使用下划线属性访问
            local_topo = self.topology_state.new_topology
            for i in range(len(route_path) - 1):
                u, v = str(route_path[i]), str(route_path[i+1])
                if u in local_topo and v in local_topo[u]:
                    link_attr = local_topo[u][v]
                    link_attr.current_flow += flow_size
                    if link_attr.current_flow < 0:
                        link_attr.current_flow = 0.0
                        
                    # (可选) 在这里基于 current_flow / link_capacity 实时计算拥塞率 (Congestion) 并更新排队丢包率

            current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
            self.topology_state.timestamp = int(current_utc.timestamp() * 1000)
            self.node_state.timestamp = int(current_utc.timestamp() * 1000)

        # 3. 注册未来自动拆除连接的任务
        if duration > 0:
            teardown_task = {
                "TaskId": f"Teardown_{id(route_path)}",
                "TaskType": "TeardownFlow",
                "route_path": route_path,
                "flow_size": -flow_size,  # 负流量，执行时相减
                "duration": 0,            # 拆除操作不再产生后续队列
                "arrival_sim_time": self.current_time + duration,
                "TaskPriority": 999       # 最高优先级，按时回收资源
            }
            with self.task_queue_lock:
                self.task_queue.append(teardown_task)
                self.task_queue.sort(key=lambda x: (-x.get("TaskPriority", 0), x.get("arrival_sim_time", 0)))

    def update_node_state(self, new_node_data: SatelliteNodeState):
        """节点状态更新入口，写入时自动更新 Timestamp 为当前帧 Unix ms"""
        with self.state_rw_lock.write_lock():
            if self.node_state is None:
                self.node_state = new_node_data
            else:
                # 更新 sat_list 内容
                self.node_state.sat_list = new_node_data.sat_list
            current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
            self.node_state.timestamp = int(current_utc.timestamp() * 1000)
        # print(f"[{self.current_time}s] 节点状态已更新")

    def _print_snapshot(self, label="微周期", prev_time=None):
        """
        物理量快照输出。每次调用打印当前帧的关键数值：
        - 卫星流量：总流量、活跃卫星数、TOP3高负载卫星（含经纬度）
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
                     sdata.flow)
                    for sid, sdata in sat_list.items()]
        active = [(sid, lat, lon, f) for sid, lat, lon, f in sat_data if f > 0]
        total_flow = sum(f for _, _, _, f in sat_data)
        top3_sats = sorted(active, key=lambda x: x[3], reverse=True)[:3]

        now = time.time()
        interval_str = f"  Δt={now - prev_time:.2f}s" if prev_time is not None else ""
        print(f"\n[T={self.current_time}s] {label}{interval_str}")
        print(f"  卫星  总流量={total_flow:.1f}Gbps  活跃={len(active)}/{total_sats}")
        rank_labels = ["第一活跃卫星", "第二活跃卫星", "第三活跃卫星"]
        for i, (sid, lat, lon, f) in enumerate(top3_sats):
            print(f"        {rank_labels[i]}: Sat#{sid:<5} 纬度={lat:+6.1f}°  经度={lon:+7.1f}°  流量={f:.3f}Gbps")

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
                    flow_links.append((u_id, target_id, flow))

        if all_links:
            min_l = min(all_links, key=lambda x: x[2])
            max_l = max(all_links, key=lambda x: x[2])
            avg_d = sum(x[2] for x in all_links) / len(all_links)
            print(f"  链路  有效={len(all_links)}条  "
                  f"最短链路={min_l[2]:.1f}km(#{min_l[0]}→#{min_l[1]})  "
                  f"最长链路={max_l[2]:.1f}km  均值={avg_d:.1f}km  "
                  f"时延样本={all_links[0][3]:.3f}ms")
        else:
            print(f"  链路  LinkDistance 全为 0，位置同步未生效")

        # ── TOP3 链路流量（来自 topology_state["newTopology"] CurrentFlow）────
        if flow_links:
            top3_lf = sorted(flow_links, key=lambda x: x[2], reverse=True)[:3]
            lf_str = "  ".join(f"#{u}→#{v}({f:.3f}Gbps)" for u, v, f in top3_lf)
            print(f"  链路流量  {lf_str}")

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
            print(f"  拓扑重构完成 | 链路={len(all_links)}条 | "
                  f"最短={min_l[2]:.1f}km(时延={min_l[3]:.3f}ms)  "
                  f"最长={max_l[2]:.1f}km(时延={max_l[3]:.3f}ms)")
        else:
            print(f"  拓扑重构完成 | 链路距离尚未更新")

    def background_traffic_update(self):
        """
        step1: 获取当前状态
        step2: 调用背景流量算法
        step3: 更新拓扑和节点状态
        返回: background_traffic_obj, ret_timestamp
        """
        topology_state, node_state = self.get_current_states()
        current_timestamp = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)

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
        current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)

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

    def _build_prediction_frame(self):
        """
        提取当前帧的全局真实物理特征
        返回 shape: (3600, 5, 5) -> 3600颗星，每颗星 [本星+4邻居]，每节点 5个特征
        """
        topo, node = self.get_current_states()
        new_topo = topo.new_topology
        sat_list = node.sat_list

        # 计算相对安全的 UTC 时间特征，并归一化到 [0, 1]
        current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
        minutes_in_day = current_utc.hour * 60 + current_utc.minute
        safe_time_feature = minutes_in_day / 1440.0

        global_frame_data = []
        total_sats = len(sat_list) # 通常是 3600

        # 严格按照 ID 顺序 (0 到 3599) 遍历，确保与 inference.py 中的 batch 索引一一对应
        for i in range(total_sats):
            target_id = str(i)
            target_links = new_topo.get(target_id, {})

            neighbors_info = []
            # 取前4个邻居进行迭代
            for j, (nid, link_attr) in enumerate(list(target_links.items())[:4]):
                neighbors_info.append({"id": str(nid), "link_attr": link_attr})

            frame_data = []

            # 1. 压入目标卫星自身的特征
            target_sat = sat_list.get(target_id)
            if target_sat:
                heat_val = target_sat.flow / 60.0
                survival = target_sat.energy_ratio
                frame_data.append([0.05, 1.0, survival, heat_val, safe_time_feature])
            else:
                frame_data.append([0.05, 1.0, 1.0, 0.0, safe_time_feature])

            # 2. 压入 4 个邻居的特征
            for n_info in neighbors_info:
                nid = n_info["id"]
                l_attr = n_info["link_attr"]
                n_sat = sat_list.get(nid)

                # 使用下划线属性访问 LinksQualitiesValue 对象
                delay = l_attr.link_propagation_delay
                cap = l_attr.link_capacity

                survival = n_sat.energy_ratio if n_sat else 1.0
                heat_val = n_sat.flow / 60.0 if n_sat else 0.0

                frame_data.append([delay, cap, survival, heat_val, safe_time_feature])

            # 3. 维度对齐安全网：如果邻居不足4个，用 0 补齐
            while len(frame_data) < 5:
                frame_data.append([0.0, 0.0, 0.0, 0.0, safe_time_feature])

            global_frame_data.append(frame_data)

        # 返回形状为 (3600, 5, 5) 的张量
        return np.array(global_frame_data, dtype=np.float32)

    def _async_rolling_prediction(self, input_window, trigger_time):
        """后台异步调用大模型进行链路预测"""
        print(f"[{trigger_time}s] [Async] 后台预测线程开始执行推理, 窗口 shape={input_window.shape}...")
        t_start = time.time()

        # 调用大模型
        prediction = self.link_prediction_execution(input_window)

        # 将最新结果挂载到引擎实例
        with self.state_rw_lock.write_lock():
            self.latest_prediction_result = prediction

        t_cost = time.time() - t_start
        print(f"[{trigger_time}s] [Async] 后台预测完成并成功挂载，耗时 {t_cost * 1000:.2f} ms")



    def link_prediction_execution(self, input_window):
        """调用封装好的预测接口，输入为 (30, 5, 5) 的 Numpy 数组"""
        curr_topo, curr_node = self.get_current_states()
        current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
        return self.link_predictor.predict(curr_topo, curr_node, current_utc, input_window)

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

        result_dict = reconstruct_topology(
            current_topo,
            node_state=current_node,
            constellation=self.constellation,
            predicted_links=predicted_links,
        )

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
            timestamp=result_dict.get("Timestamp", int(time.time() * 1000)),
            new_topology=converted_topology
        )
        
        return new_topology_state


    def path_planning_execution(self, topology_state=None, node_state=None, task_planning_params=None):
        """
        执行路径规划任务
        
        输入：
            - topology_state: 当前拓扑状态字典
            - node_state: 当前节点状态字典
            - task_planning_params: 单个任务字典 或 任务列表
            
        输出：
            - 如果输入是单个任务：返回规划结果字典
            - 如果输入是任务列表：返回规划结果列表
        """
        if task_planning_params is None:
            return []
        
        # 判断是单个任务还是任务列表
        is_single_task = isinstance(task_planning_params, dict) and "TaskId" in task_planning_params
        
        if is_single_task:
            # 处理单个任务
            try:
                result = run_task_planning(
                    topology_state=topology_state,
                    node_state=node_state,
                    task_planning_params=task_planning_params,
                    constellation=self.constellation_id
                )
                
                # 如果规划成功，更新拓扑链路流量状态
                if result.get("accepted", False):
                    allocations = result.get("allocations", [])
                    for allocation in allocations:
                        path = allocation.get("path", [])
                        ratio = allocation.get("ratio", 0.0)
                        demand_gbps = result.get("demand_gbps", 0.0)
                        duration = task_planning_params.get("Duration", 0)
                        
                        # 计算实际分配的流量
                        flow_size = demand_gbps * ratio
                        
                        if path and flow_size > 0:
                            # 更新拓扑和节点状态
                            self.update_topology_link_flow_state(
                                route_path=path,
                                timestamp=None,
                                flow_size=flow_size,
                                duration=duration
                            )
                
                return result
                
            except Exception as e:
                print(f"[path_planning_execution] 任务 {task_planning_params.get('TaskId', 'unknown')} 执行异常: {e}")
                return {
                    "accepted": False,
                    "reason": f"execution error: {str(e)}",
                    "task_id": task_planning_params.get("TaskId", "unknown"),
                    "paths": [],
                    "ratios": [],
                    "allocations": []
                }
        else:
            # 处理任务列表
            results = []
            if isinstance(task_planning_params, list):
                for task in task_planning_params:
                    result = self.path_planning_execution(
                        topology_state=topology_state,
                        node_state=node_state,
                        task_planning_params=task
                    )
                    results.append(result)
            
            return results

    def route_planning_execution(self, topology_state=None, node_state=None, task_planning_demands=None):
        """
        路由优化的执行
        上游输入：routing_demand_generation() 的输出或外部传入的任务需求
        """
        # 1. 获取当前状态
        if topology_state is None or node_state is None:
            topology_state, node_state = self.get_current_states()
        
        current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
        
        # 2. 确定路由任务列表（优先使用传入参数，否则自动生成）
        if task_planning_demands and isinstance(task_planning_demands, dict):
            tasks = task_planning_demands.get("RouteTaskList", [])
        else:
            routing_demand = routing_demand_generation(topology_state, node_state, current_utc)
            tasks = routing_demand.get("RouteTaskList", [])
        
        if not tasks:
            return []


        # 调用批量路由规划接口
        results = route_planning_batch_execution(
            topology_state=topology_state,
            node_state=node_state,
            Timestamp=current_utc.isoformat(),
            ConstellationId=topology_state.constellation_id,
            RouteTaskList=tasks
        )

        # 3. 根据结果更新拓扑链路流量状态
        for res in results:
            route_path = res.get("RoutePath")
            if not route_path:
                continue
                
            # 找到对应的任务以获取流量大小和持续时间
            task = next((t for t in tasks if t.get("TaskId") == res.get("TaskId")), None)
            if task:
                packet_size_mbits = task.get("PacketSize", 0.0)
                duration = task.get("Duration", 10)
                # Mbits -> Gbps (假设在 duration 内均匀传输)
                flow_size_gbps = packet_size_mbits / duration / 1000.0 if duration > 0 else 0.0
                
                if flow_size_gbps > 0:
                    self.update_topology_link_flow_state(
                        route_path=route_path,
                        timestamp=current_utc,
                        flow_size=flow_size_gbps,
                        duration=duration
                    )
        
        return results
    def _macro_loop(self):
        """宏周期线程：链路预测 -> 拓扑重构 -> 任务规划"""
        while not self._stop_event.is_set():
            self._macro_event.wait()      # 等待主线程触发
            self._macro_event.clear()     # 唤醒后立即清除触发信号

            if self._stop_event.is_set():
                self._macro_done.set()    # 退出前放开主线程，防死锁
                break

            try:
                print(f"\n{'━' * 54}")
                print(f"  [T={self.current_time}s] 宏周期触发")
                print(f"{'━' * 54}")

                # 1. 提取完整的 30 秒历史张量，执行链路预测
                predicted_links = None
                if len(self._prediction_buffer) == 30:
                    # 原始 stack 出来是 (30, 3600, 5, 5)
                    raw_window = np.stack(list(self._prediction_buffer), axis=0)

                    # 【核心修正】将时间维度(0)和批次维度(1)对调 -> 变成 (3600, 30, 5, 5)
                    input_window = np.transpose(raw_window, (1, 0, 2, 3))

                    print(
                        f"[{self.current_time}s] [Macro] 提取完整历史窗口 shape={input_window.shape}，执行大模型链路预测...")

                    t_start = time.time()
                    import datetime
                    curr_topo, curr_node = self.get_current_states()
                    current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)

                    predicted_links = self.link_predictor.predict(curr_topo, curr_node, current_utc, input_window)

                    t_cost = (time.time() - t_start) * 1000
                    print(f"[{self.current_time}s] [Macro] 链路预测完成，耗时 {t_cost:.2f} ms")

                    self._prediction_buffer.clear()

                # 2. 拓扑重构（强依赖第 1 步的输出）
                new_topo_data = self.topology_reconstruction_execution(predicted_links)
                self.update_topology_state(new_topo_data)
                self.notify_backend("topology_reconstruction_result", new_topo_data)
                self._print_macro_snapshot()

                # # 3. 任务规划
                # curr_topo, curr_node = self.get_current_states()
                # task_result = self.path_planning_execution(curr_topo, curr_node, self.task_queue)
                # self.notify_backend("task_planning_result", task_result)

            except Exception as e:
                print(f"[_macro_loop] 宏周期执行期间发生异常: {e}")
            finally:
                # 【核心修复】：无论正常还是异常，必须通知主线程完成，解除阻塞
                self._macro_done.set()

    def _micro_loop(self):
        """
        微周期线程：每秒执行一次，处理任务、更新流量、执行路由。
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

                # 1. 处理到达时间的规划任务
                self._process_pending_tasks()

                # 2. 背景流量生成（最新坐标）
                bg_flow_data, ret_timestamp = self.background_traffic_update()
                self.notify_backend("background_traffic_result", bg_flow_data)

                # ================= [新增] 预测窗口更新与触发 =================
                current_frame = self._build_prediction_frame()
                self._prediction_buffer.append(current_frame)

                # =============================================================

                # 3. 路由需求生成（基于最新拓扑和背景流量结果）
                #    必须在 background_traffic_update 之后调用，
                #    确保 topology_state["newTopology"] 里的 CurrentFlow 已经写回
                current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
                topo, node = self.get_current_states()
                routing_demand = routing_demand_generation(topo, node, current_utc)
                self.notify_backend("routing_demand_result", routing_demand)

                # 验证输出：打印关键字段确认执行成功
                task_count = len(routing_demand.get("RouteTaskList", []))
                ts = routing_demand.get("Timestamp", 0)
                print(f"[{self.current_time}s] 路由需求生成完成 | 生成任务数={task_count} | Timestamp={ts}")

                # 4. 路由优化执行
                self.route_planning_execution()

            except Exception as e:
                print(f"[_micro_loop] 微周期执行期间发生异常: {e}")
            finally:
                self._micro_done.set()

    def run_simulation(self, timestamp, constellation_id, tle_file_path, task_list):
        print("====== 仿真开始: 初始化阶段 ======")

        real_start_time = time.time()

        # 第0秒：初始化，内部完成拓扑构建和第0秒背景流量
        self.initialize_constellation(timestamp, constellation_id, tle_file_path, task_list)

        # 初始化完成后，赶在时间推进到 1s 之前，强行抓取 T=0 的状态推入窗口！
        initial_frame = self._build_prediction_frame()
        self._prediction_buffer.append(initial_frame)
        print(f"[0s] [Warmup] 装入 T=0 初始特征帧 {len(self._prediction_buffer)}/30")

        # 初始化完成后推进到第1秒，第0秒已处理完毕
        self.current_time = 1

        init_cost = time.time() - real_start_time
        print(f"初始化实质耗时: {init_cost:.2f} 秒")

        # 第一阶段：第1到(MACRO_PERIOD-1)秒，每秒更新卫星位置和背景流量
        print(f"====== 第一阶段: 空转预热阶段 (1s ~ {config.MACRO_PERIOD_SECONDS - 1}s) ======")
        _warmup_prev_time = None
        while self.current_time < config.MACRO_PERIOD_SECONDS:

            # 1. 更新卫星位置（基于当前逻辑时间推算真实UTC）
            self._update_satellite_positions()

            # 2. 更新背景流量
            bg_flow_data, ts = self.background_traffic_update()
            self.notify_backend("background_traffic_result", bg_flow_data)

            # ================= [新增：填入数据，不触发预测] =================
            current_frame = self._build_prediction_frame()
            self._prediction_buffer.append(current_frame)
            # 加一句打印，让你在控制台清楚看到 1s~29s 正在平稳装载数据
            # 加上 T=0 时装入的那一帧，这里打印的进度会从 2/30 一直走到 30/30
            print(f"[{self.current_time}s] [Warmup] 数据生成，用于预测: {len(self._prediction_buffer)}/30")
            # ================================================================

            # 3. 每5秒打印一次7点诊断
            if self.current_time % 5 == 0:
                _warmup_prev_time = self._print_snapshot("预热", _warmup_prev_time)

            # 4. 动态时延补偿
            target_time = real_start_time + self.current_time
            sleep_duration = target_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            self.current_time += 1

        # 第二阶段：完整宏/微周期循环
        print("====== 初始化结束，进入第二阶段: 完整周期循环 ======")

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

                # 微周期：背景流量 + 路由规划
                self._micro_done.clear()
                self._micro_event.set()
                self._micro_done.wait()

                # 每5秒输出一次物理量快照
                if self.current_time % 5 == 0:
                    _main_prev_time = self._print_snapshot("微周期", _main_prev_time)
                    #TODO 向前端推送当前状态快照数据，供可视化展示使用。

                # 动态时延补偿
                target_time = real_start_time + self.current_time
                sleep_duration = target_time - time.time()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                self.current_time += 1
        except KeyboardInterrupt:
            is_interrupted = True
            print("\n[Engine] 收到中断信号，正在终止仿真...")
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
                    
                # 关闭异步任务规划的线程池
                self.task_executor.shutdown(wait=False, cancel_futures=True)
                print("仿真系统退出。")
            else:
                print("\n====== 基础仿真已完成，底层驱动线程保持待命 ======")
                print("提示: 系统未关闭计算资源，可无缝接力调用 run_extended_simulation()。")


    def run_extended_simulation(self):
        """
        24小时扩展仿真：在60秒主仿真结束后接续运行。
        每小时（3600秒）为一个大周期，每小时输出一次物理量快照。
        卫星位置和背景流量每秒更新，宏/微周期线程继续运行。
        时延补偿已注释，以最快速度跑完（几分钟内完成24小时仿真）。
        """
        print("\n====== 开始24小时扩展仿真 ======")
    
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
                    current_utc = self.sim_start_utctime + datetime.timedelta(seconds=self.current_time)
                    print(f"\n{'═' * 54}")
                    print(f"  第 {hour:2d} 小时  UTC={current_utc.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'═' * 54}")
                    _prev_snapshot_time = self._print_snapshot("小时快照", _prev_snapshot_time)

                self.current_time += 1
        except KeyboardInterrupt:
            print("\n[Engine] 24小时扩展仿真被手动中断...")
        finally:
            print("\n====== 扩展仿真结束，清理所有底层资源 ======")
            self._stop_event.set()
            self._micro_event.set()
            self._macro_event.set()
            
            if self._macro_thread and self._macro_thread.is_alive():
                self._macro_thread.join(timeout=2.0)
            if self._micro_thread and self._micro_thread.is_alive():
                self._micro_thread.join(timeout=2.0)
                
            self.task_executor.shutdown(wait=False, cancel_futures=True)
            print("系统完全退出。")
