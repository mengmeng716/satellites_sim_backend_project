"""算法一：感知监测 —— 对外接口。

严格对应接口文档：
  * 表2-1 感知参数表（环境 / 物理 / 网络 / 业务 四组输出）
  * 表3-2 状态感知与监测查询接口（Coverage / Sampling / SensingMethod）

数据来源：仿真引擎的 ``TopologyState`` / ``SatelliteNodeState``。物理层量由仿真唯一的
链路物理实现 :func:`src.simulation.topology_link_state.evaluate_link_physics_batch` 计算；
网络层量由被动统计与主动探测两种方法对仿真真值做有限样本测量得到。

采样时序：引擎每个微周期（默认 5 s）调用一次 :func:`run_sensing_cycle`。关键参数按 1 s
间隔采样——在相邻两个微周期之间对链路距离线性插值（5 s 内轨道弧线与弦长之差小于 0.1 km），
流量取当时生效的那一周期的值，逐秒计算物理量并按真实采样时刻入库；非关键参数按 10 s 间隔采样。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .metrics import PARAM_SPECS, critical_params, non_critical_params
from .probes import ActiveProbeSensing, PassiveSensing, cross_check, sampling_impact_model
from .sampler import (
    CoverageTracker,
    SamplingPlan,
    SamplingScheduler,
    TelemetryStore,
    compute_param_health,
)

#: 表2-1 的四组输出
PARAM_GROUPS: Dict[str, List[str]] = {
    "Environment": ["AmbientTemperature", "TerminalTemperature"],
    "Physical": ["PathLoss", "SNR", "PropagateDelay", "BER", "LaserConfigRef"],
    "Network": ["PacketLossRate", "Delay", "DelayJitter", "Throughput"],
    "Business": ["HistoryFlow", "RemainingEnergy", "SatelliteLoad"],
}

#: 每个周期做交叉核对的链路数（轮转选取承载业务的链路）
CROSS_CHECK_LINKS_PER_CYCLE = 16
#: 链路可用性逐秒样本的保留时长（s），供容量评估统计可用率
AVAILABILITY_HISTORY_S = 300
#: 采样影响分析使用的参照序列长度（s）
IMPACT_HISTORY_S = 120


def collect_parameters(values: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """按表2-1 分组整理一次感知结果。"""
    out: Dict[str, Dict[str, Any]] = {}
    for group, names in PARAM_GROUPS.items():
        got = {n: values[n] for n in names if n in values}
        if got:
            out[group] = got
    return out


def query_status(
    coverage_report: Mapping[str, Any],
    health_report: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """组装表3-2 的三段输出。"""
    entity_cov = coverage_report.get("entity_coverage", {}) or {}
    coverage = [
        {
            "EntityType": entity_type,
            "Total": item.get("total", 0),
            "Sampled": item.get("sampled", 0),
            "CoverageRatio": item.get("coverage_ratio"),
        }
        for entity_type, item in entity_cov.items()
        if item.get("total", 0) > 0
    ]

    sampling = [
        {
            "ParamName": name,
            "Critical": bool(info.get("critical")),
            "PlannedInterval": info.get("planned_interval_s"),
            "ActualMaxGap": info.get("max_gap_s"),
            "LateCount": int(info.get("late_count", 0)),
            "MissingCount": int(info.get("missing_count", 0)),
        }
        for name, info in (health_report.get("params", {}) or {}).items()
    ]

    # 每种感知方法一条记录；一致性取该方法的全部交叉核对结果
    by_method: Dict[str, bool] = {}
    for item in checks:
        for key in ("passive_method", "active_method"):
            method_id = item.get(key)
            if not method_id:
                continue
            consistent = bool(item.get("consistent", False))
            by_method[method_id] = by_method.get(method_id, True) and consistent

    methods = [{"MethodId": method_id, "CrossCheckConsistent": consistent}
               for method_id, consistent in by_method.items()]

    return {"Coverage": {"EntityCoverage": coverage},
            "Sampling": sampling,
            "SensingMethod": methods}


class SensingMonitor:
    """感知监测的跨周期状态：采样计划、遥测库、调度、覆盖统计与上一帧链路快照。"""

    def __init__(self, plan: Optional[SamplingPlan] = None) -> None:
        self.plan = plan or SamplingPlan.default()
        self.store = TelemetryStore()
        self.scheduler = SamplingScheduler(self.plan)
        self.tracker = CoverageTracker()
        self.passive = PassiveSensing()
        self.active = ActiveProbeSensing()
        #: 逐秒链路可用性 [(sim_time_s, {"src->dst": bool})]
        self.availability_frames: Deque[Tuple[float, Dict[str, bool]]] = deque(maxlen=AVAILABILITY_HISTORY_S)
        #: 逐秒逐链路 SNR 参照序列 [(sim_time_s, 链路 ID 元组, 取值数组)]，用于采样频率影响分析
        self._reference: Deque[Tuple[float, Tuple[str, ...], np.ndarray]] = deque(maxlen=IMPACT_HISTORY_S)
        self._previous: Optional[Dict[str, Any]] = None
        self._cross_check_cursor = 0

    def availability_samples(self, since_s: Optional[float] = None) -> List[Dict[str, bool]]:
        """逐秒链路通断帧（Rx ≥ 灵敏度为通），供容量与效能评估统计可用率。"""
        return [frame for t, frame in self.availability_frames if since_s is None or t > since_s]


def _link_snapshot(topology_state) -> Dict[str, Any]:
    ids, keys, distances, flows, lasers, laser_types = [], [], [], [], [], []
    for src_id, links in (topology_state.new_topology or {}).items():
        for dst_id, attr in links.items():
            ids.append(f"{src_id}->{dst_id}")
            keys.append((str(src_id), str(dst_id)))
            distances.append(float(attr.link_distance))
            flows.append(float(attr.current_flow))
            lasers.append(attr.laser_config)
            laser_types.append(str(attr.laser_type))
    return {"ids": ids, "keys": keys, "distance": np.asarray(distances), "flow": np.asarray(flows),
            "lasers": lasers, "laser_types": laser_types}


def _summary(name: str, values: np.ndarray) -> Dict[str, Any]:
    spec = PARAM_SPECS[name]
    out: Dict[str, Any] = {"Unit": spec.unit, "EntityType": spec.entity_type, "Count": int(values.size)}
    if values.dtype.kind in "fiu":
        finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
        if finite.size:
            out.update(Mean=float(finite.mean()), Min=float(finite.min()), Max=float(finite.max()))
    else:
        kinds, counts = np.unique(values, return_counts=True)
        out["Distribution"] = {str(k): int(c) for k, c in zip(kinds, counts)}
    return out


def run_sensing_cycle(
    monitor: SensingMonitor,
    topology_state,
    node_state,
    sim_time_s: float,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """跑一次完整感知：逐秒采样 → 覆盖统计 → 参数健康 → 两种方法交叉核对。

    :param monitor: 跨周期保存状态的 :class:`SensingMonitor`
    :param topology_state: 仿真 ``TopologyState``
    :param node_state: 仿真 ``SatelliteNodeState``
    :param sim_time_s: 当前仿真时刻（s）
    :param seed: 测量噪声随机种子，缺省取仿真时刻
    :return: ``{"Timestamp", "Parameters"(表2-1), "Coverage", "Sampling", "SensingMethod"(表3-2),
               "SamplingImpact"}``
    """
    from src.simulation.topology_link_state import evaluate_link_physics_batch

    rng = np.random.default_rng(int(sim_time_s) if seed is None else seed)
    sim_time_s = float(sim_time_s)
    current = _link_snapshot(topology_state)
    link_ids = current["ids"]
    sat_items = sorted((node_state.sat_list or {}).items(), key=lambda item: int(item[0]))
    node_ids = [str(sid) for sid, _ in sat_items]

    # ---- 实体登记（拓扑变化时整体替换）----
    service_ids = {lid for lid, flow in zip(link_ids, current["flow"]) if flow > 0.0}
    monitor.tracker.replace_registry("node", set(node_ids))
    monitor.tracker.replace_registry("link", set(link_ids))
    monitor.tracker.replace_registry("topology", {"topology"})
    monitor.tracker.replace_registry("service", service_ids)
    for name, spec in PARAM_SPECS.items():
        monitor.store.register(name, node_ids if spec.entity_type == "node" else link_ids,
                               spec.unit, spec.entity_type)

    previous = monitor._previous
    window_start = previous["time"] if previous is not None else sim_time_s
    monitor.tracker.begin_window(window_start)

    # ---- 关键参数：上一帧之后逐秒采样到当前时刻 ----
    critical = critical_params()
    interval = min(monitor.plan.interval_of(name) for name in critical)
    if previous is not None and sim_time_s > previous["time"]:
        span = sim_time_s - previous["time"]
        sample_times = list(np.arange(previous["time"] + interval, sim_time_s + 1e-9, interval))
        prev_index = {lid: i for i, lid in enumerate(previous["ids"])}
        idx = np.array([prev_index.get(lid, -1) for lid in link_ids])
        known = idx >= 0
        prev_distance = np.where(known, previous["distance"][np.maximum(idx, 0)], current["distance"])
        prev_flow = np.where(known, previous["flow"][np.maximum(idx, 0)], current["flow"])
    else:
        span = 0.0
        sample_times = [sim_time_s]
        prev_distance = current["distance"]
        prev_flow = current["flow"]

    sampled_links = np.zeros(len(link_ids), dtype=bool)
    last_network = None
    for tau in sample_times:
        frac = (tau - (sim_time_s - span)) / span if span > 0 else 1.0
        distance = prev_distance + (current["distance"] - prev_distance) * frac
        flow = current["flow"] if tau >= sim_time_s - 1e-9 else prev_flow
        physics = evaluate_link_physics_batch(distance, current["lasers"], flow)

        base_one_way = physics["link_propagation_delay"] + physics["transmission_delay"]
        passive = monitor.passive.measure_links(flow, physics["packet_loss_rate"], base_one_way,
                                                physics["queue_delay"], interval, rng)
        active = monitor.active.measure_links(physics["packet_loss_rate"], base_one_way,
                                              physics["queue_delay"], rng)
        has = passive["has_traffic"]
        network = {
            "PacketLossRate": np.where(has, passive["packet_loss_rate"], active["packet_loss_rate"]),
            "Delay": np.where(has, passive["delay_ms"], active["delay_ms"]),
            "passive": passive,
            "physics": physics,
        }
        values = {
            "PathLoss": physics["path_loss"],
            "SNR": physics["snr"],
            "BER": physics["ber"],
            "PacketLossRate": network["PacketLossRate"],
            "Delay": network["Delay"],
        }
        frame_id = f"{tau:.0f}"
        for name in critical:
            monitor.store.update(name, values[name], tau, frame_id, monitor.plan.interval_of(name))
        monitor._reference.append((tau, tuple(link_ids), physics["snr"]))
        monitor.scheduler.mark_sampled(critical, tau)
        sampled_links |= np.all([np.isfinite(values[name]) for name in critical], axis=0)
        monitor.availability_frames.append(
            (tau, {lid: bool(up) for lid, up in zip(link_ids, physics["link_up"])}))
        last_network = network

    # ---- 非关键参数：按 10 s 间隔 ----
    due = monitor.scheduler.due(non_critical_params(), sim_time_s)
    if due:
        physics = last_network["physics"]
        passive = last_network["passive"]
        has = passive["has_traffic"]
        spread = np.sqrt(physics["queue_delay"] ** 2 + 1e-6)
        node_values = {
            "AmbientTemperature": np.array([float(a.ambient_temperature) for _, a in sat_items]),
            "TerminalTemperature": np.array([
                float(np.mean(a.terminal_temperature_list)) if a.terminal_temperature_list
                else float(a.ambient_temperature) for _, a in sat_items]),
            "HistoryFlow": np.array([float(a.flow) for _, a in sat_items]),
            "RemainingEnergy": np.array([float(a.energy_ratio) for _, a in sat_items]),
            "SatelliteLoad": np.array([float(a.node_load_ratio) for _, a in sat_items]),
        }
        link_values = {
            "PropagateDelay": physics["link_propagation_delay"],
            "LaserConfigRef": np.asarray(current["laser_types"]),
            "DelayJitter": np.where(has, passive["jitter_ms"], spread),
            "Throughput": np.where(has, passive["throughput_gbps"], 0.0),
        }
        merged = {**node_values, **link_values}
        for name in due:
            monitor.store.update(name, merged[name], sim_time_s, f"{sim_time_s:.0f}",
                                 monitor.plan.interval_of(name))
        monitor.scheduler.mark_sampled(due, sim_time_s)
        monitor.tracker.mark_sampled("node", set(node_ids))

    # ---- 覆盖 ----
    sampled_ids = {lid for lid, ok in zip(link_ids, sampled_links) if ok}
    monitor.tracker.mark_sampled("link", sampled_ids)
    monitor.tracker.mark_sampled("service", sampled_ids & service_ids)
    if sampled_ids:
        monitor.tracker.mark_sampled("topology", {"topology"})
    if not due and monitor.store.timestamps("HistoryFlow"):
        # 节点参数本周期未到采样时刻，覆盖按最近一次有效采样计
        monitor.tracker.mark_sampled("node", set(node_ids))

    # ---- 两种方法交叉核对（轮转选取承载业务的链路）----
    checks = []
    candidates = [i for i, lid in enumerate(link_ids) if lid in service_ids] or list(range(len(link_ids)))
    physics = last_network["physics"]
    for k in range(min(CROSS_CHECK_LINKS_PER_CYCLE, len(candidates))):
        i = candidates[(monitor._cross_check_cursor + k) % len(candidates)]
        base = float(physics["link_propagation_delay"][i] + physics["transmission_delay"][i])
        queue = float(physics["queue_delay"][i])
        plr = float(physics["packet_loss_rate"][i])
        window = monitor.passive.build_window(
            entity_id=link_ids[i], window_s=interval, offered_gbps=float(current["flow"][i]),
            packet_loss_rate=plr, base_one_way_ms=base, queue_delay_ms=queue, jitter_ms=0.0, rng=rng)
        p = monitor.passive.observe(window)
        a = monitor.active.probe(link_ids[i], sim_time_s, base, plr, queue, rng=rng)
        checks.append(cross_check(p, a))
    monitor._cross_check_cursor += CROSS_CHECK_LINKS_PER_CYCLE

    coverage = monitor.tracker.report()          # 先取报告：end_window 会清空本窗口的采样计数
    monitor.tracker.end_window(sim_time_s)
    health = compute_param_health(monitor.store, monitor.plan, sim_time_s)

    status = query_status(coverage, health, checks)
    status["Coverage"]["AllFullCoverage"] = bool(coverage.get("all_full_coverage"))
    status["Timestamp"] = sim_time_s
    status["Parameters"] = collect_parameters({
        name: _summary(name, monitor.store.latest(name))
        for name in monitor.store.names() if monitor.store.latest(name) is not None
    })
    # 参照序列取链路集合未变化的最近一段（拓扑重构会改变链路集合）
    segment = []
    for t, ids, snr in reversed(monitor._reference):
        if ids != tuple(link_ids):
            break
        segment.append((t, snr))
    segment.reverse()
    status["SamplingImpact"] = sampling_impact_model(
        [t for t, _ in segment], np.array([v for _, v in segment]) if segment else [],
        entity_count=len(link_ids), param_count=len(critical), unit=PARAM_SPECS["SNR"].unit)
    status["SamplingImpact"]["Param"] = "SNR"

    monitor._previous = {"time": sim_time_s, **current}
    return status
