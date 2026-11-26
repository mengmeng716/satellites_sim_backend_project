"""
功能：Walker 星座拓扑生成与状态更新模块
遵循数据结构: TopologyState.new_topology (Dict[str, Dict[str, LinksQualitiesValue]])
"""

from typing import Dict
import numpy as np
from src import config
from src.simulation.data_model import LinksQualitiesValue

SPEED_OF_LIGHT_KM_S = 299792.458


def _normalize_laser_payload(raw_payload) -> dict:
    payload = dict(raw_payload or {})
    payload.setdefault("Wavelength", float(getattr(config, "LASER_WAVELENGTH_NM", 1550.0)))
    payload.setdefault("ModulationBandwidth", float(getattr(config, "LASER_MODULATION_BANDWIDTH_HZ", 1.0e10)))
    payload.setdefault("TxPowerNominal", float(getattr(config, "LASER_TX_POWER_NOMINAL_DBM", 30.0)))
    payload.setdefault("RxSensitivity", float(getattr(config, "LASER_RX_SENSITIVITY_DBM", -35.0)))
    payload.setdefault(
        "RxPowerOperatingRange",
        list(getattr(config, "LASER_RX_POWER_OPERATING_RANGE_DBM", [-30.0, -10.0])),
    )
    payload.setdefault("OpticsEfficiency", float(getattr(config, "LASER_OPTICS_EFFICIENCY", 0.8)))
    payload.setdefault("PointingLossMax", float(getattr(config, "LASER_POINTING_LOSS_MAX_DB", 3.0)))
    return payload


def _laser_profile_by_type(laser_type: str) -> dict:
    presets = getattr(config, "LASER_CONFIG_PRESETS", {}) or {}
    default_type = str(getattr(config, "LASER_DEFAULT_TYPE", "STD"))
    selected_type = str(laser_type or default_type)
    payload = presets.get(selected_type)
    if payload is None:
        payload = presets.get(default_type, {})
    return _normalize_laser_payload(payload)


def _laser_type_by_direction(direction: str) -> str:
    direction_map = getattr(config, "SATELLITE_LASER_DIRECTION_TYPE", {}) or {}
    default_type = str(getattr(config, "LASER_DEFAULT_TYPE", "STD"))
    return str(direction_map.get(direction, default_type))


def laser_type_for_link(src_id, dst_id) -> str:
    """按链路方向确定激光器类型（与 init_walker_topology 的方向约定一致）。

    同轨面：目的槽位 = 源槽位 + 1 为 RIGHT，否则 LEFT；
    异轨面：目的轨道面 = 源轨道面 + 1 为 UP，否则 DOWN。
    """
    sats_per_plane = max(int(getattr(config, "SATS_PER_PLANE", 1)), 1)
    num_planes = max(int(getattr(config, "NUM_ORBIT_PLANES", 1)), 1)
    src_plane, src_slot = divmod(int(src_id), sats_per_plane)
    dst_plane, dst_slot = divmod(int(dst_id), sats_per_plane)
    if src_plane == dst_plane:
        direction = "RIGHT" if (dst_slot - src_slot) % sats_per_plane == 1 else "LEFT"
    else:
        direction = "UP" if (dst_plane - src_plane) % num_planes == 1 else "DOWN"
    return _laser_type_by_direction(direction)


def _laser_param_arrays(laser_cfgs):
    """把逐链路的 LaserConfig 展开成参数数组（同一对象只解析一次）。"""
    by_id = {}
    rows = []
    for cfg in laser_cfgs:
        key = id(cfg)
        if key not in by_id:
            by_id[key] = (
                float(cfg.wavelength) * 1e-9,
                float(cfg.optics_efficiency),
                float(cfg.tx_power_nominal),
                float(cfg.pointing_loss_max),
                float(cfg.rx_sensitivity),
                float(cfg.rx_power_operating_range[1]),
                float(cfg.modulation_bandwidth),
            )
        rows.append(by_id[key])
    table = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    names = ("wavelength_m", "optics_efficiency", "tx_dbm", "pointing_db",
             "sensitivity_dbm", "rx_upper_dbm", "modulation_bandwidth_hz")
    return {name: table[:, i] for i, name in enumerate(names)}


def evaluate_link_physics_batch(distance_km, laser_cfgs, current_flow_gbps) -> dict:
    """由激光器配置、链路距离与当前流量批量计算星间激光链路的物理层/网络层状态。

    全工程唯一的链路物理实现：引擎状态同步、感知监测、链路状态预测适配都调用本函数。

    链路预算（收发同型终端）::

        G        = 10·lg((π·D/λ)²)                     D = LASER_APERTURE_DIAMETER_M
        FSPL     = 20·lg(4π·d/λ)
        Rx       = Tx + 2·G + 20·lg(η) − FSPL − PointingLossMax
        PathLoss = Tx − Rx                              （净链路损耗，SignalIntensity = Tx − PathLoss）

    接收与速率::

        SNR    = min(Rx, Rx上限) − RxSensitivity + LASER_SNR_AT_SENSITIVITY_DB
                 （超过工作范围上限时按终端可调衰减器压到上限计算）
        BER    = QPSK(SNR + LASER_FEC_NET_CODING_GAIN_DB)       纠错后误码率
        Rate   = ModulationBandwidth                            Rx ≥ 灵敏度
               = ModulationBandwidth·10^((Rx−灵敏度)/10)        Rx < 灵敏度（每比特光子数不变的速率回退）
        AchievableRate = min(Rate, MAX_LINK_SPEED_GBPS)

    网络层::

        PLR  = 1 − (1 − (1−(1−BER)^帧长))·(1 − 过载丢弃比例)
        时延 = 传播 d/c + 发送 帧长/速率 + 排队 M/M/1 ρ/(1−ρ)·服务时间   （ms）

    :param distance_km: (N,) 链路距离 km
    :param laser_cfgs: 长度 N 的 LaserConfig 序列
    :param current_flow_gbps: (N,) 链路流量 Gbps
    :return: 各字段为 (N,) 数组的字典
    """
    from scipy.special import erfc

    distance = np.maximum(np.asarray(distance_km, dtype=np.float64), 1e-3)
    flow = np.maximum(np.asarray(current_flow_gbps, dtype=np.float64), 0.0)
    laser = _laser_param_arrays(laser_cfgs)
    aperture_m = float(getattr(config, "LASER_APERTURE_DIAMETER_M", 0.085))
    wavelength_m = laser["wavelength_m"]

    gain_db = 20.0 * np.log10(np.pi * aperture_m / wavelength_m)
    optics_db = 20.0 * np.log10(np.maximum(laser["optics_efficiency"], 1e-6))
    fspl_db = 20.0 * np.log10(4.0 * np.pi * distance * 1e3 / wavelength_m)
    tx_dbm = laser["tx_dbm"]
    rx_dbm = tx_dbm + 2.0 * gain_db + optics_db - fspl_db - laser["pointing_db"]

    sensitivity = laser["sensitivity_dbm"]
    rx_upper = laser["rx_upper_dbm"]
    snr_db = (np.minimum(rx_dbm, rx_upper) - sensitivity
              + float(getattr(config, "LASER_SNR_AT_SENSITIVITY_DB", 8.5)))
    coded_snr_linear = 10.0 ** ((snr_db + float(getattr(config, "LASER_FEC_NET_CODING_GAIN_DB", 9.0))) / 10.0)
    ber_post = np.clip(0.5 * erfc(np.sqrt(coded_snr_linear / 2.0)), 1e-15, 0.5)

    margin_db = rx_dbm - sensitivity
    line_rate_gbps = laser["modulation_bandwidth_hz"] / 1e9 * np.where(
        margin_db < 0.0, 10.0 ** (np.minimum(margin_db, 0.0) / 10.0), 1.0)
    rated_gbps = float(getattr(config, "MAX_LINK_SPEED_GBPS", 100.0))
    achievable_gbps = np.clip(line_rate_gbps, 0.0, rated_gbps)

    packet_bits = float(getattr(config, "LINK_PACKET_BITS", 12000))
    frame_loss = -np.expm1(packet_bits * np.log1p(-ber_post))
    safe_flow = np.where(flow > 0.0, flow, 1.0)
    overload_loss = np.where(flow > 0.0, np.maximum(0.0, (flow - achievable_gbps) / safe_flow), 0.0)
    packet_loss = np.clip(1.0 - (1.0 - frame_loss) * (1.0 - overload_loss), 0.0, 1.0)

    propagation_ms = distance / SPEED_OF_LIGHT_KM_S * 1e3
    safe_rate = np.where(achievable_gbps > 0.0, achievable_gbps, 1.0)
    service_ms = np.where(achievable_gbps > 0.0, packet_bits / (safe_rate * 1e9) * 1e3, 0.0)
    rho = np.where(achievable_gbps > 0.0, np.minimum(flow / safe_rate, 0.99), 0.0)
    queue_ms = rho / (1.0 - rho) * service_ms

    return {
        "path_loss": tx_dbm - rx_dbm,
        "rx_power_dbm": rx_dbm,
        "snr": snr_db,
        "ber": ber_post,
        "achievable_rate": achievable_gbps,
        "packet_loss_rate": packet_loss,
        "link_propagation_delay": propagation_ms,
        "transmission_delay": service_ms,
        "queue_delay": queue_ms,
        "link_up": margin_db >= 0.0,
        "rx_saturated": rx_dbm > rx_upper,
    }


def evaluate_link_physics(distance_km: float, laser_cfg, current_flow_gbps: float = 0.0) -> dict:
    """单条链路版本，公式见 :func:`evaluate_link_physics_batch`。"""
    batch = evaluate_link_physics_batch([distance_km], [laser_cfg], [current_flow_gbps])
    return {key: (bool(value[0]) if value.dtype == bool else float(value[0])) for key, value in batch.items()}


def _get_default_link_attr(laser_type: str) -> LinksQualitiesValue:
    """生成默认链路属性对象"""
    normalized_type = str(laser_type or getattr(config, "LASER_DEFAULT_TYPE", "STD"))
    return LinksQualitiesValue(
        link_distance=0.0,
        max_capacity=config.MAX_LINK_SPEED_GBPS,
        link_capacity=config.MAX_LINK_SPEED_GBPS,
        left_capacity=1.0,
        current_flow=0.0,
        cum_flow=0.0,
        link_propagation_delay=0.0,
        delay=0.0,
        delay_jitter=0.0,
        queue_delay=0.0,
        transmission_delay=0.0,
        path_loss=0.0,
        snr=0.0,
        ber=0.0,
        packet_loss_rate=0.0,
        throughput=0.0,
        heat_value=0.0,
        achievable_rate=config.MAX_LINK_SPEED_GBPS,
        load_fluctuation=0.0,
        laser_type=normalized_type,
        laser_config=_laser_profile_by_type(normalized_type),
    )


def init_walker_topology(constellation) -> Dict[str, Dict[str, LinksQualitiesValue]]:
    """
    初始化 Walker 星座拓扑 (仅确定连接关系)
    自动适配 3600 星座 (60x60) 和 432 星座 (24x18)
    """
    num_planes = constellation.numOrbitPlane
    sats_per_plane = constellation.numSatsInPlane
    P = num_planes
    S = sats_per_plane
    
    new_topology: Dict[str, Dict[str, LinksQualitiesValue]] = {}

    for sat_id in range(P * S):
        p = sat_id // S  # 轨道面索引
        s = sat_id % S   # 面内索引
        links: Dict[str, LinksQualitiesValue] = {}
        
        # 辅助函数：添加邻居
        def add_neighbor(target_id: int, direction: str):
            laser_type = _laser_type_by_direction(direction)
            links[str(target_id)] = _get_default_link_attr(laser_type)

        # 1. 面内连接 - 下一个
        s_next = (s + 1) % S
        add_neighbor(p * S + s_next, "RIGHT")

        # 2. 面内连接 - 上一个
        s_prev = (s - 1) % S
        add_neighbor(p * S + s_prev, "LEFT")

        # 3. 面间连接 - 下一个轨道面 & 4. 上一个轨道面
        p_next = (p + 1) % P
        p_prev = (p - 1) % P

        if P == 24 and S == 18:
            # --- 432 星座逻辑 (24x18: 奇偶面交替偏移 + 24 面闭环边界) ---
            if p == P - 1:
                target_s_next = (s + 4) % S
            elif p % 2 == 0:
                target_s_next = s
            else:
                target_s_next = (s - 1) % S

            if p == 0:
                target_s_prev = (s - 4) % S
            elif p % 2 == 0:
                target_s_prev = (s + 1) % S
            else:
                target_s_prev = s

            add_neighbor(p_next * S + target_s_next, "UP")
            add_neighbor(p_prev * S + target_s_prev, "DOWN")

        elif P == 60 and S == 60:
            # 3600 constellation: same alternating inter-plane pattern as 432,
            # with the 60-plane closing seam shifted by 38 satellites.
            if p == P - 1:
                target_s_next = (s - 38) % S
            elif p % 2 == 0:
                target_s_next = s
            else:
                target_s_next = (s - 1) % S

            if p == 0:
                target_s_prev = (s + 38) % S
            elif p % 2 == 0:
                target_s_prev = (s + 1) % S
            else:
                target_s_prev = s

            add_neighbor(p_next * S + target_s_next, "UP")
            add_neighbor(p_prev * S + target_s_prev, "DOWN")

        else:
            # --- 3600 星座及其他标准 Walker 逻辑 (保留原代码精髓) ---
            # 3. 面间连接 - 下一个轨道面
            if p == P - 1 and p_next == 0:
                target_s = (s - 38) % S  # 异向缝隙特化 (针对 3600 星座)
            else:
                target_s = (s + 59) % S  # 常规偏移 (59 等价于 -1 mod 60)
            add_neighbor(p_next * S + target_s, "UP")

            # 4. 面间连接 - 上一个轨道面
            if p == 0 and p_prev == P - 1:
                target_s = (s + 38) % S  # 异向缝隙特化 (针对 3600 星座)
            else:
                target_s = (s - 59) % S  # 常规偏移 (-59 等价于 +1 mod 60)
            add_neighbor(p_prev * S + target_s, "DOWN")

        new_topology[str(sat_id)] = links

    return new_topology


def satellite_cartesian_positions(constellation):
    """Return satellite positions in kilometers at the constellation's current time."""
    if constellation is None:
        return {}
    orbit_height_km = float(getattr(constellation, "orbitHeight", config.ORBIT_HEIGHT_KM))
    r = config.EARTH_RADIUS_KM + orbit_height_km
    coords = {}
    for sat in getattr(constellation, "satelliteList", []) or []:
        try:
            sid_int = int(sat.satId)
            phi = np.radians(sat.latitude)
            lam = np.radians(sat.longitude)
            x = r * np.cos(phi) * np.cos(lam)
            y = r * np.cos(phi) * np.sin(lam)
            z = r * np.sin(phi)
            coords[sid_int] = np.array([x, y, z])
        except (ValueError, AttributeError):
            continue
    return coords


def update_topology_snapshot(constellation, current_utc, topology: Dict[str, Dict[str, LinksQualitiesValue]]):
    """
    基于当前时间更新所有链路的距离与传播时延。
    直接在传入的 topology 字典中原地更新 LinksQualitiesValue 对象
    """
    coords = satellite_cartesian_positions(constellation)

    # 更新距离和时延
    for source_id_str, links in topology.items():
        try:
            source_id = int(source_id_str)
        except ValueError:
            continue
            
        if source_id not in coords:
            continue
        vec_source = coords[source_id]

        for target_id_str, link_attr in links.items():
            try:
                target_id = int(target_id_str)
            except ValueError:
                continue
                
            if target_id not in coords:
                continue
            vec_target = coords[target_id]

            # 计算欧氏距离
            distance = float(np.linalg.norm(vec_source - vec_target))
            
            # 原地更新 LinksQualitiesValue 对象属性
            link_attr.link_distance = distance
            link_attr.link_propagation_delay = (distance / SPEED_OF_LIGHT_KM_S) * 1000.0
            link_attr.max_capacity = config.MAX_LINK_SPEED_GBPS
            link_attr.link_capacity = config.MAX_LINK_SPEED_GBPS
            link_attr.current_flow = max(0.0, float(link_attr.current_flow))
            utilization = link_attr.current_flow / max(config.MAX_LINK_SPEED_GBPS, 1e-9)
            link_attr.left_capacity = max(0.0, min(1.0, 1.0 - utilization))
            link_attr.heat_value = max(0.0, min(1.0, utilization))
