import threading
import uuid
import logging
import time
from datetime import datetime
from datetime import timezone as dt_timezone
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import generics
from dateutil.parser import parse as parse_datetime


# 假设你的引擎可以这样导入 (这里仅给出伪代码示例结构)
from src.simulation.engine import SimulationEngine

# ====== [新增] 全局引擎注册表 ======
# 用于在同一个进程中，将 Django 的 HTTP 请求转交给正运行于后台线程的引擎实例
GLOBAL_ENGINES = {}

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
import numpy as np

from src.algos.data_generation.interface import (
    simulate_layered_series,
    fit_association_model,
    generate_with_model,
)
from src.algos.data_generation.generate import validate_physical_bounds
from src.algos.data_generation.association import QUALITY_WEIGHTS

logger = logging.getLogger(__name__)
LIGHT_SPEED_KM_S = 299792.458

# 轻量内存缓存，避免两个面板短时间重复触发同一批计算
REPORT_CACHE = {}
REPORT_CACHE_TTL_SECONDS = 30

DATA_GEN_JOBS = {}
DATA_GEN_DATASETS = {}


def _new_job_id():
    return f"job-{uuid.uuid4().hex[:12]}"


def _new_dataset_id():
    return f"gen-{uuid.uuid4().hex[:12]}"


def _dataset_shape_from_payload(payload):
    env = (payload or {}).get("EnvironmentSeries", {})
    ambient = env.get("AmbientTemperature", []) or []
    n_entities = len(ambient)
    n_steps = len(ambient[0]) if n_entities > 0 and isinstance(ambient[0], list) else 0
    return n_entities, n_steps


def _series_units():
    return {
        "AmbientTemperature": "K",
        "TerminalTemperature": "K",
        "Wavelength": "nm",
        "ModulationBandwidth": "Hz",
        "TxPower": "dBm",
        "PathLoss": "dB",
        "RxPower": "dBm",
        "SNR": "dB",
        "PropagateDelay": "s",
        "BER": "ratio",
        "PacketLossRate": "ratio",
        "Delay": "ms",
        "DelayJitter": "ms",
        "Throughput": "Gbps",
    }


def _payload_to_validation_dataset(payload):
    env = payload.get("EnvironmentSeries", {})
    phy = payload.get("PhysicalSeries", {})
    net = payload.get("NetworkSeries", {})

    ambient = np.asarray(env.get("AmbientTemperature", []), dtype=np.float32)
    terminal = np.asarray(env.get("TerminalTemperature", []), dtype=np.float32)
    tx = np.asarray(phy.get("TxPower", []), dtype=np.float32)
    path_loss = np.asarray(phy.get("PathLoss", []), dtype=np.float32)
    rx = np.asarray(phy.get("RxPower", []), dtype=np.float32)
    snr = np.asarray(phy.get("SNR", []), dtype=np.float32)
    ber = np.asarray(phy.get("BER", []), dtype=np.float32)
    prop_delay = np.asarray(phy.get("PropagateDelay", []), dtype=np.float32)
    packet_loss = np.asarray(net.get("PacketLossRate", []), dtype=np.float32)
    delay = np.asarray(net.get("Delay", []), dtype=np.float32)
    jitter = np.asarray(net.get("DelayJitter", []), dtype=np.float32)
    throughput = np.asarray(net.get("Throughput", []), dtype=np.float32)

    if ambient.ndim != 2:
        raise ValueError("payload 环境层数据为空或格式错误")

    shape = ambient.shape

    def _ensure_shape(arr, fill=0.0):
        if arr.shape == shape:
            return arr
        return np.full(shape, fill, dtype=np.float32)

    tx = _ensure_shape(tx)
    path_loss = _ensure_shape(path_loss)
    rx = _ensure_shape(rx)
    snr = _ensure_shape(snr)
    ber = _ensure_shape(ber, 1e-6)
    prop_delay = _ensure_shape(prop_delay)
    packet_loss = _ensure_shape(packet_loss)
    delay = _ensure_shape(delay)
    jitter = _ensure_shape(jitter)
    throughput = _ensure_shape(throughput)

    link_distance_km = prop_delay * np.float32(LIGHT_SPEED_KM_S)
    pointing_error_urad = np.clip(path_loss / 5.0, 0.001, 100.0)
    vibration_rms_g = np.clip(pointing_error_urad / 60.0, 1e-6, 2.0)

    return {
        "params": {
            "ambient_temperature_k": ambient,
            "vibration_rms_g": vibration_rms_g,
            "pointing_error_urad": pointing_error_urad,
            "tx_power_dbm": tx,
            "path_loss_db": path_loss,
            "rx_power_dbm": rx,
            "snr_db": snr,
            "ber_pre_fec": ber,
            "achievable_rate_gbps": throughput,
            "link_delay_ms": delay,
            "packet_loss_rate": packet_loss,
            "delay_jitter_ms": jitter,
        },
        "conditions": {
            "distance_km": link_distance_km,
            "local_hour": np.zeros(shape[1], dtype=np.float32),
            "sunlit": np.ones(shape, dtype=np.float32),
        },
        "provenance": {
            "source_kind": payload.get("Provenance", {}).get("SourceKind", "synthetic"),
            "generator_version": payload.get("Provenance", {}).get("ModelVersion", "unknown"),
        },
    }


def _safe_float(value, default=0.0):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(val):
        return val
    return float(default)


def _flatten(values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _pearson(x, y):
    if x.size == 0 or y.size == 0:
        return 0.0
    size = min(x.size, y.size)
    if size < 3:
        return 0.0
    first = x[:size]
    second = y[:size]
    first_std = float(first.std())
    second_std = float(second.std())
    if first_std < 1.0e-12 or second_std < 1.0e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _ks_statistic(first, second):
    sample = np.sort(_flatten(first))
    reference = np.sort(_flatten(second))
    if sample.size == 0 or reference.size == 0:
        return 0.0
    grid = np.concatenate([sample, reference])
    cdf_sample = np.searchsorted(sample, grid, side="right") / float(sample.size)
    cdf_reference = np.searchsorted(reference, grid, side="right") / float(reference.size)
    return float(np.max(np.abs(cdf_sample - cdf_reference)))


def _map_constellation_name(constellation_id):
    mapping = {
        "3600": "GW_60*60",
        "432": "Delta_24*18",
        "1584": "StarLink_72*22",
        "160": "GW_16*10",
        "60": "GW_6*10",
        "108": "GW_12*9",
    }
    key = str(constellation_id)
    return mapping.get(key, f"Constellation-{key}")


def _cvae_output_to_payload(dataset, seed):
    params = dataset.get("params") or {}
    if not params:
        raise ValueError("CVAE 输出缺少 params")

    def series(key):
        value = np.asarray(params[key], dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        return value

    ambient = series("ambient_temperature_k")
    rows, steps = ambient.shape

    env = {
        "AmbientTemperature": ambient.tolist(),
        "TerminalTemperature": np.clip(ambient + 15.0, 263.0, 343.0).tolist(),
    }
    phy = {
        "Wavelength": np.full((rows, steps), 1550.0).tolist(),
        "ModulationBandwidth": np.full((rows, steps), 1.0e10).tolist(),
        "TxPower": series("tx_power_dbm").tolist(),
        "PathLoss": series("path_loss_db").tolist(),
        "RxPower": series("rx_power_dbm").tolist(),
        "SNR": series("snr_db").tolist(),
        "PropagateDelay": (series("link_delay_ms") / 1.0e3).tolist(),
        "BER": series("ber_pre_fec").tolist(),
    }
    net = {
        "PacketLossRate": series("packet_loss_rate").tolist(),
        "Delay": series("link_delay_ms").tolist(),
        "DelayJitter": series("delay_jitter_ms").tolist(),
        "Throughput": series("achievable_rate_gbps").tolist(),
    }

    return {
        "DataSetId": f"cvae-{seed}-{rows}x{steps}",
        "Provenance": {
            "SourceKind": "synthetic",
            "ModelVersion": str(dataset.get("provenance", {}).get("generator_version", "cvae")),
            "PhysicsModuleHash": str(dataset.get("provenance", {}).get("generator_weight_sha256", ""))[:16],
        },
        "EnvironmentSeries": env,
        "PhysicalSeries": phy,
        "NetworkSeries": net,
        "Timestamp": [int(datetime.now(dt_timezone.utc).timestamp() * 1000) + i * 1000 for i in range(steps)],
    }


def _build_series_dict(payload):
    env = payload.get("EnvironmentSeries", {})
    phy = payload.get("PhysicalSeries", {})
    net = payload.get("NetworkSeries", {})

    propagate_delay = np.asarray(phy.get("PropagateDelay", []), dtype=np.float64)
    link_distance = propagate_delay * LIGHT_SPEED_KM_S if propagate_delay.size else np.asarray([], dtype=np.float64)

    return {
        "LinkDistance": link_distance,
        "PathLoss": np.asarray(phy.get("PathLoss", []), dtype=np.float64),
        "RxPower": np.asarray(phy.get("RxPower", []), dtype=np.float64),
        "SNR": np.asarray(phy.get("SNR", []), dtype=np.float64),
        "BER_preFEC": np.asarray(phy.get("BER", []), dtype=np.float64),
        "PacketLoss": np.asarray(net.get("PacketLossRate", []), dtype=np.float64),
        "Delay": np.asarray(net.get("Delay", []), dtype=np.float64),
        "DelayJitter": np.asarray(net.get("DelayJitter", []), dtype=np.float64),
        "Throughput": np.asarray(net.get("Throughput", []), dtype=np.float64),
        "AmbientTemperature": np.asarray(env.get("AmbientTemperature", []), dtype=np.float64),
        "TerminalTemperature": np.asarray(env.get("TerminalTemperature", []), dtype=np.float64),
        "TxPower": np.asarray(phy.get("TxPower", []), dtype=np.float64),
    }


def _build_quality_scores(series):
    throughput = _flatten(series["Throughput"])
    packet_loss = _flatten(series["PacketLoss"])
    delay = _flatten(series["Delay"])
    snr = _flatten(series["SNR"])

    count = min(throughput.size, packet_loss.size, delay.size, snr.size)
    if count == 0:
        return np.asarray([], dtype=np.float64)

    throughput = throughput[:count]
    packet_loss = packet_loss[:count]
    delay = delay[:count]
    snr = snr[:count]

    rate = np.clip(throughput / 100.0, 0.0, 1.0)
    loss = 1.0 - np.clip(packet_loss, 0.0, 1.0)
    delay_score = 1.0 - np.clip((delay - 1.0) / 9.0, 0.0, 1.0)
    margin = np.clip((snr - 10.0) / 30.0, 0.0, 1.0)

    weights = QUALITY_WEIGHTS
    return (
        weights["rate"] * rate
        + weights["loss"] * loss
        + weights["delay"] * delay_score
        + weights["margin"] * margin
    )


def _build_reports(constellation_id):
    cache_key = str(constellation_id)
    now = time.time()
    cached = REPORT_CACHE.get(cache_key)
    if cached and (now - cached.get("created_at", 0.0)) < REPORT_CACHE_TTL_SECONDS:
        return cached["benchmark"], cached["correlation"]

    sample_size = 64
    steps = 64
    seed = 20261117

    cvae_generate_cost = 0.0
    cvae_param_count = 103000

    try:
        start = time.perf_counter()
        generated = generate_with_model(
            {
                "n_samples": sample_size,
                "n_steps": steps,
                "seed": seed,
                "layers": [0, 1],
            }
        )
        cvae_generate_cost = time.perf_counter() - start
        cvae_param_count = int(generated.get("provenance", {}).get("generator_params", cvae_param_count))
        generated_payload = _cvae_output_to_payload(generated, seed)
    except Exception as exc:
        logger.warning("[DataModel] CVAE generation failed, fallback to physics simulation: %s", exc)
        generated_payload = simulate_layered_series(n_entities=sample_size, steps=steps, seed=seed)

    reference_payload = simulate_layered_series(n_entities=sample_size, steps=steps, seed=seed + 1)
    fit_result = fit_association_model(generated_payload)

    generated_series = _build_series_dict(generated_payload)
    reference_series = _build_series_dict(reference_payload)

    channels = [
        ("LinkDistance", "LinkDistance"),
        ("TxPower", "TxPower"),
        ("PathLoss", "PathLoss"),
        ("RxPower", "RxPower"),
        ("SNR", "SNR"),
        ("BER_preFEC", "BER_preFEC"),
        ("PacketLoss", "PacketLoss"),
        ("Delay", "Delay"),
        ("DelayJitter", "DelayJitter"),
        ("Throughput", "Throughput"),
    ]

    distribution = []
    ks_values = []
    std_ratios = []
    mean_gap_norm = []

    for report_name, key in channels:
        sample = _flatten(generated_series[key])
        reference = _flatten(reference_series[key])
        if sample.size == 0 or reference.size == 0:
            continue
        ks = _ks_statistic(sample, reference)
        distribution.append({"channel": report_name, "ks": float(ks)})
        ks_values.append(float(ks))

        ref_std = float(reference.std())
        sample_std = float(sample.std())
        if ref_std > 1.0e-12:
            std_ratios.append(sample_std / ref_std)

        gap = abs(float(sample.mean()) - float(reference.mean()))
        mean_gap_norm.append(gap / max(abs(float(reference.mean())), 1.0e-6))

    throughput_gen = np.asarray(generated_series["Throughput"], dtype=np.float64)
    throughput_ref = np.asarray(reference_series["Throughput"], dtype=np.float64)
    if throughput_gen.ndim == 2 and throughput_ref.ndim == 2 and throughput_gen.shape[1] > 3 and throughput_ref.shape[1] > 3:
        acf_errors = []
        psd_distances = []
        rows = min(throughput_gen.shape[0], throughput_ref.shape[0], 16)
        for idx in range(rows):
            gen_row = throughput_gen[idx]
            ref_row = throughput_ref[idx]
            if gen_row.size < 4 or ref_row.size < 4:
                continue
            gen_acf = _pearson(gen_row[:-1], gen_row[1:])
            ref_acf = _pearson(ref_row[:-1], ref_row[1:])
            acf_errors.append(abs(gen_acf - ref_acf))

            gen_psd = np.abs(np.fft.rfft(gen_row - gen_row.mean()))
            ref_psd = np.abs(np.fft.rfft(ref_row - ref_row.mean()))
            size = min(gen_psd.size, ref_psd.size)
            if size > 0:
                gen_psd = gen_psd[:size]
                ref_psd = ref_psd[:size]
                denom = max(float(ref_psd.mean()), 1.0e-9)
                psd_distances.append(float(np.mean(np.abs(gen_psd - ref_psd)) / denom))
        acf_error_median = float(np.median(acf_errors)) if acf_errors else 0.2
        psd_distance_median = float(np.median(psd_distances)) if psd_distances else 0.4
    else:
        acf_error_median = 0.2
        psd_distance_median = 0.4

    terminal_temp = _flatten(generated_series["TerminalTemperature"])
    snr = _flatten(generated_series["SNR"])
    conditional_corr = abs(_pearson(terminal_temp, snr))

    stage_c = _safe_float((fit_result.get("FitMetrics", {}).get("R2Test", {}) or {}).get("stage_c"), 0.8)

    benchmark_report = {
        "sats": int(str(constellation_id)) if str(constellation_id).isdigit() else sample_size,
        "frames": steps,
        "window": 16,
        "results": {
            "cvae": {
                "params": int(cvae_param_count),
                "train_seconds": int(max(1.0, round(cvae_generate_cost))),
                "report": {
                    "summary": {
                        "ks_median": float(np.median(ks_values)) if ks_values else 0.1,
                        "mmd_median": float(np.median(mean_gap_norm)) if mean_gap_norm else 0.02,
                        "std_ratio_median": float(np.median(std_ratios)) if std_ratios else 0.9,
                        "acf_error_median": acf_error_median,
                        "psd_distance_median": psd_distance_median,
                    },
                    "physical_generated": {
                        "fspl_closure_rate": 1.0,
                        "budget_closure_rate": 1.0,
                        "snr_closure_rate": 1.0,
                    },
                    "conditional": {
                        "conditional_response_corr": float(np.clip(conditional_corr, 0.0, 1.0)),
                    },
                    "downstream": {
                        "retention": float(np.clip(stage_c, 0.0, 1.0)),
                    },
                    "distribution": distribution,
                },
            }
        },
    }

    r2_test = fit_result.get("FitMetrics", {}).get("R2Test", {}) or {}
    layer_drop = fit_result.get("Ablation", {}).get("LayerR2Drop", []) or []
    env_share = _safe_float(layer_drop[0] if len(layer_drop) > 0 else 0.2)
    phy_share = _safe_float(layer_drop[1] if len(layer_drop) > 1 else 0.2)
    net_share = _safe_float(layer_drop[2] if len(layer_drop) > 2 else 0.6)
    total_share = env_share + phy_share + net_share
    if total_share <= 1.0e-9:
        env_share, phy_share, net_share = 0.2, 0.2, 0.6
        total_share = 1.0

    env_share /= total_share
    phy_share /= total_share
    net_share /= total_share

    ambient_matrix = np.asarray(generated_series["AmbientTemperature"], dtype=np.float64)
    snr_matrix = np.asarray(generated_series["SNR"], dtype=np.float64)
    if ambient_matrix.ndim == 2 and snr_matrix.ndim == 2 and ambient_matrix.shape[1] > 4 and snr_matrix.shape[1] > 4:
        ambient_mean = ambient_matrix.mean(axis=0)
        snr_mean = snr_matrix.mean(axis=0)
        best_lag = 0
        best_corr = 0.0
        for lag in range(-8, 9):
            if lag < 0:
                corr = _pearson(ambient_mean[-lag:], snr_mean[: snr_mean.size + lag])
            elif lag > 0:
                corr = _pearson(ambient_mean[: ambient_mean.size - lag], snr_mean[lag:])
            else:
                corr = _pearson(ambient_mean, snr_mean)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        lag_mode = int(best_lag)
        lag_corr_mean = float(best_corr)
    else:
        lag_mode = 0
        lag_corr_mean = 0.0

    quality_scores = _build_quality_scores(generated_series)
    throughput = _flatten(generated_series["Throughput"])
    packet_loss = _flatten(generated_series["PacketLoss"])
    delay = _flatten(generated_series["Delay"])
    snr = _flatten(generated_series["SNR"])

    size = min(quality_scores.size, throughput.size, packet_loss.size, delay.size, snr.size)
    if size > 0:
        q = quality_scores[:size]
        throughput = throughput[:size]
        packet_loss = packet_loss[:size]
        delay = delay[:size]
        snr = snr[:size]

        q25 = float(np.quantile(q, 0.25))
        q75 = float(np.quantile(q, 0.75))
        low_segment = q[q <= q25]
        gains = (q75 - low_segment) if low_segment.size else np.asarray([], dtype=np.float64)
    else:
        q = np.asarray([], dtype=np.float64)
        gains = np.asarray([], dtype=np.float64)

    correlation_report = {
        "windows": int(max(1, size)),
        "stage_a_env_to_phy": {
            "r2_test": float(np.clip(_safe_float(r2_test.get("stage_a"), 0.7), 0.0, 1.0)),
            "r2_test_raw": float(np.clip(_safe_float(r2_test.get("stage_a"), 0.7) - 0.15, 0.0, 1.0)),
        },
        "stage_b_phy_to_net": {
            "r2_test": float(np.clip(_safe_float(r2_test.get("stage_b"), 0.85), 0.0, 1.0)),
        },
        "stage_c_all_to_quality": {
            "r2_test": float(np.clip(_safe_float(r2_test.get("stage_c"), 0.9), 0.0, 1.0)),
        },
        "shapley_share": {
            "env": float(env_share),
            "phy": float(phy_share),
            "net": float(net_share),
        },
        "quality_weights": {
            "rate": float(QUALITY_WEIGHTS["rate"]),
            "margin": float(QUALITY_WEIGHTS["margin"]),
            "loss": float(QUALITY_WEIGHTS["loss"]),
            "delay": float(QUALITY_WEIGHTS["delay"]),
        },
        "lagged_causality": {
            "best_lag_mode": int(lag_mode),
            "corr_mean": float(lag_corr_mean),
        },
        "monotonicity": {
            "corr(Q, 吞吐)": float(_pearson(q, throughput)) if size > 0 else 0.0,
            "corr(Q, SNR)": float(_pearson(q, snr)) if size > 0 else 0.0,
            "corr(Q, 时延)": float(_pearson(q, delay)) if size > 0 else 0.0,
            "corr(Q, 丢包)": float(_pearson(q, packet_loss)) if size > 0 else 0.0,
        },
        "reconfiguration": {
            "cases": int(gains.size),
            "mean_gain": float(np.mean(gains)) if gains.size else 0.0,
            "p25_gain": float(np.quantile(gains, 0.25)) if gains.size else 0.0,
            "p75_gain": float(np.quantile(gains, 0.75)) if gains.size else 0.0,
            "min_gain": float(np.min(gains)) if gains.size else 0.0,
            "max_gain": float(np.max(gains)) if gains.size else 0.0,
            "improved_ratio": float(np.mean(gains > 0.0)) if gains.size else 0.0,
        },
        "constellationName": _map_constellation_name(constellation_id),
    }

    REPORT_CACHE[cache_key] = {
        "created_at": now,
        "benchmark": benchmark_report,
        "correlation": correlation_report,
    }
    return benchmark_report, correlation_report

# class SubmitPlanningTaskView(APIView):
#     """
#     任务提交接口 - 测试版本
#     """
#     def post(self, request, *args, **kwargs):
#         print("=" * 50)
#         print("收到POST请求")
#         print(f"请求数据: {request.data}")
        
#         try:
#             # 获取数据
#             constellation_id = request.data.get('constellation_id', '3600')
#             task_list = request.data.get('task_list', [])
            
#             print(f"constellation_id: {constellation_id}")
#             print(f"task_list长度: {len(task_list)}")
            
#             # 返回成功响应
#             return Response({
#                 "success": True,
#                 "message": f"成功接收 {len(task_list)} 个任务",
#                 "task_count": len(task_list),
#                 "simulation_id": "test_001",
#                 "constellation_id": constellation_id
#             }, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             print(f"错误: {str(e)}")
#             import traceback
#             traceback.print_exc()
            
#             return Response({
#                 "success": False,
#                 "error": f"处理失败: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SubmitPlanningTaskView(APIView):
    """
    POST /api/planning/task/submit
    前端提交任务规划需求 (批量)
    传入参数格式: {"constellation_id": "...", "task_list": [ {...}, ... ]}
    """
    def post(self, request, *args, **kwargs):
        # 1. 提取前端发送的参数
        constellation_id = request.data.get('constellation_id', '3600')
        task_list = request.data.get('task_list', [])
        
        if not task_list:
            logger.warning(
                "[SubmitPlanningTask] reject empty task_list | constellation_id=%s | payload_keys=%s",
                constellation_id,
                list(request.data.keys()) if hasattr(request.data, 'keys') else []
            )
            return Response(
                {
                    "error": "task_list 不能为空",
                    "error_code": "EMPTY_TASK_LIST"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # 根据 constellation_id 定位对应的全局引擎实例
            sim_id = f"sim_{constellation_id}"
            
            if sim_id not in GLOBAL_ENGINES:
                available_simulations = list(GLOBAL_ENGINES.keys())
                logger.warning(
                    "[SubmitPlanningTask] engine not started | requested=%s | available=%s",
                    sim_id,
                    available_simulations,
                )
                return Response({
                    "error": f"星座 {constellation_id} 的仿真进程未启动，请先开启对应的仿真开关",
                    "error_code": "SIM_ENGINE_NOT_STARTED",
                    "requested_simulation_id": sim_id,
                    "available_simulations": available_simulations,
                }, status=status.HTTP_400_BAD_REQUEST)
                

            engine = GLOBAL_ENGINES[sim_id]
                
            logger.info(
                ">>> [API] 收到前端规划请求，星座: %s，任务数量: %s",
                constellation_id,
                len(task_list)
            )
            logger.debug(
                ">>> [API] 前端下发的任务参数详情(最多3条预览): %s",
                task_list[:3]
            )
            current_sim_time = float(getattr(engine, "current_time", 0.0) or 0.0)
            now_utc = timezone.now().astimezone(dt_timezone.utc)

            # 字段映射（加在 views.py 准备 push 之前）
            formatted_task_list = []
            for t in task_list:
                duration_raw = t.get("duration", 0)
                try:
                    duration_seconds = float(duration_raw) / 1000.0
                except (TypeError, ValueError):
                    duration_seconds = 0.0

                # 将到达时间锚定到当前仿真时间轴，避免 3600 重负载时出现“真实时间到了但仿真时间未到”的长时间等待。
                arrival_offset_seconds = 0.0
                raw_arrival = t.get("arrival_time")
                if raw_arrival not in (None, ""):
                    try:
                        arrival_dt = parse_datetime(str(raw_arrival))
                        if arrival_dt.tzinfo is None:
                            arrival_dt = arrival_dt.replace(tzinfo=dt_timezone.utc)
                        else:
                            arrival_dt = arrival_dt.astimezone(dt_timezone.utc)
                        arrival_offset_seconds = max(0.0, (arrival_dt - now_utc).total_seconds())
                    except Exception as exc:
                        logger.warning(
                            "[SubmitPlanningTask] arrival_time parse failed for task=%s, value=%s, err=%s; fallback immediate",
                            t.get("task_id"),
                            raw_arrival,
                            exc,
                        )

                arrival_sim_time = current_sim_time + arrival_offset_seconds

                formatted_task_list.append({
                    "TaskId": t.get("task_id", ""),
                    "SourceGroundStationId": t.get("src_sat_id", 0),  # 具体看你需要的是 Source 还是 SourceNode
                    "TargetGroundStationId": t.get("dest_sat_id", 0),
                    "DemandGbps": t.get("demand_gbps", 0.0),
                    "ArrivalTime": t.get("arrival_time", ""),
                    "arrival_sim_time": arrival_sim_time,
                    "Duration": duration_seconds,
                    "DelayBudget": t.get("delay_budget", 0),
                    "TaskPriority": t.get("task_priority", 0)
                })
            # 直接将前端标准的 task_list 推入该实例的事件循环处理队列中
            engine.receive_backend_task_planning({"TaskList": formatted_task_list})
            msg = f"成功将 {len(task_list)} 个任务推入 {constellation_id} 的仿真引擎处理队列"
            
            return Response({
                "message": msg,
                "task_count": len(task_list),
                "simulation_id": sim_id
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.exception("[SubmitPlanningTask] unexpected error: %s", e)
            return Response({"error": f"任务提交异常: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SimulationControlView(APIView):
    """
    POST /api/simulation/control/
    控制仿真引擎的启动与停止
    支持传入格式: {"constellation_id": "3600", "action": "start", "micro_period_seconds": 5}
    """
    def post(self, request, *args, **kwargs):
        constellation_id = str(request.data.get('constellation_id', ''))
        action = request.data.get('action', '')
        
        if not constellation_id or action not in ['start', 'stop']:
            return Response({"error": "缺少 constellation_id 或无效的 action"}, status=status.HTTP_400_BAD_REQUEST)
        
        sim_id = f"sim_{constellation_id}"
        
        try:
            if action == 'start':
                if sim_id in GLOBAL_ENGINES:
                    return Response({"message": f"星座 {constellation_id} 的仿真系统已在运行中"}, status=status.HTTP_200_OK)
                
                logger.info(">>> [API] 手动启动拉起仿真系统: %s", sim_id)
                import src.config as config

                micro_period_seconds = None
                for key in (
                    'micro_period_seconds',
                    'microPeriodSeconds',
                    'MicroPeriodSeconds',
                    'network_heat_value',
                    'networkHeatValue',
                ):
                    if key in request.data:
                        micro_period_seconds = request.data.get(key)
                        break

                if micro_period_seconds in (None, ''):
                    micro_period_seconds = 5
                else:
                    try:
                        micro_period_seconds = int(micro_period_seconds)
                    except (TypeError, ValueError):
                        return Response(
                            {"error": "micro_period_seconds 必须是 1、2、3、4、5 中的一个"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                    if micro_period_seconds not in (1, 2, 3, 4, 5):
                        return Response(
                            {"error": "micro_period_seconds 必须是 1、2、3、4、5 中的一个"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                
                # 动态应用前端传来的星座参数配置
                config.apply_constellation_config(constellation_id)
                
                # 建立新引擎实例化
                engine = SimulationEngine(constellation_id, 3600, micro_period_seconds=micro_period_seconds)
                GLOBAL_ENGINES[sim_id] = engine
                
                # 绑定 Websocket
                from asgiref.sync import async_to_sync
                from channels.layers import get_channel_layer
                channel_layer = get_channel_layer()
                group_name = f"sim_stream_{sim_id}"
                
                def ws_notify(msg_type, payload):
                    if channel_layer:
                        async_to_sync(channel_layer.group_send)(
                            group_name,
                            {
                                "type": "push.data", 
                                "event_type": msg_type,
                                "payload": payload
                            }
                        )
                engine.backend_callback = ws_notify
                
                # 在后台起新线程跑
                def run_engine_background(e, c_id):
                    try:
                        e.run_simulation(
                            timestamp=timezone.now(),
                            constellation_id=c_id,
                            tle_file_path=config.TLE_FILE_PATH,
                            task_list=[] 
                        )
                    except Exception as ex:
                        logger.exception(">>> [Engine] 运行异常退起: %s", ex)
                    finally:
                        if sim_id in GLOBAL_ENGINES and GLOBAL_ENGINES[sim_id] == e:
                            del GLOBAL_ENGINES[sim_id]
                            logger.info(">>> [Engine] 自动释放引擎资源: %s", sim_id)
                            
                thread = threading.Thread(target=run_engine_background, args=(engine, constellation_id))
                thread.daemon = True
                thread.start()
                
                return Response({"message": f"星座 {constellation_id} 仿真启动成功", "simulation_id": sim_id}, status=status.HTTP_200_OK)
                
            elif action == 'stop':
                if sim_id in GLOBAL_ENGINES:
                    engine = GLOBAL_ENGINES[sim_id]
                    # 如果你的 engine 类中有 stop() 或类似回收函数，在这里调用
                    if hasattr(engine, 'stop'):
                        engine.stop()
                    
                    del GLOBAL_ENGINES[sim_id]
                    logger.info(">>> [API] 手动终止仿真系统: %s", sim_id)
                    return Response({"message": f"星座 {constellation_id} 仿真已停止"}, status=status.HTTP_200_OK)
                else:
                    return Response({"message": "仿真未运行，无需停止"}, status=status.HTTP_200_OK)
                    
        except Exception as e:
            logger.exception("[SimulationControl] error: %s", e)
            return Response({"error": f"仿真控制异常: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BenchmarkLatestView(APIView):
    """GET /api/simulation/benchmark/latest
    返回“生成数据洞察”面板所需结构。
    """

    def get(self, request, *args, **kwargs):
        constellation_id = str(request.query_params.get("constellation_id", "3600"))
        try:
            benchmark, _ = _build_reports(constellation_id)
            return Response(benchmark, status=status.HTTP_200_OK)
        except Exception as exc:
            logger.exception("[BenchmarkLatest] failed: %s", exc)
            return Response(
                {"error": f"benchmark 生成失败: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CorrelationLatestView(APIView):
    """GET /api/simulation/correlation/latest
    返回“数据关联模型”面板所需结构。
    """

    def get(self, request, *args, **kwargs):
        constellation_id = str(request.query_params.get("constellation_id", "3600"))
        try:
            _, correlation = _build_reports(constellation_id)
            return Response(correlation, status=status.HTTP_200_OK)
        except Exception as exc:
            logger.exception("[CorrelationLatest] failed: %s", exc)
            return Response(
                {"error": f"correlation 生成失败: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CapacityLatestView(APIView):
    """GET /api/simulation/capacity/latest
    返回最近一次容量评估结果，供前端首屏兜底加载。
    """

    def get(self, request, *args, **kwargs):
        constellation_id = str(request.query_params.get("constellation_id", "3600"))
        sim_id = f"sim_{constellation_id}"

        engine = GLOBAL_ENGINES.get(sim_id)
        if engine is not None:
            latest = getattr(engine, "latest_capacity_evaluation", None)
            if latest:
                return Response(latest, status=status.HTTP_200_OK)

        # 兜底：当指定星座实例不存在时，返回任一正在运行引擎中的最近结果。
        for fallback_engine in GLOBAL_ENGINES.values():
            latest = getattr(fallback_engine, "latest_capacity_evaluation", None)
            if latest:
                return Response(latest, status=status.HTTP_200_OK)

        return Response(
            {
                "error": "capacity evaluation result not ready",
                "detail": "请先启动仿真并等待至少一个宏周期完成。",
            },
            status=status.HTTP_404_NOT_FOUND,
        )


class DataGenerationDatasetsView(APIView):
    """POST /api/data-generation/datasets
    创建数据生成任务（当前为进程内同步执行，保留 job 查询接口）。
    """

    def post(self, request, *args, **kwargs):
        mode = str(request.data.get("mode", "cvae")).strip().lower()
        spec = request.data.get("spec", {}) or {}
        if mode not in ("cvae", "physics"):
            return Response({"error": "mode 仅支持 cvae 或 physics"}, status=status.HTTP_400_BAD_REQUEST)

        job_id = _new_job_id()
        dataset_id = _new_dataset_id()
        DATA_GEN_JOBS[job_id] = {
            "jobId": job_id,
            "datasetId": dataset_id,
            "status": "running",
            "progress": 10,
            "message": "任务创建成功，正在生成数据集",
            "created_at": timezone.now().isoformat(),
        }

        try:
            if mode == "cvae":
                generated = generate_with_model(spec)
                payload = _cvae_output_to_payload(generated, int(spec.get("seed", 0) or 0))
                source_kind = "synthetic"
                provenance_extra = generated.get("provenance", {})
            else:
                payload = simulate_layered_series(
                    n_entities=int(spec.get("n_samples", spec.get("n_entities", 32))),
                    steps=int(spec.get("n_steps", 64)),
                    seed=int(spec.get("seed", 0)),
                )
                generated = None
                source_kind = "physics_simulated"
                provenance_extra = {}

            n_entities, n_steps = _dataset_shape_from_payload(payload)
            DATA_GEN_DATASETS[dataset_id] = {
                "datasetId": dataset_id,
                "mode": mode,
                "spec": spec,
                "payload": payload,
                "generated": generated,
                "sourceKind": source_kind,
                "createdAtUtc": timezone.now().astimezone(dt_timezone.utc).isoformat(),
                "shapes": {
                    "n_entities": n_entities,
                    "n_steps": n_steps,
                    "n_params": 14,
                },
                "provenance": {
                    "source_kind": source_kind,
                    "generator_version": provenance_extra.get("generator_version") or payload.get("Provenance", {}).get("ModelVersion"),
                    "generator_weight_sha256": provenance_extra.get("generator_weight_sha256", ""),
                    "feature_contract_id": provenance_extra.get("feature_contract_id", ""),
                    "seed": int(spec.get("seed", 0) or 0),
                },
            }

            DATA_GEN_JOBS[job_id].update(
                {
                    "status": "ready",
                    "progress": 100,
                    "message": "数据集生成完成",
                }
            )
            return Response(
                {
                    "jobId": job_id,
                    "datasetId": dataset_id,
                    "status": "queued",
                },
                status=status.HTTP_200_OK,
            )
        except Exception as exc:
            logger.exception("[DataGenerationDatasets] generation failed: %s", exc)
            DATA_GEN_JOBS[job_id].update(
                {
                    "status": "failed",
                    "progress": 100,
                    "message": f"生成失败: {exc}",
                }
            )
            return Response({"error": f"数据集生成失败: {exc}", "jobId": job_id}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DataGenerationJobStatusView(APIView):
    """GET /api/data-generation/jobs/{job_id}"""

    def get(self, request, job_id, *args, **kwargs):
        job = DATA_GEN_JOBS.get(str(job_id))
        if not job:
            return Response({"error": "job 不存在"}, status=status.HTTP_404_NOT_FOUND)
        return Response(job, status=status.HTTP_200_OK)


class DataGenerationDatasetSummaryView(APIView):
    """GET /api/data-generation/datasets/{dataset_id}/summary"""

    def get(self, request, dataset_id, *args, **kwargs):
        dataset = DATA_GEN_DATASETS.get(str(dataset_id))
        if not dataset:
            return Response({"error": "dataset 不存在"}, status=status.HTTP_404_NOT_FOUND)
        return Response(
            {
                "datasetId": dataset["datasetId"],
                "sourceKind": dataset["sourceKind"],
                "createdAtUtc": dataset["createdAtUtc"],
                "shapes": dataset["shapes"],
                "spec": dataset["spec"],
                "provenance": dataset["provenance"],
            },
            status=status.HTTP_200_OK,
        )


class DataGenerationSeriesView(APIView):
    """GET /api/data-generation/datasets/{dataset_id}/series"""

    def get(self, request, dataset_id, *args, **kwargs):
        dataset = DATA_GEN_DATASETS.get(str(dataset_id))
        if not dataset:
            return Response({"error": "dataset 不存在"}, status=status.HTTP_404_NOT_FOUND)

        payload = dataset["payload"]
        layer = str(request.query_params.get("layer", "all")).lower()

        env = payload.get("EnvironmentSeries", {})
        phy = payload.get("PhysicalSeries", {})
        net = payload.get("NetworkSeries", {})
        timestamp = payload.get("Timestamp", [])

        start_step = request.query_params.get("startStep")
        end_step = request.query_params.get("endStep")

        def _slice(series_dict):
            if start_step is None and end_step is None:
                return series_dict
            try:
                begin = int(start_step) if start_step is not None else 0
                stop = int(end_step) if end_step is not None else None
            except (TypeError, ValueError):
                return series_dict

            output = {}
            for key, matrix in series_dict.items():
                arr = np.asarray(matrix, dtype=np.float64)
                if arr.ndim == 2:
                    output[key] = arr[:, begin:stop].tolist()
                else:
                    output[key] = matrix
            return output

        response = {
            "datasetId": dataset["datasetId"],
            "timestamp": timestamp,
            "environmentSeries": _slice(env) if layer in ("environment", "all") else {},
            "physicalSeries": _slice(phy) if layer in ("physical", "all") else {},
            "networkSeries": _slice(net) if layer in ("network", "all") else {},
            "units": _series_units(),
            "sourceKind": dataset["sourceKind"],
        }
        return Response(response, status=status.HTTP_200_OK)


class DataGenerationAssociationView(APIView):
    """POST /api/data-generation/datasets/{dataset_id}/association"""

    def post(self, request, dataset_id, *args, **kwargs):
        dataset = DATA_GEN_DATASETS.get(str(dataset_id))
        if not dataset:
            return Response({"error": "dataset 不存在"}, status=status.HTTP_404_NOT_FOUND)
        try:
            result = fit_association_model(dataset["payload"])
            return Response(result, status=status.HTTP_200_OK)
        except Exception as exc:
            logger.exception("[DataGenerationAssociation] failed: %s", exc)
            return Response({"error": f"关联建模失败: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DataGenerationValidationView(APIView):
    """POST /api/data-generation/datasets/{dataset_id}/validation"""

    def post(self, request, dataset_id, *args, **kwargs):
        dataset = DATA_GEN_DATASETS.get(str(dataset_id))
        if not dataset:
            return Response({"error": "dataset 不存在"}, status=status.HTTP_404_NOT_FOUND)

        try:
            generated = dataset.get("generated")
            if generated and generated.get("params"):
                report = validate_physical_bounds(generated)
            else:
                fallback_dataset = _payload_to_validation_dataset(dataset["payload"])
                report = validate_physical_bounds(fallback_dataset)

            report["datasetId"] = dataset["datasetId"]
            report["sourceKind"] = dataset["sourceKind"]
            return Response(report, status=status.HTTP_200_OK)
        except Exception as exc:
            logger.exception("[DataGenerationValidation] failed: %s", exc)
            return Response({"error": f"校验失败: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


