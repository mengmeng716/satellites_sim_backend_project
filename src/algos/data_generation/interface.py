"""算法二：多层关联数据模拟参数生成 —— 对外接口。

严格对应接口文档：
  * 表3-3 生成式多层数据模拟输出接口
  * 表3-4 跨层数据关联与质量综合评估输出接口

职责：以物理链路模型为骨架生成三层参数序列，并建立层间关联模型。
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import hashlib
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

try:  # 作为包导入：src.algos.data_generation.interface
    from . import association, physics
except ImportError:  # 作为扁平模块导入：sys.path 指向本目录
    import association  # type: ignore[no-redef]
    import physics  # type: ignore[no-redef]

FREE_CHANNELS = physics.FREE_CHANNELS


def simulate_layered_series(
    n_entities: int = 32,
    steps: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """生成一份符合表3-3 的多层参数序列。

    做法：先随机产生 4 个自由量（距离/指向损耗/发射功率/到达负载），
    其余 9 个观测量由链路公式推导，因此天然满足物理闭合。
    """
    rng = np.random.default_rng(seed)
    shape = (n_entities, steps)

    free = {
        "LinkDistance_km": rng.uniform(300.0, 1900.0, shape),
        "PointingLoss_dB": rng.uniform(0.2, 3.0, shape),
        "TxPower_dBm": rng.normal(30.0, 2.0, shape),
        "OfferedLoadRatio": rng.uniform(0.2, 2.0, shape),
    }
    derived = physics.derive(free)

    ambient = rng.uniform(233.0, 313.0, shape)
    terminal = ambient + rng.uniform(5.0, 30.0, shape)

    payload = {
        "DataSetId": f"gen-{seed}-{n_entities}x{steps}",
        "Provenance": {
            "SourceKind": "synthetic",
            "ModelVersion": "physics+4free-v1",
            "PhysicsModuleHash": hashlib.sha256(
                b"physics.derive:v1").hexdigest()[:16],
        },
        "EnvironmentSeries": {
            "AmbientTemperature": ambient.tolist(),
            "TerminalTemperature": terminal.tolist(),
        },
        "PhysicalSeries": {
            "Wavelength": np.full(shape, physics.LAMBDA_NM).tolist(),
            "ModulationBandwidth": np.full(shape, 50e9).tolist(),
            "TxPower": free["TxPower_dBm"].tolist(),
            "PathLoss": derived["PathLoss_dB"].tolist(),
            "RxPower": derived["RxPower_dBm"].tolist(),
            "SNR": derived["SNR_dB"].tolist(),
            "PropagateDelay": (free["LinkDistance_km"] / physics.C_KM_S).tolist(),
            "BER": derived["BER_preFEC"].tolist(),
        },
        "NetworkSeries": {
            "PacketLossRate": derived["PacketLossRate"].tolist(),
            "Delay": derived["Delay_ms"].tolist(),
            "DelayJitter": derived["DelayJitter_ms"].tolist(),
            "Throughput": derived["Throughput_Gbps"].tolist(),
        },
        "Timestamp": [int(time.time() * 1000) + i * 1000 for i in range(steps)],
    }
    return payload


def _to_frame(payload: Mapping[str, Any]):
    """把表3-3 结构摊平成 DataFrame，供关联模型使用。"""
    import pandas as pd

    env, phy, net = (payload["EnvironmentSeries"], payload["PhysicalSeries"],
                     payload["NetworkSeries"])
    rows = len(env["AmbientTemperature"])
    steps = len(env["AmbientTemperature"][0]) if rows else 0
    flat = {"SatelliteId": [],
            "AmbientTemperature_K": [], "TerminalTemperature_K": [],
            "PathLoss_dB": [], "SNR_dB": [], "BER_preFEC": [],
            "PacketLossRate": [], "Delay_ms": [], "DelayJitter_ms": [],
            "Throughput_Gbps": [], "LinkDistance_km": []}
    for i in range(rows):
        for t in range(steps):
            flat["SatelliteId"].append(f"sat-{i}")
            flat["AmbientTemperature_K"].append(env["AmbientTemperature"][i][t])
            flat["TerminalTemperature_K"].append(env["TerminalTemperature"][i][t])
            flat["PathLoss_dB"].append(phy["PathLoss"][i][t])
            flat["SNR_dB"].append(phy["SNR"][i][t])
            flat["BER_preFEC"].append(phy["BER"][i][t])
            flat["PacketLossRate"].append(net["PacketLossRate"][i][t])
            flat["Delay_ms"].append(net["Delay"][i][t])
            flat["DelayJitter_ms"].append(net["DelayJitter"][i][t])
            flat["Throughput_Gbps"].append(net["Throughput"][i][t])
            flat["LinkDistance_km"].append(phy["PropagateDelay"][i][t] * physics.C_KM_S)
    return pd.DataFrame(flat)


def fit_association_model(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """按表3-4 输出跨层关联模型结果。"""
    df = _to_frame(payload)
    env_ch = ["AmbientTemperature_K", "TerminalTemperature_K"]
    phy_ch = ["PathLoss_dB", "SNR_dB", "LinkDistance_km"]
    net_ch = ["PacketLossRate", "Delay_ms", "DelayJitter_ms", "Throughput_Gbps"]

    steps = len(payload["Timestamp"])
    window = max(4, min(16, steps // 2))
    stride = max(1, window // 2)
    fe, _ = association.window_features(df, env_ch, window=window, stride=stride)
    fp, _ = association.window_features(df, phy_ch, window=window, stride=stride)
    fn, ip = association.window_features(df, net_ch, window=window, stride=stride)
    n = min(len(fe), len(fp), len(fn))
    fe, fp, fn = fe[:n], fp[:n], fn[:n]

    quality = association.quality_target(df.loc[ip[:n]].reset_index(drop=True))
    stage_a = association.fit_stage(fe, fp[:, :4])
    stage_b = association.fit_stage(fp, fn[:, :4])
    stage_c = association.fit_stage(np.concatenate([fe, fp, fn], axis=1),
                                    quality.reshape(-1, 1))

    parts = {"env": fe, "phy": fp, "net": fn}
    baseline = {k: v.mean(axis=0) for k, v in parts.items()}
    shares, _ = association.shapley_layers(stage_c, parts, baseline)

    return {
        "ModelType": "layered_regression",
        "ParamCoverage": {"ParamCount": len(env_ch + phy_ch + net_ch), "LayerCount": 3},
        "Correlation": {"ParamPairs": [
            {"Pair": ["TerminalTemperature_K", "SNR_dB"],
             "Corr": float(np.corrcoef(df["TerminalTemperature_K"], df["SNR_dB"])[0, 1])},
            {"Pair": ["SNR_dB", "PacketLossRate"],
             "Corr": float(np.corrcoef(df["SNR_dB"], df["PacketLossRate"])[0, 1])},
        ]},
        "Ablation": {"LayerR2Drop": [shares["env"], shares["phy"], shares["net"]]},
        "LinkQuality": [
            {"EntityId": f"link-{i}", "QualityScore": float(q),
             "AchievableRateScore": float(np.clip(q, 0, 1)),
             "DelayScore": float(np.clip(1.0 - q * 0.5, 0, 1))}
            for i, q in enumerate(quality[:8])
        ],
        "FitMetrics": {
            "WindowCount": n,
            "WindowFrames": window,
            "WindowStrideFrames": stride,
            "R2Test": {"stage_a": stage_a.extra.get("r2_test_mean_per_target"),
                       "stage_b": stage_b.extra.get("r2_test_mean_per_target"),
                       "stage_c": stage_c.extra.get("r2_test_mean_per_target")},
            "RMSETest": None,
        },
    }


def generate_with_model(spec, model_dir=None, device=None):
    """用训练好的生成模型（CVAE）生成数据集。

    :param spec: 生成场景，字段见 ``generate.generate_dataset``
                 （n_samples / n_steps / dt_s / layers / hours / seed / temperature）
    :param model_dir: 模型目录，需包含 ``cvae_multilayer.pt``
    :param device: 计算设备，缺省自动选择
    :return: ``{"params", "conditions", "provenance", ...}``

    模型文件不存在时抛出 ``FileNotFoundError``。
    """
    try:
        from . import generate as generator_module
    except ImportError:  # 扁平导入
        import generate as generator_module  # type: ignore[no-redef]

    generator = generator_module.load_generator(model_dir, device)
    return generator_module.generate_dataset(spec, generator)
