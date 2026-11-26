"""离线参数生成入口。

用途：**离线**批量生成多层关联参数数据集，落盘为 CSV + meta.json，
供链路预测模型训练/评估使用。不参与在线仿真主循环。

用法::

    python -m src.algos.data_generation.offline_generate --entities 512 --steps 180 --out out/generated
    python -m src.algos.data_generation.offline_generate --entities 64 --steps 32 --model-dir <dir>

产物：

* ``params.csv``：长表，列为 ``entity_id, step, <参数名...>``
* ``meta.json``：表3-3 的 Provenance 段 + 数据形状 + 列清单 + 校验和
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import physics
from .interface import simulate_layered_series

#: 表3-3 中带序列的字段（顺序固定，写入 CSV 时按此列序）
CSV_COLUMNS = [
    "AmbientTemperature", "TerminalTemperature",
    "Wavelength", "ModulationBandwidth", "TxPower", "PathLoss", "RxPower",
    "SNR", "PropagateDelay", "BER",
    "PacketLossRate", "Delay", "DelayJitter", "Throughput",
]


def flatten(payload: Dict[str, Any]) -> List[Dict[str, float]]:
    """把表3-3 的嵌套结构摊平为逐实体逐步的行。"""
    env = payload["EnvironmentSeries"]
    phy = payload["PhysicalSeries"]
    net = payload["NetworkSeries"]
    entities = len(env["AmbientTemperature"])
    steps = len(env["AmbientTemperature"][0]) if entities else 0

    rows: List[Dict[str, float]] = []
    for i in range(entities):
        for t in range(steps):
            row = {"entity_id": f"link-{i}", "step": t}
            row["AmbientTemperature"] = env["AmbientTemperature"][i][t]
            row["TerminalTemperature"] = env["TerminalTemperature"][i][t]
            for name in ("Wavelength", "ModulationBandwidth", "TxPower", "PathLoss",
                         "RxPower", "SNR", "PropagateDelay", "BER"):
                row[name] = phy[name][i][t]
            for name in ("PacketLossRate", "Delay", "DelayJitter", "Throughput"):
                row[name] = net[name][i][t]
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, float]], path: Path) -> str:
    """写 CSV（不依赖 pandas，保持轻量），返回内容 sha256。"""
    header = ["entity_id", "step"] + CSV_COLUMNS
    lines = [",".join(header)]
    for row in rows:
        values = [str(row["entity_id"]), str(row["step"])]
        for name in CSV_COLUMNS:
            values.append(f"{float(row[name]):.6g}")
        lines.append(",".join(values))
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_meta(payload: Dict[str, Any], rows: List[Dict[str, float]],
               csv_sha256: str, source: str) -> Dict[str, Any]:
    entities = len(payload["EnvironmentSeries"]["AmbientTemperature"])
    steps = len(payload["Timestamp"])
    return {
        "DataSetId": payload["DataSetId"],
        "Source": source,
        "GeneratedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "Shape": {"entities": entities, "steps": steps, "rows": len(rows)},
        "Columns": CSV_COLUMNS,
        "CsvSha256": csv_sha256,
        "Provenance": payload["Provenance"],
        "PhysicsClosed": True,
        "FreeChannels": list(getattr(physics, "FREE_CHANNELS", [])),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线生成多层关联参数数据集")
    parser.add_argument("--entities", type=int, default=64, help="实体（链路）条数")
    parser.add_argument("--steps", type=int, default=120, help="每条序列时间步")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out", type=str, default="out/generated", help="输出目录")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="可选：用 CVAE 权重生成；缺省走物理公式生成")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        from .interface import generate_with_model
        dataset = generate_with_model(
            {"n_samples": args.entities, "n_steps": args.steps, "seed": args.seed},
            model_dir=args.model_dir,
        )
        payload = _model_output_to_payload(dataset, args.seed)
        source = "cvae"
    else:
        payload = simulate_layered_series(args.entities, args.steps, args.seed)
        source = "physics"

    rows = flatten(payload)
    csv_path = out_dir / "params.csv"
    csv_sha = write_csv(rows, csv_path)

    meta = build_meta(payload, rows, csv_sha, source)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8", newline="\n")

    print(f"[offline_generate] source={source}")
    print(f"  实体 {meta['Shape']['entities']} × 步数 {meta['Shape']['steps']}"
          f" = {meta['Shape']['rows']} 行")
    print(f"  CSV : {csv_path}  (sha256 {csv_sha[:12]})")
    print(f"  Meta: {meta_path}")
    return 0


def _model_output_to_payload(dataset: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """把 CVAE 输出（params/conditions）转成表3-3 结构。

    模型输出 12 个参数（小写键，形状 ``(n_samples, n_steps)``），
    其余 4 个字段由固定配置或派生得到，并在 ``Provenance.DerivedFields`` 中如实标注。
    """
    params = dataset.get("params") or {}
    if not params:
        raise ValueError("CVAE 输出未包含 params，无法转换为表3-3 结构")

    def series(key: str) -> np.ndarray:
        value = np.asarray(params[key], dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        return value

    base = series("ambient_temperature_k")
    rows, steps = base.shape

    #: 模型直接产生的 12 个字段
    mapping = {
        "ambient_temperature_k": "AmbientTemperature",
        "tx_power_dbm": "TxPower",
        "path_loss_db": "PathLoss",
        "rx_power_dbm": "RxPower",
        "snr_db": "SNR",
        "ber_pre_fec": "BER",
        "link_delay_ms": "Delay",
        "delay_jitter_ms": "DelayJitter",
        "packet_loss_rate": "PacketLossRate",
        "achievable_rate_gbps": "Throughput",
    }
    env, phy, net = {}, {}, {}
    for key, target in mapping.items():
        if key not in params:
            continue
        value = series(key).tolist()
        if target in ("AmbientTemperature", "TerminalTemperature"):
            env[target] = value
        elif target in ("PacketLossRate", "Delay", "DelayJitter", "Throughput"):
            net[target] = value
        else:
            phy[target] = value

    #: 派生/固定字段（模型不产生）
    derived: List[str] = []
    if "TerminalTemperature" not in env:
        env["TerminalTemperature"] = np.clip(base + 15.0, 263.0, 343.0).tolist()
        derived.append("TerminalTemperature=AmbientTemperature+15K")
    if "Wavelength" not in phy:
        phy["Wavelength"] = np.full((rows, steps), 1550.0).tolist()
        derived.append("Wavelength=1550nm（固定配置）")
    if "ModulationBandwidth" not in phy:
        phy["ModulationBandwidth"] = np.full((rows, steps), 1.0e10).tolist()
        derived.append("ModulationBandwidth=1e10Hz（固定配置）")
    if "PropagateDelay" not in phy:
        phy["PropagateDelay"] = (series("link_delay_ms") / 1.0e3).tolist()
        derived.append("PropagateDelay=link_delay_ms/1000（近似）")

    for group, required in ((env, ("AmbientTemperature", "TerminalTemperature")),
                            (phy, ("Wavelength", "ModulationBandwidth", "TxPower", "PathLoss",
                                   "RxPower", "SNR", "PropagateDelay", "BER")),
                            (net, ("PacketLossRate", "Delay", "DelayJitter", "Throughput"))):
        for name in required:
            if name not in group:
                raise ValueError(f"表3-3 缺少必需字段: {name}")

    return {
        "DataSetId": f"cvae-{seed}-{rows}x{steps}",
        "Provenance": {
            "SourceKind": "synthetic",
            "ModelVersion": str(dataset.get("provenance", {}).get("generator_version", "cvae")),
            "PhysicsModuleHash": str(
                dataset.get("provenance", {}).get("generator_weight_sha256", ""))[:16],
            "DerivedFields": derived,
        },
        "EnvironmentSeries": env,
        "PhysicalSeries": phy,
        "NetworkSeries": net,
        "Timestamp": [int(datetime.now(timezone.utc).timestamp() * 1000) + i * 1000
                      for i in range(steps)],
    }


if __name__ == "__main__":
    sys.exit(main())
