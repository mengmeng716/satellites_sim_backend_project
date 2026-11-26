"""链路预测接口自测：SCFE 输出 → 表2-2 记录。"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
# 本目录下有一个同名的 src/ 子包（SCFE 模型），会遮蔽工程根的 src 命名空间包，
# 因此先把脚本目录从 sys.path 移除，再插入工程根。
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
while _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.algos.link_prediction.interface import (  # noqa: E402
    OUTPUT_FIELDS,
    adapt_scfe_output,
    analyze_link_quality,
    predict_link_state,
)

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def fake_scfe_output():
    t0 = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    t1 = datetime(2026, 11, 16, 12, 0, 5, tzinfo=timezone.utc).isoformat()
    snapshot = {
        "0": [["1", {"Capacity": 82.0, "Survival": 1.0, "HeatValue": 0.18,
                      "LinkAvailability": 0.95}]],
        "1": [["2", {"Capacity": 55.0, "Survival": 0.9, "HeatValue": 0.45,
                      "LinkAvailability": 0.72}]],
    }
    return {
        "Timestamp": [t0, t1],
        "ConstellationId": "3600",
        "PredictionHorizon": 2,
        "LinksPredTopology": [
            {"Timestamp": t0, "Topology": snapshot},
            {"Timestamp": t1, "Topology": snapshot},
        ],
    }


def fake_link_states():
    """两条仿真链路：0->1 同轨 719 km，1->2 异轨 900 km（3600 星座编号）。"""
    from src.simulation.data_model import LinksQualitiesValue
    from src.simulation.topology_link_state import _laser_profile_by_type, laser_type_for_link

    def link(src, dst, distance, flow):
        laser_type = laser_type_for_link(src, dst)
        return LinksQualitiesValue(link_distance=distance, current_flow=flow, delay=distance / 299.792458,
                                   laser_type=laser_type, laser_config=_laser_profile_by_type(laser_type))

    return {("0", "1"): link("0", "1", 719.0, 12.0), ("1", "2"): link("1", "62", 900.0, 30.0)}


def main() -> int:
    states = fake_link_states()
    records = adapt_scfe_output(fake_scfe_output(), states, max_capacity_gbps=100.0)

    print("== 1. 记录数量与字段 ==")
    check("2 步 × 2 链路 = 4 条", len(records) == 4, f"{len(records)}")
    check("12 字段齐全", set(records[0]) == set(OUTPUT_FIELDS),
          str(set(records[0]) ^ set(OUTPUT_FIELDS)))

    print("\n== 2. 单位口径 ==")
    first = records[0]
    check("Capacity 为 Gbps（模型值直传）", first["Capacity"] == 82.0, str(first["Capacity"]))
    check("PredFlow = 额定 − 剩余", first["PredFlow"] == 18.0, str(first["PredFlow"]))
    check("LoadFluctuation = 本帧 − 仿真当前", first["LoadFluctuation"] == 6.0,
          str(first["LoadFluctuation"]))
    check("Delay 单位为秒，≥ 传播时延", first["Delay"] >= 719.0 / 299792.458, str(first["Delay"]))
    check("SignalIntensity = Tx − PathLoss，落在 [-80,-9] dBm", -80 <= first["SignalIntensity"] <= -9,
          str(first["SignalIntensity"]))
    check("AchievableRate ∈(0,100]", 0 < first["AchievableRate"] <= 100,
          str(first["AchievableRate"]))
    check("Timestamp 已转毫秒", first["Timestamp"] > 1_700_000_000_000, str(first["Timestamp"]))
    second_step = records[2]
    check("第 2 步负载波动相对第 1 步", second_step["LoadFluctuation"] == 0.0,
          str(second_step["LoadFluctuation"]))

    print("\n== 3. 不同距离得到不同物理量 ==")
    near = next(r for r in records if r["DstSatID"] == "1")
    far = next(r for r in records if r["DstSatID"] == "2")
    check("远链路传播时延更大", far["Delay"] > near["Delay"], f"{far['Delay']} > {near['Delay']}")

    print("\n== 4. 仿真拓扑缺链路时报错清晰 ==")
    try:
        adapt_scfe_output(fake_scfe_output(), {("0", "1"): states[("0", "1")]})
    except KeyError as exc:
        check("缺链路抛 KeyError 且含链路名", "1" in str(exc) and "2" in str(exc), str(exc))
    else:
        check("缺链路抛 KeyError", False, "未抛异常")

    print("\n== 5. 质量分析 ==")
    quality = analyze_link_quality(records)
    check("质量条目数与记录一致", len(quality) == len(records))
    check("质量字段齐全",
          set(quality[0]) == {"EntityId", "QualityScore", "AchievableRateScore", "DelayScore"})
    check("质量分 ∈[0,1]", all(0 <= q["QualityScore"] <= 1 for q in quality))

    print("\n== 6. 单链路解析路径仍可用 ==")
    single = predict_link_state("0", "9", 800.0, capacity_gbps=100.0, flow_gbps=40.0)
    check("单条记录 12 字段", set(single) == set(OUTPUT_FIELDS))
    check("单条确定性", single == predict_link_state("0", "9", 800.0, capacity_gbps=100.0,
                                                     flow_gbps=40.0))

    print("\n== 汇总 ==")
    if fails:
        print(f"失败 {len(fails)} 项: {fails}")
        return 1
    print("链路预测接口自测全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
