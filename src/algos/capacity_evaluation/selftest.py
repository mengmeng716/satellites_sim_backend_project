"""容量与效能评估自测。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
while _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.algos.capacity_evaluation import evaluate, lambda_star  # noqa: E402
from src.algos.capacity_evaluation.capacity_interface import (  # noqa: E402
    compare_reconstruction_candidate,
    reconstruction_decision,
)

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def ring(n=8, distance_km=1000.0, loss=0.0, ber=1e-9):
    """n 节点环形拓扑，链路属性用接口文档字段名。"""
    topo = {}
    for i in range(n):
        topo[str(i)] = {
            str((i + 1) % n): {"LinkDistance": distance_km, "PacketLossRate": loss, "BER": ber},
            str((i - 1) % n): {"LinkDistance": distance_km, "PacketLossRate": loss, "BER": ber},
        }
    return topo


def main() -> int:
    print("== 1. λ* 精确性（环形，容量 100，单条需求 50）==")
    topo = ring(8, distance_km=1000.0)
    single = [{"src": "0", "dst": "4", "demand_gbps": 50.0}]
    result = lambda_star(topo, single, capacity_gbps=100.0, precision=0.01)
    check("单条需求 λ* = 2.0", abs(result["lambda_star"] - 2.0) < 0.02,
          str(result["lambda_star"]))
    check("容量 = Σ需求×λ*", abs(result["capacity_gbps"] - 100.0) < 1.0,
          str(result["capacity_gbps"]))
    check("全部可达", result["unroutable"] == 0)

    print("\n== 1b. 两条互不共享链路的业务 ==")
    disjoint = {
        "0": {"1": {"LinkDistance": 1000.0}},
        "1": {"0": {"LinkDistance": 1000.0}},
        "2": {"3": {"LinkDistance": 1000.0}},
        "3": {"2": {"LinkDistance": 1000.0}},
    }
    demands = [
        {"src": "0", "dst": "1", "demand_gbps": 50.0},
        {"src": "2", "dst": "3", "demand_gbps": 50.0},
    ]
    r0 = lambda_star(disjoint, demands, capacity_gbps=100.0, precision=0.01)
    check("互不影响时 λ* = 2.0", abs(r0["lambda_star"] - 2.0) < 0.02, str(r0["lambda_star"]))

    print("\n== 2. 共享链路成为瓶颈 ==")
    shared = [
        {"src": "0", "dst": "2", "demand_gbps": 40.0},
        {"src": "0", "dst": "3", "demand_gbps": 40.0},
    ]
    r2 = lambda_star(topo, shared, capacity_gbps=100.0, precision=0.01)
    check("λ* 受 0-1 链路限制（100/80=1.25）", abs(r2["lambda_star"] - 1.25) < 0.03,
          str(r2["lambda_star"]))
    check("识别出瓶颈链路", len(r2["bottleneck_links"]) > 0, str(r2["bottleneck_links"][:3]))

    print("\n== 3. 断网时不可达 ==")
    broken = {"0": {"1": {"LinkDistance": 1000.0}}, "1": {"0": {"LinkDistance": 1000.0}},
              "2": {"3": {"LinkDistance": 1000.0}}, "3": {"2": {"LinkDistance": 1000.0}}}
    r3 = lambda_star(broken, [{"src": "0", "dst": "2", "demand_gbps": 10.0}], 100.0)
    check("不可达时 λ*=0", r3["lambda_star"] == 0.0)
    check("不可达计数正确", r3["unroutable"] == 1, str(r3["unroutable"]))

    print("\n== 4. 完整评估输出（表2-20 字段）==")
    samples = [
        {"0->1": True, "1->2": True, "2->3": True},
        {"0->1": True, "1->2": False, "2->3": True},
        {"0->1": True, "1->2": True, "2->3": True},
        {"0->1": True, "1->2": True, "2->3": True},
    ]
    eval_demands = [{"src": "0", "dst": "4", "demand_gbps": 50.0}]
    out = evaluate(
        topology=topo,
        traffic_matrix=eval_demands,
        qos={"MaxDelay": 300.0, "MaxLoss": 0.05, "MaxBER": 1e-6},
        capacity_protocol={"LinkCapacity": 100.0, "SearchPrecision": 0.01},
        availability_samples=samples,
        planned_links=["0->1", "1->2", "2->3"],
        frame_seconds=5.0,
        timestamp_ms=1700000000000,
    )
    check("Capacity 四字段齐全",
          {"CapacityBefore", "CapacityAfter", "CapacityGainPct", "BottleneckLinks"} <= set(out["Capacity"]))
    check("Availability 四字段齐全",
          {"Availability", "MinLinkAvailability", "DowntimeSeconds", "OutageEvents"} <= set(out["Availability"]))
    check("Performance 五字段齐全",
          {"MeanEndToEndDelay", "P95EndToEndDelay", "MeanPacketLossRate",
           "MeanBERPostFEC", "AggregateGoodput"} <= set(out["Performance"]))
    check("Verdict 六项判定", out["Verdict"]["Total"] == 6, str(out["Verdict"]["Total"]))
    check("可用率 ∈[0,1]", 0 <= out["Availability"]["Availability"] <= 1,
          str(out["Availability"]["Availability"]))
    check("时延 = 4 跳 × 3.34 ms", abs(out["Performance"]["MeanEndToEndDelay"] - 13.34) < 0.1,
          str(out["Performance"]["MeanEndToEndDelay"]))
    check("平均丢包为 0", out["Performance"]["MeanPacketLossRate"] == 0.0,
          str(out["Performance"]["MeanPacketLossRate"]))
    check("有效吞吐 = 需求", abs(out["Performance"]["AggregateGoodput"] - 50.0) < 1e-6,
          str(out["Performance"]["AggregateGoodput"]))
    check("中断次数 = 1", out["Availability"]["OutageEvents"] == 1,
          str(out["Availability"]["OutageEvents"]))

    print("\n== 5. 容量提升（基准 vs 优化后）==")
    better = ring(8, distance_km=1000.0)
    for i in range(8):
        better[str(i)][str((i + 2) % 8)] = {"LinkDistance": 2000.0, "PacketLossRate": 0.0}
    gain = evaluate(topology=better, traffic_matrix=shared, baseline_topology=ring(4),
                    capacity_protocol={"LinkCapacity": 100.0})["Capacity"]["CapacityGainPct"]
    check("提升百分比为数值或 None", gain is None or isinstance(gain, float), str(gain))

    print("\n== 6. 确定性 ==")
    a = evaluate(topo, demands, capacity_protocol={"LinkCapacity": 100.0})
    b = evaluate(topo, demands, capacity_protocol={"LinkCapacity": 100.0})
    check("两次结果一致", a["Capacity"] == b["Capacity"] and a["Performance"] == b["Performance"])

    print("\n== 7. 固定 Walker 基线与验收判定 ==")
    walker = ring(8)
    previous = ring(8)
    for neighbors in walker.values():
        for attr in neighbors.values():
            attr["LinkCapacity"] = 50.0
    for neighbors in previous.values():
        for attr in neighbors.values():
            attr["LinkCapacity"] = 70.0
    compared = evaluate(
        topo,
        single,
        baseline_topology=previous,
        walker_baseline_topology=walker,
        walker_gain_applicable=True,
    )
    capacity = compared["Capacity"]
    check("重构前相对 Walker +40%", capacity["CapacityBeforeGainVsBaselinePct"] == 40.0)
    check("重构后相对 Walker +100%", capacity["CapacityGainVsBaselinePct"] == 100.0)
    check("相对上一拓扑增益单独计算", capacity["CapacityGainPct"] == 42.8571)
    check("无可用率样本时判定为 NO_DATA", compared["Verdict"]["Results"][3]["Result"] == "NO_DATA")

    unchanged = evaluate(topo, single, baseline_topology=topo, walker_baseline_topology=walker)
    check("拓扑未重构时相对上一拓扑增益为 0", unchanged["Capacity"]["CapacityGainPct"] == 0.0)
    check("非验收星座保留对照数值", unchanged["Capacity"]["CapacityGainVsBaselinePct"] == 100.0)
    check("非验收星座容量阈值不适用", unchanged["Verdict"]["Results"][4]["Result"] == "N/A")

    print("\n== 8. 容量下降触发与候选约束 ==")

    def evaluate_cycle(rate, delay=1.0, loss=0.0):
        edge = {"LinkCapacity": rate, "LinkPropagationDelay": delay, "PacketLossRate": loss}
        graph = {"0": {"1": edge}, "1": {"0": edge}}
        return evaluate(graph, [{"src": "0", "dst": "1", "demand_gbps": 1.0}])

    def assess(report, reference=None, *, prediction_ready=True, cooling_down=False):
        return reconstruction_decision(
            report,
            reference,
            prediction_ready=prediction_ready,
            cooling_down=cooling_down,
        )

    healthy = assess(evaluate_cycle(100.0))
    check("首个健康窗口建立可比容量", healthy["HealthyCapacityGbps"] == 100.0)
    boundary = assess(evaluate_cycle(80.0), healthy["HealthyCapacityGbps"])
    check("下降恰好 20% 不触发", boundary["State"] == "healthy" and boundary["CapacityDropPct"] == 20.0)
    degraded_report = evaluate_cycle(79.0)
    degraded = assess(degraded_report, healthy["HealthyCapacityGbps"])
    check(
        "下降超过 20% 进入候选评估",
        degraded["State"] == "candidate_unavailable" and degraded["CapacityDropPct"] == 21.0,
    )
    check(
        "无预测窗口时等待",
        assess(degraded_report, 100.0, prediction_ready=False)["State"] == "waiting_prediction",
    )
    check(
        "刚重构的拓扑保持冷却期",
        assess(degraded_report, 100.0, cooling_down=True)["State"] == "cooldown",
    )
    scheduled = reconstruction_decision(
        evaluate_cycle(100.0), 100.0, prediction_ready=False, cooling_down=False, periodic_due=True
    )
    check(
        "定时评估允许在健康且预测未预热时检查候选",
        scheduled["State"] == "candidate_unavailable" and scheduled["CapacityDropPct"] == 0.0,
    )

    recovered = compare_reconstruction_candidate(degraded, degraded_report, evaluate_cycle(100.0))
    check(
        "更高容量且保持 QoS 的候选可提交",
        recovered["State"] == "applied" and recovered["CandidateGainPct"] == 26.5823,
    )
    half_capacity = evaluate_cycle(50.0)
    gain_boundary = compare_reconstruction_candidate(assess(half_capacity), half_capacity, evaluate_cycle(60.0))
    check("增益恰好 20% 不提交", gain_boundary["RejectionReasons"] == ["capacity_gain_below_target"])
    unchanged_candidate = compare_reconstruction_candidate(degraded, degraded_report, degraded_report)
    check("容量未提升则拒绝", unchanged_candidate["RejectionReasons"] == ["no_capacity_gain"])
    worse_qos = compare_reconstruction_candidate(degraded, degraded_report, evaluate_cycle(100.0, 2.0, 0.01))
    check("时延或丢包变差则拒绝", worse_qos["RejectionReasons"] == ["delay_regression", "loss_regression"])

    broken_report = evaluate({}, [{"src": "0", "dst": "1", "demand_gbps": 1.0}])
    disconnected = compare_reconstruction_candidate(degraded, degraded_report, broken_report)
    check("候选需求不可达则拒绝", "unroutable_demands" in disconnected["RejectionReasons"])
    check("无健康参考时不触发", assess(broken_report)["State"] == "no_reference")

    print("\n== 汇总 ==")
    if fails:
        print(f"失败 {len(fails)} 项: {fails}")
        return 1
    print("容量与效能评估自测全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
