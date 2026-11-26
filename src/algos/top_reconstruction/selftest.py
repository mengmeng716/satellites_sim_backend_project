"""拓扑重构接口自测：表2-11 四项指标 + 原有字段不回归。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
while _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.algos.top_reconstruction import reconstruct_topology  # noqa: E402


fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def walker(n_planes=6, per_plane=10, alt=500.0):
    radius = 6371.0 + alt
    speed = math.sqrt(398600.4418 / radius)
    positions = {}
    for p in range(n_planes):
        raan = 2 * math.pi * p / n_planes
        for s in range(per_plane):
            u = 2 * math.pi * s / per_plane + 2 * math.pi * p / (n_planes * per_plane)
            cu, su = math.cos(u), math.sin(u)
            co, so = math.cos(raan), math.sin(raan)
            ci, si = math.cos(math.radians(58)), math.sin(math.radians(58))
            node = p * per_plane + s
            positions[node] = [
                radius * (co * cu - so * su * ci),
                radius * (so * cu + co * su * ci),
                radius * su * si,
            ]
    return positions


def build(positions, k=3):
    ids = sorted(positions)
    coords = np.array([positions[i] for i in ids])
    topo = {i: [] for i in ids}
    for i in ids:
        dist = np.linalg.norm(coords - coords[ids.index(i)], axis=1)
        for j in np.argsort(dist)[1:k + 1]:
            neighbour = ids[int(j)]
            d = float(dist[int(j)])
            attr = {
                "LinkDistance": d,
                "LinkCapacity": 100.0,
                "LeftCapacity": 100.0,
                "LinkPropagationDelay": d / 299792.458 * 1000.0,
            }
            if not any(t == neighbour for t, _ in topo[i]):
                topo[i].append((neighbour, attr))
            if not any(t == i for t, _ in topo[neighbour]):
                topo[neighbour].append((i, attr))
    return topo


def main() -> int:
    positions = walker()
    topology = build(positions)

    class DummyConstellation:
        constellationId = "test"
        numOrbitPlane = 6
        numSatsInPlane = 10
        orbitHeight = 500.0

    class DummyNode:
        sat_list = {}

    result = reconstruct_topology(
        topology_state={"newTopology": topology},
        node_state=None,
        constellation=DummyConstellation(),
        predicted_links={"ConstellationId": "test"},
        previous_topology_update=0,
    )

    print("== 1. 原有字段不回归 ==")
    for key in ("ConstellationId", "Timestamp", "TopologyId", "TopologyUpdate",
                "newTopology", "TopQualities", "TopDifference"):
        check(f"字段 {key}", key in result)

    print("\n== 2. 表2-11 四项指标 ==")
    tq = result["TopQualities"]
    for key in ("AvgLinkDistanceScore", "AvgLinkAvailableDurationScore",
                "TotalTransmissionCapacity", "AvgLinkPropagationDelay"):
        check(f"新增 {key}", key in tq, str(tq.get(key)))
    check("距离评分 ∈[0,1]", 0 <= tq["AvgLinkDistanceScore"] <= 1, str(tq["AvgLinkDistanceScore"]))
    check("时长评分 ∈[0,1]", 0 <= tq["AvgLinkAvailableDurationScore"] <= 1,
          str(tq["AvgLinkAvailableDurationScore"]))
    check("总容量 > 0", tq["TotalTransmissionCapacity"] > 0, str(tq["TotalTransmissionCapacity"]))
    check("平均传播时延 > 0", tq["AvgLinkPropagationDelay"] > 0,
          str(tq["AvgLinkPropagationDelay"]))

    print("\n== 3. 容量按去重链路统计 ==")
    unique = set()
    for src, links in topology.items():
        for dst, attr in links:
            unique.add((min(src, dst), max(src, dst)))
    expected = 100.0 * len(unique)
    check("总容量 = 去重链路数 × 100", abs(tq["TotalTransmissionCapacity"] - expected) < 1e-6,
          f"{tq['TotalTransmissionCapacity']} vs {expected}")

    print("\n== 4. 近似口径已标注 ==")
    check("TopQualitiesNote 存在且说明近似", "近似" in (result.get("TopQualitiesNote") or ""),
          str(result.get("TopQualitiesNote"))[:40])

    print("\n== 5. 确定性 ==")
    again = reconstruct_topology(
        topology_state={"newTopology": topology},
        node_state=None,
        constellation=DummyConstellation(),
        predicted_links={"ConstellationId": "test"},
        previous_topology_update=0,
    )
    def stable(metrics):
        return {k: v for k, v in metrics.items() if k != "TopologyDecisionTime"}

    check("两次 TopQualities 一致（除决策耗时）",
          stable(again["TopQualities"]) == stable(tq))

    print("\n== 汇总 ==")
    if fails:
        print(f"失败 {len(fails)} 项: {fails}")
        return 1
    print("拓扑重构接口自测全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
