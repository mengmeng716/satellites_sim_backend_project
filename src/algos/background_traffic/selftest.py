"""背景流量模块自测：网格、分配、路由、接口。"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)

import numpy as np

from src.algos.background_traffic import background_traffic_generation, background_traffic_snapshot
from src.algos.background_traffic.flow_allocation import allocate_upload_flows
from src.algos.background_traffic.traffic_grid import active_cells, get_traffic_grid_buffer, grid_source
from src.simulation.data_model import LinksQualitiesValue, NodeAttribute, SatelliteNodeState, TopologyState

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def build_ring(n=24, lat0=0.0):
    """构造一个环形小星座，用于验证分配与路由。"""
    node_state = SatelliteNodeState(constellation_id="test")
    for i in range(n):
        lon = -180.0 + 360.0 * i / n
        node_state.sat_list[str(i)] = NodeAttribute(
            Latitude=lat0, Longitude=lon, Flow=0.0, EnergyRatio=1.0
        )
    topo = {}
    for i in range(n):
        topo[str(i)] = {
            str((i + 1) % n): LinksQualitiesValue(CurrentFlow=0.0),
            str((i - 1) % n): LinksQualitiesValue(CurrentFlow=0.0),
        }
    return SatelliteNodeState(
        constellation_id="test", sat_list=node_state.sat_list
    ), TopologyState(constellation_id="test", timestamp=0, new_topology=topo)


def main() -> int:
    print("== 1. 网格加载 ==")
    grid, lat, lon = get_traffic_grid_buffer()
    check("网格非空", grid.size > 0, f"{grid.size} 格")
    check("网格与经纬度等长", grid.size == lat.size == lon.size)
    check("来源可诊断", grid_source() != "", grid_source())

    print("\n== 2. 时段因子 ==")
    ts_noon = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
    ts_night = datetime(2026, 11, 16, 2, 0, 0, tzinfo=timezone.utc)
    cells_noon = active_cells(ts_noon)
    cells_night = active_cells(ts_night)
    check("正午有活跃网格", len(cells_noon[2]) > 0, f"{len(cells_noon[2])} 格")
    check("夜间流量低于正午",
          float(np.sum(cells_night[2])) < float(np.sum(cells_noon[2])),
          f"{np.sum(cells_night[2]):.1f} < {np.sum(cells_noon[2]):.1f}")

    print("\n== 3. 分配守恒 ==")
    sat_lats = np.array([0.0, 10.0, -10.0])
    sat_lons = np.array([0.0, 0.0, 0.0])
    upload = allocate_upload_flows(sat_lats, sat_lons, [0.0], [0.0], [100.0])
    check("单网格总流量守恒", abs(upload.sum() - 100.0) < 1e-6, f"{upload.sum():.6f}")
    check("近距卫星分得更多", upload[0] > upload[1] and upload[0] > upload[2],
          str(np.round(upload, 3)))

    print("\n== 4. 接口：返回结构与回写 ==")
    node_state, topo = build_ring(24)
    t0 = time.perf_counter()
    new_topo, new_node, payload, utc_str = background_traffic_generation(
        topo, node_state, ts_noon
    )
    cost = (time.perf_counter() - t0) * 1000
    check("返回四元组", isinstance(payload, list) and len(payload) == 2)
    sat_flows, link_flows = payload
    check("卫星流量每颗一条", len(sat_flows) == 24, f"{len(sat_flows)}")
    check("单条为 [id, 上传, 总流量]", len(sat_flows[0]) == 3, str(sat_flows[0]))
    check("节点流量已回写",
          all(new_node.sat_list[str(i)].flow == sat_flows[i][2] for i in range(24)))
    check("时间戳格式", len(utc_str) == 19, utc_str)
    print(f"       链路流量 {len(link_flows)} 条，耗时 {cost:.1f} ms")

    print("\n== 5. 只读摘要不改状态 ==")
    node_state2, topo2 = build_ring(12)
    before = np.array([node_state2.sat_list[str(i)].flow for i in range(12)])
    snap = background_traffic_snapshot(topo2, node_state2, ts_noon)
    after = np.array([node_state2.sat_list[str(i)].flow for i in range(12)])
    check("摘要字段齐全",
          {"Timestamp", "GridSource", "ActiveCells", "OfferedGbps",
           "SatelliteFlow", "LinkFlow", "ElapsedMs"} <= set(snap))
    check("摘要不修改节点流量", np.array_equal(before, after))

    print("\n== 6. 与旧入口等价 ==")
    from src.simulation.background_traffic_generation import (
        background_traffic_generation as legacy_entry,
    )
    check("兼容层指向同一函数", legacy_entry is background_traffic_generation)

    print("\n== 汇总 ==")
    if fails:
        print(f"失败 {len(fails)} 项: {fails}")
        return 1
    print("背景流量模块自测全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
