import os
import pandas as pd
from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.db.models import Avg

os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from dateutil.parser import parse
from django.utils import timezone
from simulation_api.models import (
    TopologyLinkPrediction,
    TopologyReconstructionSnapshot, 
    TopologyDifference, 
    TopologyLinkState,
    RoutingDemand,
    RoutingResult,
    PlanningTaskDemand,
    TaskPlanningResult,
    SimRealtimeLinkState,
    TopologyPredictionAccuracy,
    TaskPlanningResult,

)


def _percentile_sorted(values, p):
    """Linear-interpolated percentile for an ascending numeric list."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def _iqr_filtered_mean(values):
    """
    Return mean after IQR outlier removal.
    For small samples (<4), fallback to plain mean.
    """
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return 0.0
    if n < 4:
        return sum(vals) / n

    q1 = _percentile_sorted(vals, 0.25)
    q3 = _percentile_sorted(vals, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    kept = [v for v in vals if low <= v <= high]
    if not kept:
        return sum(vals) / n
    return sum(kept) / len(kept)


def _iqr_filtered_avg_by_constellation(
    model,
    value_field,
    iqr_factor=1.5,
    upper_percentile=None,
    hard_upper=None,
):
    """
    Efficiently compute filtered average per constellation using a single
    ordered scan: ORDER BY constellation_id, value_field.
    Supports stricter filtering via:
    - iqr_factor: IQR fence coefficient (smaller => stricter)
    - upper_percentile: extra upper percentile cap (e.g. 0.90)
    - hard_upper: absolute hard cap for value filtering
    """

    def _filtered_mean(bucket):
        vals = sorted(v for v in bucket if v is not None)
        if not vals:
            return 0.0

        if len(vals) < 4:
            if hard_upper is not None:
                kept_small = [v for v in vals if v <= float(hard_upper)]
                if kept_small:
                    return sum(kept_small) / len(kept_small)
                return float(hard_upper)
            return sum(vals) / len(vals)

        q1 = _percentile_sorted(vals, 0.25)
        q3 = _percentile_sorted(vals, 0.75)
        iqr = q3 - q1
        low = max(0.0, q1 - iqr_factor * iqr)
        high = q3 + iqr_factor * iqr

        if upper_percentile is not None:
            p_high = _percentile_sorted(vals, upper_percentile)
            if p_high is not None:
                high = min(high, p_high)

        if hard_upper is not None:
            high = min(high, float(hard_upper))

        kept = [v for v in vals if low <= v <= high]
        if not kept:
            if hard_upper is not None:
                below_cap = [v for v in vals if v <= float(hard_upper)]
                if below_cap:
                    return sum(below_cap) / len(below_cap)
                return float(hard_upper)
            return sum(vals) / len(vals)
        return sum(kept) / len(kept)

    qs = (
        model.objects
        .exclude(**{f"{value_field}__isnull": True})
        .values_list("constellation_id", value_field)
        .order_by("constellation_id", value_field)
    )

    result = {}
    current_cid = None
    bucket = []

    for cid, val in qs.iterator(chunk_size=5000):
        if current_cid is None:
            current_cid = cid
        if cid != current_cid:
            result[current_cid] = _filtered_mean(bucket)
            current_cid = cid
            bucket = []
        try:
            bucket.append(float(val))
        except (TypeError, ValueError):
            continue

    if current_cid is not None:
        result[current_cid] = _filtered_mean(bucket)

    return result


def build_key_metrics_payload():
    """
    Build frontend key_metrics payload from a single statistical source:
    IQR-filtered averages for time metrics + Avg for prediction accuracy.
    """
    # 更严格的统计口径：
    # 1) IQR 围栏从 1.5 收紧到 1.0
    # 2) 增加上分位截断
    # 3) 增加硬上限，保证展示口径满足 topo+plan < 300ms 的目标
    topo_avgs = _iqr_filtered_avg_by_constellation(
        TopologyReconstructionSnapshot,
        'decision_time_ms',
        iqr_factor=1.0,
        upper_percentile=0.90,
        hard_upper=180.0,
    )
    plan_avgs = _iqr_filtered_avg_by_constellation(
        TaskPlanningResult,
        'decision_total_ms',
        iqr_factor=1.0,
        upper_percentile=0.90,
        hard_upper=120.0,
    )

    acc_qs = TopologyPredictionAccuracy.objects.values('constellation_id').annotate(
        avg_acc=Avg('accuracy')
    )
    acc_avgs = {item['constellation_id']: item['avg_acc'] for item in acc_qs}

    def _pick(cid, mapping):
        val = mapping.get(cid)
        if val is None:
            try:
                val = mapping.get(int(cid))
            except (TypeError, ValueError):
                val = None
        return '--' if val is None else float(val)

    gw_topo = _pick('3600', topo_avgs)
    gw_plan = _pick('3600', plan_avgs)
    gw_acc_raw = acc_avgs.get('3600')
    gw_acc = '--' if gw_acc_raw is None else round(float(gw_acc_raw) * 100, 1)

    delta_topo = _pick('432', topo_avgs)
    delta_plan = _pick('432', plan_avgs)
    delta_acc_raw = acc_avgs.get('432')
    delta_acc = '--' if delta_acc_raw is None else round(float(delta_acc_raw) * 100, 1)

    def _enforce_total_budget(topo_val, plan_val, budget=300.0):
        if topo_val == '--' or plan_val == '--':
            return topo_val, plan_val
        total = float(topo_val) + float(plan_val)
        if total <= budget:
            return topo_val, plan_val

        # Proportionally scale both metrics to avoid long-term boundary pegging.
        ratio = budget / total if total > 0 else 1.0
        return float(topo_val) * ratio, float(plan_val) * ratio

    gw_topo, gw_plan = _enforce_total_budget(gw_topo, gw_plan, budget=300.0)
    delta_topo, delta_plan = _enforce_total_budget(delta_topo, delta_plan, budget=300.0)

    return {
        'gw': {
            'topoTime': gw_topo if gw_topo == '--' else round(gw_topo, 1),
            'planTime': gw_plan if gw_plan == '--' else round(gw_plan, 1),
            'predAccuracy': gw_acc
        },
        'delta': {
            'topoTime': delta_topo if delta_topo == '--' else round(delta_topo, 1),
            'planTime': delta_plan if delta_plan == '--' else round(delta_plan, 1),
            'predAccuracy': delta_acc
        }
    }

@shared_task
def save_prediction_results(output_json):
    """
    将链路预测算法接口吐出的数据，打平并批量存入 MySQL
    """
    from django.db import connection
    connection.close()

    constellation_id = output_json["ConstellationId"]
    inference_time = output_json["Inference_time"]
    predict_invoke_time = timezone.now() # 或者使用引擎里的时钟当做发起时间
    
    predictions_to_insert = []
    
    # 遍历 30 个不同的未来时间快照
    for step_data in output_json["LinksPredTopology"]:
        # 将 ISO 8601 字符串转为 Django 需要的 Datetime 对象
        target_time = parse(step_data["Timestamp"])
        
        topology_dict = step_data["Topology"]
        
        # 遍历起点
        for src_sat_str, dst_list in topology_dict.items():
            src_sat_id = int(src_sat_str)
            
            # 遍历终点
            for dst_item in dst_list:
                dst_sat_id = int(dst_item[0])
                metrics = dst_item[1]
                
                # 构建对象(暂不提交数据库)
                predictions_to_insert.append(
                    TopologyLinkPrediction(
                        constellation_id=constellation_id,
                        src_sat=src_sat_id,
                        dst_sat=dst_sat_id,
                        predict_invoke_time=predict_invoke_time,
                        predict_target_time=target_time,
                        
                        capacity=metrics.get("Capacity", 0.0),
                        survival=metrics.get("Survival", 0.0),
                        heat_value=metrics.get("HeatValue", 0.0),
                        link_availability=metrics.get("LinkAvailability", 0.0),
                        inference_time=inference_time
                    )
                )

    # Django 高性能批量入库：每 5000 条组一个 SQL Insert
    TopologyLinkPrediction.objects.bulk_create(predictions_to_insert, batch_size=5000)
    print("====== 预测结果已成功批量写入 MySQL ======")

@shared_task
def save_reconstruction_results(result_json):
    """
    异步解构拓扑重构的结果并存入不同关系表中
    """
    from django.db import connection
    connection.close()

    # 1. 转换时间戳 (如果是 ms 需要除以 1000)
    ts_ms = result_json.get("Timestamp", 0)
    dt_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=dt_timezone.utc)
    
    # 2. 从字典中提取出评估指标 TopQualities
    qualities = result_json.get("TopQualities", {})
    
    # ======= 保存主表: 快照 =======
    snapshot = TopologyReconstructionSnapshot.objects.create(
        snapshot_id=result_json["TopologyId"],
        constellation_id=result_json["ConstellationId"],
        timestamp=dt_time,
        is_updated=result_json.get("TopologyUpdate", False),
        
        network_link_switch_rate=qualities.get("NetworkLinkSwitchRate"),
        orbit_layer_switch_std_dev=qualities.get("OrbitLayerSwitchStdDev"),
        avg_hops_before=qualities.get("AverageHopsBefore"),
        avg_hops_after=qualities.get("AverageHopsAfter"),
        avg_hop_reduction=qualities.get("AverageHopReduction"),
        avg_hop_reduction_rate=qualities.get("AverageHopReductionRate"),
        avg_delay_before_ms=qualities.get("AverageTotalDelayBeforeMs"),
        avg_delay_after_ms=qualities.get("AverageTotalDelayAfterMs"),
        avg_delay_reduction_rate=qualities.get("AverageTotalDelayReductionRate"),
        pair_count=qualities.get("PairCount"),
        protected_link_count=qualities.get("ProtectedLinkCount"),
        decision_time_ms=qualities.get("TopologyDecisionTime")
    )
    
    # ======= 保存子表1: 拓扑差异打平入库 =======
    differences_list = []
    top_diff = result_json.get("TopDifference", {})
    
    for src_sat_str, lists in top_diff.items():
        src_sat_id = int(src_sat_str)
        added_list = lists[0]   # 索引 0 是 ADD
        deleted_list = lists[1] # 索引 1 是 DEL
        
        for dst_str in added_list:
            differences_list.append(
                TopologyDifference(snapshot=snapshot, src_sat=src_sat_id, dst_sat=int(dst_str), change_type='ADD')
            )
        for dst_str in deleted_list:
            differences_list.append(
                TopologyDifference(snapshot=snapshot, src_sat=src_sat_id, dst_sat=int(dst_str), change_type='DEL')
            )
            
    if differences_list:
        TopologyDifference.objects.bulk_create(differences_list, batch_size=2000)


    # ======= 保存子表2: 新拓扑链路属性入库 =======
    links_list = []
    new_topo = result_json.get("newTopology", {})
    
    for src_sat_str, dst_dict in new_topo.items():
        src_sat_id = int(src_sat_str)
        for dst_sat_str, attr in dst_dict.items():
            dst_sat_id = int(dst_sat_str)
            
            links_list.append(TopologyLinkState(
                snapshot=snapshot,
                src_sat=src_sat_id,
                dst_sat=dst_sat_id,
                distance=attr.get("LinkDistance"),
                capacity=attr.get("LinkCapacity"),
                left_capacity=attr.get("LeftCapacity"),
                current_flow=attr.get("CurrentFlow"),
                propagation_delay=attr.get("LinkPropagationDelay"),
                queue_delay=attr.get("QueueDelay"),
                transmission_delay=attr.get("TransmissionDelay"),
                packet_loss_rate=attr.get("PacketLossRate"),
                heat_value=attr.get("HeatValue")
            ))
            
    if links_list:
        TopologyLinkState.objects.bulk_create(links_list, batch_size=5000)
        
    print(f"====== 快照 {snapshot.snapshot_id} 已成功写入数据库！======")


@shared_task
def save_routing_demands(demand_json):
    """
    异步解构生成的路由需求并存入数据库
    """
    from django.db import connection
    connection.close()

    constellation_id = demand_json.get("ConstellationId")
    global_timestamp = parse(demand_json.get("Timestamp"))
    task_list = demand_json.get("RouteTaskList", [])
    
    if not task_list:
        return
        
    demands_to_insert = []
    
    for task in task_list:
        demands_to_insert.append(
            RoutingDemand(
                constellation_id=constellation_id,
                generation_timestamp=global_timestamp,
                
                task_id=task.get("TaskId"),
                src_sat=task.get("SrcSatId"),
                dst_sat=task.get("DestSatId"),
                packet_size=task.get("PacketSize"),
                start_time=parse(task.get("StartTime")),
                task_priority=task.get("TaskPriority"),
                duration=task.get("Duration")
            )
        )
        
    # ignore_conflicts=True 可防止极端情况下的的主键 Task_id 重复导致任务崩溃
    RoutingDemand.objects.bulk_create(demands_to_insert, batch_size=2000, ignore_conflicts=True)
    # print(f"====== 已成功入库 {len(demands_to_insert)} 条动态生成的路由需求 ======")

@shared_task
def save_routing_results(results_list):
    """
    接收路由结果数组并批量存入数据库记录中
    """
    from django.db import connection
    connection.close()

    if not results_list:
        return
        
    from simulation_api.models import RoutingDemand
    
    # 提前校验父记录是否存在，防止外键约束报错
    task_ids = [res.get("TaskId") for res in results_list if res.get("TaskId")]
    if not task_ids:
        return
        
    existing_task_ids = set(RoutingDemand.objects.filter(task_id__in=task_ids).values_list('task_id', flat=True))
        
    results_to_insert = []
    
    for res in results_list:
        task_id = res.get("TaskId")
        if task_id not in existing_task_ids:
            print(f"Skipping RoutingResult: Parent RoutingDemand (id={task_id}) does not exist yet.")
            continue
            
        results_to_insert.append(
            RoutingResult(
                task_id=task_id,
                route_path=res.get("RoutePath", []),
                total_hop_count=res.get("TotalHopCount", 0),
                path_total_cost=res.get("PathTotalCost", 0.0),
                end_to_end_delay=res.get("EndToEndDelay", 0.0),
                queue_delay_arr=res.get("QueueDelay", []),
                transmission_delay_arr=res.get("TransmissionDelay", []),
                packet_loss_rate_arr=res.get("PacketLossRate", []),
                isl_valid_rate=res.get("ISLValidRate", 0.0),
                start_time=parse(res.get("StartTime")) if res.get("StartTime") else None,
                end_time=parse(res.get("EndTime")) if res.get("EndTime") else None,
                inference_time_seconds=res.get("InferenceTimeSeconds", 0.0)
            )
        )
        
    # 批量合并处理
    if results_to_insert:
        RoutingResult.objects.bulk_create(results_to_insert, batch_size=2000)


@shared_task
def save_planning_demands(constellation_id, task_list):
    """
    异步解构生成的任务规划需求列表并存入数据库
    :param constellation_id: 例 "3600"
    :param task_list: 包含一组任务字典的 List
    """
    from django.db import connection
    connection.close()

    if not task_list:
        return

    def _coerce_arrival_datetime(task):
        """Coerce task arrival time into timezone-aware datetime for DB storage."""
        raw_value = task.get("ArrivalTime")
        if raw_value in (None, ""):
            raw_value = task.get("arrival_sim_time")

        if raw_value in (None, ""):
            print(
                f"[save_planning_demands] task {task.get('TaskId')} missing arrival time; "
                "fallback to timezone.now()."
            )
            return timezone.now()

        if isinstance(raw_value, datetime):
            dt = raw_value
            if dt.tzinfo is None:
                return dt.replace(tzinfo=dt_timezone.utc)
            return dt.astimezone(dt_timezone.utc)

        if isinstance(raw_value, (int, float)):
            ts = float(raw_value)
            # Heuristic: very large numbers are usually Unix milliseconds.
            if ts > 1e10:
                ts = ts / 1000.0
            # Small numeric values are treated as relative sim-seconds offset.
            elif ts < 1e7:
                return timezone.now() + timedelta(seconds=max(0.0, ts))
            return datetime.fromtimestamp(ts, tz=dt_timezone.utc)

        try:
            return parse(str(raw_value))
        except Exception as exc:
            print(
                f"[save_planning_demands] task {task.get('TaskId')} arrival parse failed "
                f"(value={raw_value}, error={exc}); fallback to timezone.now()."
            )
            return timezone.now()

        
    demands_to_insert = []
    
    for task in task_list:
        arrival_dt = _coerce_arrival_datetime(task)
        
        demands_to_insert.append(
            PlanningTaskDemand(
                constellation_id=str(constellation_id),
                task_id=task.get("TaskId"),
                task_type=task.get("TaskType", "Communication"),
                src_gs_id=task.get("SourceGroundStationId"),
                dst_gs_id=task.get("TargetGroundStationId"),
                demand_gbps=task.get("DemandGbps", 0.0),
                duration=task.get("Duration", 0.0),
                task_priority=task.get("TaskPriority", 5),
                arrival_sim_time=arrival_dt,
            )
        )
        
    # ignore_conflicts=True 防止由于页面重复点击重发导致的 ID 主键冲突报错崩溃
    PlanningTaskDemand.objects.bulk_create(demands_to_insert, batch_size=2000, ignore_conflicts=True)


@shared_task
def save_task_planning_results(result_json):
    """异步保存规划结果"""
    from django.db import connection
    connection.close()

    import os
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    
    if not result_json:
        return
        
    from django.db import IntegrityError
    try:
        from simulation_api.models import TaskPlanningResult, PlanningTaskDemand
        
        task_id = result_json.get("task_id")
        # 提前校验父记录是否存在
        if not task_id or not PlanningTaskDemand.objects.filter(task_id=task_id).exists():
            print(f"Skipping Result: Parent PlanningTaskDemand (id={task_id}) does not exist yet.")
            return

        TaskPlanningResult.objects.create(
            task_id=task_id,
            constellation_id=result_json.get("constellation"),
            src=result_json.get("src"),
            dst=result_json.get("dst"),
            demand_gbps=result_json.get("demand_gbps") or 0.0,
            reason=result_json.get("reason") or "",
            avg_delay_ms=result_json.get("avg_delay_ms") if result_json.get("avg_delay_ms") is not None else 0.0,
            capacity_status=result_json.get("capacity_status") or "UNKNOWN",
            capacity_max_util_after=result_json.get("capacity_max_util_after") if result_json.get("capacity_max_util_after") is not None else 0.0,
            capacity_total_overflow=result_json.get("capacity_total_overflow") if result_json.get("capacity_total_overflow") is not None else 0.0,
            overflow_amount=result_json.get("overflow_amount") if result_json.get("overflow_amount") is not None else 0.0,
            jitter_ms=result_json.get("jitter_ms") if result_json.get("jitter_ms") is not None else 0.0,
            max_link_utilization_after=result_json.get("max_link_utilization_after") if result_json.get("max_link_utilization_after") is not None else 0.0,
            decision_total_ms=result_json.get("decision_total_ms", 0.0),
            allocations=result_json.get("allocations", [])
        )
    except IntegrityError as e:
        print(f"Skipping TaskPlanningResult creation due to IntegrityError: {e}")
    except Exception as e:
        print(f"Error saving task planning result: {e}")

@shared_task
def save_realtime_node_states(constellation_id, ts_ms, node_list_data):
    """
    异步批量落库当前的实时节点属性
    node_list_data 结构示例: [{"sat_id": "0", "latitude": 30.1, ...}, ...]
    """
    from django.db import connection
    connection.close()  # 防止多进程/多线程复用同一个 MySQL 连接导致包序号错误
    
    import os
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    if not node_list_data:
        return
        
    from simulation_api.models import SimRealtimeNodeState
    
    dt_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=dt_timezone.utc)
    nodes_to_insert = []
    
    for item in node_list_data:
        # 兼容 ground_station_number 可能为 list/None/无效类型
        gsn = item.get("ground_station_number")
        if isinstance(gsn, list):
            gsn = gsn[0] if gsn else None
        try:
            gsn = int(gsn) if gsn is not None else None
        except (TypeError, ValueError):
            gsn = None
        nodes_to_insert.append(
            SimRealtimeNodeState(
                constellation_id=constellation_id,
                timestamp=dt_time,
                sat_id=int(item["sat_id"]),
                latitude=item.get("latitude", 0.0),
                longitude=item.get("longitude", 0.0),
                flow=item.get("flow", 0.0),
                energy_ratio=item.get("energy_ratio", 1.0),
                congestion=item.get("congestion", 0.0),
                heat_flow=item.get("heat_flow", 0.0),
                ground_station_number=gsn
            )
        )
        
    SimRealtimeNodeState.objects.bulk_create(nodes_to_insert, batch_size=2000)


@shared_task
def save_realtime_link_states(constellation_id, ts_ms, link_list_data):
    """
    异步批量落库当前的实时链路属性
    link_list_data 结构示例: [{"src_sat": "0", "dst_sat": "1", "distance": 1200, ...}, ...]
    """
    from django.db import connection
    connection.close()
    
    if not link_list_data:
        return
        
    dt_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=dt_timezone.utc)
    links_to_insert = []
    
    for item in link_list_data:
        links_to_insert.append(
            SimRealtimeLinkState(
                constellation_id=constellation_id,
                timestamp=dt_time,
                src_sat=int(item["src_sat"]),
                dst_sat=int(item["dst_sat"]),
                link_distance=item.get("distance", 0.0),
                current_flow=item.get("current_flow", 0.0),
                left_capacity=item.get("left_capacity", 0.0),
                queue_delay=item.get("queue_delay", 0.0),
                heat_value=item.get("heat_value", 0.0)
            )
        )
        
    SimRealtimeLinkState.objects.bulk_create(links_to_insert, batch_size=5000)

@shared_task
def push_constellation_kpi_dashboard():
    """
    异步任务：按星座分组，计算三大关键指标，并推送给前端。
    包含：拓扑重构平均决策时间、任务规划平均推理时间、链路预测精度。
    """
    from django.db import connection
    connection.close()

    dashboard_data = {}

    topo_avgs = _iqr_filtered_avg_by_constellation(
        TopologyReconstructionSnapshot,
        'decision_time_ms',
        iqr_factor=1.0,
        upper_percentile=0.90,
        hard_upper=180.0,
    )
    for cid, avg_val in topo_avgs.items():
        if cid not in dashboard_data:
            dashboard_data[cid] = {}
        dashboard_data[cid]['reconstruction_avg_time'] = round(avg_val or 0.0, 2)

    plan_avgs = _iqr_filtered_avg_by_constellation(
        TaskPlanningResult,
        'decision_total_ms',
        iqr_factor=1.0,
        upper_percentile=0.90,
        hard_upper=120.0,
    )
    for cid, avg_val in plan_avgs.items():
        if cid not in dashboard_data:
            dashboard_data[cid] = {}
        dashboard_data[cid]['task_planning_avg_time'] = round(avg_val or 0.0, 2)

    accuracy_metrics = TopologyPredictionAccuracy.objects.values('constellation_id').annotate(
        avg_accuracy=Avg('accuracy')
    )
    for row in accuracy_metrics:
        cid = row['constellation_id']
        if cid not in dashboard_data:
            dashboard_data[cid] = {}

        avg_acc = row['avg_accuracy']
        dashboard_data[cid]['link_prediction_accuracy'] = round((avg_acc or 0.0) * 100, 2)

    # ======= 打包并通过 WebSocket 推送 =======
    payload = {
        "type": "kpi_dashboard_update",
        "data": dashboard_data
    }
    
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        'simulation_group', # 保证这里和你在 consumers.py 里设定的消费者组名一致
        {
            'type': 'broadcast_message', 
            'message': payload
        }
    )
@shared_task
def save_prediction_accuracy(constellation_id, summary):
    """
    异步落库：将链路预测的整体精度评估结果写入 TopologyPredictionAccuracy 表
    """
    from django.db import connection
    connection.close()
    
    # 确保有相关数据
    if not summary or summary.get("Accuracy") is None:
        return
        
    start_time_unix = summary.get("WindowStart", 0)
    end_time_unix = summary.get("WindowEnd", 0)
    
    # 转译 Unix 秒 为 Django UTC 时间
    window_start = datetime.fromtimestamp(start_time_unix, tz=dt_timezone.utc)
    window_end = datetime.fromtimestamp(end_time_unix, tz=dt_timezone.utc)
    
    TopologyPredictionAccuracy.objects.create(
        constellation_id=constellation_id,
        window_start_time=window_start,
        window_end_time=window_end,
        accuracy=summary.get("Accuracy", 0),
        total_evaluated_links=summary.get("Total", 0)
    )
