from django.db import models

class Constellation(models.Model):
    """
    6.3.1. 星座信息表
    星座基础配置参数，拓扑生成基础
    """
    constellation_id = models.CharField(
        max_length=64, primary_key=True, verbose_name="星座ID"
    )
    name = models.CharField(max_length=100, verbose_name="星座名称")
    orbit_height = models.IntegerField(verbose_name="轨道高度（km）")
    orbit_inclination = models.FloatField(verbose_name="轨道倾角（度）")
    num_orbit_plane = models.IntegerField(verbose_name="轨道面数量")
    num_sats_in_plane = models.IntegerField(verbose_name="单轨道面卫星数")
    phase_factor = models.IntegerField(verbose_name="相位因子")
    created_at = models.DateTimeField(db_index=True, verbose_name="创建时间")

    def __str__(self):
        return self.name

    class Meta:
        db_table = "constellation"
        verbose_name = "星座信息"
        verbose_name_plural = verbose_name


class Satellite(models.Model):
    """
    6.3.2. 卫星信息表
    卫星详细信息、轨道参数、实时状态
    """
    id = models.BigAutoField(primary_key=True, verbose_name="卫星ID")
    norad_id = models.CharField(
        max_length=50, unique=True, verbose_name="NORAD编号"
    )
    name = models.CharField(
        max_length=100, null=True, blank=True, verbose_name="卫星名称"
    )
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="所属星座"
    )
    tle_line1 = models.CharField(
        max_length=200, null=True, blank=True, verbose_name="TLE第一行"
    )
    tle_line2 = models.CharField(
        max_length=200, null=True, blank=True, verbose_name="TLE第二行"
    )
    orbit_plane = models.IntegerField(db_index=True, verbose_name="所在轨道面")
    orbit_plane_index = models.IntegerField(verbose_name="轨道面内序号")
    altitude = models.FloatField(verbose_name="轨道高度（km）")
    inclination = models.FloatField(verbose_name="轨道倾角（度）")
    remaining_energy = models.FloatField(
        null=True, blank=True, verbose_name="剩余能量 [0,1]"
    )
    satellite_load = models.FloatField(
        null=True, blank=True, verbose_name="卫星负载 [0,1]"
    )
    history_flow = models.FloatField(
        null=True, blank=True, verbose_name="历史累计业务流量（GB）"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(
        auto_now=True, null=True, blank=True, verbose_name="更新时间"
    )

    def __str__(self):
        return f"{self.name} ({self.norad_id})"

    class Meta:
        db_table = "satellite"
        verbose_name = "卫星信息"
        verbose_name_plural = verbose_name


class TopologySnapshot(models.Model):
    """
    6.3.3. 拓扑快照主表
    拓扑重构版本管理核心
    """
    snapshot_id = models.CharField(
        max_length=64, primary_key=True, verbose_name="拓扑快照ID"
    )
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    prev_snapshot = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True,
        db_column="prev_snapshot_id", db_index=True, verbose_name="上一版本快照ID"
    )
    version = models.BigIntegerField(db_index=True, verbose_name="版本号")
    timestamp = models.DateTimeField(db_index=True, verbose_name="拓扑时间戳")
    is_active = models.BooleanField(
        db_index=True, default=0, verbose_name="是否当前生效"
    )
    algorithm_version = models.CharField(
        max_length=50, null=True, blank=True, verbose_name="算法版本"
    )
    execution_time_ms = models.FloatField(
        null=True, blank=True, verbose_name="执行耗时（ms）"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="入库时间")

    class Meta:
        db_table = "topology_snapshot"
        verbose_name = "拓扑快照"
        verbose_name_plural = verbose_name


class TopologyLinkRelation(models.Model):
    """
    6.3.4. 拓扑链路关系表
    卫星间稳定链路连接关系，静态属性
    """
    relation_id = models.CharField(
        max_length=64, primary_key=True, verbose_name="链路关系ID"
    )
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    src_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="src_links",
        db_column="src_sat_id", db_index=True, verbose_name="源卫星ID"
    )
    dst_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="dst_links",
        db_column="dst_sat_id", db_index=True, verbose_name="目标卫星ID"
    )
    link_type = models.CharField(max_length=20, verbose_name="链路类型")
    is_bidirectional = models.BooleanField(default=1, verbose_name="是否双向链路")
    valid_from = models.DateTimeField(verbose_name="生效开始时间")
    valid_to = models.DateTimeField(null=True, blank=True, verbose_name="生效结束时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "topology_link_relation"
        verbose_name = "拓扑链路关系"
        verbose_name_plural = verbose_name


class TopologyLinkAttribute(models.Model):
    """
    6.3.5. 拓扑链路属性表
    时变链路属性，拓扑重构核心输出
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    snapshot = models.ForeignKey(
        TopologySnapshot, on_delete=models.CASCADE,
        db_column="snapshot_id", db_index=True, verbose_name="拓扑快照ID"
    )
    relation = models.ForeignKey(
        TopologyLinkRelation, on_delete=models.CASCADE,
        db_column="relation_id", db_index=True, verbose_name="链路关系ID"
    )
    link_distance = models.FloatField(null=True, blank=True, verbose_name="空间距离（km）")
    link_capacity = models.FloatField(null=True, blank=True, verbose_name="链路容量（Gbps）")
    link_left_capacity = models.FloatField(null=True, blank=True, verbose_name="链路剩余容量（Gbps）") 
    link_current_flow = models.FloatField(null=True, blank=True, verbose_name="链路当前流量（Gbps）")

    link_available_duration_score = models.FloatField(null=True, blank=True, verbose_name="可用维持时间评分")

    link_propagation_delay = models.FloatField(null=True, blank=True, verbose_name="传播时延（ms）")
    link_queue_delay = models.FloatField(null=True, blank=True, verbose_name="排队时延（ms）")
    link_transmission_delay = models.FloatField(null=True, blank=True, verbose_name="传输时延（ms）")
    link_packet_loss_rate = models.FloatField(null=True, blank=True, verbose_name="丢包率")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="入库时间")

    class Meta:
        db_table = "topology_link_attribute"
        verbose_name = "拓扑链路属性"
        verbose_name_plural = verbose_name


class TopologyQuality(models.Model):
    """
    6.3.6. 拓扑质量属性表
    全网质量评估指标
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    snapshot = models.OneToOneField(
        TopologySnapshot, on_delete=models.CASCADE,
        db_column="snapshot_id", unique=True, db_index=True, verbose_name="拓扑快照ID"
    )
    network_link_switch_rate = models.FloatField(null=True, blank=True, verbose_name="全网链路切换率")
    orbit_layer_switch_stddev = models.FloatField(null=True, blank=True, verbose_name="轨道层切换标准差")
    average_hop_before = models.FloatField(null=True, blank=True, verbose_name="重构前平均跳数")
    average_hop_after = models.FloatField(null=True, blank=True, verbose_name="重构后平均跳数")
    average_hop_reduction = models.FloatField(null=True, blank=True, verbose_name="平均跳数减少量")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="入库时间")

    class Meta:
        db_table = "topology_quality"
        verbose_name = "拓扑质量评估"
        verbose_name_plural = verbose_name


class TopologyDifference(models.Model):
    """
    6.3.7. 拓扑差异表
    相邻拓扑快照链路变化
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    snapshot = models.ForeignKey(
        TopologySnapshot, on_delete=models.CASCADE,
        db_column="snapshot_id", db_index=True, related_name="current_changes", verbose_name="新拓扑ID"
    )
    prev_snapshot = models.ForeignKey(
        TopologySnapshot, on_delete=models.CASCADE,
        db_column="prev_snapshot_id", db_index=True, related_name="prev_changes", verbose_name="上一拓扑ID"
    )
    sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE,
        db_column="sat_id", db_index=True, verbose_name="卫星ID"
    )
    change_type = models.CharField(max_length=10, verbose_name="变化类型 ADD/DEL")
    target_sat = models.BigIntegerField(verbose_name="目标卫星ID")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="记录时间")

    class Meta:
        db_table = "topology_difference"
        verbose_name = "拓扑差异"
        verbose_name_plural = verbose_name


class TopologyLinkPrediction(models.Model):
    """
    6.3.8. 链路状态预测表
    链路预测算法输出结果
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    topology = models.ForeignKey(
        TopologySnapshot, on_delete=models.CASCADE,
        db_column="topology_id", db_index=True, verbose_name="拓扑ID"
    )
    prediction_id = models.CharField(max_length=64, db_index=True, verbose_name="预测批次ID")
    src_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="pred_src",
        db_column="src_sat_id", db_index=True, verbose_name="源卫星ID"
    )
    dst_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="pred_dst",
        db_column="dst_sat_id", db_index=True, verbose_name="目标卫星ID"
    )
    predict_start_time = models.DateTimeField(db_index=True, verbose_name="预测发起时间")
    predict_target_time = models.DateTimeField(db_index=True, verbose_name="预测目标时间")
    pred_flow = models.FloatField(null=True, blank=True, verbose_name="预测业务流量")
    delay = models.FloatField(null=True, blank=True, verbose_name="预测综合时延")
    capacity = models.FloatField(null=True, blank=True, verbose_name="预测剩余容量")
    survival = models.FloatField(null=True, blank=True, verbose_name="预测连续生存度")
    heat_value = models.FloatField(null=True, blank=True, verbose_name="预测流量热力值")
    signal_intensity = models.FloatField(null=True, blank=True, verbose_name="预测信号强度")
    confidence_score = models.FloatField(null=True, blank=True, verbose_name="预测置信度")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "topology_link_prediction"
        verbose_name = "链路状态预测"
        verbose_name_plural = verbose_name


class LinkQualityHistory(models.Model):
    """
    6.3.9. 链路质量历史表
    链路质量时序数据，用于趋势分析、模型训练
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    link_relation = models.ForeignKey(
        TopologyLinkRelation, on_delete=models.CASCADE,
        db_column="link_id", db_index=True, verbose_name="链路关系ID"
    )
    signal_intensity = models.FloatField(null=True, blank=True, verbose_name="信号强度")
    ber = models.FloatField(null=True, blank=True, verbose_name="误码率")
    packet_loss_rate = models.FloatField(null=True, blank=True, verbose_name="丢包率")
    delay = models.FloatField(null=True, blank=True, verbose_name="综合时延")
    pred_flow = models.FloatField(null=True, blank=True, verbose_name="预测业务流量")
    capacity = models.FloatField(null=True, blank=True, verbose_name="剩余容量")
    survival = models.FloatField(null=True, blank=True, verbose_name="连续生存度")
    heat_value = models.FloatField(null=True, blank=True, verbose_name="流量热力值")
    timestamp = models.DateTimeField(db_index=True, verbose_name="时间戳")

    class Meta:
        db_table = "link_quality_history"
        verbose_name = "链路质量历史"
        verbose_name_plural = verbose_name


class PlanningTask(models.Model):
    """
    6.3.10. 规划任务信息表
    任务规划算法输入：业务流任务参数
    """
    task_id = models.CharField(
        max_length=64, primary_key=True, verbose_name="任务ID"
    )
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    src_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="plan_src",
        db_column="src_sat_id", db_index=True, verbose_name="源卫星ID"
    )
    dst_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="plan_dst",
        db_column="dst_sat_id", db_index=True, verbose_name="目的卫星ID"
    )
    demand_gbps = models.FloatField(null=True, blank=True, verbose_name="带宽需求（Gbps）")
    packet_size = models.FloatField(null=True, blank=True, verbose_name="数据块大小（Mbits）")
    arrival_time = models.DateTimeField(verbose_name="业务流到达时间")
    duration = models.IntegerField(null=True, blank=True, verbose_name="持续时间（ms）")
    delay_budget = models.FloatField(null=True, blank=True, verbose_name="最大可容忍时延（ms）")
    task_priority = models.IntegerField(null=True, blank=True, verbose_name="业务优先级")
    status = models.CharField(max_length=20, db_index=True, verbose_name="任务状态")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="完成时间")

    class Meta:
        db_table = "planning_task"
        verbose_name = "任务规划输入"
        verbose_name_plural = verbose_name


class TaskPlanningResult(models.Model):
    """
    6.3.11. 任务规划结果表
    任务规划输出：3条路径 + 任务级指标
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    task = models.ForeignKey(
        PlanningTask, on_delete=models.CASCADE,
        db_column="task_id", db_index=True, verbose_name="任务ID"
    )
    path_rank = models.IntegerField(verbose_name="路径排名 1/2/3")
    path_nodes = models.TextField(verbose_name="路径节点序列 JSON数组")
    path_cost = models.FloatField(null=True, blank=True, verbose_name="路径总代价")
    path_split_ratio = models.FloatField(null=True, blank=True, verbose_name="分流比例")
    path_delay = models.FloatField(null=True, blank=True, verbose_name="路径时延")
    total_delay = models.FloatField(null=True, blank=True, verbose_name="总完成时延")
    jitter = models.FloatField(null=True, blank=True, verbose_name="多路径时延差")
    overflow = models.FloatField(null=True, blank=True, verbose_name="超载量")
    algorithm_version = models.CharField(max_length=50, null=True, blank=True, verbose_name="算法版本")
    execution_time_ms = models.FloatField(null=True, blank=True, verbose_name="执行耗时")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "task_planning_result"
        unique_together = [["task", "path_rank"]]
        verbose_name = "任务规划结果"
        verbose_name_plural = verbose_name


class RoutingTask(models.Model):
    """
    6.3.12. 路由任务信息表
    单播路由任务输入
    """
    task_id = models.CharField(
        max_length=64, primary_key=True, verbose_name="任务ID"
    )
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    src_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="route_src",
        db_column="src_sat_id", db_index=True, verbose_name="源卫星ID"
    )
    dst_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="route_dst",
        db_column="dst_sat_id", db_index=True, verbose_name="目的卫星ID"
    )
    packet_size = models.FloatField(null=True, blank=True, verbose_name="数据块大小")
    arrival_time = models.DateTimeField(verbose_name="到达时间")
    task_priority = models.IntegerField(null=True, blank=True, verbose_name="业务优先级")
    status = models.CharField(max_length=20, db_index=True, verbose_name="任务状态")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="完成时间")

    class Meta:
        db_table = "routing_task"
        verbose_name = "路由任务输入"
        verbose_name_plural = verbose_name


class RoutingResult(models.Model):
    """
    6.3.13. 路由结果表
    单路径路由算法输出
    """
    id = models.BigAutoField(primary_key=True, verbose_name="记录ID")
    task = models.ForeignKey(
        RoutingTask, on_delete=models.CASCADE,
        db_column="task_id", db_index=True, verbose_name="任务ID"
    )
    route_path = models.TextField(verbose_name="路由路径节点 JSON")
    total_hop_count = models.IntegerField(null=True, blank=True, verbose_name="总跳数")
    path_total_cost = models.FloatField(null=True, blank=True, verbose_name="路径总代价")
    end_to_end_delay = models.FloatField(null=True, blank=True, verbose_name="端到端时延")
    isl_valid_rate = models.FloatField(null=True, blank=True, verbose_name="星间链路有效率")
    overflow = models.FloatField(null=True, blank=True, verbose_name="超载量")
    created_at = models.DateTimeField(db_index=True, auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "routing_result"
        verbose_name = "路由结果"
        verbose_name_plural = verbose_name


# ===================== 补充的2张核心表（项目必须）=====================
class SimulationTask(models.Model):
    """
    补充：仿真主任务表
    顶层任务，区分每一次独立仿真
    """
    simulation_id = models.CharField(max_length=64, primary_key=True, verbose_name="仿真任务ID")
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    status = models.CharField(max_length=20, default="PENDING", db_index=True, verbose_name="仿真状态")
    start_time = models.DateTimeField(null=True, blank=True, verbose_name="开始时间")
    end_time = models.DateTimeField(null=True, blank=True, verbose_name="结束时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "simulation_task"
        verbose_name = "仿真主任务"
        verbose_name_plural = verbose_name


class BackgroundTraffic(models.Model):
    """
    补充：背景流量表
    仿真初始化生成的背景流量存储
    """
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    simulation = models.ForeignKey(
        SimulationTask, on_delete=models.CASCADE,
        db_column="simulation_id", db_index=True, verbose_name="仿真任务ID"
    )
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    src_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="bg_src",
        db_column="src_sat_id", verbose_name="源卫星"
    )
    dst_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE, related_name="bg_dst",
        db_column="dst_sat_id", verbose_name="目标卫星"
    )
    flow_gbps = models.FloatField(verbose_name="流量大小（Gbps）")
    start_time = models.DateTimeField(verbose_name="生效开始时间")
    end_time = models.DateTimeField(verbose_name="生效结束时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "background_traffic"
        verbose_name = "背景流量"
        verbose_name_plural = verbose_name


class SatelliteRealTimeState(models.Model):
    """
    卫星节点实时状态（1秒更新）
    对应：SatelliteNodeState + NodeAttribute
    """
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    
    # 关联
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    satellite = models.ForeignKey(
        Satellite, on_delete=models.CASCADE,
        db_column="sat_id", db_index=True, verbose_name="卫星ID"
    )
    
    # 实时位置
    latitude = models.FloatField(verbose_name="纬度")
    longitude = models.FloatField(verbose_name="经度")
    
    # 实时负载
    flow = models.FloatField(verbose_name="当前流量")
    energy_ratio = models.FloatField(verbose_name="能量比例 [0,1]")
    congestion = models.FloatField(verbose_name="拥塞程度")
    heat_flow = models.FloatField(verbose_name="热流量")
    
    # 时间戳（1秒时序核心）
    timestamp = models.BigIntegerField(db_index=True, verbose_name="毫秒时间戳")
    created_at = models.DateTimeField(auto_now=True, verbose_name="入库时间")

    class Meta:
        db_table = "satellite_real_time_state"
        verbose_name = "卫星实时状态"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=["constellation", "satellite", "timestamp"]),
        ]


class TopologyRealTimeState(models.Model):
    """
    拓扑链路实时状态（1秒更新）
    对应：TopologyState + LinksQualitiesValue
    """
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    
    # 关联
    constellation = models.ForeignKey(
        Constellation, on_delete=models.CASCADE,
        db_column="constellation_id", db_index=True, verbose_name="星座ID"
    )
    src_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE,
        db_column="src_sat_id", related_name="rt_src", verbose_name="源卫星"
    )
    dst_sat = models.ForeignKey(
        Satellite, on_delete=models.CASCADE,
        db_column="dst_sat_id", related_name="rt_dst", verbose_name="目标卫星"
    )
    
    # 链路实时属性（完全对应你的 LinksQualitiesValue）
    link_distance = models.FloatField(verbose_name="链路距离")
    link_capacity = models.FloatField(verbose_name="额定容量")
    left_capacity = models.FloatField(verbose_name="剩余容量")
    current_flow = models.FloatField(verbose_name="当前流量负载")
    link_propagation_delay = models.FloatField(verbose_name="传播时延")
    queue_delay = models.FloatField(verbose_name="排队时延")
    transmission_delay = models.FloatField(verbose_name="传输时延")
    packet_loss_rate = models.FloatField(verbose_name="丢包率")
    heat_value = models.FloatField(verbose_name="热力值")
    
    # 时间戳
    timestamp = models.BigIntegerField(db_index=True, verbose_name="毫秒时间戳")
    created_at = models.DateTimeField(auto_now=True, verbose_name="入库时间")

    class Meta:
        db_table = "topology_real_time_state"
        verbose_name = "拓扑实时状态"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=["constellation", "src_sat", "dst_sat", "timestamp"]),
        ]