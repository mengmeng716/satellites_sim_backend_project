from django.db import models

class TopologyLinkPrediction(models.Model):
    """【预测值】链路预测算法输出表（打平存储，支持未来时序分析）"""
    
    # --- 维度：实体标识 ---
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    src_sat = models.IntegerField(verbose_name="源卫星 ID")
    dst_sat = models.IntegerField(verbose_name="目的卫星 ID")
    
    # --- 维度：时间轴 ---
    predict_invoke_time = models.DateTimeField(db_index=True, verbose_name="发起预测的时间")
    predict_target_time = models.DateTimeField(verbose_name="预测发生的目标长时刻")
    
    # --- 指标：拓扑参数 ---
    capacity = models.FloatField(verbose_name="预测剩余容量(Capacity)")
    survival = models.FloatField(verbose_name="预测链路存活度(Survival)")
    heat_value = models.FloatField(verbose_name="预测热力值(HeatValue)")
    link_availability = models.FloatField(verbose_name="可用性评分(LinkAvailability)")
    
    # --- 指标：执行性能 ---
    inference_time = models.FloatField(null=True, blank=True, verbose_name="算法批量推理耗时(s)")

    class Meta:
        # 为 KPI-1 分析创建高效率的复合索引
        # 由于做 KPI 统计时常用: WHERE constellation_id = ? AND src_sat = ? AND dst_sat = ? AND predict_target_time = ?
        indexes = [
            models.Index(fields=['constellation_id', 'src_sat', 'dst_sat', 'predict_target_time']),
        ]

class TopologyReconstructionSnapshot(models.Model):
    """
    【主表】拓扑重构快照与质量评估主表
    记录每次重构决定的整体表现，包含 KPI-3（决策时间）
    """
    snapshot_id = models.CharField(max_length=128, primary_key=True, verbose_name="拓扑快照ID")
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    timestamp = models.DateTimeField(db_index=True, verbose_name="重构发生时间")
    
    is_updated = models.BooleanField(default=False, verbose_name="拓扑是否真的发生了更新")
    
    # === 以下直接映射 TopQualities 评估指标 ===
    network_link_switch_rate = models.FloatField(null=True, blank=True, verbose_name="全网链路切换率")
    orbit_layer_switch_std_dev = models.FloatField(null=True, blank=True, verbose_name="轨道层间切换标准差")
    
    avg_hops_before = models.FloatField(null=True, blank=True, verbose_name="重构前平均跳数")
    avg_hops_after = models.FloatField(null=True, blank=True, verbose_name="重构后平均跳数")
    avg_hop_reduction = models.FloatField(null=True, blank=True, verbose_name="平均跳数减少量")
    avg_hop_reduction_rate = models.FloatField(null=True, blank=True, verbose_name="平均跳数减少率")
    
    avg_delay_before_ms = models.FloatField(null=True, blank=True, verbose_name="重构前平均总时延(ms)")
    avg_delay_after_ms = models.FloatField(null=True, blank=True, verbose_name="重构后平均总时延(ms)")
    avg_delay_reduction_rate = models.FloatField(null=True, blank=True, verbose_name="平均总时延减少率")
    
    pair_count = models.IntegerField(null=True, blank=True, verbose_name="业务需求对数量")
    protected_link_count = models.FloatField(null=True, blank=True, verbose_name="受保护链路数量")
    
    # KPI-3: 拓扑重构决策时间
    decision_time_ms = models.FloatField(null=True, blank=True, verbose_name="重构算法决策时间(ms)")
    
    created_at = models.DateTimeField(auto_now_add=True)


class TopologyDifference(models.Model):
    """
    【子表1】拓扑差异表
    专门记录相对于上一个版本，哪些边被 ADD，哪些被 DEL，方便前端快速渲染断链动效
    """
    snapshot = models.ForeignKey(TopologyReconstructionSnapshot, on_delete=models.CASCADE, related_name="differences", db_index=True)
    src_sat = models.IntegerField(db_index=True, verbose_name="源卫星")
    dst_sat = models.IntegerField(db_index=True, verbose_name="目标卫星")
    change_type = models.CharField(max_length=10, choices=(('ADD', 'Added'), ('DEL', 'Deleted')), verbose_name="变化类型")

    class Meta:
        indexes = [
            models.Index(fields=['snapshot', 'change_type']),
        ]

class TopologyLinkState(models.Model):
    """
    【子表2】新拓扑网络状态表
    记录重构后的新拓扑结构中，每一条留存连接的详细属性
    """
    snapshot = models.ForeignKey(TopologyReconstructionSnapshot, on_delete=models.CASCADE, related_name="links", db_index=True)
    src_sat = models.IntegerField(db_index=True, verbose_name="源卫星")
    dst_sat = models.IntegerField(db_index=True, verbose_name="目标卫星")
    
    # 物理与网络属性
    distance = models.FloatField(null=True, blank=True, verbose_name="距离")
    capacity = models.FloatField(null=True, blank=True, verbose_name="总容量")
    left_capacity = models.FloatField(null=True, blank=True, verbose_name="剩余容量")
    current_flow = models.FloatField(null=True, blank=True, verbose_name="当前流量")
    
    propagation_delay = models.FloatField(null=True, blank=True, verbose_name="传播时延")
    queue_delay = models.FloatField(null=True, blank=True, verbose_name="排队时延")
    transmission_delay = models.FloatField(null=True, blank=True, verbose_name="传输时延")
    
    packet_loss_rate = models.FloatField(null=True, blank=True, verbose_name="丢包率")
    heat_value = models.FloatField(null=True, blank=True, verbose_name="热力值")

class RoutingDemand(models.Model):
    """【路由需求】模拟背景或用户生成的离散路由请求表"""
    
    # 1. 继承下来的全局/环境参数
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    generation_timestamp = models.DateTimeField(db_index=True, verbose_name="仿真生成批次时刻")
    
    # 2. 任务专属属性 (Task Attributes)
    task_id = models.CharField(max_length=128, primary_key=True, verbose_name="路由任务ID") # 对应 "RouteGen_..."
    src_sat = models.IntegerField(db_index=True, verbose_name="源卫星ID")
    dst_sat = models.IntegerField(db_index=True, verbose_name="目的卫星ID")
    
    # 3. 业务指标
    packet_size = models.FloatField(verbose_name="业务包大小(PacketSize)")
    task_priority = models.IntegerField(default=5, verbose_name="业务优先级(TaskPriority)")
    duration = models.FloatField(verbose_name="持续占用时长(Duration)")
    
    # 4. 时间指标
    # 注意把 StartTime 转成数据库的 datetime
    start_time = models.DateTimeField(db_index=True, verbose_name="任务开始执行时间(StartTime)") 
    
    # 记录该条记录何时写入数据库的
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 增加复合索引加快查询
        indexes = [
            models.Index(fields=['constellation_id', 'generation_timestamp']),
        ]

    def __str__(self):
        return f"RouteDemand {self.task_id} (Sat {self.src_sat} -> Sat {self.dst_sat})"


class RoutingResult(models.Model):
    """【路由结果】存储路由优化算法根据需求生成的对应求路结果"""
    id = models.BigAutoField(primary_key=True)
    
    # 核心外键关联，严格指向 RoutingDemand 表中的 task_id 主键
    task = models.ForeignKey(
        'RoutingDemand', 
        on_delete=models.CASCADE, 
        db_column="task_id", 
        related_name="routing_results",
        verbose_name="关联的路由任务ID"
    )
    
    # ==== 路由核心结构 (List -> JSON) ====
    route_path = models.JSONField(verbose_name="规划的路径节点数组")
    queue_delay_arr = models.JSONField(verbose_name="每一跳的排队时延(数组)")
    transmission_delay_arr = models.JSONField(verbose_name="每一跳的传输时延(数组)")
    packet_loss_rate_arr = models.JSONField(verbose_name="每一跳的丢包率(数组)")
    
    # ==== 指标量化统计 ====
    total_hop_count = models.IntegerField(verbose_name="总跳数")
    path_total_cost = models.FloatField(verbose_name="路径总代价")
    end_to_end_delay = models.FloatField(verbose_name="端到端总时延")
    isl_valid_rate = models.FloatField(verbose_name="星间链路有效率")
    
    # ==== 性能与时间指标 ====
    start_time = models.DateTimeField(null=True, blank=True, verbose_name="路由起算/生效时间")
    end_time = models.DateTimeField(null=True, blank=True, verbose_name="路由结束/失效时间")
    inference_time_seconds = models.FloatField(null=True, blank=True, verbose_name="算法推理耗时(秒)")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 为高频的业务指标建立索引
        indexes = [
            models.Index(fields=['task']),
        ]

    def __str__(self):
        return f"Result for Task: {self.task_id} (Hops: {self.total_hop_count})"

class PlanningTaskDemand(models.Model):
    """【任务规划需求】前端下发或离线配置的高级调度任务表"""
    
    # === 环境关联 ===
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    
    # === 任务基础指纹 ===
    task_id = models.CharField(max_length=128, primary_key=True, verbose_name="规划任务ID")
    task_type = models.CharField(max_length=50, default="Communication", verbose_name="任务类型")
    
    # === 任务物理起止点 (地面对地面的复杂调度) ===
    src_gs_id = models.IntegerField(db_index=True, verbose_name="源地面站ID")
    dst_gs_id = models.IntegerField(db_index=True, verbose_name="目的地面站ID")
    
    # === 业务需求参数 ===
    demand_gbps = models.FloatField(verbose_name="带宽需求(Gbps)")
    duration = models.FloatField(verbose_name="持续占用时长(s)")
    task_priority = models.IntegerField(default=5, verbose_name="任务优先级")
    
    # === 时间流 ===
    arrival_sim_time = models.DateTimeField(db_index=True, verbose_name="仿真到达时间") 
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 高频条件索引：通常我们会查“某星座下随着时间轴逐步抵达的任务” 
        indexes = [
            models.Index(fields=['constellation_id', 'arrival_sim_time']),
        ]

    def __str__(self):
        return f"PlanTask {self.task_id} (Type: {self.task_type}, GS {self.src_gs_id} -> GS {self.dst_gs_id})"
    
class TaskPlanningResult(models.Model):
    """【任务规划结果】记录全局多路规划/分流调度的详细输出结果"""
    id = models.BigAutoField(primary_key=True)
    
    # 关联回上一步我们建立的任务需求表（通过 task_id 这个外键）
    task = models.ForeignKey(
        'PlanningTaskDemand', 
        on_delete=models.CASCADE, 
        db_column="task_id", 
        related_name="planning_results",
        verbose_name="关联的规划需求ID"
    )
    
    # === 基础回传信息 ===
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    src = models.CharField(max_length=64, verbose_name="源端点ID")
    dst = models.CharField(max_length=64, verbose_name="目标端点ID")
    demand_gbps = models.FloatField(verbose_name="分配的带宽(Gbps)")
    reason = models.CharField(max_length=255, null=True, blank=True, verbose_name="失败或降级等原因")
    
    # === KPI 与系统状态指标 ===
    avg_delay_ms = models.FloatField(verbose_name="加权平均时延(ms)")
    capacity_status = models.CharField(max_length=20, verbose_name="容量分配状态(OK/FAILED)")
    capacity_max_util_after = models.FloatField(verbose_name="分配后最大容量占用率")
    capacity_total_overflow = models.FloatField(verbose_name="总超载量")
    overflow_amount = models.FloatField(verbose_name="网络溢出量")
    jitter_ms = models.FloatField(verbose_name="抖动时延估计(ms)")
    max_link_utilization_after = models.FloatField(verbose_name="链路最大占用率")
    
    # KPI 2：任务规划决的推理决策耗时
    decision_total_ms = models.FloatField(verbose_name="推理决策总耗时(ms)")
    
    # === 核心分流结果 ===
    # 存储你的 allocations 数组，包含字典如: 
    # [{"bandwidth_gbps": 0.829, 'path': ['119', '0', '1'], 'ratio': 0.33, 'start_time':...}, ... ]
    allocations = models.JSONField(verbose_name="具体路径和分流方案数组")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['task']),
            models.Index(fields=['constellation_id']),
        ]

    def __str__(self):
        return f"PlanResult for: {self.task_id} (Status: {self.capacity_status})"
class SimRealtimeNodeState(models.Model):
    """【实时节点状态】记录每个微周期或触发更新时的卫星节点负载等属性"""
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    timestamp = models.DateTimeField(db_index=True, verbose_name="仿真更新时刻")
    
    sat_id = models.IntegerField(db_index=True, verbose_name="卫星ID")
    latitude = models.FloatField(verbose_name="纬度")
    longitude = models.FloatField(verbose_name="经度")
    
    flow = models.FloatField(verbose_name="当前总流量(Gbps)")
    energy_ratio = models.FloatField(verbose_name="电池电量(%)")
    congestion = models.FloatField(verbose_name="节点拥塞程度")
    heat_flow = models.FloatField(verbose_name="节点热力值")
    ground_station_number = models.IntegerField(null=True, blank=True, verbose_name="直连地面站数量")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['constellation_id', 'timestamp']),
            models.Index(fields=['sat_id', 'timestamp']),
        ]

class SimRealtimeLinkState(models.Model):
    """【实时链路状态】记录每个微周期或触发更新时的链路属性与流量变动"""
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    timestamp = models.DateTimeField(db_index=True, verbose_name="仿真更新时刻")
    
    src_sat = models.IntegerField(db_index=True, verbose_name="源卫星ID")
    dst_sat = models.IntegerField(db_index=True, verbose_name="目的卫星ID")
    
    link_distance = models.FloatField(verbose_name="链路距离(km)")
    current_flow = models.FloatField(verbose_name="当前流量(Gbps)")
    left_capacity = models.FloatField(verbose_name="剩余容量")
    queue_delay = models.FloatField(verbose_name="排队时延(ms)")
    heat_value = models.FloatField(verbose_name="链路热力值")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['constellation_id', 'timestamp']),
        ]

# satellites_sim_backend_project/simulation_api/models.py 新增一表
class TopologyPredictionAccuracy(models.Model):
    """【统计值】链路预测整体精度评估结果"""
    constellation_id = models.CharField(max_length=64, db_index=True, verbose_name="星座ID")
    
    # 评估的时间窗口
    window_start_time = models.DateTimeField(db_index=True, verbose_name="评估窗口起始时间")
    window_end_time = models.DateTimeField(verbose_name="评估窗口结束时间")
    
    # 精度指标
    accuracy = models.FloatField(verbose_name="总体预测准确率")
    precision = models.FloatField(null=True, blank=True, verbose_name="精确率")
    recall = models.FloatField(null=True, blank=True, verbose_name="召回率")
    
    # 采样规模
    total_evaluated_links = models.IntegerField(verbose_name="参与评估的总链路数")

    class Meta:
        db_table = "sim_prediction_accuracy"