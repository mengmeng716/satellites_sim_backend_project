from rest_framework import serializers
from .models import PlanningTask, TopologyLinkAttribute

class PlanningTaskSerializer(serializers.ModelSerializer):
    """
    任务规划提交序列化器
    自动校验源卫星、目标卫星id是否存在，转化前端时间格式
    """
    class Meta:
        model = PlanningTask
        # 注意此处的 src_sat 和 dst_sat 对应的是外键的 ID
        fields = ['src_sat', 'dst_sat', 'demand_gbps', 'arrival_time', 'delay_budget']

    def create(self, validated_data):
        import uuid
        # 自动生成 task_id，分配到一个挂起的仿真批次中 (逻辑根据实际情况可调整)
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # 插入额外的默认字段
        validated_data['task_id'] = task_id
        validated_data['status'] = 'PENDING'
        
        # 假设当前有一个正在运行的 simulation，业务中可能需要通过查询绑定当前 active 的 Constellation/Simulation
        # validated_data['constellation_id'] = '3600' 
        
        return super().create(validated_data)

class TopologyLinkAttributeSerializer(serializers.ModelSerializer):
    """
    拓扑结构链路历史数据序列化器
    """
    src_sat_id = serializers.CharField(source='relation.src_sat.id', read_only=True)
    dst_sat_id = serializers.CharField(source='relation.dst_sat.id', read_only=True)

    class Meta:
        model = TopologyLinkAttribute
        fields = [
            'src_sat_id', 'dst_sat_id', 
            'link_capacity', 'link_left_capacity', 
            'link_current_flow', 'link_packet_loss_rate',
            'link_propagation_delay', 'link_queue_delay', 'link_transmission_delay'
        ]