from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # 捕获 url中的 sim_id 给 consumer
    re_path(r'^ws/simulation/(?P<sim_id>\w+)/$', consumers.SimulationConsumer.as_asgi()),
    # 兼容旧前端固定路径
    re_path(r'^ws/node_status/$', consumers.SimulationConsumer.as_asgi()),
    re_path(r'^ws/topology_status/$', consumers.SimulationConsumer.as_asgi()),
    re_path(r'^ws/route_optimization/$', consumers.SimulationConsumer.as_asgi()),
    re_path(r'^ws/topology_reconstruction/$', consumers.SimulationConsumer.as_asgi()),
    re_path(r'^ws/key_metrics/$', consumers.KeyMetricsConsumer.as_asgi()),
]
