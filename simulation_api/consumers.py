import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class SimulationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self._is_connected = False
        # 1. 从 websocket的url路由参数中获取 sim_id
        self.simulation_id = self.scope['url_route']['kwargs'].get('sim_id', 'default_sim')
        # 2. 动态生成这个防真的专属组名
        self.group_name = f"sim_stream_{self.simulation_id}"
        
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        self._is_connected = True

    async def disconnect(self, close_code):
        self._is_connected = False
        # 断开连接时，将其从组里移除
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def push_data(self, event):
        if not getattr(self, '_is_connected', False):
            return

        # 接收到后端用 channel_layer 传过来的数据
        event_type = event.get('event_type', 'UNKNOWN_EVENT')
        payload = event.get('payload', {})
        
        # 统一打包成标准格式推给前端
        try:
            await self.send(text_data=json.dumps({
                "event": event_type,
                "data": payload
            }))
        except Exception as e:
            # 连接关闭与 group_send 并发时，可能出现 closed protocol 竞态。
            if 'closed protocol' in str(e).lower() or 'disconnected' in str(e).lower():
                logger.debug("Skip push_data after websocket closed: %s", e)
                return
            logger.warning("SimulationConsumer push_data failed: %s", e)

# 新增：KPI/大屏指标推送专用 Consumer
class KeyMetricsConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self._is_connected = False
        await self.accept()
        self._is_connected = True

    async def disconnect(self, close_code):
        self._is_connected = False

    async def receive(self, text_data=None, bytes_data=None):
        # 可选：处理前端发来的请求
        pass

    async def broadcast_message(self, event):
        if not getattr(self, '_is_connected', False):
            return

        # 后端 group_send 通过 type='broadcast_message' 推送 KPI 数据
        try:
            await self.send(text_data=json.dumps(event.get('message', {})))
        except Exception as e:
            if 'closed protocol' in str(e).lower() or 'disconnected' in str(e).lower():
                logger.debug("Skip broadcast_message after websocket closed: %s", e)
                return
            logger.warning("KeyMetricsConsumer broadcast_message failed: %s", e)