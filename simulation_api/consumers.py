import json
from channels.generic.websocket import AsyncWebsocketConsumer

class SimulationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "sim_stream"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def push_data(self, event):
        # 接收引擎广播，直接推给前端
        await self.send(text_data=json.dumps(event['payload']))