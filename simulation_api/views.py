import threading
import uuid
import logging
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import generics


# 假设你的引擎可以这样导入 (这里仅给出伪代码示例结构)
from src.simulation.engine import SimulationEngine

# ====== [新增] 全局引擎注册表 ======
# 用于在同一个进程中，将 Django 的 HTTP 请求转交给正运行于后台线程的引擎实例
GLOBAL_ENGINES = {}

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json

logger = logging.getLogger(__name__)

# class SubmitPlanningTaskView(APIView):
#     """
#     任务提交接口 - 测试版本
#     """
#     def post(self, request, *args, **kwargs):
#         print("=" * 50)
#         print("收到POST请求")
#         print(f"请求数据: {request.data}")
        
#         try:
#             # 获取数据
#             constellation_id = request.data.get('constellation_id', '3600')
#             task_list = request.data.get('task_list', [])
            
#             print(f"constellation_id: {constellation_id}")
#             print(f"task_list长度: {len(task_list)}")
            
#             # 返回成功响应
#             return Response({
#                 "success": True,
#                 "message": f"成功接收 {len(task_list)} 个任务",
#                 "task_count": len(task_list),
#                 "simulation_id": "test_001",
#                 "constellation_id": constellation_id
#             }, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             print(f"错误: {str(e)}")
#             import traceback
#             traceback.print_exc()
            
#             return Response({
#                 "success": False,
#                 "error": f"处理失败: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SubmitPlanningTaskView(APIView):
    """
    POST /api/planning/task/submit
    前端提交任务规划需求 (批量)
    传入参数格式: {"constellation_id": "...", "task_list": [ {...}, ... ]}
    """
    def post(self, request, *args, **kwargs):
        # 1. 提取前端发送的参数
        constellation_id = request.data.get('constellation_id', '3600')
        task_list = request.data.get('task_list', [])
        
        if not task_list:
            logger.warning(
                "[SubmitPlanningTask] reject empty task_list | constellation_id=%s | payload_keys=%s",
                constellation_id,
                list(request.data.keys()) if hasattr(request.data, 'keys') else []
            )
            return Response(
                {
                    "error": "task_list 不能为空",
                    "error_code": "EMPTY_TASK_LIST"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # 根据 constellation_id 定位对应的全局引擎实例
            sim_id = f"sim_{constellation_id}"
            
            if sim_id not in GLOBAL_ENGINES:
                available_simulations = list(GLOBAL_ENGINES.keys())
                logger.warning(
                    "[SubmitPlanningTask] engine not started | requested=%s | available=%s",
                    sim_id,
                    available_simulations,
                )
                return Response({
                    "error": f"星座 {constellation_id} 的仿真进程未启动，请先开启对应的仿真开关",
                    "error_code": "SIM_ENGINE_NOT_STARTED",
                    "requested_simulation_id": sim_id,
                    "available_simulations": available_simulations,
                }, status=status.HTTP_400_BAD_REQUEST)
                

            engine = GLOBAL_ENGINES[sim_id]
                
            logger.info(
                ">>> [API] 收到前端规划请求，星座: %s，任务数量: %s",
                constellation_id,
                len(task_list)
            )
            logger.debug(
                ">>> [API] 前端下发的任务参数详情(最多3条预览): %s",
                task_list[:3]
            )
            # 字段映射（加在 views.py 准备 push 之前）
            formatted_task_list = []
            for t in task_list:
                duration_raw = t.get("duration", 0)
                try:
                    duration_seconds = float(duration_raw) / 1000.0
                except (TypeError, ValueError):
                    duration_seconds = 0.0

                formatted_task_list.append({
                    "TaskId": t.get("task_id", ""),
                    "SourceGroundStationId": t.get("src_sat_id", 0),  # 具体看你需要的是 Source 还是 SourceNode
                    "TargetGroundStationId": t.get("dest_sat_id", 0),
                    "DemandGbps": t.get("demand_gbps", 0.0),
                    "ArrivalTime": t.get("arrival_time", ""),
                    "Duration": duration_seconds,
                    "DelayBudget": t.get("delay_budget", 0),
                    "TaskPriority": t.get("task_priority", 0)
                })
            # 直接将前端标准的 task_list 推入该实例的事件循环处理队列中
            engine.receive_backend_task_planning({"TaskList": formatted_task_list})
            msg = f"成功将 {len(task_list)} 个任务推入 {constellation_id} 的仿真引擎处理队列"
            
            return Response({
                "message": msg,
                "task_count": len(task_list),
                "simulation_id": sim_id
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.exception("[SubmitPlanningTask] unexpected error: %s", e)
            return Response({"error": f"任务提交异常: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SimulationControlView(APIView):
    """
    POST /api/simulation/control/
    控制仿真引擎的启动与停止
    支持传入格式: {"constellation_id": "3600", "action": "start", "micro_period_seconds": 5}
    """
    def post(self, request, *args, **kwargs):
        constellation_id = str(request.data.get('constellation_id', ''))
        action = request.data.get('action', '')
        
        if not constellation_id or action not in ['start', 'stop']:
            return Response({"error": "缺少 constellation_id 或无效的 action"}, status=status.HTTP_400_BAD_REQUEST)
        
        sim_id = f"sim_{constellation_id}"
        
        try:
            if action == 'start':
                if sim_id in GLOBAL_ENGINES:
                    return Response({"message": f"星座 {constellation_id} 的仿真系统已在运行中"}, status=status.HTTP_200_OK)
                
                logger.info(">>> [API] 手动启动拉起仿真系统: %s", sim_id)
                import src.config as config

                micro_period_seconds = None
                for key in (
                    'micro_period_seconds',
                    'microPeriodSeconds',
                    'MicroPeriodSeconds',
                    'network_heat_value',
                    'networkHeatValue',
                ):
                    if key in request.data:
                        micro_period_seconds = request.data.get(key)
                        break

                if micro_period_seconds in (None, ''):
                    micro_period_seconds = 5
                else:
                    try:
                        micro_period_seconds = int(micro_period_seconds)
                    except (TypeError, ValueError):
                        return Response(
                            {"error": "micro_period_seconds 必须是 1、2、3、4、5 中的一个"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                    if micro_period_seconds not in (1, 2, 3, 4, 5):
                        return Response(
                            {"error": "micro_period_seconds 必须是 1、2、3、4、5 中的一个"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                
                # 动态应用前端传来的星座参数配置
                config.apply_constellation_config(constellation_id)
                
                # 建立新引擎实例化
                engine = SimulationEngine(constellation_id, 3600, micro_period_seconds=micro_period_seconds)
                GLOBAL_ENGINES[sim_id] = engine
                
                # 绑定 Websocket
                from asgiref.sync import async_to_sync
                from channels.layers import get_channel_layer
                channel_layer = get_channel_layer()
                group_name = f"sim_stream_{sim_id}"
                
                def ws_notify(msg_type, payload):
                    if channel_layer:
                        async_to_sync(channel_layer.group_send)(
                            group_name,
                            {
                                "type": "push.data", 
                                "event_type": msg_type,
                                "payload": payload
                            }
                        )
                engine.backend_callback = ws_notify
                
                # 在后台起新线程跑
                def run_engine_background(e, c_id):
                    try:
                        e.run_simulation(
                            timestamp=timezone.now(),
                            constellation_id=c_id,
                            tle_file_path=config.TLE_FILE_PATH,
                            task_list=[] 
                        )
                    except Exception as ex:
                        logger.exception(">>> [Engine] 运行异常退起: %s", ex)
                    finally:
                        if sim_id in GLOBAL_ENGINES and GLOBAL_ENGINES[sim_id] == e:
                            del GLOBAL_ENGINES[sim_id]
                            logger.info(">>> [Engine] 自动释放引擎资源: %s", sim_id)
                            
                thread = threading.Thread(target=run_engine_background, args=(engine, constellation_id))
                thread.daemon = True
                thread.start()
                
                return Response({"message": f"星座 {constellation_id} 仿真启动成功", "simulation_id": sim_id}, status=status.HTTP_200_OK)
                
            elif action == 'stop':
                if sim_id in GLOBAL_ENGINES:
                    engine = GLOBAL_ENGINES[sim_id]
                    # 如果你的 engine 类中有 stop() 或类似回收函数，在这里调用
                    if hasattr(engine, 'stop'):
                        engine.stop()
                    
                    del GLOBAL_ENGINES[sim_id]
                    logger.info(">>> [API] 手动终止仿真系统: %s", sim_id)
                    return Response({"message": f"星座 {constellation_id} 仿真已停止"}, status=status.HTTP_200_OK)
                else:
                    return Response({"message": "仿真未运行，无需停止"}, status=status.HTTP_200_OK)
                    
        except Exception as e:
            logger.exception("[SimulationControl] error: %s", e)
            return Response({"error": f"仿真控制异常: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


