import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulation.engine import SimulationEngine
import src.config as config


def main():
    selected_constellation_id = '3600'  # 可切换为 '432'
    config.apply_constellation_config(selected_constellation_id)
    tle_file_path = config.TLE_FILE_PATH

    if not os.path.exists(tle_file_path):
        print(f"[严重错误] 找不到 TLE 文件，请检查路径: {tle_file_path}")
        print("请确保把文件放在了项目的 data/ 目录下！")
        return

    # 仿真参数初始化
    constellation_id = config.CONSTELLATION_ID
    start_timestamp = datetime(2026, 4, 24, 12, 0, 0)
    # 示例任务列表（用于测试任务规划功能）
    task_list = [
        {
            "TaskId": "Task_001",
            "TaskType": "Communication",
            "SourceSatId": 0,           # 源卫星ID
            "TargetSatId": 100,         # 目标卫星ID
            "DemandGbps": 2.5,          # 带宽需求 2.5 Gbps
            "Duration": 10,             # 持续10秒
            "TaskPriority": 5,          # 优先级5
            "arrival_sim_time": 0,  # 立即执行
        },
    ]


    print(f">>> 正在初始化仿真系统主控 (设置时长: {config.SIMULATION_DURATION}s)...")

    # 实例化引擎
    simulation_engine = SimulationEngine(constellation_id, config.SIMULATION_DURATION)

    # 启动引擎
    simulation_engine.run_simulation(
        timestamp=start_timestamp,#此处的时间类型需要讨论，============================================================================
        constellation_id=constellation_id,
        tle_file_path=tle_file_path,
        task_list=task_list
    )
    # # 接续运行24小时扩展仿真
    # simulation_engine.run_extended_simulation()


if __name__ == "__main__":
    main()
