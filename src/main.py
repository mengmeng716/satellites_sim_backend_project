import os
import random
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulation.engine import SimulationEngine
import src.config as config


class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def main():
    selected_constellation_id = "3600"  # can switch to "432"
    config.apply_constellation_config(selected_constellation_id)
    tle_file_path = config.TLE_FILE_PATH

    if not os.path.exists(tle_file_path):
        print(f"[Error] TLE file not found: {tle_file_path}")
        print("Please put the TLE file under the project data directory.")
        return

    constellation_id = config.CONSTELLATION_ID
    start_timestamp = datetime.now(timezone.utc)
    task_list = [
        {
            "TaskId": "Task_001",
            "TaskType": "Communication",
            "SourceGroundStationId": random.randint(10, 20),
            "TargetGroundStationId": random.randint(70, 80),
            "DemandGbps": 2.5,
            "Duration": 10,
            "TaskPriority": 5,
            "arrival_sim_time": start_timestamp.isoformat(),
        },
    ]

    simulation_duration = 300

    # 创建日志目录
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp_str = start_timestamp.strftime("%Y%m%d_%H%M%S")
    log_filename = f"simulation_{constellation_id}_{timestamp_str}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    print(f">>> 日志文件: {log_filepath}")
    print(f">>> Initializing simulation controller "
          f"(duration: {simulation_duration}s)...")

    # 重定向标准输出到日志文件（同时保留终端输出）
    tee_output = TeeOutput(log_filepath)
    old_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        simulation_engine = SimulationEngine(constellation_id, simulation_duration)

        simulation_engine.run_simulation(
            timestamp=start_timestamp,
            constellation_id=constellation_id,
            tle_file_path=tle_file_path,
            task_list=task_list,
        )
        # simulation_engine.run_extended_simulation()
    except Exception as e:
        print(f"\n[错误] 仿真过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复标准输出
        sys.stdout = old_stdout
        tee_output.close()
        print(f"\n>>> 仿真完成，日志已保存到: {log_filepath}")


if __name__ == "__main__":
    main()