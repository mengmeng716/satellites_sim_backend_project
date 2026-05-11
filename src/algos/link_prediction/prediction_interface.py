import os
import time
import math
import torch
import numpy as np
from src.algos.link_prediction.src.scfe_model import SCFE_without_Future_Clean

class LinkPredictor:
    def __init__(self, constellation_id):
        self.constellation_id = str(constellation_id)
        self.device = self._select_device()

        # 动态获取当前文件所在目录，拒绝相对路径硬编码
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.constellation_id == "3600":
            weight_path = os.path.join(base_dir, "models-weights", "SCFE_clean_weights_3600.pth")
            self.sat_num = 3600
        elif self.constellation_id == "432":
            weight_path = os.path.join(base_dir, "models-weights", "SCFE_clean_weights_432.pth")
            self.sat_num = 432
        else:
            raise ValueError(f"Unsupported constellation_id: {constellation_id}")

        # 模型实例化（系统生命周期内只执行一次）
        self.model = SCFE_without_Future_Clean(
            input_len=30, input_dim=25, output_len=30, hidden_dim=512
        )

        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            print(f"[LinkPredictor] 成功加载模型权重: {weight_path} 到 {self.device}")
        else:
            print(f"[LinkPredictor] [警告] 找不到权重文件 {weight_path}")

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _select_device():
        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            major, minor = torch.cuda.get_device_capability()
            current_arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            if supported_arches and current_arch not in supported_arches:
                print(
                    f"[LinkPredictor] [Warning] CUDA device {current_arch} is not supported "
                    "by this PyTorch build, falling back to CPU."
                )
                return torch.device("cpu")
        except Exception as e:
            print(f"[LinkPredictor] [Warning] CUDA probe failed, falling back to CPU: {e}")
            return torch.device("cpu")

        return torch.device("cuda")

    def predict(self, current_topology, current_node_state, current_utc_time, input_window):
        batch_size = self.sat_num
        flattened_input = input_window.reshape(batch_size, 30, 25)
        xb = torch.tensor(flattened_input, dtype=torch.float32).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            out = self.model(xb)
            
        pred_step_1 = out[:, 0, :].cpu().numpy()
        end_time = time.time()   

        output_interface = {
            "Timestamp": int(current_utc_time.timestamp() * 1000),
            "ConstellationId": self.constellation_id,
            "Inference_time": end_time - start_time,
            "LinksPredTopology": {}
        }

        # 兼容 data_model 新格式（TopologyState / SatelliteNodeState）和旧字典格式
        if hasattr(current_topology, 'new_topology'):
            topo_dict = current_topology.new_topology
        elif isinstance(current_topology, dict):
            topo_dict = current_topology.get("newTopology", {})
        else:
            topo_dict = {}

        if hasattr(current_node_state, 'sat_list'):
            sat_list = current_node_state.sat_list
        elif isinstance(current_node_state, dict):
            sat_list = current_node_state.get("SatList", {})
        else:
            sat_list = {}

        for sat_idx in range(batch_size):
            sat_id = str(sat_idx)
            links_info = []
            pred_values = pred_step_1[sat_idx]
            
            # 兼容字典键和整数键
            neighbors = topo_dict.get(sat_idx, topo_dict.get(sat_id, {}))
            
            # 如果 neighbors 是 dict[str, LinksQualitiesValue]，转换为列表
            if isinstance(neighbors, dict):
                neighbors_list = list(neighbors.items())
            else:
                neighbors_list = list(neighbors) if neighbors else []

            for link_idx, (target_sat_id_raw, link_attr) in enumerate(neighbors_list):
                if link_idx >= 4:
                    break
                    
                target_sat_id = str(target_sat_id_raw)
                
                # 统一读取链路属性：兼容字典和 LinksQualitiesValue 对象
                def get_link_attr(attr, key_camel, key_snake, default):
                    if isinstance(attr, dict):
                        return attr.get(key_camel, attr.get(key_snake, default))
                    else:
                        return getattr(attr, key_snake, getattr(attr, key_camel, default))
                
                delay = get_link_attr(link_attr, "LinkPropagationDelay", "link_propagation_delay", 0.05)
                capacity = get_link_attr(link_attr, "LinkCapacity", "link_capacity", 1.0)
                dist_km = get_link_attr(link_attr, "LinkDistance", "link_distance", 500.0)

                # 读取卫星属性：兼容字典和 NodeAttribute 对象
                target_sat = sat_list.get(target_sat_id)
                if target_sat is not None:
                    if isinstance(target_sat, dict):
                        survival = target_sat.get("EnergyRatio", target_sat.get("energy_ratio", 1.0))
                    else:
                        survival = getattr(target_sat, 'energy_ratio', getattr(target_sat, 'EnergyRatio', 1.0))
                else:
                    survival = 1.0

                if link_idx < 4:
                    pred_flow = max(0.0, float(pred_values[link_idx * 2]))
                    pred_heat = max(0.0, float(pred_values[link_idx * 2 + 1]))
                else:
                    pred_flow, pred_heat = 0.0, 0.0

                signal_intensity = -30.0 - 50.0 * (math.log10(max(dist_km, 100)) - 2)
                signal_intensity = max(-80.0, min(-30.0, float(signal_intensity)))

                links_info.append([
                    target_sat_id,
                    {
                        "PredFlow": pred_flow, 
                        "Delay": float(delay), 
                        "Capacity": float(capacity),
                        "Survival": float(survival), 
                        "HeatValue": pred_heat, 
                        "LinkAvailability": 1.0 if capacity > 0 else 0.0
                    },
                    signal_intensity
                ])

            output_interface["LinksPredTopology"][sat_id] = links_info

        return output_interface
