import os
import sys
import time
import torch
import numpy as np
from datetime import timezone, timedelta

_THIS_FILE     = os.path.abspath(__file__)
_LINK_PRED_DIR = os.path.dirname(_THIS_FILE)
_ALGOS_DIR     = os.path.dirname(_LINK_PRED_DIR)
_SRC_DIR       = os.path.dirname(_ALGOS_DIR)
_PROJECT_ROOT  = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.algos.link_prediction.src.scfe_model import SCFE_without_Future_Clean
from src.algos.link_prediction.src.classifier_model import LinkActivityClassifier
import src.config as config


class LinkPredictor:
    """
    双阶段链路流量预测器 v3
    
    输出未来6个时间步的链路状态，格式符合 LinksPredTopology 接口规范。
    模型仍使用 output_len=30 训练的权重，取前6步输出。
    """

    FLOW_NORM_FACTOR        = 60.0
    LINK_AVAILABILITY_ALPHA = 0.5
    INPUT_STEPS             = 30
    OUTPUT_STEPS            = 6      # 对外输出的时间步数
    TIME_STEP_SECONDS       = float(getattr(config, "MICRO_PERIOD_SECONDS", 5))

    def __init__(self, constellation_id):
        self.constellation_id = str(constellation_id)
        self.device = self._select_device()
        self.input_len = self.INPUT_STEPS
        self.output_steps = self.OUTPUT_STEPS
        self.time_step_seconds = self.TIME_STEP_SECONDS
        self.inference_batch_size = max(
            1,
            int(getattr(config, "LINK_PREDICTION_BATCH_SIZE", 4096)),
        )

        if self.constellation_id == "3600":
            regressor_path  = os.path.join(_LINK_PRED_DIR, "models-weights", "SCFE_link_weights_3600.pth")
            classifier_path = os.path.join(_LINK_PRED_DIR, "models-weights", "classifier_weights.pth")
        elif self.constellation_id == "432":
            regressor_path  = os.path.join(_LINK_PRED_DIR, "models-weights", "SCFE_link_weights_432.pth")
            classifier_path = os.path.join(_LINK_PRED_DIR, "models-weights", "classifier_weights.pth")
        else:
            raise ValueError(f"Unsupported constellation_id: {constellation_id}")

        # Stage 1: 分类器
        self.classifier = LinkActivityClassifier(input_len=30, input_dim=11, hidden_dim=64)
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=self.device)
            )
            print(f"[LinkPredictor] 分类器加载成功: {classifier_path}")
            self._classifier_available = True
        else:
            print(f"[LinkPredictor] 警告: 分类器权重不存在, 降级到单阶段模式")
            self._classifier_available = False
        self.classifier.to(self.device)
        self.classifier.eval()

        # Stage 2: 回归器（保持 output_len=30，取前5步）
        self.model = SCFE_without_Future_Clean(
            input_len=30,
            input_dim=11,
            output_len=30,
            output_dim=1,
            hidden_dim=512,
        )
        if os.path.exists(regressor_path):
            self.model.load_state_dict(
                torch.load(regressor_path, map_location=self.device)
            )
            print(f"[LinkPredictor] 回归器加载成功: {regressor_path}")
        else:
            print(f"[LinkPredictor] 警告: 回归器权重不存在")
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[LinkPredictor] device={self.device}, "
            f"batch_size={self.inference_batch_size}"
        )

    @staticmethod
    def _select_device():
        if not torch.cuda.is_available():
            print("[LinkPredictor] 未检测到可用 GPU, 使用 CPU")
            return torch.device("cpu")
        try:
            major, minor = torch.cuda.get_device_capability()
            print(f"[LinkPredictor] 成功调用 CUDA sm_{major}{minor}: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        return torch.device("cuda")

    def _predict_classifier_probabilities(self, xb, batch_size):
        outputs = []
        with torch.inference_mode():
            for start in range(0, xb.shape[0], batch_size):
                stop = min(start + batch_size, xb.shape[0])
                outputs.append(
                    self.classifier.predict_proba(xb[start:stop]).detach().cpu()
                )
        return torch.cat(outputs, dim=0).numpy()

    def _predict_regression_batches(self, xb_active, batch_size):
        outputs = []
        with torch.inference_mode():
            for start in range(0, xb_active.shape[0], batch_size):
                stop = min(start + batch_size, xb_active.shape[0])
                out = self.model(xb_active[start:stop])
                outputs.append(out[:, :self.OUTPUT_STEPS, 0].detach().cpu())
        return torch.cat(outputs, dim=0).numpy()

    def predict(self, current_utc_time, input_window, link_index, step_seconds=None):
        """
        预测未来6个时间步的链路状态。

        Returns:
            output_interface: 符合接口规范的预测结果字典，包含6个时间步的拓扑
        """
        # 输入验证
        if input_window.ndim != 3 or input_window.shape[1] != 30 or input_window.shape[2] != 11:
            raise ValueError(f"输入维度错误: 期望 (link_num, 30, 11), 实际 {input_window.shape}")
        if input_window.dtype not in (np.float32, np.float64):
            raise ValueError(f"数据类型错误: 期望 float32, 实际 {input_window.dtype}")

        input_window = np.ascontiguousarray(input_window.astype(np.float32, copy=False))
        link_num = input_window.shape[0]

        if len(link_index) != link_num:
            raise ValueError(f"链路索引长度 {len(link_index)} 与窗口行数 {link_num} 不匹配")
        if link_num == 0:
            raise ValueError("链路索引为空")
        if current_utc_time.tzinfo is None or current_utc_time.tzinfo.utcoffset(current_utc_time) is None:
            raise ValueError("时刻必须为带时区的 aware datetime")

        max_cap = float(getattr(config, "MAX_LINK_SPEED_GBPS", 25.0))
        batch_size = self.inference_batch_size
        start_time = time.perf_counter()

        # 提取历史特征
        history_start = time.perf_counter()
        hist_last     = input_window[:, -1, 10] * max_cap
        hist_recent_3 = input_window[:, -3:, 10].mean(axis=1) * max_cap
        hist_recent_5 = input_window[:, -5:, 10].mean(axis=1) * max_cap
        hist_early_5  = input_window[:, -10:-5, 10].mean(axis=1) * max_cap
        hist_max      = input_window[:, :, 10].max(axis=1) * max_cap
        hist_std      = input_window[:, -5:, 10].std(axis=1) * max_cap
        decay_ratio   = hist_recent_5 / np.maximum(hist_early_5, 0.1)
        history_ms = (time.perf_counter() - history_start) * 1000.0

        tensor_start = time.perf_counter()
        xb = torch.from_numpy(input_window).to(self.device)
        tensor_ms = (time.perf_counter() - tensor_start) * 1000.0

        # ── Stage 1: 活跃度判定 ──────────────────────────────────────────────
        classifier_ms = 0.0
        if self._classifier_available:
            classifier_start = time.perf_counter()
            active_prob = self._predict_classifier_probabilities(xb, batch_size)
            classifier_ms = (time.perf_counter() - classifier_start) * 1000.0

            hist_stability = np.divide(
                hist_std,
                hist_recent_5,
                out=np.zeros_like(hist_std, dtype=np.float32),
                where=hist_recent_5 > 0.1,
            )

            active_mask = np.zeros(link_num, dtype=bool)

            stable_alive = (
                (hist_last > 0.8) &
                (hist_recent_5 > 0.6) &
                (active_prob > 0.90) &
                (decay_ratio > 0.3) &
                (hist_stability < 0.8)
            )
            active_mask |= stable_alive

            reconnect_mask = (hist_last <= 0.8) & (hist_max > 5.0) & (active_prob > 0.95)
            active_mask |= reconnect_mask

            decay_kill = (decay_ratio < 0.4) & (active_prob < 0.90)
            active_mask[decay_kill] = False

            high_decay_kill = (hist_last > 2.0) & (decay_ratio < 0.25) & (active_prob < 0.95)
            active_mask[high_decay_kill] = False

            kill_mask = (hist_recent_3 < 0.2) & (active_prob < 0.99)
            active_mask[kill_mask] = False

            high_volatility_kill = (hist_stability > 1.5) & (active_prob < 0.99)
            active_mask[high_volatility_kill] = False

            print(f"[LinkPredictor] Stage1 活跃={int(active_mask.sum())}/{link_num} "
                  f"(稳定={stable_alive.sum()} 恢复={reconnect_mask.sum()} "
                  f"衰减抑={decay_kill.sum()} 高值衰减抑={high_decay_kill.sum()} "
                  f"静默抑={kill_mask.sum()} 高波动抑={high_volatility_kill.sum()})")
        else:
            active_mask = hist_last > 1.0
            active_prob = np.zeros(link_num, dtype=np.float32)

        # ── Stage 2: 条件回归，输出6步 ──────────────────────────────────────
        # pred_flow shape: (link_num, OUTPUT_STEPS)
        pred_flow = np.zeros((link_num, self.OUTPUT_STEPS), dtype=np.float32)

        active_count = int(active_mask.sum())
        regression_ms = 0.0
        post_ms = 0.0
        if active_count > 0:
            xb_active = xb[active_mask]
            regression_start = time.perf_counter()
            pred_norm = self._predict_regression_batches(xb_active, batch_size)
            regression_ms = (time.perf_counter() - regression_start) * 1000.0

            # 取前6步，shape: (active_num, 6)
            post_start = time.perf_counter()
            if not np.all(np.isfinite(pred_norm)):
                pred_norm = np.where(np.isfinite(pred_norm), pred_norm, 0.0)

            raw_pred_flow = np.maximum(pred_norm * self.FLOW_NORM_FACTOR, 0.0)

            # 极值压制：不超过历史最大值的 1.2 倍
            hist_max_active = hist_max[active_mask][:, np.newaxis]  # (active_num, 1) 广播
            bounded_pred_flow = np.minimum(raw_pred_flow, hist_max_active * 1.2)

            # 有条件 T-1 平滑：仅对 T-1 高值链路应用
            HIST_ALPHA = 0.5
            hist_last_active = hist_last[active_mask][:, np.newaxis]  # (active_num, 1) 广播
            smooth_mask = hist_last_active > 2.0
            smoothed = np.where(
                smooth_mask,
                HIST_ALPHA * hist_last_active + (1.0 - HIST_ALPHA) * bounded_pred_flow,
                bounded_pred_flow
            )
            pred_flow[active_mask] = smoothed

            # 硬阈值清洗
            SILENCE_THRESHOLD = 2.0
            pred_flow[pred_flow < SILENCE_THRESHOLD] = 0.0

            # 后置概率验证
            active_prob_active = active_prob[active_mask][:, np.newaxis]
            hist_recent_5_active = hist_recent_5[active_mask][:, np.newaxis]
            mid_conf_suppress = (
                (pred_flow[active_mask] > 6.0) &
                (active_prob_active < 0.85) &
                (hist_recent_5_active < pred_flow[active_mask] * 0.3)
            )
            tmp = pred_flow[active_mask]
            tmp[mid_conf_suppress] = 0.0
            pred_flow[active_mask] = tmp

            suppressed = int(mid_conf_suppress.any(axis=1).sum())
            if suppressed > 0:
                print(f"[LinkPredictor] Stage2 后置验证抑制={suppressed}条")

            post_ms = (time.perf_counter() - post_start) * 1000.0

        pred_flow = np.clip(pred_flow, 0.0, max_cap)

        derive_start = time.perf_counter()

        # ── 派生字段，shape 均为 (link_num, OUTPUT_STEPS) ────────────────────
        pred_capacity     = np.clip(max_cap - pred_flow, 0.0, max_cap)
        pred_heat         = np.clip(pred_flow / max_cap, 0.0, 1.0)
        survival          = 1.0
        capacity_norm     = np.clip(pred_capacity / max_cap, 0.0, 1.0)
        pred_availability = np.clip(
            self.LINK_AVAILABILITY_ALPHA * capacity_norm +
            (1.0 - self.LINK_AVAILABILITY_ALPHA) * survival,
            0.0, 1.0
        )
        derive_ms = (time.perf_counter() - derive_start) * 1000.0
        format_start = time.perf_counter()

        # ── 生成每个时间步的时间戳 ────────────────────────────────────────────
        effective_step_seconds = float(
            self.TIME_STEP_SECONDS if step_seconds is None else step_seconds
        )
        step_timestamps = [
            (current_utc_time + timedelta(seconds=effective_step_seconds * step)).isoformat()
            for step in range(self.OUTPUT_STEPS)
        ]

        # ── 组装输出 ──────────────────────────────────────────────────────────
        output_interface = {
            "Timestamp"        : step_timestamps,
            "ConstellationId"  : self.constellation_id,
            "Inference_time"   : 0.0,
            "PredictionHorizon": self.OUTPUT_STEPS,
            "TimeStepSeconds"  : effective_step_seconds,
            "LinksPredTopology": [],
            "_debug_info": {
                "hist_last"   : hist_last,
                "hist_recent_5": hist_recent_5,
                "hist_std"    : hist_std,
                "decay_ratio" : decay_ratio,
                "active_prob" : active_prob if self._classifier_available else None,
                "timing_ms"   : {},
            }
        }

        # 每个时间步单独组装一个 Topology 快照
        for step in range(self.OUTPUT_STEPS):
            step_topology = {}

            for i, (src_sat_id, dst_sat_id) in enumerate(link_index):
                src_key = str(src_sat_id)
                dst_key = str(dst_sat_id)

                if src_key in step_topology and len(step_topology[src_key]) >= 4:
                    continue
                

                ##输出接口规范要求的字段，链路状态向量包含：剩余容量、热度、可用性、生存概率（目前固定为1.0）
                link_state_vector = {
                    "Capacity"        : float(pred_capacity[i, step]),
                    "Survival"        : survival,
                    "HeatValue"       : float(pred_heat[i, step]),
                    "LinkAvailability": float(pred_availability[i, step]),
                }

                if src_key not in step_topology:
                    step_topology[src_key] = []
                step_topology[src_key].append([dst_key, link_state_vector])

            output_interface["LinksPredTopology"].append({
                "Timestamp": step_timestamps[step],
                "Topology" : step_topology,
            })

        format_ms = (time.perf_counter() - format_start) * 1000.0
        total_time = time.perf_counter() - start_time
        timing_ms = {
            "history": history_ms,
            "tensor": tensor_ms,
            "classifier": classifier_ms,
            "regressor": regression_ms,
            "post": post_ms,
            "derive": derive_ms,
            "format": format_ms,
            "total": total_time * 1000.0,
        }
        output_interface["Inference_time"] = total_time
        output_interface["_debug_info"]["timing_ms"] = timing_ms

        if bool(getattr(config, "LINK_PREDICTION_TIMING_LOG", True)):
            print(
                f"[LinkPredictor] Timing device={self.device}, "
                f"links={link_num}, active={active_count}, batch={batch_size}, "
                f"history={history_ms:.2f}ms, tensor={tensor_ms:.2f}ms, "
                f"classifier={classifier_ms:.2f}ms, regressor={regression_ms:.2f}ms, "
                f"post={post_ms:.2f}ms, derive={derive_ms:.2f}ms, "
                f"format={format_ms:.2f}ms, total={total_time * 1000.0:.2f}ms"
            )

        return output_interface
