import logging
import os
import sys
import time
import torch
import numpy as np
from datetime import datetime, timedelta, timezone
from itertools import chain
from math import ceil, isfinite
from threading import Thread

_THIS_FILE     = os.path.abspath(__file__)
_LINK_PRED_DIR = os.path.dirname(_THIS_FILE)
_ALGOS_DIR     = os.path.dirname(_LINK_PRED_DIR)
_SRC_DIR       = os.path.dirname(_ALGOS_DIR)
_PROJECT_ROOT  = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.algos.link_prediction.delay_jitter import estimate_delay_jitter
from src.algos.link_prediction.baseline import MovingAverageBaseline
from src.algos.link_prediction.src.scfe_model import SCFE_without_Future_Clean
# from src.algos.link_prediction.src.classifier_model import LinkActivityClassifier
import src.config as config

logger = logging.getLogger(__name__)


class LinkPredictor:
    """
    全网链路流量预测器
    
    输出未来6个时间步的链路状态，格式符合 LinksPredTopology 接口规范。
    模型仍使用 output_len=30 训练的权重，取前6步输出。
    """

    FLOW_NORM_FACTOR        = float(getattr(config, "MAX_LINK_SPEED_GBPS", 100.0))  # 与输入特征 flow/MAX_LINK_SPEED_GBPS 同一归一化
    LINK_AVAILABILITY_ALPHA = 0.5
    JITTER_WINDOW_SECONDS   = 15.0
    FLOW_TOLERANCE_GBPS     = {"432": 8.5, "3600": 18.0}
    JITTER_TOLERANCE_MS     = 0.001
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
            # classifier_path = os.path.join(_LINK_PRED_DIR, "models-weights", "classifier_weights.pth")
        elif self.constellation_id == "432":
            regressor_path  = os.path.join(_LINK_PRED_DIR, "models-weights", "SCFE_link_weights_432.pth")
            # classifier_path = os.path.join(_LINK_PRED_DIR, "models-weights", "classifier_weights.pth")
        else:
            raise ValueError(f"Unsupported constellation_id: {constellation_id}")

        # 旧版分类器已停用
        # self.classifier = LinkActivityClassifier(input_len=30, input_dim=11, hidden_dim=64)
        # if os.path.exists(classifier_path):
        #     self.classifier.load_state_dict(
        #         torch.load(classifier_path, map_location=self.device)
        #     )
        #     print(f"[LinkPredictor] 分类器加载成功: {classifier_path}")
        #     self._classifier_available = True
        # else:
        #     print(f"[LinkPredictor] 警告: 分类器权重不存在, 降级到单阶段模式")
        #     self._classifier_available = False
        # self.classifier.to(self.device)
        # self.classifier.eval()

        # SCFE 回归器使用原训练权重，输出取前 6 步。
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
        self.baseline = MovingAverageBaseline(
            input_len=self.INPUT_STEPS, input_dim=11, output_len=self.OUTPUT_STEPS, output_dim=1
        ).to(self.device).eval()
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

    # def _predict_classifier_probabilities(self, xb, batch_size):
    #     outputs = []
    #     with torch.inference_mode():
    #         for start in range(0, xb.shape[0], batch_size):
    #             stop = min(start + batch_size, xb.shape[0])
    #             outputs.append(
    #                 self.classifier.predict_proba(xb[start:stop]).detach().cpu()
    #             )
    #     return torch.cat(outputs, dim=0).numpy()

    def _predict_regression_batches(self, xb, batch_size, model=None):
        model = self.model if model is None else model
        outputs = []
        with torch.inference_mode():
            for start in range(0, xb.shape[0], batch_size):
                stop = min(start + batch_size, xb.shape[0])
                out = model(xb[start:stop])
                outputs.append(out[:, :self.OUTPUT_STEPS, 0].detach().cpu())
        return torch.cat(outputs, dim=0).numpy()

    @staticmethod
    def _flow_snapshot_time(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value)
        if value.utcoffset() is None:
            raise ValueError("链路流量时间戳必须带时区")
        # 与引擎落库使用的 Unix 毫秒精度一致。
        return datetime.fromtimestamp(int(value.timestamp() * 1000) / 1000, timezone.utc)

    @staticmethod
    def _link_distances_km(frames):
        coordinates = np.deg2rad(frames[..., [2, 3, 7, 8]].astype(np.float64))
        src_lat, src_lon, dst_lat, dst_lon = np.moveaxis(coordinates, -1, 0)
        haversine = (
            np.sin((dst_lat - src_lat) / 2.0) ** 2
            + np.cos(src_lat) * np.cos(dst_lat) * np.sin((dst_lon - src_lon) / 2.0) ** 2
        )
        radius_km = config.EARTH_RADIUS_KM + config.ORBIT_HEIGHT_KM
        return 2.0 * radius_km * np.sqrt(np.clip(haversine, 0.0, 1.0))

    def _jitter_rates_gbps(self, distances, links):
        """用最近一帧的几何与仿真配置估算各链路的共同参考速率。"""
        from src.simulation.data_model import LaserConfig
        from src.simulation.topology_link_state import (
            _laser_profile_by_type,
            evaluate_link_physics_batch,
            laser_type_for_link,
        )

        laser_types = []
        for src, dst in links:
            try:
                laser_types.append(laser_type_for_link(src, dst))
            except (TypeError, ValueError):
                laser_types.append(config.LASER_DEFAULT_TYPE)
        profiles = {
            name: LaserConfig.model_validate(_laser_profile_by_type(name))
            for name in set(laser_types)
        }
        physics = evaluate_link_physics_batch(
            distances, [profiles[name] for name in laser_types], np.zeros(len(links))
        )
        return physics["achievable_rate"]

    def _jitter_series_ms(
        self, links, rates, target_times, flows, history_times, history, step_seconds,
        *, target_propagation_ms=None, history_propagation_ms=None,
    ):
        target_ms = [round(self._flow_snapshot_time(t).timestamp() * 1000) for t in target_times]
        history_ms = [round(self._flow_snapshot_time(t).timestamp() * 1000) for t in history_times]
        if target_propagation_ms is None:
            target_propagation_ms = [(None,) * len(target_times)] * len(links)
        if history_propagation_ms is None:
            history_propagation_ms = [(None,) * len(history_times)] * len(links)
        history_records = (
            {"SrcSatID": src, "DstSatID": dst, "Timestamp": stamp, "AchievableRate": rate,
             "CurrentFlow": flow, "LinkPropagationDelay": delay}
            for (src, dst), rate, series, propagation in zip(links, rates, history, history_propagation_ms)
            for stamp, flow, delay in zip(history_ms, series, propagation)
        )
        prediction_records = (
            {"SrcSatID": src, "DstSatID": dst, "Timestamp": stamp, "AchievableRate": rate,
             "PredFlow": flow, "LinkPropagationDelay": delay}
            for (src, dst), rate, series, propagation in zip(links, rates, flows, target_propagation_ms)
            for stamp, flow, delay in zip(target_ms, series, propagation)
        )
        estimates = estimate_delay_jitter(
            prediction_records, step_seconds,
            history_records=history_records, window_seconds=self.JITTER_WINDOW_SECONDS,
        )
        return np.asarray([item["DelayJitter"] for item in estimates], dtype=np.float64).reshape(
            len(links), len(target_times)
        )

    @staticmethod
    def _network_wape(baseline, scfe, actual, mae_unit):
        paired = np.isfinite(baseline) & np.isfinite(scfe) & np.isfinite(actual)
        observed = actual[paired]
        samples = len(observed)
        total_actual = float(observed.sum(dtype=np.float64))
        baseline_error = float(np.abs(baseline[paired] - observed).sum(dtype=np.float64))
        scfe_error = float(np.abs(scfe[paired] - observed).sum(dtype=np.float64))
        return {
            "status": "scored" if total_actual > 0.0 else "zero_actual" if samples else "no_matches",
            "expected_samples": actual.size,
            "matched_samples": samples,
            f"baseline_mae_{mae_unit}": baseline_error / samples if samples else None,
            f"scfe_mae_{mae_unit}": scfe_error / samples if samples else None,
            "baseline_wape": baseline_error / total_actual if total_actual > 0.0 else None,
            "scfe_wape": scfe_error / total_actual if total_actual > 0.0 else None,
            "improvement_pct": (
                (baseline_error - scfe_error) / baseline_error * 100.0
                if total_actual > 0.0 and baseline_error > 0.0 else None
            ),
        }

    @staticmethod
    def _tolerance_evaluation(baseline, scfe, actual, tolerance, unit):
        paired = np.isfinite(baseline) & np.isfinite(scfe) & np.isfinite(actual)
        observed = actual[paired]
        matched = observed.size
        baseline_misses = int(np.count_nonzero(np.abs(baseline[paired] - observed) > tolerance))
        scfe_misses = int(np.count_nonzero(np.abs(scfe[paired] - observed) > tolerance))
        if not actual.size:
            status = "no_loaded_links"
        elif not matched:
            status = "no_matches"
        elif not baseline_misses:
            status = "baseline_no_misses"
        else:
            status = "scored"
        return {
            "status": status,
            "evaluated_step": 1,
            "tolerance": tolerance,
            "unit": unit,
            "expected_samples": actual.size,
            "matched_samples": matched,
            "baseline_accuracy": (matched - baseline_misses) / matched if matched else None,
            "scfe_accuracy": (matched - scfe_misses) / matched if matched else None,
            "baseline_misses": baseline_misses,
            "scfe_misses": scfe_misses,
            "improvement_pct": (
                100.0 * (baseline_misses - scfe_misses) / baseline_misses if baseline_misses else None
            ),
            "meets_30_pct_target": 10 * scfe_misses <= 7 * baseline_misses if baseline_misses else None,
        }

    @staticmethod
    def _pending_tolerance(tolerance, unit, eligible_count):
        return {
            "status": "pending" if eligible_count else "no_loaded_links",
            "evaluated_step": 1,
            "tolerance": tolerance,
            "unit": unit,
            "expected_samples": eligible_count,
            "matched_samples": 0,
            "baseline_accuracy": None,
            "scfe_accuracy": None,
            "baseline_misses": None,
            "scfe_misses": None,
            "improvement_pct": None,
            "meets_30_pct_target": None,
        }

    @staticmethod
    def _current_flow_values(snapshots):
        return snapshots.values_list(
            "timestamp", "src_sat", "dst_sat", "current_flow", "link_distance"
        ).iterator(
            chunk_size=5000
        )

    @staticmethod
    def _evaluation_output(prediction, flow_scores, jitter_scores):
        scores_by_field = {"FlowEvaluation": flow_scores, "JitterEvaluation": jitter_scores}
        for field, scores in scores_by_field.items():
            tolerance = scores.get("tolerance_evaluation")
            status = scores["status"]
            if status in {"invalid_timestamps", "database_unavailable"} and tolerance and tolerance["status"] == "pending":
                scores_by_field[field] = dict(scores, tolerance_evaluation=dict(tolerance, status=status))
        debug = dict(
            prediction["_debug_info"], flow_evaluation=scores_by_field["FlowEvaluation"],
            jitter_evaluation=scores_by_field["JitterEvaluation"],
        )
        return dict(prediction, **scores_by_field, _debug_info=debug)

    def evaluate_prediction(self, prediction, current_flow_rows=None):
        """评估全网六步流量 WAPE，以及带载链路的首步流量和抖动容差。

        current_flow_rows 可提供 (时刻, 源卫星, 目标卫星, Gbps, 星距 km)。
        旧的四列记录使用本轮预测的传播时延轨迹作为共同参照。
        """
        return self._evaluate_prediction(prediction, current_flow_rows, score_flow=True, score_jitter=True)

    def _evaluate_prediction(self, prediction, current_flow_rows, *, score_flow, score_jitter):
        debug = prediction["_debug_info"]
        forecasts = debug["flow_forecasts_gbps"]
        jitter = debug.get("jitter_forecasts") if score_jitter else None
        target_times = [self._flow_snapshot_time(value) for value in prediction["Timestamp"]]
        if not score_flow:
            target_times = target_times[:1]
        history_times = [] if jitter is None else [
            self._flow_snapshot_time(value) for value in jitter["history_timestamps"]
        ]
        step_indices = {stamp: i for i, stamp in enumerate(target_times + history_times)}
        if len(step_indices) != len(target_times) + len(history_times):
            raise ValueError("毫秒精度下的预测与历史时间戳必须互异")

        link_rows = {}
        for index, (src, dst) in enumerate(forecasts["link_index"]):
            link_rows.setdefault((src, dst), index)
        horizon = len(target_times)

        if current_flow_rows is None:
            from simulation_api.models import SimRealtimeLinkState

            snapshots = SimRealtimeLinkState.objects.filter(
                constellation_id=prediction["ConstellationId"], timestamp__in=list(step_indices)
            )
            current_flow_rows = self._current_flow_values(snapshots)

        actual = np.full((len(forecasts["link_index"]), len(step_indices)), np.nan)
        has_propagation = jitter is not None and "target_propagation_ms" in jitter
        actual_distances = np.full_like(actual, np.nan) if has_propagation else None
        observed_distances_provided = False
        conflicts = set()
        distance_conflicts = set()
        for timestamp, src, dst, flow, *distance_column in current_flow_rows:
            step = step_indices.get(self._flow_snapshot_time(timestamp))
            row = link_rows.get((str(src), str(dst)))
            if step is None or row is None or flow is None:
                continue
            flow = float(flow)
            if not isfinite(flow) or flow < 0.0:
                continue
            previous = actual[row, step]
            if not isfinite(previous):
                actual[row, step] = flow
            elif previous != flow:
                conflicts.add((row, step))
            if actual_distances is not None and distance_column:
                observed_distances_provided = True
                distance = distance_column[0]
                if distance is None:
                    continue
                distance = float(distance)
                if not isfinite(distance) or distance <= 0.0:
                    continue
                previous_distance = actual_distances[row, step]
                if not isfinite(previous_distance):
                    actual_distances[row, step] = distance
                elif previous_distance != distance:
                    distance_conflicts.add((row, step))

        rows = list(link_rows.values())
        observed = actual[rows, :horizon]
        missing = observed.size - int(np.count_nonzero(np.isfinite(observed)))
        for row, step in conflicts:
            actual[row, step] = np.nan

        flow_scores = debug.get("flow_evaluation", {"status": "pending"})
        jitter_scores = debug.get("jitter_evaluation", {"status": "pending"})
        last_flow = np.asarray(debug["hist_last"])
        loaded = np.isfinite(last_flow) & (last_flow > 0.0)
        eligible_links = [link for link, row in link_rows.items() if loaded[row]]
        eligible_rows = [link_rows[link] for link in eligible_links]
        if jitter is not None:
            reference_target_delay = None
            reference_history_delay = None
            propagation_source = "queue_only"
            if observed_distances_provided:
                for row, step in distance_conflicts:
                    actual_distances[row, step] = np.nan
                observed_delay = actual_distances[eligible_rows] * (1000.0 / config.SPEED_OF_LIGHT_KM_S)
                reference_target_delay = observed_delay[:, :1]
                reference_history_delay = observed_delay[:, horizon:]
                propagation_source = "observed_link_distance"
            elif has_propagation:
                reference_target_delay = np.asarray(jitter["target_propagation_ms"])[eligible_rows, :1]
                reference_history_delay = np.asarray(jitter["history_propagation_ms"])[eligible_rows]
                propagation_source = "forecast_geometry"

            reference = self._jitter_series_ms(
                eligible_links, np.asarray(jitter["rates_gbps"])[eligible_rows], target_times[:1],
                actual[eligible_rows, :1], history_times, actual[eligible_rows, horizon:],
                prediction["TimeStepSeconds"], target_propagation_ms=reference_target_delay,
                history_propagation_ms=reference_history_delay,
            )
            baseline_jitter = np.asarray(jitter["baseline_ms"])[eligible_rows, :1]
            scfe_jitter = np.asarray(jitter["scfe_ms"])[eligible_rows, :1]
            jitter_scores = self._network_wape(baseline_jitter, scfe_jitter, reference, "ms")
            jitter_scores["tolerance_evaluation"] = self._tolerance_evaluation(
                baseline_jitter, scfe_jitter, reference, self.JITTER_TOLERANCE_MS, "ms"
            )
            if not eligible_links:
                jitter_scores["status"] = "no_loaded_links"
            improvement = jitter_scores["improvement_pct"]
            jitter_scores.update(
                evaluated_step=1,
                target_timestamp=prediction["Timestamp"][0],
                eligible_links=len(eligible_links),
                propagation_source=propagation_source,
                meets_30_pct_target=None if improvement is None else improvement >= 30.0,
            )

        baseline_flow = np.asarray(forecasts["baseline"])
        scfe_flow = np.asarray(forecasts["scfe"])
        if score_flow:
            flow_scores = self._network_wape(baseline_flow[rows], scfe_flow[rows], actual[rows, :horizon], "gbps")
            flow_scores["missing_samples"] = missing
            flow_scores["conflicting_samples"] = sum(step < horizon for _, step in conflicts)
        else:
            flow_scores = dict(flow_scores)

        flow_tolerance = self._tolerance_evaluation(
            baseline_flow[eligible_rows, 0], scfe_flow[eligible_rows, 0], actual[eligible_rows, 0],
            self.FLOW_TOLERANCE_GBPS[prediction["ConstellationId"]], "Gbps",
        )
        flow_scores["tolerance_evaluation"] = flow_tolerance
        fields = {
            "BaselinePrecision": flow_tolerance["baseline_accuracy"] or 0.0,
            "SCFEPrecision": flow_tolerance["scfe_accuracy"] or 0.0,
        }
        evaluated = self._evaluation_output(prediction, flow_scores, jitter_scores)
        steps = prediction["LinksPredTopology"]
        if not steps or all(steps[0].get(key) == value for key, value in fields.items()):
            return evaluated
        evaluated["LinksPredTopology"] = [
            dict(
                step,
                Topology={
                    src: [[dst, dict(state, **fields)] for dst, state in links]
                    for src, links in step["Topology"].items()
                },
                **fields,
            )
            for step in steps
        ]
        return evaluated

    def _schedule_flow_evaluation(self, prediction, issued_at):
        try:
            from django.apps import apps
        except ModuleNotFoundError:
            return
        if apps.ready:
            Thread(
                target=self._await_current_flows,
                args=(prediction, issued_at),
                name="link-flow-evaluation",
                daemon=True,
            ).start()

    def _await_current_flows(self, prediction, issued_at):
        from django.db import DatabaseError, connections
        from simulation_api.models import SimRealtimeLinkState

        target_times = [self._flow_snapshot_time(value) for value in prediction["Timestamp"]]
        flow_scores = prediction["FlowEvaluation"]
        jitter_scores = prediction["JitterEvaluation"]
        if len(set(target_times)) != len(target_times):
            prediction.update(self._evaluation_output(
                prediction, dict(flow_scores, status="invalid_timestamps"),
                dict(jitter_scores, status="invalid_timestamps"),
            ))
            return

        snapshots = SimRealtimeLinkState.objects.filter(
            constellation_id=prediction["ConstellationId"],
            timestamp__in=target_times,
            created_at__gte=issued_at,
        )
        first_snapshot = snapshots.filter(timestamp=target_times[0])
        history_times = [
            self._flow_snapshot_time(value)
            for value in prediction["_debug_info"]["jitter_forecasts"]["history_timestamps"]
        ]
        history = SimRealtimeLinkState.objects.filter(
            constellation_id=prediction["ConstellationId"], timestamp__in=history_times
        )
        step_seconds = prediction["TimeStepSeconds"]
        poll_interval = max(0.25, min(step_seconds, 5.0))
        deadline = time.monotonic() + step_seconds * len(target_times) + 30.0

        try:
            while flow_scores["status"] == "pending" or jitter_scores["status"] == "pending":
                expired = time.monotonic() >= deadline
                try:
                    if jitter_scores["status"] == "pending":
                        # 等下一微周期有记录，降低把同一时刻的重构快照当作首步真值的风险。
                        next_time = target_times[min(1, len(target_times) - 1)]
                        if expired or (
                            snapshots.filter(timestamp=next_time).exists()
                            and first_snapshot.exists()
                            and history.values_list("timestamp", flat=True).distinct().count() == len(history_times)
                        ):
                            rows = chain(self._current_flow_values(first_snapshot), self._current_flow_values(history))
                            prediction.update(self._evaluate_prediction(
                                prediction, rows, score_flow=False, score_jitter=True
                            ))
                            flow_scores = prediction["FlowEvaluation"]
                            jitter_scores = prediction["JitterEvaluation"]
                            tolerance_scores = jitter_scores["tolerance_evaluation"]
                            logger.info(
                                "首步抖动评估 %s %s：%s，配对 %s/%s，Baseline WAPE=%s，SCFE WAPE=%s，"
                                "Baseline 超差=%s，SCFE 超差=%s，超差降幅=%s%%，达到30%%=%s",
                                prediction["ConstellationId"], jitter_scores["target_timestamp"],
                                tolerance_scores["status"], tolerance_scores["matched_samples"],
                                tolerance_scores["expected_samples"], jitter_scores["baseline_wape"],
                                jitter_scores["scfe_wape"], tolerance_scores["baseline_misses"],
                                tolerance_scores["scfe_misses"], tolerance_scores["improvement_pct"],
                                tolerance_scores["meets_30_pct_target"],
                            )
                    if flow_scores["status"] == "pending" and (
                        expired or (
                            snapshots.filter(timestamp=target_times[-1]).exists()
                            and snapshots.values_list("timestamp", flat=True).distinct().count() == len(target_times)
                        )
                    ):
                        prediction.update(self._evaluate_prediction(
                            prediction, self._current_flow_values(snapshots), score_flow=True, score_jitter=False
                        ))
                        flow_scores = prediction["FlowEvaluation"]
                        tolerance_scores = flow_scores["tolerance_evaluation"]
                        logger.info(
                            "全网流量评估 %s %s：%s，配对 %s/%s，Baseline WAPE=%s，SCFE WAPE=%s，"
                            "首步 Baseline 超差=%s，SCFE 超差=%s，超差降幅=%s%%，达到30%%=%s",
                            prediction["ConstellationId"], prediction["Timestamp"][0], flow_scores["status"],
                            flow_scores["matched_samples"], flow_scores["expected_samples"],
                            flow_scores["baseline_wape"], flow_scores["scfe_wape"], tolerance_scores["baseline_misses"],
                            tolerance_scores["scfe_misses"], tolerance_scores["improvement_pct"],
                            tolerance_scores["meets_30_pct_target"],
                        )
                except DatabaseError:
                    connections.close_all()
                    if expired:
                        if flow_scores["status"] == "pending":
                            flow_scores = dict(flow_scores, status="database_unavailable")
                        if jitter_scores["status"] == "pending":
                            jitter_scores = dict(jitter_scores, status="database_unavailable")
                        prediction.update(self._evaluation_output(prediction, flow_scores, jitter_scores))
                        logger.warning("链路评估未能读取 CurrentFlow：%s", prediction["Timestamp"][0])
                        return
                if flow_scores["status"] == "pending" or jitter_scores["status"] == "pending":
                    time.sleep(poll_interval)
        finally:
            connections.close_all()

    def _simple_step1_metrics(self, baseline, scfe, actual, tolerance, unit):
        baseline = np.asarray(baseline, dtype=np.float64)
        scfe = np.asarray(scfe, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)

        paired = np.isfinite(baseline) & np.isfinite(scfe) & np.isfinite(actual)
        if not np.any(paired):
            return {
                "status": "no_matches",
                "matched_samples": 0,
                "expected_samples": int(actual.size),
                "tolerance": float(tolerance),
                "unit": unit,
                "baseline_mae": None,
                "scfe_mae": None,
                "baseline_wape": None,
                "scfe_wape": None,
                "BaselinePrecision": 0.0,
                "SCFEPrecision": 0.0,
                "improvement_pct": None,
            }

        b = baseline[paired]
        s = scfe[paired]
        y = actual[paired]

        baseline_err = np.abs(b - y)
        scfe_err = np.abs(s - y)
        denom = float(np.sum(np.abs(y)))

        baseline_mae = float(np.mean(baseline_err))
        scfe_mae = float(np.mean(scfe_err))
        baseline_wape = float(np.sum(baseline_err) / denom) if denom > 0.0 else None
        scfe_wape = float(np.sum(scfe_err) / denom) if denom > 0.0 else None

        baseline_precision = float(np.mean(baseline_err <= tolerance))
        scfe_precision = float(np.mean(scfe_err <= tolerance))
        improvement_pct = None
        if baseline_wape is not None and scfe_wape is not None and baseline_wape > 0.0:
            improvement_pct = float((baseline_wape - scfe_wape) / baseline_wape * 100.0)

        return {
            "status": "scored",
            "matched_samples": int(y.size),
            "expected_samples": int(actual.size),
            "tolerance": float(tolerance),
            "unit": unit,
            "baseline_mae": baseline_mae,
            "scfe_mae": scfe_mae,
            "baseline_wape": baseline_wape,
            "scfe_wape": scfe_wape,
            "BaselinePrecision": baseline_precision,
            "SCFEPrecision": scfe_precision,
            "improvement_pct": improvement_pct,
        }

    def evaluate_prediction_simple(self, prediction, current_flow_rows):
        """同步单步评估：由调用方直接传入首步真值，返回 SCFE 与 Baseline 对比结果。"""
        if not isinstance(prediction, dict):
            return prediction
        debug = prediction.get("_debug_info", {})
        forecasts = debug.get("flow_forecasts_gbps") or {}
        link_index = forecasts.get("link_index") or []
        baseline = np.asarray(forecasts.get("baseline"), dtype=np.float64)
        scfe = np.asarray(forecasts.get("scfe"), dtype=np.float64)

        if len(link_index) == 0 or baseline.ndim != 2 or scfe.ndim != 2:
            prediction["FlowEvaluation"] = {
                "status": "invalid_prediction_payload",
                "matched_samples": 0,
                "expected_samples": 0,
                "BaselinePrecision": 0.0,
                "SCFEPrecision": 0.0,
            }
            logger.warning(
                "[LinkPredictor] simple-eval skipped: invalid payload "
                "(links=%s, baseline_ndim=%s, scfe_ndim=%s)",
                len(link_index),
                getattr(baseline, "ndim", None),
                getattr(scfe, "ndim", None),
            )
            return prediction

        target_timestamp = self._flow_snapshot_time(prediction["Timestamp"][0])
        truth_map = {}
        for row in current_flow_rows or []:
            if len(row) < 4:
                continue
            ts, src, dst, flow = row[0], row[1], row[2], row[3]
            if flow is None:
                continue
            ts_dt = self._flow_snapshot_time(ts)
            if ts_dt != target_timestamp:
                continue
            try:
                flow_val = float(flow)
            except (TypeError, ValueError):
                continue
            if not isfinite(flow_val):
                continue
            truth_map[(str(src), str(dst))] = max(0.0, flow_val)

        logger.info(
            "[LinkPredictor] simple-eval prepare: ts=%s, link_index=%d, current_flow_rows=%d, matched_truth_links=%d",
            prediction["Timestamp"][0],
            len(link_index),
            len(current_flow_rows or []),
            len(truth_map),
        )

        actual = np.full((len(link_index),), np.nan, dtype=np.float64)
        for i, key in enumerate(link_index):
            if key in truth_map:
                actual[i] = truth_map[key]

        tolerance = self.FLOW_TOLERANCE_GBPS[prediction["ConstellationId"]]
        metrics = self._simple_step1_metrics(
            baseline=baseline[:, 0],
            scfe=scfe[:, 0],
            actual=actual,
            tolerance=tolerance,
            unit="Gbps",
        )
        metrics.update({
            "evaluated_step": 1,
            "target_timestamp": prediction["Timestamp"][0],
        })

        logger.info(
            "[LinkPredictor] simple-eval result: ts=%s, status=%s, matched=%s/%s, "
            "baseline_precision=%.6f, scfe_precision=%.6f, baseline_wape=%s, scfe_wape=%s",
            prediction["Timestamp"][0],
            metrics.get("status"),
            metrics.get("matched_samples"),
            metrics.get("expected_samples"),
            float(metrics.get("BaselinePrecision") or 0.0),
            float(metrics.get("SCFEPrecision") or 0.0),
            metrics.get("baseline_wape"),
            metrics.get("scfe_wape"),
        )

        prediction["FlowEvaluation"] = metrics
        prediction["JitterEvaluation"] = {
            "status": "skipped_in_simple_mode",
            "reason": "simple mode only evaluates step-1 flow precision",
        }

        fields = {
            "BaselinePrecision": float(metrics.get("BaselinePrecision") or 0.0),
            "SCFEPrecision": float(metrics.get("SCFEPrecision") or 0.0),
        }
        updated_steps = []
        for step in prediction.get("LinksPredTopology", []) or []:
            updated_steps.append(
                dict(
                    step,
                    Topology={
                        src: [[dst, dict(state, **fields)] for dst, state in links]
                        for src, links in (step.get("Topology") or {}).items()
                    },
                    **fields,
                )
            )
        prediction["LinksPredTopology"] = updated_steps
        debug["flow_evaluation"] = metrics
        prediction["_debug_info"] = debug
        return prediction

    def predict(self, current_utc_time, input_window, link_index, step_seconds=None, current_flow_rows=None):
        """
        预测未来6个时间步的链路状态。

        Returns:
            output_interface: 符合接口规范的预测结果字典，包含6个时间步的拓扑
        """
        effective_step_seconds = float(
            self.TIME_STEP_SECONDS if step_seconds is None else step_seconds
        )
        if not isfinite(effective_step_seconds) or effective_step_seconds < 0.001:
            raise ValueError("step_seconds 必须是至少 0.001 秒的有限值")

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

        max_cap = float(config.MAX_LINK_SPEED_GBPS)
        batch_size = self.inference_batch_size
        issued_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()

        # 提取历史特征
        history_start = time.perf_counter()
        hist_last     = input_window[:, -1, 10] * max_cap
        # hist_recent_3 = input_window[:, -3:, 10].mean(axis=1) * max_cap
        hist_recent_5 = input_window[:, -5:, 10].mean(axis=1) * max_cap
        hist_early_5  = input_window[:, -10:-5, 10].mean(axis=1) * max_cap
        hist_max      = input_window[:, :, 10].max(axis=1) * max_cap
        hist_std      = input_window[:, -5:, 10].std(axis=1) * max_cap
        decay_ratio   = hist_recent_5 / np.maximum(hist_early_5, 0.1)
        history_ms = (time.perf_counter() - history_start) * 1000.0

        tensor_start = time.perf_counter()
        xb = torch.from_numpy(input_window).to(self.device)
        tensor_ms = (time.perf_counter() - tensor_start) * 1000.0

        # 旧版活跃度筛选已停用，两个模型均对全部输入链路推理。
        # # ── Stage 1: 活跃度判定 ──────────────────────────────────────────────
        # classifier_ms = 0.0
        # if self._classifier_available:
        #     classifier_start = time.perf_counter()
        #     active_prob = self._predict_classifier_probabilities(xb, batch_size)
        #     classifier_ms = (time.perf_counter() - classifier_start) * 1000.0

        #     hist_stability = np.divide(
        #         hist_std,
        #         hist_recent_5,
        #         out=np.zeros_like(hist_std, dtype=np.float32),
        #         where=hist_recent_5 > 0.1,
        #     )

        #     active_mask = np.zeros(link_num, dtype=bool)

        #     stable_alive = (
        #         (hist_last > 0.8) &
        #         (hist_recent_5 > 0.6) &
        #         (active_prob > 0.90) &
        #         (decay_ratio > 0.3) &
        #         (hist_stability < 0.8)
        #     )
        #     active_mask |= stable_alive

        #     reconnect_mask = (hist_last <= 0.8) & (hist_max > 5.0) & (active_prob > 0.95)
        #     active_mask |= reconnect_mask

        #     decay_kill = (decay_ratio < 0.4) & (active_prob < 0.90)
        #     active_mask[decay_kill] = False

        #     high_decay_kill = (hist_last > 2.0) & (decay_ratio < 0.25) & (active_prob < 0.95)
        #     active_mask[high_decay_kill] = False

        #     kill_mask = (hist_recent_3 < 0.2) & (active_prob < 0.99)
        #     active_mask[kill_mask] = False

        #     high_volatility_kill = (hist_stability > 1.5) & (active_prob < 0.99)
        #     active_mask[high_volatility_kill] = False

        #     print(f"[LinkPredictor] Stage1 活跃={int(active_mask.sum())}/{link_num} "
        #           f"(稳定={stable_alive.sum()} 恢复={reconnect_mask.sum()} "
        #           f"衰减抑={decay_kill.sum()} 高值衰减抑={high_decay_kill.sum()} "
        #           f"静默抑={kill_mask.sum()} 高波动抑={high_volatility_kill.sum()})")
        # else:
        #     active_mask = hist_last > 1.0
        #     active_prob = np.zeros(link_num, dtype=np.float32)

        regression_start = time.perf_counter()
        pred_norm = self._predict_regression_batches(xb, batch_size)
        regression_ms = (time.perf_counter() - regression_start) * 1000.0

        post_start = time.perf_counter()
        if not np.all(np.isfinite(pred_norm)):
            pred_norm = np.where(np.isfinite(pred_norm), pred_norm, 0.0)

        raw_pred_flow = np.maximum(pred_norm * self.FLOW_NORM_FACTOR, 0.0)

        # 极值压制：不超过历史最大值的 1.2 倍
        bounded_pred_flow = np.minimum(raw_pred_flow, hist_max[:, np.newaxis] * 1.2)

        # 有条件 T-1 平滑：仅对 T-1 高值链路应用
        HIST_ALPHA = 0.5
        hist_last_column = hist_last[:, np.newaxis]
        pred_flow = np.where(
            hist_last_column > 2.0,
            HIST_ALPHA * hist_last_column + (1.0 - HIST_ALPHA) * bounded_pred_flow,
            bounded_pred_flow,
        )

        # # 后置概率验证
        # active_prob_active = active_prob[active_mask][:, np.newaxis]
        # hist_recent_5_active = hist_recent_5[active_mask][:, np.newaxis]
        # mid_conf_suppress = (
        #     (pred_flow[active_mask] > 6.0) &
        #     (active_prob_active < 0.85) &
        #     (hist_recent_5_active < pred_flow[active_mask] * 0.3)
        # )
        # tmp = pred_flow[active_mask]
        # tmp[mid_conf_suppress] = 0.0
        # pred_flow[active_mask] = tmp

        # suppressed = int(mid_conf_suppress.any(axis=1).sum())
        # if suppressed > 0:
        #     print(f"[LinkPredictor] Stage2 后置验证抑制={suppressed}条")

        post_ms = (time.perf_counter() - post_start) * 1000.0
        pred_flow = np.clip(pred_flow, 0.0, max_cap)

        baseline_start = time.perf_counter()
        baseline_flow = self._predict_regression_batches(xb, batch_size, self.baseline) * max_cap
        baseline_ms = (time.perf_counter() - baseline_start) * 1000.0

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
        step_timestamps = [
            (current_utc_time + timedelta(seconds=effective_step_seconds * step)).isoformat()
            for step in range(self.OUTPUT_STEPS)
        ]

        normalized_links = [(str(src), str(dst)) for src, dst in link_index]
        history_count = min(
            input_window.shape[1], ceil(self.JITTER_WINDOW_SECONDS / effective_step_seconds) - 1
        )
        history_timestamps = [
            (current_utc_time - timedelta(seconds=effective_step_seconds * offset)).isoformat()
            for offset in range(history_count, 0, -1)
        ]
        history_flow = input_window[:, input_window.shape[1] - history_count:, 10] * max_cap
        distances = self._link_distances_km(input_window[:, -max(2, history_count):])
        latest_distance = distances[:, -1]
        rates = self._jitter_rates_gbps(latest_distance, normalized_links)

        # 最新观测对应预测起点之前一帧。按该帧间的距离差外推未来星距。
        distance_change = latest_distance - distances[:, -2]
        target_steps = np.arange(1, self.OUTPUT_STEPS + 1)
        target_distance = np.maximum(latest_distance[:, None] + distance_change[:, None] * target_steps, 0.0)
        ms_per_km = 1000.0 / config.SPEED_OF_LIGHT_KM_S
        target_propagation = target_distance * ms_per_km
        history_propagation = distances[:, distances.shape[1] - history_count:] * ms_per_km
        scfe_jitter = self._jitter_series_ms(
            normalized_links, rates, step_timestamps, pred_flow,
            history_timestamps, history_flow, effective_step_seconds,
            target_propagation_ms=target_propagation, history_propagation_ms=history_propagation,
        )
        baseline_jitter = self._jitter_series_ms(
            normalized_links, rates, step_timestamps, baseline_flow,
            history_timestamps, history_flow, effective_step_seconds,
            target_propagation_ms=target_propagation, history_propagation_ms=history_propagation,
        )

        loaded = np.isfinite(hist_last) & (hist_last > 0.0)
        eligible_count = int(np.count_nonzero(loaded))
        flow_scores = {
            "status": "pending", "baseline_wape": None, "scfe_wape": None,
            "tolerance_evaluation": self._pending_tolerance(
                self.FLOW_TOLERANCE_GBPS[self.constellation_id], "Gbps", eligible_count,
            ),
        }
        jitter_scores = {
            "status": "pending" if eligible_count else "no_loaded_links",
            "evaluated_step": 1,
            "target_timestamp": step_timestamps[0],
            "eligible_links": eligible_count,
            "valid_scfe_samples": int(np.count_nonzero(np.isfinite(scfe_jitter[loaded, 0]))),
            "baseline_wape": None,
            "scfe_wape": None,
            "improvement_pct": None,
            "meets_30_pct_target": None,
            "tolerance_evaluation": self._pending_tolerance(self.JITTER_TOLERANCE_MS, "ms", eligible_count),
        }
        # ── 组装输出 ──────────────────────────────────────────────────────────
        output_interface = {
            "Timestamp"        : step_timestamps,
            "ConstellationId"  : self.constellation_id,
            "Inference_time"   : 0.0,
            "PredictionHorizon": self.OUTPUT_STEPS,
            "TimeStepSeconds"  : effective_step_seconds,
            "LinksPredTopology": [],
            "FlowEvaluation": flow_scores,
            "JitterEvaluation": jitter_scores,
            "_debug_info": {
                "hist_last"   : hist_last,
                "hist_recent_5": hist_recent_5,
                "hist_std"    : hist_std,
                "decay_ratio" : decay_ratio,
                "active_prob" : None,
                "flow_forecasts_gbps": {
                    "link_index": normalized_links,
                    "baseline": baseline_flow,
                    "scfe": pred_flow,
                },
                "flow_evaluation": flow_scores,
                "jitter_forecasts": {
                    "rates_gbps": rates,
                    "history_timestamps": history_timestamps,
                    "history_propagation_ms": history_propagation,
                    "target_propagation_ms": target_propagation,
                    "scfe_ms": scfe_jitter,
                    "baseline_ms": baseline_jitter,
                },
                "jitter_evaluation": jitter_scores,
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
                

                # 真值到齐前使用浮点占位，评分状态见 FlowEvaluation.tolerance_evaluation。
                jitter_ms = float(scfe_jitter[i, step])
                link_state_vector = {
                    "BERPreFEC"        : 0.0,
                    "PacketLossRate"   : 0.0,
                    "SignalIntensity"  : 0.0,
                    "Delay"            : 0.0,
                    "AchievableRate"   : 0.0,
                    "DelayJitter"      : jitter_ms if isfinite(jitter_ms) else 0.0,
                    "LoadFluctuation"  : 0.0,
                    "Capacity"         : float(pred_capacity[i, step]),
                    "Survival"         : survival,
                    "HeatValue"        : float(pred_heat[i, step]),
                    "PredFlow"         : float(pred_flow[i, step]) * (effective_step_seconds / 8.0),
                    "LinkAvailability" : float(pred_availability[i, step]),
                }

                if src_key not in step_topology:
                    step_topology[src_key] = []
                step_topology[src_key].append([dst_key, link_state_vector])

            # TODO 计算整个链路上的精度指标
            baseline_precision,scfe_precision = 0.0,0.0

            output_interface["LinksPredTopology"].append({
                "Timestamp": step_timestamps[step],
                "Topology" : step_topology,
                "BaselinePrecision":baseline_precision,
                "SCFEPrecision":scfe_precision,
            })

        format_ms = (time.perf_counter() - format_start) * 1000.0
        total_time = time.perf_counter() - start_time
        timing_ms = {
            "history": history_ms,
            "tensor": tensor_ms,
            "classifier": 0.0,
            "regressor": regression_ms,
            "baseline": baseline_ms,
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
                f"links={link_num}, batch={batch_size}, "
                f"history={history_ms:.2f}ms, tensor={tensor_ms:.2f}ms, "
                f"regressor={regression_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
                f"post={post_ms:.2f}ms, derive={derive_ms:.2f}ms, "
                f"format={format_ms:.2f}ms, total={total_time * 1000.0:.2f}ms"
            )

        if current_flow_rows is not None:
            return self.evaluate_prediction_simple(output_interface, current_flow_rows)

        # 简化模式：不再在接口内启动后台线程轮询数据库。
        # 若需要精度评估，请调用方在拿到首步真值后显式调用 evaluate_prediction_simple。
        return output_interface
