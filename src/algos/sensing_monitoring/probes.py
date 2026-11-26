"""网络层参数的两种感知方法与交叉核对。

* **被动感知**（passive-flow-stats）：读取链路收发计数器，按统计窗口内的业务报文
  统计丢包率、时延、抖动与吞吐。只有承载业务的链路才有报文可统计。
* **主动探测**（active-probe）：每条链路按固定速率发送探测报文，测往返时延与探测丢包。
  空闲链路同样可测。

两种方法测的是同一条链路的同一组真值（来自仿真链路状态），测量误差来自有限样本：
丢包为二项分布，时延为 M/M/1 排队（指数分布）加时间戳噪声。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

PASSIVE_METHOD_ID = "passive-flow-stats"
ACTIVE_METHOD_ID = "active-probe"

#: 被动统计用于时延估计的最大样本报文数（每个窗口）
PASSIVE_DELAY_SAMPLES = 1000
#: 主动探测速率（包/秒/链路）
PROBE_RATE_PPS = 10
#: 时间戳噪声（ms）
TIMESTAMP_NOISE_MS = 1e-3
#: 报文长度（bit），与仿真 LINK_PACKET_BITS 一致
DEFAULT_PACKET_BITS = 12000


def _packet_bits() -> float:
    try:
        from src import config

        return float(getattr(config, "LINK_PACKET_BITS", DEFAULT_PACKET_BITS))
    except ImportError:
        return float(DEFAULT_PACKET_BITS)


class PassiveSensing:
    method_id = PASSIVE_METHOD_ID

    def build_window(self, entity_id: str, window_s: float, offered_gbps: float, packet_loss_rate: float,
                     base_one_way_ms: float, queue_delay_ms: float, jitter_ms: float,
                     rng: np.random.Generator) -> Dict[str, Any]:
        """生成一个统计窗口的计数器读数。"""
        sent = int(round(max(offered_gbps, 0.0) * 1e9 * window_s / _packet_bits()))
        lost = int(rng.binomial(sent, min(max(packet_loss_rate, 0.0), 1.0))) if sent else 0
        k = min(max(sent - lost, 0), PASSIVE_DELAY_SAMPLES)
        delays = (base_one_way_ms
                  + (rng.exponential(queue_delay_ms, k) if queue_delay_ms > 0 else np.zeros(k))
                  + rng.normal(0.0, max(jitter_ms, TIMESTAMP_NOISE_MS), k))
        return {"entity_id": str(entity_id), "window_s": float(window_s), "sent": sent, "lost": lost,
                "delays_ms": delays}

    def observe(self, window: Mapping[str, Any]) -> Dict[str, Any]:
        sent, lost = int(window["sent"]), int(window["lost"])
        delays = np.asarray(window["delays_ms"], dtype=float)
        delivered_bits = (sent - lost) * _packet_bits()
        return {
            "method_id": self.method_id,
            "entity_id": window["entity_id"],
            "samples": sent,
            "packet_loss_rate": (lost / sent) if sent else None,
            "delay_ms": float(delays.mean()) if delays.size else None,
            "delay_std_ms": float(delays.std(ddof=1)) if delays.size > 1 else None,
            "delay_samples": int(delays.size),
            "jitter_ms": float(np.mean(np.abs(np.diff(delays)))) if delays.size > 1 else None,
            "throughput_gbps": delivered_bits / float(window["window_s"]) / 1e9,
        }

    def measure_links(self, offered_gbps: np.ndarray, packet_loss_rate: np.ndarray,
                      base_one_way_ms: np.ndarray, queue_delay_ms: np.ndarray, window_s: float,
                      rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """全网链路批量被动测量。无业务的链路各项为 NaN（无报文可统计）。"""
        sent = np.round(np.maximum(offered_gbps, 0.0) * 1e9 * window_s / _packet_bits()).astype(np.int64)
        lost = rng.binomial(sent, np.clip(packet_loss_rate, 0.0, 1.0))
        delivered = sent - lost
        k = np.minimum(delivered, PASSIVE_DELAY_SAMPLES)
        has = sent > 0
        safe_k = np.maximum(k, 1)
        spread = np.sqrt(queue_delay_ms ** 2 + TIMESTAMP_NOISE_MS ** 2)
        delay = base_one_way_ms + queue_delay_ms + rng.normal(0.0, 1.0, sent.shape) * spread / np.sqrt(safe_k)
        nan = np.full(sent.shape, np.nan)
        return {
            "has_traffic": has,
            "packet_loss_rate": np.where(has, lost / np.maximum(sent, 1), nan),
            "delay_ms": np.where(has & (k > 0), delay, nan),
            "jitter_ms": np.where(has & (k > 1), spread, nan),
            "throughput_gbps": np.where(has, delivered * _packet_bits() / window_s / 1e9, nan),
        }


class ActiveProbeSensing:
    method_id = ACTIVE_METHOD_ID

    def probe(self, link_id: str, sim_time_s: float, base_one_way_ms: float, packet_loss_rate: float,
              queue_delay_ms: float, rng: np.random.Generator, probes: int = PROBE_RATE_PPS) -> Dict[str, Any]:
        """发送 ``probes`` 个探测报文，按往返时延的一半估计单向时延。"""
        lost = int(rng.binomial(probes, min(max(packet_loss_rate, 0.0), 1.0)))
        received = probes - lost
        one_way = (base_one_way_ms
                   + (rng.exponential(queue_delay_ms, received) if queue_delay_ms > 0 else np.zeros(received))
                   + rng.normal(0.0, TIMESTAMP_NOISE_MS, received))
        return {
            "method_id": self.method_id,
            "entity_id": str(link_id),
            "sim_time_s": float(sim_time_s),
            "samples": probes,
            "packet_loss_rate": lost / probes,
            "delay_ms": float(one_way.mean()) if received else None,
            "delay_std_ms": float(one_way.std(ddof=1)) if received > 1 else None,
            "delay_samples": int(received),
        }

    def measure_links(self, packet_loss_rate: np.ndarray, base_one_way_ms: np.ndarray,
                      queue_delay_ms: np.ndarray, rng: np.random.Generator,
                      probes: int = PROBE_RATE_PPS) -> Dict[str, np.ndarray]:
        """全网链路批量主动探测。"""
        lost = rng.binomial(probes, np.clip(packet_loss_rate, 0.0, 1.0))
        received = probes - lost
        spread = np.sqrt(queue_delay_ms ** 2 + TIMESTAMP_NOISE_MS ** 2)
        delay = base_one_way_ms + queue_delay_ms + rng.normal(0.0, 1.0, lost.shape) * spread / np.sqrt(
            np.maximum(received, 1))
        return {
            "packet_loss_rate": lost / probes,
            "delay_ms": np.where(received > 0, delay, np.nan),
        }


def cross_check(passive: Mapping[str, Any], active: Mapping[str, Any]) -> Dict[str, Any]:
    """同一链路两种方法的一致性核对（3σ 判据）。"""
    def _sigma(result):
        std = result.get("delay_std_ms") or TIMESTAMP_NOISE_MS
        return std / np.sqrt(max(int(result.get("delay_samples") or 1), 1))

    delay_ok = True
    delay_diff = None
    if passive.get("delay_ms") is not None and active.get("delay_ms") is not None:
        delay_diff = abs(passive["delay_ms"] - active["delay_ms"])
        tolerance = 3.0 * float(np.hypot(_sigma(passive), _sigma(active))) + TIMESTAMP_NOISE_MS
        delay_ok = delay_diff <= tolerance

    loss_ok = True
    loss_diff = None
    if passive.get("packet_loss_rate") is not None and active.get("packet_loss_rate") is not None:
        p_hat = float(passive["packet_loss_rate"])
        n_p = max(int(passive.get("samples") or 1), 1)
        n_a = max(int(active.get("samples") or 1), 1)
        loss_diff = abs(p_hat - float(active["packet_loss_rate"]))
        tolerance = 3.0 * np.sqrt(max(p_hat * (1 - p_hat), 0.0) * (1 / n_p + 1 / n_a)) + 1.0 / n_a
        loss_ok = loss_diff <= tolerance

    return {
        "entity_id": passive.get("entity_id"),
        "passive_method": passive.get("method_id"),
        "active_method": active.get("method_id"),
        "delay_diff_ms": delay_diff,
        "loss_diff": loss_diff,
        "consistent": bool(delay_ok and loss_ok),
    }


def sampling_impact_model(times_s: Sequence[float], values: Sequence[float],
                          intervals_s: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
                          entity_count: int = 1, param_count: int = 1,
                          bytes_per_value: int = 8, unit: str = "") -> Dict[str, Any]:
    """采样频率对感知结果与网络开销的影响分析。

    以 1 s 采样序列为参照（``values`` 可为 (T,) 或 (T, 实体数)），按各候选间隔降采样后
    用零阶保持重建，统计：

    * ``RMSE`` / ``MaxAbsError``：重建误差（与参照序列同单位）；
    * ``TelemetryKbps``：遥测上报开销 = 实体数 × 参数数 × 字节数 × 8 / 间隔。
    """
    t = np.asarray(times_s, dtype=float)
    v = np.asarray(values, dtype=float)
    rows = []
    if t.size >= 2:
        for interval in intervals_s:
            kept = np.zeros(t.size, dtype=bool)
            next_time = t[0]
            for i, ti in enumerate(t):
                if ti >= next_time - 1e-9:
                    kept[i] = True
                    next_time = ti + float(interval)
            held_idx = np.maximum.accumulate(np.where(kept, np.arange(t.size), 0))
            err = v - v[held_idx]
            rows.append({
                "Interval": float(interval),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MaxAbsError": float(np.max(np.abs(err))),
                "TelemetryKbps": entity_count * param_count * bytes_per_value * 8.0 / float(interval) / 1e3,
            })
    return {"Unit": unit, "ReferenceSamples": int(t.size), "Results": rows}
