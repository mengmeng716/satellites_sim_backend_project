"""可用率评估：时间加权可用率、最差链路、中断时长与次数。

输入 ``samples`` 支持两种形式：

1. ``[{链接: bool}, ...]``：等间隔时间帧，逐帧记录链路通断；
2. ``{"link_id": [bool, bool, ...]}``：按链路给出通断序列。

统计口径：

* 单链路可用率 = 通帧数 / 总帧数
* 全网可用率 = 计划服务链路可用率的算术平均
* 不可用时长 = 总时长 × (1 − 全网可用率)
* 中断次数 = 链路由通到断的跳变数之和
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _normalize_samples(samples: Any) -> Dict[str, List[bool]]:
    """统一成 ``{link_id: [bool, ...]}``。"""
    if not samples:
        return {}

    if isinstance(samples, Mapping):
        result: Dict[str, List[bool]] = {}
        for link_id, series in samples.items():
            result[str(link_id)] = [bool(x) for x in (series or [])]
        return result

    frames = list(samples)
    if not frames:
        return {}

    result = {}
    for frame in frames:
        for link_id, up in (frame or {}).items():
            result.setdefault(str(link_id), []).append(bool(up))
    return result


def measure_availability(
    samples: Any = None,
    planned_links: Optional[Iterable[str]] = None,
    frame_seconds: float = 5.0,
) -> Dict[str, Any]:
    """计算可用率指标。

    :param samples: 时间帧或按链路的通断序列
    :param planned_links: 计划服务链路清单（可用率分母）；缺省用样本中的全部链路
    :param frame_seconds: 单帧时长（秒），用于换算不可用时长
    :return: ``{"Availability", "MinLinkAvailability", "DowntimeSeconds",
                "OutageEvents", "LinkCount", "FrameCount", "PerLink"}``
    """
    series_map = _normalize_samples(samples)
    if planned_links:
        keep = [str(x) for x in planned_links]
        series_map = {k: v for k, v in series_map.items() if k in keep}

    if not series_map:
        return {
            "Availability": None, "MinLinkAvailability": None,
            "DowntimeSeconds": None, "OutageEvents": None,
            "LinkCount": 0, "FrameCount": 0, "PerLink": {},
        }

    per_link: Dict[str, Dict[str, Any]] = {}
    total_frames = 0
    outages = 0

    for link_id, series in series_map.items():
        if not series:
            continue
        frames = len(series)
        up = sum(1 for x in series if x)
        availability = up / frames
        transitions = sum(
            1 for i in range(1, frames) if series[i - 1] and not series[i]
        )
        outages += transitions
        total_frames = max(total_frames, frames)
        per_link[link_id] = {
            "Frames": frames,
            "UpFrames": up,
            "Availability": round(availability, 6),
            "OutageEvents": transitions,
            "DowntimeSeconds": round((frames - up) * frame_seconds, 3),
        }

    if not per_link:
        return {
            "Availability": None, "MinLinkAvailability": None,
            "DowntimeSeconds": None, "OutageEvents": None,
            "LinkCount": 0, "FrameCount": 0, "PerLink": {},
        }

    values = [item["Availability"] for item in per_link.values()]
    mean_availability = sum(values) / len(values)

    return {
        "Availability": round(mean_availability, 6),
        "MinLinkAvailability": round(min(values), 6),
        "DowntimeSeconds": round(total_frames * frame_seconds * (1.0 - mean_availability), 3),
        "OutageEvents": int(outages),
        "LinkCount": len(per_link),
        "FrameCount": total_frames,
        "PerLink": per_link,
    }
