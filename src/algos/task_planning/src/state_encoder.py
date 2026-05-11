# src/algos/task_planning/src/state_encoder.py

from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import networkx as nx

from src.algos.task_planning.config import (
    NUM_CANDIDATE_PATHS,
    HIST_NUM_BINS,
    HIST_VALUE_MIN,
    HIST_VALUE_MAX,
    DEMAND_SCALE,
    INCLUDE_VALID_MASK,
)


def encode_state(
    G: nx.Graph,
    candidate_paths: Sequence[Sequence[int | str]],
    valid_mask: Sequence[int],
    demand_gbps: float,
) -> np.ndarray:
    """
    将当前网络状态、候选路径和业务需求编码成 SAC 模型输入 obs。

    状态结构与训练时保持一致：

        [global_hist, path1_hist, path2_hist, path3_hist, demand]

    如果 INCLUDE_VALID_MASK=True，则为：

        [global_hist, path1_hist, path2_hist, path3_hist, demand, valid_mask]

    当前默认：
        NUM_CANDIDATE_PATHS = 3
        HIST_NUM_BINS = 10
        INCLUDE_VALID_MASK = False

    因此默认 obs 维度为：
        (3 + 1) * 10 + 1 = 41
    """

    paths = _pad_paths(candidate_paths, NUM_CANDIDATE_PATHS)
    mask = _pad_mask(valid_mask, NUM_CANDIDATE_PATHS)

    # 1. 全局链路利用率直方图
    global_utils = [
        float(data.get("utilization", 0.0))
        for _, _, data in G.edges(data=True)
    ]
    global_hist = _build_histogram(global_utils)

    # 2. 每条候选路径上的链路利用率直方图
    path_hists = []
    for path, is_valid in zip(paths, mask):
        if int(is_valid) == 0 or len(path) < 2:
            path_hists.append(np.zeros(HIST_NUM_BINS, dtype=np.float32))
            continue

        path_utils = _path_edge_utils(G, path)
        path_hists.append(_build_histogram(path_utils))

    # 3. 业务需求归一化
    demand_feat = np.array(
        [float(demand_gbps) / max(float(DEMAND_SCALE), 1e-12)],
        dtype=np.float32,
    )

    parts = [global_hist, *path_hists, demand_feat]

    if INCLUDE_VALID_MASK:
        parts.append(np.asarray(mask, dtype=np.float32))

    obs = np.concatenate(parts, axis=0).astype(np.float32)

    return obs


def expected_obs_dim() -> int:
    base_dim = (NUM_CANDIDATE_PATHS + 1) * HIST_NUM_BINS + 1
    if INCLUDE_VALID_MASK:
        base_dim += NUM_CANDIDATE_PATHS
    return base_dim


def _build_histogram(values: Sequence[float]) -> np.ndarray:
    """
    构造归一化直方图。
    """

    if len(values) == 0:
        return np.zeros(HIST_NUM_BINS, dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)
    arr = np.clip(arr, HIST_VALUE_MIN, HIST_VALUE_MAX)

    hist, _ = np.histogram(
        arr,
        bins=HIST_NUM_BINS,
        range=(HIST_VALUE_MIN, HIST_VALUE_MAX),
    )

    hist = hist.astype(np.float32)
    total = float(hist.sum())

    if total > 0:
        hist /= total

    return hist


def _path_edge_utils(
    G: nx.Graph,
    path: Sequence[int | str],
) -> list[float]:
    utils: list[float] = []

    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            utils.append(float(G[u][v].get("utilization", 0.0)))
        elif not G.is_directed() and G.has_edge(v, u):
            utils.append(float(G[v][u].get("utilization", 0.0)))
        else:
            # 如果路径和图不一致，保持鲁棒性
            utils.append(0.0)

    return utils


def _pad_paths(
    candidate_paths: Sequence[Sequence[Any]],
    num_paths: int,
) -> list[list[Any]]:
    paths = [list(p) for p in candidate_paths]

    if len(paths) == 0:
        paths = [[]]

    while len(paths) < num_paths:
        paths.append([])

    return paths[:num_paths]


def _pad_mask(
    valid_mask: Sequence[int],
    num_paths: int,
) -> list[int]:
    mask = [int(x) for x in valid_mask]

    while len(mask) < num_paths:
        mask.append(0)

    return mask[:num_paths]