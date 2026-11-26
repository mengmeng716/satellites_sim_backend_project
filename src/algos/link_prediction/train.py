"""训练入口：AdamW + ReduceLROnPlateau + 梯度裁剪，保存模型/标准化器/契约/训练元数据。

默认超参数（可被 CLI 覆盖）
--------------------------
* 优化器 AdamW(lr=1e-3, weight_decay=1e-4)
* 调度器 ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5)
* 梯度裁剪 1.0
* batch 64，epochs 80
* 早停 patience 12（只按验证集损失选最佳轮次，测试集从不参与选模）

产物目录 ``models/predictor``：
* ``multitarget_lstm_attn.pt`` 最佳权重 + 模型配置 + 契约 + 版本号
* ``scaler.json`` 训练集统计量
* ``feature_contract.json`` 特征/目标契约
* ``train_meta.json`` 训练曲线、数据来源、切分、参数量、单 seed 声明
"""

from __future__ import annotations

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from contracts import utc_iso
from dataset import (
    AUX_TARGET_NAMES,
    MODEL_DIR,
    TARGET_LOSS_WEIGHTS,
    DatasetBundle,
    FeatureContract,
    Standardizer,
    WindowedDataset,
    load_contract,
    load_scaler,
    save_contract,
    save_scaler,
    target_base_from_features,
)
from model import (
    ModelConfig,
    MultiTargetSpatioTemporalNet,
    build_model,
    count_parameters,
    masked_huber_loss,
    resolve_device,
    set_global_seed,
)

CHECKPOINT_NAME = "multitarget_lstm_attn.pt"


@dataclass
class TrainConfig:
    """训练超参数。"""

    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 12
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    min_learning_rate: float = 1e-5
    dropout: float = 0.15
    device: str = "auto"
    seed: int = 20260101
    verbose: bool = True
    #: 物理合成模式：``"none"`` / ``"jitter"`` / ``"full"``（见 ModelConfig）
    composition_mode: str = "full"
    #: 输入归一化方式：``"layer"`` / ``"none"``（见 ModelConfig）
    input_norm: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    """训练结果（模型、曲线、最佳轮次、耗时）。"""

    model: MultiTargetSpatioTemporalNet
    history: List[Dict[str, float]]
    best_epoch: int
    best_val_loss: float
    best_val_mae: float
    epochs_run: int
    train_seconds: float
    parameter_count: int
    device: str
    model_version: str = ""
    checkpoint_path: str = ""
    jitter_mode: str = "direct"
    val_jitter_mae_direct: float = 0.0
    val_jitter_mae_composed: float = 0.0
    aux_val_mae: Dict[str, float] = field(default_factory=dict)


def _state_dict_digest(state_dict: Dict[str, torch.Tensor]) -> str:
    """对权重内容求稳定摘要，用于生成可追溯的模型版本号。"""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:8]


def _make_batches(n_samples: int, batch_size: int, rng: np.random.Generator) -> List[np.ndarray]:
    order = rng.permutation(n_samples)
    return [order[start : start + batch_size] for start in range(0, n_samples, batch_size)]


def evaluate_metrics(
    model: nn.Module,
    dataset: WindowedDataset,
    scaler: Standardizer,
    contract: FeatureContract,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[float, float]:
    """返回 ``(加权 Huber, 标准化 MAE)``；两者都只在有效目标上统计。

    验证集用标准化 MAE 做调度与选模：它与验收口径（原始 MAE）单调一致，
    且不会被 BER 等重尾目标的 Huber 长尾主导。
    """
    model.eval()
    X = scaler.transform_features(dataset.X)
    Y = scaler.transform_targets(dataset.Y)
    mask = dataset.mask.astype(np.float32)
    base = target_base_from_features(dataset.X, scaler, contract)
    weights = torch.tensor(TARGET_LOSS_WEIGHTS, dtype=torch.float32, device=device)
    total_weighted, total_weight, total_abs, total_count = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            yb = torch.from_numpy(Y[start : start + batch_size]).to(device)
            mb = torch.from_numpy(mask[start : start + batch_size]).to(device)
            bb = torch.from_numpy(base[start : start + batch_size]).to(device)
            predictions = model(xb, bb)
            elementwise = torch.nn.functional.smooth_l1_loss(predictions, yb, reduction="none", beta=0.5)
            weight = mb.unsqueeze(1) * weights.view(1, 1, -1)
            total_weighted += float((elementwise * weight).sum().item())
            total_weight += float(weight.sum().item())
            total_abs += float((torch.abs(predictions - yb) * mb.unsqueeze(1)).sum().item())
            total_count += float(mb.sum().item()) * contract.horizon
    mean_loss = total_weighted / max(total_weight, 1.0)
    mean_mae = total_abs / max(total_count, 1.0)
    return mean_loss, mean_mae


def evaluate_loss(
    model: nn.Module,
    dataset: WindowedDataset,
    scaler: Standardizer,
    contract: FeatureContract,
    device: torch.device,
    batch_size: int = 2048,
) -> float:
    """只返回加权 Huber 损失（兼容旧调用）。"""
    return evaluate_metrics(model, dataset, scaler, contract, device, batch_size)[0]


def evaluate_jitter_variants(
    model: nn.Module,
    dataset: WindowedDataset,
    scaler: Standardizer,
    contract: FeatureContract,
    device: torch.device,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """在**验证集**上比较直接 jitter 头与物理合成头的原始 MAE。"""
    X = scaler.transform_features(dataset.X)
    base = target_base_from_features(dataset.X, scaler, contract)
    truth = dataset.Y[..., 1].astype(np.float32)
    valid = dataset.mask[:, 1] > 0.5
    predictions: Dict[bool, List[np.ndarray]] = {False: [], True: []}
    for mode in (False, True):
        model.use_composed_jitter = mode
        model.eval()
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).to(device)
                bb = torch.from_numpy(base[start : start + batch_size]).to(device)
                output = model(xb, bb)
                predictions[mode].append(output.detach().cpu().numpy())
    direct_raw = scaler.inverse_targets(np.concatenate(predictions[False], axis=0))[..., 1]
    composed_raw = scaler.inverse_targets(np.concatenate(predictions[True], axis=0))[..., 1]
    model.use_composed_jitter = False
    return {
        "direct_mae": float(np.mean(np.abs(direct_raw[valid] - truth[valid]))),
        "composed_mae": float(np.mean(np.abs(composed_raw[valid] - truth[valid]))),
    }


def select_composed_targets(
    model: nn.Module,
    dataset: WindowedDataset,
    scaler: Standardizer,
    contract: FeatureContract,
    device: torch.device,
    batch_size: int = 2048,
    margin: float = 0.002,
) -> List[bool]:
    """在验证集上逐目标比较「直接回归」与「物理合成」，只保留确实更优的目标。

    采用 ``margin``（默认 0.2%）迟滞，避免把数值噪声当成改进。
    合成是**绝对覆盖**，不会与残差 base 叠加，因此两侧都按模型最终输出比较。
    """
    X = scaler.transform_features(dataset.X)
    Y = scaler.transform_targets(dataset.Y)
    base = target_base_from_features(dataset.X, scaler, contract)
    truth = scaler.inverse_targets(Y)
    mask = np.asarray(getattr(dataset, "mask", np.ones(Y.shape[:1] + (Y.shape[-1],))), dtype=np.float32)

    original_mode = getattr(model.config, "composition_mode", "none")
    original_mask = tuple(getattr(model.config, "composed_target_mask", ()) or ())
    n_targets = contract.n_targets

    def _predictions(mode: str, target_mask) -> np.ndarray:
        # forward 读的是实例属性 self.composition_mode，必须两处一起改
        model.config.composition_mode = mode
        model.config.composed_target_mask = target_mask
        model.composition_mode = mode
        outputs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).to(device)
                bb = torch.from_numpy(base[start : start + batch_size]).to(device)
                out = model(xb, bb)
                outputs.append(out.detach().cpu().numpy())
        return scaler.inverse_targets(np.concatenate(outputs, axis=0))

    direct = _predictions("none", ())
    composed = _predictions("full", tuple([True] * n_targets))

    selection: List[bool] = []
    for index in range(n_targets):
        valid = mask[:, index] > 0.5
        if not valid.any():
            selection.append(False)
            continue
        direct_mae = float(np.mean(np.abs(direct[valid, :, index] - truth[valid, :, index])))
        composed_mae = float(np.mean(np.abs(composed[valid, :, index] - truth[valid, :, index])))
        selection.append(bool(composed_mae < direct_mae * (1.0 - margin)))

    model.config.composition_mode = original_mode
    model.config.composed_target_mask = original_mask
    model.composition_mode = original_mode
    return selection


def evaluate_aux_mae(
    model: nn.Module,
    dataset: WindowedDataset,
    scaler: Standardizer,
    contract: FeatureContract,
    device: torch.device,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """辅助头在验证集上的原始 MAE（队列占用、振动）。"""
    X = scaler.transform_features(dataset.X)
    base = target_base_from_features(dataset.X, scaler, contract)
    truth = np.asarray(dataset.C["aux"], dtype=np.float32)
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            bb = torch.from_numpy(base[start : start + batch_size]).to(device)
            _, aux_scaled, _ = model(xb, bb, return_aux=True)
            outputs.append(aux_scaled.detach().cpu().numpy())
    aux_raw = scaler.inverse_aux(np.concatenate(outputs, axis=0))
    return {
        name: float(np.mean(np.abs(aux_raw[..., index] - truth[..., index])))
        for index, name in enumerate(AUX_TARGET_NAMES)
    }


def train_model(
    bundle: DatasetBundle,
    config: Optional[TrainConfig] = None,
    progress: Optional[Callable[[Dict[str, float]], None]] = None,
) -> TrainResult:
    """训练多目标模型并返回最佳权重。"""
    config = config or TrainConfig()
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    contract = bundle.contract
    scaler = bundle.scaler
    if len(scaler.feature_names) != contract.feature_dim:
        raise ValueError("标准化器与特征契约维度不一致")

    model_config = ModelConfig(
        input_dim=contract.feature_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        horizon=contract.horizon,
        n_targets=contract.n_targets,
        dropout=config.dropout,
        enable_aux=True,
        aux_dim=len(AUX_TARGET_NAMES),
        composition_mode=getattr(config, "composition_mode", "full"),
        input_norm=getattr(config, "input_norm", "none"),
    )
    model = build_model(model_config).to(device)
    model.set_composition_stats(
        scaler.aux_mean, scaler.aux_std, scaler.target_mean, scaler.target_std,
        feature_mean=scaler.feature_mean, feature_std=scaler.feature_std,
    )
    # 训练期直接头与物理合成同时训练（合成通道绝对覆盖），验证集再比较是否启用合成
    model.use_composed_jitter = False
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    weights = torch.tensor(TARGET_LOSS_WEIGHTS, dtype=torch.float32, device=device)

    X_train = scaler.transform_features(bundle.train.X)
    Y_train = scaler.transform_targets(bundle.train.Y)
    mask_train = bundle.train.mask.astype(np.float32)
    base_train = target_base_from_features(bundle.train.X, scaler, contract)
    aux_train = scaler.transform_aux(bundle.train.C["aux"])
    aux_mask_train = bundle.train.C.get(
        "aux_mask", np.ones((len(bundle.train), len(AUX_TARGET_NAMES)), dtype=np.float32)
    ).astype(np.float32)
    aux_weights = torch.ones(len(AUX_TARGET_NAMES), dtype=torch.float32, device=device)
    n_train = int(X_train.shape[0])

    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started_all = time.perf_counter()

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        epoch_started = time.perf_counter()
        rng = np.random.default_rng(config.seed + epoch * 1009)
        total_loss, total_count = 0.0, 0.0
        for indices in _make_batches(n_train, config.batch_size, rng):
            xb = torch.from_numpy(X_train[indices]).to(device)
            yb = torch.from_numpy(Y_train[indices]).to(device)
            mb = torch.from_numpy(mask_train[indices]).to(device)
            bb = torch.from_numpy(base_train[indices]).to(device)
            aux_b = torch.from_numpy(aux_train[indices]).to(device)
            aux_mb = torch.from_numpy(aux_mask_train[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, aux_predictions, _ = model(xb, bb, return_aux=True)
            main_loss = masked_huber_loss(predictions, yb, mb, weights)
            aux_loss = masked_huber_loss(aux_predictions, aux_b, aux_mb, aux_weights)
            # 直接头的辅助损失：保证物理合成覆盖的通道上直接头也被训练，
            # 这样验证集逐目标选择“直接 vs 合成”才有意义、才能退回。
            direct_output = getattr(model, "last_direct_output", None)
            direct_loss = (
                masked_huber_loss(direct_output, yb, mb, weights)
                if direct_output is not None
                else torch.zeros((), device=device)
            )
            loss = main_loss + 0.5 * aux_loss + 0.5 * direct_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_weight = float((mb.unsqueeze(1) * weights.view(1, 1, -1)).sum().item())
            total_loss += float(loss.item()) * batch_weight
            total_count += batch_weight
        train_loss = total_loss / max(total_count, 1.0)
        val_loss, val_mae = evaluate_metrics(model, bundle.val, scaler, contract, device)
        scheduler.step(val_mae)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "learning_rate": learning_rate,
            "seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(record)
        if progress is not None:
            progress(record)
        elif config.verbose:
            print(
                f"[train] epoch {epoch:03d} train={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_mae={val_mae:.5f} lr={learning_rate:.2e}",
                flush=True,
            )
        if val_mae < best_val_mae - 1e-6:
            best_val_mae = float(val_mae)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.patience):
                if config.verbose:
                    print(f"[train] early stop at epoch {epoch} (best epoch={best_epoch})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # 只在验证集上做逐目标选择：直接回归 vs 物理合成，谁 MAE 小用谁。
    # 合成是绝对覆盖，一旦启用就不再叠加残差 base，因此两侧都按最终输出比较。
    if str(getattr(model.config, "composition_mode", "none")) == "full":
        target_mask = select_composed_targets(model, bundle.val, scaler, contract, device)
        model.config.composed_target_mask = tuple(target_mask)
        selected = [contract.target_names[i] for i, flag in enumerate(target_mask) if flag]
        if config.verbose:
            print(f"[train] 逐目标合成选择: {selected}", flush=True)
    # 只在验证集上选择 jitter 直接头 / 物理合成头；测试集不参与。
    jitter_variants = evaluate_jitter_variants(model, bundle.val, scaler, contract, device)
    use_composed = jitter_variants["composed_mae"] < jitter_variants["direct_mae"]
    model.use_composed_jitter = bool(use_composed)
    model.config.use_composed_jitter = bool(use_composed)
    aux_val_mae = evaluate_aux_mae(model, bundle.val, scaler, contract, device)
    train_seconds = float(time.perf_counter() - started_all)
    version = f"mtp-v1-seed{config.seed}-{_state_dict_digest(model.state_dict())}"
    return TrainResult(
        model=model,
        history=history,
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val_loss),
        best_val_mae=float(best_val_mae),
        epochs_run=len(history),
        train_seconds=train_seconds,
        parameter_count=parameter_count,
        device=str(device),
        model_version=version,
        jitter_mode="composed" if use_composed else "direct",
        val_jitter_mae_direct=float(jitter_variants["direct_mae"]),
        val_jitter_mae_composed=float(jitter_variants["composed_mae"]),
        aux_val_mae=aux_val_mae,
    )


def save_model_bundle(
    result: TrainResult,
    bundle: DatasetBundle,
    config: TrainConfig,
    model_dir: Path = MODEL_DIR,
    extra_seed_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """保存权重、标准化器、特征契约与训练元数据；返回各文件路径。"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / CHECKPOINT_NAME
    payload = {
        "state_dict": result.model.state_dict(),
        "model_config": result.model.config.to_dict(),
        "contract": bundle.contract.to_dict(),
        "model_version": result.model_version,
        "target_names": list(bundle.contract.target_names),
        "use_composed_jitter": bool(result.model.use_composed_jitter),
    }
    torch.save(payload, checkpoint_path)
    save_scaler(bundle.scaler, model_dir / "scaler.json")
    save_contract(bundle.contract, model_dir / "feature_contract.json")

    target_stats = {}
    for index, name in enumerate(bundle.contract.target_names):
        valid = bundle.train.mask[:, index] > 0.5
        values = bundle.train.Y[valid, :, index].astype(np.float64).reshape(-1)
        target_stats[name] = {
            "train_mean": float(values.mean()) if values.size else 0.0,
            "train_std": float(values.std()) if values.size else 0.0,
            "n_train_values": int(values.size),
            "scaler_mean": float(bundle.scaler.target_mean[index]),
            "scaler_std": float(bundle.scaler.target_std[index]),
        }
    train_meta: Dict[str, Any] = {
        "model_version": result.model_version,
        "model_file": CHECKPOINT_NAME,
        "created_utc": utc_iso(),
        "contract_id": bundle.contract.contract_id,
        "parameter_count": result.parameter_count,
        "model_config": result.model.config.to_dict(),
        "target_names": list(bundle.contract.target_names),
        "target_units": list(bundle.contract.target_units),
        "target_entity_types": list(bundle.contract.target_entity_types),
        "target_loss_weights": list(TARGET_LOSS_WEIGHTS),
        "target_statistics": target_stats,
        "training": {
            **config.to_dict(),
            "best_epoch": result.best_epoch,
            "best_val_loss_scaled": result.best_val_loss,
            "best_val_mae_scaled": result.best_val_mae,
            "epochs_run": result.epochs_run,
            "train_seconds": result.train_seconds,
            "device_used": result.device,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "grad_clip": config.grad_clip,
            "loss": "masked SmoothL1(Huber, beta=0.5), target-weighted; 多步残差预测（base=最后观测值）",
            "aux_loss": "queue_occupancy_ratio / vibration_rms_g 辅助头，权重 0.5",
            "jitter_mode": result.jitter_mode,
            "val_jitter_mae_direct": result.val_jitter_mae_direct,
            "val_jitter_mae_composed": result.val_jitter_mae_composed,
            "aux_val_mae": result.aux_val_mae,
            "jitter_selection": "只在验证集比较 direct/composed，测试集不参与选择",
            "model_selection": "仅按验证集标准化 MAE 选择最佳轮次；测试集/跨场景不参与选模",
        },
        "history": result.history,
        "data": bundle.summary(),
        "seed_policy": {
            "primary_seed": int(config.seed),
            "declared_before_training": True,
            "seed_selection": "单 seed 主结果；未按测试结果挑 seed",
            "extra_seed_results": extra_seed_results or [],
        },
        "notes": [
            "输入为含传感/遥测误差的观测历史，标签为 physics 物理真值；观测误差表在 dataset.OBSERVATION_NOISE_SPEC 中冻结。",
            "标签与输入均不包含预测发起时刻之后的未来信息。",
            "标准化统计量只来自训练集。",
        ],
    }
    (model_dir / "train_meta.json").write_text(
        json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "checkpoint": str(checkpoint_path),
        "scaler": str(model_dir / "scaler.json"),
        "contract": str(model_dir / "feature_contract.json"),
        "train_meta": str(model_dir / "train_meta.json"),
    }


def load_model_bundle(
    model_dir: Path = MODEL_DIR, device: str = "auto"
) -> Dict[str, Any]:
    """加载模型、标准化器、契约与训练元数据。"""
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - 兼容旧版 torch 的 weights_only 行为
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = load_contract(model_dir / "feature_contract.json")
    scaler = load_scaler(model_dir / "scaler.json")
    model_config = ModelConfig.from_dict(payload["model_config"])
    model = build_model(model_config)
    model.load_state_dict(payload["state_dict"])
    resolved = resolve_device(device)
    model.to(resolved).eval()
    model.set_composition_stats(
        scaler.aux_mean, scaler.aux_std, scaler.target_mean, scaler.target_std,
        feature_mean=scaler.feature_mean, feature_std=scaler.feature_std,
    )
    model.use_composed_jitter = bool(payload.get("use_composed_jitter", False))
    model.composition_mode = str(model_config.composition_mode)
    model.config.use_composed_jitter = model.use_composed_jitter
    meta_path = model_dir / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "model": model,
        "model_config": model_config,
        "contract": contract,
        "scaler": scaler,
        "meta": meta,
        "model_version": str(payload.get("model_version", meta.get("model_version", "unknown"))),
        "device": resolved,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
