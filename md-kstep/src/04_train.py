"""Supervised training loop for the equivariant k-step model."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from utils import (configure_logging, ensure_dir, load_yaml, set_seed,
                   LOGGER)

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "03_model.py"
_spec = importlib.util.spec_from_file_location("md_kstep_model", MODEL_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
import sys

sys.modules[_spec.name] = _module  # ensure dataclasses can resolve module
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]
build_model_from_config = _module.build_model_from_config  # type: ignore[attr-defined]


@dataclass
class TrainConfig:
    seed: int
    k_steps: int
    batch_size: int
    grad_accum: int
    num_workers: int
    lr: float
    lr_min: float
    weight_decay: float
    max_epochs: int
    steps_per_epoch: int
    amp: bool
    grad_clip: float
    lambda_vel: float
    lambda_com: float
    lambda_force: float
    val_every_steps: int
    checkpoint_every_steps: int
    checkpoint_dir: str
    log_dir: str
    resume: str | None
    wandb: Dict
    random_rotate: bool

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainConfig":
        return cls(**data)


class KStepDataset(Dataset):
    def __init__(self, dataset_path: Path, allowed_molecules: Iterable[str] | None = None):
        data = np.load(dataset_path, allow_pickle=True)
        self.x_t = data["x_t"]
        self.v_t = data["v_t"]
        self.x_tk = data["x_tk"]
        self.v_tk = data["v_tk"]
        self.masses = data["masses"]
        self.atom_types = data["atom_types"]
        self.molecule = data["molecule"]
        self.k_steps = int(data["k_steps"])

        if allowed_molecules is not None:
            mask = np.isin(self.molecule, list(allowed_molecules))
            self.x_t = self.x_t[mask]
            self.v_t = self.v_t[mask]
            self.x_tk = self.x_tk[mask]
            self.v_tk = self.v_tk[mask]
            self.masses = self.masses[mask]
            self.atom_types = self.atom_types[mask]
            self.molecule = self.molecule[mask]

    def __len__(self) -> int:
        return len(self.x_t)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "x_t": self.x_t[idx],
            "v_t": self.v_t[idx],
            "x_tk": self.x_tk[idx],
            "v_tk": self.v_tk[idx],
            "masses": self.masses[idx],
            "atom_types": self.atom_types[idx],
            "molecule": self.molecule[idx],
        }


def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    x_t = [torch.from_numpy(item["x_t"]).to(device) for item in batch]
    v_t = [torch.from_numpy(item["v_t"]).to(device) for item in batch]
    x_tk = [torch.from_numpy(item["x_tk"]).to(device) for item in batch]
    v_tk = [torch.from_numpy(item["v_tk"]).to(device) for item in batch]
    masses = [torch.from_numpy(item["masses"]).to(device) for item in batch]
    atom_types = [torch.from_numpy(item["atom_types"]).long().to(device) for item in batch]

    node_counts = [item.shape[0] for item in x_t]
    batch_index = torch.cat([torch.full((count,), i, dtype=torch.long) for i, count in enumerate(node_counts)])

    def _cat(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=0)

    return {
        "x_t": _cat(x_t),
        "v_t": _cat(v_t),
        "x_tk": _cat(x_tk),
        "v_tk": _cat(v_tk),
        "masses": _cat(masses),
        "atom_types": _cat(atom_types),
        "batch": batch_index,
        "node_slices": torch.tensor(node_counts, dtype=torch.long),
    }


def _split_by_batch(tensor: torch.Tensor, node_slices: torch.Tensor) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    start = 0
    for count in node_slices.tolist():
        outputs.append(tensor[start:start + count])
        start += count
    return outputs


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if dim_size <= 0:
        return torch.zeros((0,) + src.shape[1:], device=src.device, dtype=src.dtype)
    out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def _com_loss(pred: torch.Tensor, target: torch.Tensor, masses: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    masses = masses.unsqueeze(-1)
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    total_mass = _scatter_sum(masses, batch_index, num_graphs)
    pred_com = _scatter_sum(pred * masses, batch_index, num_graphs) / torch.clamp(total_mass, min=1e-8)
    target_com = _scatter_sum(target * masses, batch_index, num_graphs) / torch.clamp(total_mass, min=1e-8)
    return torch.mean((pred_com - target_com) ** 2)


def _momentum_loss(pred_vel: torch.Tensor, target_vel: torch.Tensor, masses: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    masses = masses.unsqueeze(-1)
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    pred_momentum = _scatter_sum(pred_vel * masses, batch_index, num_graphs)
    target_momentum = _scatter_sum(target_vel * masses, batch_index, num_graphs)
    return torch.mean((pred_momentum - target_momentum) ** 2)


def _move_to(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in batch.items()}


def _random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Shoemake (1992) uniform quaternions
    u1 = torch.rand((), device=device, dtype=dtype)
    u2 = torch.rand((), device=device, dtype=dtype)
    u3 = torch.rand((), device=device, dtype=dtype)
    sqrt_u1 = torch.sqrt(u1)
    sqrt_one_minus_u1 = torch.sqrt(1 - u1)
    theta1 = 2 * math.pi * u2
    theta2 = 2 * math.pi * u3
    x = sqrt_one_minus_u1 * torch.sin(theta1)
    y = sqrt_one_minus_u1 * torch.cos(theta1)
    z = sqrt_u1 * torch.sin(theta2)
    w = sqrt_u1 * torch.cos(theta2)
    rot = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])
    return rot.to(device=device, dtype=dtype)


def _apply_random_rotations(batch: Dict[str, torch.Tensor]) -> None:
    counts = batch["node_slices"].detach().cpu().tolist()
    if not counts:
        return
    device = batch["x_t"].device
    dtype = batch["x_t"].dtype
    start = 0
    for count in counts:
        end = start + count
        rot = _random_rotation_matrix(device, dtype)
        for key in ("x_t", "v_t", "x_tk", "v_tk"):
            batch[key][start:end] = batch[key][start:end] @ rot.T
        start = end


def run_validation(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_to(batch, device)
            outputs = model(batch)
            pred_pos = batch["x_t"] + outputs["delta_pos"]
            pred_vel = batch["v_t"] + outputs["delta_vel"]
            loss_pos = torch.mean((pred_pos - batch["x_tk"]) ** 2)
            loss_vel = torch.mean((pred_vel - batch["v_tk"]) ** 2)
            loss = loss_pos + cfg.lambda_vel * loss_vel
            losses.append(loss.item())
    model.train()
    return {"val_loss": float(np.mean(losses)) if losses else float("nan")}


def save_checkpoint(state: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset npz path (single k)")
    parser.add_argument("--model-config", type=Path, required=True, help="Model YAML config")
    parser.add_argument("--train-config", type=Path, required=True, help="Training YAML config")
    parser.add_argument("--splits", type=Path, nargs=2, required=True, help="Train and val split JSON files")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return parser


def load_split(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    return data.get("molecules", [])


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()

    model_cfg = load_yaml(args.model_config)
    train_cfg = TrainConfig.from_dict(load_yaml(args.train_config))
    set_seed(train_cfg.seed)

    log_dir = Path(train_cfg.log_dir)
    ensure_dir(log_dir)
    train_log_path = log_dir / "train_metrics.jsonl"
    val_log_path = log_dir / "val_metrics.jsonl"
    train_log_path.unlink(missing_ok=True)
    val_log_path.unlink(missing_ok=True)

    train_metrics: List[Dict[str, float]] = []
    val_metrics: List[Dict[str, float]] = []

    train_split = load_split(args.splits[0])
    val_split = load_split(args.splits[1])

    train_dataset = KStepDataset(args.dataset, train_split)
    val_dataset = KStepDataset(args.dataset, val_split)

    device = torch.device(args.device)

    model = build_model_from_config(model_cfg)
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = train_cfg.max_epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=train_cfg.lr_min)
    scaler = GradScaler("cuda", enabled=train_cfg.amp)

    global_step = 0
    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(train_cfg.max_epochs):
        LOGGER.info("Epoch %d/%d", epoch + 1, train_cfg.max_epochs)
        for batch in train_loader:
            batch = _move_to(batch, device)
            if train_cfg.random_rotate:
                _apply_random_rotations(batch)
            com_pos_loss = None
            momentum_loss = None
            force_reg = None
            with autocast("cuda", enabled=train_cfg.amp):
                outputs = model(batch)
                pred_pos = batch["x_t"] + outputs["delta_pos"]
                pred_vel = batch["v_t"] + outputs["delta_vel"]
                loss_pos = torch.mean((pred_pos - batch["x_tk"]) ** 2)
                loss_vel = torch.mean((pred_vel - batch["v_tk"]) ** 2)
                loss = loss_pos + train_cfg.lambda_vel * loss_vel
                if train_cfg.lambda_com > 0:
                    com_pos_loss = _com_loss(pred_pos, batch["x_tk"], batch["masses"], batch["batch"])
                    momentum_loss = _momentum_loss(pred_vel, batch["v_tk"], batch["masses"], batch["batch"])
                    loss = loss + train_cfg.lambda_com * (com_pos_loss + momentum_loss)
                if train_cfg.lambda_force > 0 and "force_pred" in outputs:
                    force_reg = torch.mean(outputs["force_pred"] ** 2)
                    loss = loss + train_cfg.lambda_force * force_reg

            current_step = global_step + 1
            metrics_entry: Dict[str, float | int] = {
                "step": int(current_step),
                "epoch": int(epoch + 1),
                "loss": float(loss.detach().item()),
                "loss_pos": float(loss_pos.detach().item()),
                "loss_vel": float(loss_vel.detach().item()),
            }
            if com_pos_loss is not None and momentum_loss is not None:
                metrics_entry["loss_com"] = float(com_pos_loss.detach().item())
                metrics_entry["loss_momentum"] = float(momentum_loss.detach().item())
            if force_reg is not None:
                metrics_entry["loss_force"] = float(force_reg.detach().item())
            train_metrics.append(metrics_entry)

            scaler.scale(loss / train_cfg.grad_accum).backward()

            if (global_step + 1) % train_cfg.grad_accum == 0:
                if train_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1

            if train_cfg.val_every_steps > 0 and global_step % train_cfg.val_every_steps == 0:
                metrics = run_validation(model, val_loader, device, train_cfg)
                LOGGER.info("Step %d: val_loss=%.6f", global_step, metrics["val_loss"])
                val_metrics.append({
                    "step": int(global_step),
                    "epoch": int(epoch + 1),
                    "val_loss": float(metrics["val_loss"]),
                })
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    checkpoint_path = Path(train_cfg.checkpoint_dir) / "best.pt"
                    save_checkpoint({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": best_val,
                    }, checkpoint_path)

            if train_cfg.checkpoint_every_steps > 0 and global_step % train_cfg.checkpoint_every_steps == 0:
                checkpoint_path = Path(train_cfg.checkpoint_dir) / f"step_{global_step}.pt"
                save_checkpoint({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                }, checkpoint_path)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    _write_jsonl(train_log_path, train_metrics)
    _write_jsonl(val_log_path, val_metrics)

    LOGGER.info("Training complete. Best val_loss=%.6f", best_val)


if __name__ == "__main__":
    main()
