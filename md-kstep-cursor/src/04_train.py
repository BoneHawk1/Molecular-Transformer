from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import get_logger, load_yaml, ensure_dir, set_seed
from utils import pad_collate
import importlib.util
import sys
from pathlib import Path as _Path

# Dynamically import build_model from 03_model.py
_MODEL_FILE = _Path(__file__).with_name("03_model.py")
_spec = importlib.util.spec_from_file_location("kstep_model", _MODEL_FILE)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_model = _mod.build_model


class KStepDataset(Dataset):
    def __init__(self, dataset_npz: str):
        with np.load(dataset_npz) as d:
            self.x_t = d["x_t"].astype(np.float32)
            self.v_t = d["v_t"].astype(np.float32)
            self.x_tk = d["x_tk"].astype(np.float32)
            self.v_tk = d["v_tk"].astype(np.float32)
            self.atom_types = d["atom_types"].astype(np.int64)
        self.num_atoms = self.x_t.shape[1]

    def __len__(self) -> int:
        return self.x_t.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        return {
            "x_t": self.x_t[idx],
            "v_t": self.v_t[idx],
            "x_tk": self.x_tk[idx],
            "v_tk": self.v_tk[idx],
            "atom_types": self.atom_types,
            "num_atoms": self.num_atoms,
        }


def train_one_epoch(model, loader, opt, scaler, device, cfg_train, logger):
    model.train()
    total = 0.0
    for step, batch in enumerate(loader, 1):
        x_t = batch["x_t"].to(device)
        v_t = batch["v_t"].to(device)
        x_tk = batch["x_tk"].to(device)
        v_tk = batch["v_tk"].to(device)
        atom_types = batch["atom_types"].to(device)
        mask = batch["mask"].to(device)

        with torch.autocast(device_type=device if device != "mps" else "cpu", enabled=bool(cfg_train.get("amp", True))):
            out = model(x_t, atom_types, mask)
            pred_x = x_t + out["delta_x"]
            pred_v = v_t + out["delta_v"]
            loss_pos = ((pred_x - x_tk).square() * mask[..., None]).sum() / mask.sum().clamp(min=1)
            loss_vel = ((pred_v - v_tk).square() * mask[..., None]).sum() / mask.sum().clamp(min=1)
            loss = loss_pos + float(cfg_train.get("lambda_vel", 0.5)) * loss_vel

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_train.get("grad_clip", 1.0)))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_train.get("grad_clip", 1.0)))
            opt.step()

        total += loss.item()
        if step % 50 == 0:
            logger.info(f"step {step}: loss {loss.item():.6f}")
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser(description="Train k-step jump model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    logger = get_logger()
    cfg_model = load_yaml(args.model)
    cfg_train = load_yaml(args.train)
    set_seed(int(cfg_train.get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(cfg_model).to(device)

    ds = KStepDataset(args.dataset)
    loader = DataLoader(ds, batch_size=int(cfg_train.get("batch_size", 32)), shuffle=True, num_workers=0, collate_fn=pad_collate)

    opt = optim.AdamW(model.parameters(), lr=float(cfg_train.get("lr", 2e-4)), weight_decay=float(cfg_train.get("weight_decay", 1e-2)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_train.get("amp", True)) and device == "cuda")

    epochs = int(cfg_train.get("epochs", 50))
    save_dir = Path(cfg_train.get("save_dir", "outputs")) / "ckpts"
    ensure_dir(save_dir)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, opt, scaler, device, cfg_train, logger)
        logger.info(f"epoch {epoch}: loss {loss:.6f}")
        torch.save({"model": model.state_dict(), "cfg": cfg_model}, save_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()

