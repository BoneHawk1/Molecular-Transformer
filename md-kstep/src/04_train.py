"""Train an equivariant k-step model (deterministic ``mean`` or stochastic ``flow``).

Example::

    python src/04_train.py --dataset data/md/dataset_k4.npz \\
        --model-config configs/model_flow.yaml --train-config configs/train.yaml \\
        --splits data/splits/train.json data/splits/val.json \\
        --out outputs/v2/flow_k4

The whole dataset lives on the training device and batches are gathered there, so
there are no DataLoader workers. Per-element normalisation tables are fitted on the
training split and stored inside the checkpoint, which makes checkpoints
self-contained (``kstep.model.load_predictor`` needs no YAML).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List

import torch

from kstep.common import LOGGER, configure_logging, load_yaml, set_seed, write_json
from kstep.data import KStepData
from kstep.model import build_model, save_checkpoint


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    lr: float = 3e-4
    lr_min: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.0
    max_steps: int = 30000
    grad_clip: float = 5.0
    lambda_vel: float = 1.0
    ema_decay: float = 0.999
    val_every_steps: int = 1000
    max_val_batches: int = 50
    log_every_steps: int = 100
    bf16: bool = False
    wandb: Dict | None = None

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        ignored = sorted(set(data) - names)
        if ignored:
            LOGGER.warning("Ignoring train config keys not used by this trainer: %s", ", ".join(ignored))
        return cls(**{k: v for k, v in data.items() if k in names})


def load_split(path: Path) -> List[str]:
    return json.loads(Path(path).read_text())["molecules"]


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model, data: KStepData, cfg: TrainConfig, device) -> Dict[str, float]:
    """Validation loss with a fixed noise/time seed so values are comparable across steps."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    order_gen = torch.Generator().manual_seed(1234)
    totals = {"loss": 0.0, "loss_pos": 0.0, "loss_vel": 0.0}
    n = 0
    for batch in data.iterate(cfg.batch_size, shuffle=True, generator=order_gen, limit=cfg.max_val_batches):
        loss, parts = model.loss(batch, cfg.lambda_vel, generator=gen)
        totals["loss"] += float(loss)
        totals["loss_pos"] += parts["loss_pos"]
        totals["loss_vel"] += parts["loss_vel"]
        n += 1
    model.train()
    return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--splits", type=Path, nargs=2, required=True, help="train.json val.json")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt checkpoint")
    args = parser.parse_args()
    configure_logging()

    cfg = TrainConfig.from_dict(load_yaml(args.train_config))
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    model_cfg = load_yaml(args.model_config)
    set_seed(cfg.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train = KStepData(args.dataset, load_split(args.splits[0]), device)
    val_data = KStepData(args.dataset, load_split(args.splits[1]), device)
    LOGGER.info("k=%d | train %d samples (%s) | val %d samples (%s)", train.k_steps, len(train),
                ", ".join(train.molecules), len(val_data), ", ".join(val_data.molecules))

    model = build_model(model_cfg).to(device)
    model.set_normalization(train.normalization_tables(model.cfg.max_atomic_number))
    ema = copy.deepcopy(model).eval() if cfg.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("%s model with %s parameters", model.cfg.kind, f"{n_params:,}")

    start_step = 0
    best = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if ema is not None and payload.get("ema"):
            ema.load_state_dict(payload["ema"])
        opt.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        best = float(payload.get("best_val", best))
        LOGGER.info("Resumed from %s at step %d", args.resume, start_step)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    meta = {
        "k_steps": train.k_steps,
        "frame_dt_ps": train.frame_dt_ps,
        "dataset": str(args.dataset),
        "train_molecules": train.molecules,
        "val_molecules": val_data.molecules,
        "train_config": asdict(cfg),
    }
    write_json({"model_config": model_cfg, **meta, "n_params": n_params}, out / "run_config.json")

    wandb_run = None
    if cfg.wandb and cfg.wandb.get("enabled"):
        import wandb

        wandb_run = wandb.init(project=cfg.wandb.get("project", "md-kstep"), entity=cfg.wandb.get("entity"),
                               name=cfg.wandb.get("run_name") or out.name, config={**meta, "model": model_cfg})

    def checkpoint(path: Path, step: int, val_loss: float) -> None:
        save_checkpoint(path, model, ema=ema.state_dict() if ema is not None else None,
                        optimizer=opt.state_dict(), step=step, val_loss=val_loss, best_val=best, **meta)

    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    idx_gen = torch.Generator().manual_seed(cfg.seed)
    model.train()
    t0 = time.time()
    running: Dict[str, float] = {}
    for step in range(start_step, cfg.max_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        idx = torch.randint(len(train), (cfg.batch_size,), generator=idx_gen)
        batch = train.gather(idx)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16):
            loss, parts = model.loss(batch, cfg.lambda_vel, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        if not torch.isfinite(loss):
            LOGGER.warning("Non-finite loss at step %d; skipping update", step)
            continue
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - cfg.ema_decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

        for key, value in {"loss": loss.item(), **parts, "grad_norm": grad_norm.item()}.items():
            running[key] = running.get(key, 0.0) + value
        if (step + 1) % cfg.log_every_steps == 0:
            rec = {k: v / cfg.log_every_steps for k, v in running.items()}
            rec.update(step=step + 1, lr=opt.param_groups[0]["lr"], elapsed_min=(time.time() - t0) / 60)
            running = {}
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if wandb_run:
                wandb_run.log(rec, step=step + 1)
            if (step + 1) % (cfg.log_every_steps * 10) == 0:
                LOGGER.info("step %d | loss %.4f (pos %.4f vel %.4f) | %.1f min", step + 1, rec["loss"],
                            rec["loss_pos"], rec["loss_vel"], rec["elapsed_min"])

        if (step + 1) % cfg.val_every_steps == 0 or step + 1 == cfg.max_steps:
            metrics = evaluate(ema if ema is not None else model, val_data, cfg, device)
            metrics["step"] = step + 1
            with log_path.open("a") as fh:
                fh.write(json.dumps(metrics) + "\n")
            if wandb_run:
                wandb_run.log(metrics, step=step + 1)
            improved = metrics["val_loss"] < best
            if improved:
                best = metrics["val_loss"]
                checkpoint(out / "best.pt", step + 1, best)
            checkpoint(out / "last.pt", step + 1, metrics["val_loss"])
            LOGGER.info("step %d | val %.4f (pos %.4f vel %.4f)%s", step + 1, metrics["val_loss"],
                        metrics["val_loss_pos"], metrics["val_loss_vel"], " *" if improved else "")

    write_json({"best_val_loss": best, "steps": cfg.max_steps, "minutes": (time.time() - t0) / 60}, out / "summary.json")
    LOGGER.info("Done. Best val loss %.4f; checkpoints in %s", best, out)


if __name__ == "__main__":
    main()
