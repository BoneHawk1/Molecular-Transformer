"""Model entry point: the equivariant k-step models live in :mod:`kstep.model`.

Run directly to print a parameter count and verify equivariance of a config::

    python src/03_model.py --model-config configs/model_flow.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kstep.common import load_yaml
from kstep.model import KStepModel, ModelConfig, build_model, load_predictor, save_checkpoint  # noqa: F401


def _random_rotation(gen: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen))
    return q * torch.sign(torch.diagonal(r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()
    model = build_model(load_yaml(args.model_config)).eval()
    n_params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(0)
    n = 12
    batch = {
        "x_t": torch.randn(n, 3, generator=gen) * 0.15,
        "v_t": torch.randn(n, 3, generator=gen),
        "atom_types": torch.tensor([6, 6, 8, 7] + [1] * (n - 4)),
        "masses": torch.tensor([12.0, 12.0, 16.0, 14.0] + [1.008] * (n - 4)),
        "batch": torch.zeros(n, dtype=torch.long),
    }
    rot = _random_rotation(gen)
    nx, nv = torch.randn(n, 3, generator=gen), torch.randn(n, 3, generator=gen)
    dx, dv = model.predict(batch, noise=(nx, nv))
    rb = dict(batch, x_t=batch["x_t"] @ rot.T, v_t=batch["v_t"] @ rot.T)
    rdx, rdv = model.predict(rb, noise=(nx @ rot.T, nv @ rot.T))
    err = max(((rdx - dx @ rot.T).norm() / dx.norm()).item(), ((rdv - dv @ rot.T).norm() / dv.norm()).item())
    print(f"{model.cfg.kind} model: {n_params:,} parameters, relative equivariance error {err:.2e}")


if __name__ == "__main__":
    main()
