"""Hybrid integrator: learned k-step jump + short OpenMM Langevin corrector.

Each macro-step covers ``k × save_interval`` baseline micro-steps of learned jump plus
``--corrector-steps`` real micro-steps, so hybrid frames are
``(k·S + c)·dt`` apart and are compared against every k-th baseline frame.

Predictors (``--predictor``):
  model      a trained checkpoint (``--checkpoint``; legacy 2025 checkpoints also need
             ``--model-config configs/legacy/...``)
  zero       control: no jump, corrector only
  ballistic  control: free flight Δx = v·T

Example::

    python src/06_hybrid_integrate.py --checkpoint outputs/v2/flow_k4/best.pt \\
        --md-config configs/md.yaml --molecule "data/raw/CC(C)CO" \\
        --initial-md "data/md/CC(C)CO/trajectory.npz" --steps 2500 \\
        --out "outputs/v2/hybrid/flow_k4/CC(C)CO.npz"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kstep.common import LOGGER, configure_logging, load_trajectory, load_yaml, set_seed
from kstep.hybrid import BallisticPredictor, CudaGraphPredictor, HybridOptions, ZeroPredictor, run_hybrid
from kstep.model import KStepModel, load_predictor
from kstep.openmm_corrector import OpenMMCorrector


def build_predictor(args, jump_time_ps: float, device: torch.device):
    if args.predictor == "zero":
        return ZeroPredictor(), {"predictor": "zero"}
    if args.predictor == "ballistic":
        return BallisticPredictor(jump_time_ps), {"predictor": "ballistic"}
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --predictor model")
    legacy_cfg = load_yaml(args.model_config) if args.model_config else None
    pred = load_predictor(args.checkpoint, device, legacy_model_config=legacy_cfg)
    info = {"predictor": "model", "checkpoint": str(args.checkpoint),
            "model_kind": getattr(pred.cfg, "kind", getattr(pred.cfg, "arch", "legacy"))}
    if isinstance(pred, KStepModel) and device.type == "cuda" and not args.no_cuda_graph:
        pred = CudaGraphPredictor(pred)
    return pred, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictor", choices=["model", "zero", "ballistic"], default="model")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-config", type=Path, default=None, help="Only for legacy (2025) checkpoints")
    p.add_argument("--md-config", type=Path, required=True)
    p.add_argument("--molecule", type=Path, required=True, help="data/raw/<mol> with structure.pdb + forcefield.xml")
    p.add_argument("--initial-md", type=Path, required=True, help="Baseline trajectory.npz for the initial state")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--steps", type=int, default=2500, help="Number of macro-steps")
    p.add_argument("--k-steps", type=int, default=None, help="Jump length in stored frames (default: from checkpoint, else 4)")
    p.add_argument("--corrector-steps", type=int, default=0, help="Corrector micro-steps per macro-step (0: use fraction)")
    p.add_argument("--corrector-fraction", type=float, default=0.05, help="Corrector steps as fraction of k·S")
    p.add_argument("--delta-scale", type=float, default=1.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-bond-strain", type=float, default=0.25)
    p.add_argument("--max-delta-pos", type=float, default=0.0, help="Hard cap on |Δx| (nm); legacy runs used 0.2")
    p.add_argument("--max-delta-vel", type=float, default=0.0, help="Hard cap on |Δv| (nm/ps); legacy runs used 4.5")
    p.add_argument("--uq-samples", type=int, default=1, help="Flow samples per step for the uncertainty estimate")
    p.add_argument("--uq-threshold", type=float, default=float("inf"), help="Escalate when Δx spread (nm) exceeds this")
    p.add_argument("--flow-steps", type=int, default=None, help="Override ODE steps of the flow sampler")
    p.add_argument("--platform", default=None, help="Override OpenMM platform")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    configure_logging()
    set_seed(args.seed)
    device = torch.device(args.device)

    md_cfg = load_yaml(args.md_config)
    save_interval = int(md_cfg.get("save_interval_steps", 50))
    dt_ps = float(md_cfg.get("dt_fs", 2.0)) * 1e-3

    k = args.k_steps
    if k is None and args.checkpoint is not None and args.predictor == "model":
        k = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("k_steps")
    k = int(k or 4)
    jump_steps = k * save_interval
    corrector_steps = args.corrector_steps or max(1, int(round(args.corrector_fraction * jump_steps)))
    jump_time_ps = jump_steps * dt_ps

    predictor, info = build_predictor(args, jump_time_ps, device)
    corrector = OpenMMCorrector(args.molecule, md_cfg, args.seed, args.platform)
    init = load_trajectory(args.initial_md)
    sampler_kwargs = {"steps": args.flow_steps} if args.flow_steps else {}
    opts = HybridOptions(
        jump_time_ps=jump_time_ps,
        corrector_steps=corrector_steps,
        delta_scale=args.delta_scale,
        max_attempts=args.max_attempts,
        max_bond_strain=args.max_bond_strain,
        max_delta_pos=args.max_delta_pos,
        max_delta_vel=args.max_delta_vel,
        uq_samples=args.uq_samples,
        uq_threshold=args.uq_threshold,
        sampler_kwargs=sampler_kwargs,
        log_every=max(args.steps // 10, 1),
    )
    LOGGER.info("%s | k=%d (%.3f ps jump) + %d corrector steps | %d macro-steps", info["predictor"], k,
                jump_time_ps, corrector_steps, args.steps)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    res = run_hybrid(predictor, corrector, init["pos"][args.frame], init["vel"][args.frame], init["masses"],
                     init["atom_types"], args.steps, opts, device, gen)

    meta = {**res["summary"], **info, "k_steps": k, "save_interval_steps": save_interval, "dt_fs": dt_ps * 1e3,
            "corrector_steps": corrector_steps, "jump_time_ps": jump_time_ps,
            "args": {key: str(v) for key, v in vars(args).items()},
            "initial_metadata": init.get("metadata", {})}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pos=res["pos"], vel=res["vel"], Ekin=res["Ekin"], Epot=res["Epot"], Etot=res["Etot"],
             time_ps=res["time_ps"], status=res["status"], attempts=res["attempts"], uq_spread=res["uq_spread"],
             atom_types=init["atom_types"], masses=init["masses"], metadata=json.dumps(meta))
    s = res["summary"]
    LOGGER.info("Saved %s | %.1f ps in %.1f s (model %.1f s, corrector %.1f s) | force-call savings %.1fx | "
                "accepted %.1f%%, escalated %.1f%%, fallback %.1f%%", args.out, s["simulated_ps"], s["wall_clock_s"],
                s["model_time_s"], s["corrector_time_s"], s["force_call_savings"], 100 * s["accepted_fraction"],
                100 * s["escalated_fraction"], 100 * s["fallback_fraction"])


if __name__ == "__main__":
    main()
