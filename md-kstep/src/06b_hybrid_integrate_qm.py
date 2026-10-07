"""QM hybrid integrator: learned k-step jump + short xTB / PySCF velocity-Verlet corrector.

Uses the same predictor/corrector loop as the classical integrator
(:func:`kstep.hybrid.run_hybrid`), so acceptance checks, uncertainty-triggered
escalation and force-call accounting are identical. Note that every corrector call
from a new geometry costs ``corrector_steps + 1`` QM gradients (ASE must evaluate the
forces at the jumped-to geometry before the first Verlet step); this is included in
``force_calls``.

Example::

    python src/06b_hybrid_integrate_qm.py --checkpoint outputs/v2/qm_flow_k40/best.pt \\
        --qm-config configs/qm.yaml --qm-traj data/qm/ethanol/trajectory.npz \\
        --steps 200 --corrector-steps 2 --out outputs/v2/hybrid_qm/ethanol.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kstep.common import (LOGGER, ase_velocity_to_nm_per_ps, configure_logging, load_trajectory, load_yaml,
                          nm_per_ps_to_ase_velocity, set_seed)
from kstep.hybrid import CorrectorResult, CudaGraphPredictor, HybridOptions, ZeroPredictor, run_hybrid
from kstep.model import KStepModel, load_predictor

EV_TO_KJ_PER_MOL = 96.48533212


def make_calculator(qm_cfg: dict, backend: str):
    if backend == "pyscf":
        from pyscf_gpu_calculator import PySCFGPUCalculator

        return PySCFGPUCalculator(method=qm_cfg.get("method", "rhf"), basis=qm_cfg.get("basis", "sto-3g"),
                                  xc=qm_cfg.get("xc", "pbe"), charge=qm_cfg.get("charge", 0), spin=qm_cfg.get("spin", 0),
                                  use_gpu=qm_cfg.get("use_gpu", False), conv_tol=qm_cfg.get("conv_tol", 1e-8),
                                  max_cycle=qm_cfg.get("max_cycle", 100))
    from xtb.ase.calculator import XTB

    return XTB(method=qm_cfg.get("method", "GFN2-xTB"), charge=qm_cfg.get("charge", 0))


class ASECorrector:
    """Velocity-Verlet steps with a QM calculator (one calculator reused for all calls)."""

    extra_force_calls_per_run = 1

    def __init__(self, atomic_numbers: np.ndarray, qm_cfg: dict, backend: str) -> None:
        from ase import Atoms, units

        self.units = units
        self.atoms = Atoms(numbers=np.asarray(atomic_numbers), positions=np.zeros((len(atomic_numbers), 3)))
        self.atoms.calc = make_calculator(qm_cfg, backend)
        self.dt_fs = float(qm_cfg.get("dt_fs", 0.25))
        self.micro_dt_ps = self.dt_fs * 1e-3

    def run(self, positions_nm, velocities, steps: int) -> CorrectorResult:
        from ase.md.verlet import VelocityVerlet

        atoms = self.atoms
        atoms.set_positions(np.asarray(positions_nm) * 10.0)
        atoms.set_velocities(nm_per_ps_to_ase_velocity(velocities))
        try:
            VelocityVerlet(atoms, timestep=self.dt_fs * self.units.fs).run(int(steps))
            return CorrectorResult(
                positions=atoms.get_positions() / 10.0,
                velocities=ase_velocity_to_nm_per_ps(atoms.get_velocities()),
                ekin=atoms.get_kinetic_energy() * EV_TO_KJ_PER_MOL,
                epot=atoms.get_potential_energy() * EV_TO_KJ_PER_MOL,
            )
        except Exception as exc:  # SCF failures etc.
            LOGGER.warning("QM corrector failed: %s", exc)
            nan = np.full((len(atoms), 3), np.nan)
            return CorrectorResult(nan, nan, float("nan"), float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictor", choices=["model", "zero"], default="model",
                   help="'zero' with --k-steps 0 --corrector-steps N runs plain QM Verlet (timing reference)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-config", type=Path, default=None, help="Only for legacy (2025) checkpoints")
    p.add_argument("--qm-config", type=Path, required=True)
    p.add_argument("--qm-traj", type=Path, required=True, help="QM trajectory npz (initial state)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--steps", type=int, default=100, help="Macro-steps")
    p.add_argument("--k-steps", type=int, default=None, help="Jump length in stored frames (default: from checkpoint)")
    p.add_argument("--corrector-steps", type=int, default=1)
    p.add_argument("--qm-backend", choices=["xtb", "pyscf"], default=None, help="Default: 'backend' in the QM config, else xtb")
    p.add_argument("--delta-scale", type=float, default=1.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-bond-strain", type=float, default=0.2)
    p.add_argument("--max-epot-rise-kt", type=float, default=25.0,
                   help="Reject learned steps whose potential energy rises > this many kT above the start (0 = off)")
    p.add_argument("--uq-samples", type=int, default=1)
    p.add_argument("--uq-threshold", type=float, default=float("inf"))
    p.add_argument("--flow-steps", type=int, default=None)
    p.add_argument("--energy-rescale", action="store_true",
                   help="Rescale velocities after each learned step to keep the initial total energy (NVE reference)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    configure_logging()
    set_seed(args.seed)
    device = torch.device(args.device)

    qm_cfg = load_yaml(args.qm_config)
    backend = args.qm_backend or qm_cfg.get("backend", qm_cfg.get("qm_backend", "xtb"))
    init = load_trajectory(args.qm_traj)
    frame_dt_ps = float(np.median(np.diff(init["time_ps"]))) if len(init.get("time_ps", [])) > 1 else \
        float(qm_cfg.get("dt_fs", 0.25)) * int(qm_cfg.get("save_interval_steps", 1)) * 1e-3

    if args.predictor == "zero":
        predictor, info = ZeroPredictor(), {"predictor": "zero"}
        k = 4 if args.k_steps is None else int(args.k_steps)
    else:
        legacy_cfg = load_yaml(args.model_config) if args.model_config else None
        predictor = load_predictor(args.checkpoint, device, legacy_model_config=legacy_cfg)
        k = int(args.k_steps if args.k_steps is not None else (getattr(predictor, "k_steps", None) or 4))
        info = {"predictor": "model", "checkpoint": str(args.checkpoint)}
        if isinstance(predictor, KStepModel) and device.type == "cuda":
            predictor = CudaGraphPredictor(predictor)

    corrector = ASECorrector(init["atom_types"], qm_cfg, backend)
    jump_time_ps = k * frame_dt_ps
    opts = HybridOptions(jump_time_ps=jump_time_ps, corrector_steps=args.corrector_steps, delta_scale=args.delta_scale,
                         max_attempts=args.max_attempts, max_bond_strain=args.max_bond_strain,
                         uq_samples=args.uq_samples, uq_threshold=args.uq_threshold,
                         sampler_kwargs={"steps": args.flow_steps} if args.flow_steps else {},
                         log_every=max(args.steps // 10, 1), energy_rescale=args.energy_rescale,
                         max_epot_rise_kT=args.max_epot_rise_kt, temperature_K=float(qm_cfg.get("temperature_K", 300.0)))
    LOGGER.info("%s | %s corrector | k=%d (%.1f fs jump) + %d Verlet steps", info["predictor"], backend, k,
                jump_time_ps * 1e3, args.corrector_steps)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    res = run_hybrid(predictor, corrector, init["pos"][args.frame], init["vel"][args.frame], init["masses"],
                     init["atom_types"], args.steps, opts, device, gen)

    meta = {**res["summary"], **info, "qm_backend": backend, "k_steps": k, "jump_time_ps": jump_time_ps,
            "corrector_steps": args.corrector_steps, "velocity_units": "nm/ps",
            "args": {key: str(v) for key, v in vars(args).items()}, "initial_metadata": init.get("metadata", {})}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pos=res["pos"], vel=res["vel"], Ekin=res["Ekin"], Epot=res["Epot"], Etot=res["Etot"],
             time_ps=res["time_ps"], status=res["status"], attempts=res["attempts"], uq_spread=res["uq_spread"],
             atom_types=init["atom_types"], masses=init["masses"], metadata=json.dumps(meta))
    s = res["summary"]
    LOGGER.info("Saved %s | %.2f ps in %.1f s | QM gradients %d vs %d reference (%.1fx) | accepted %.1f%%",
                args.out, s["simulated_ps"], s["wall_clock_s"], s["force_calls"], s["force_calls_reference"],
                s["force_call_savings"], 100 * s["accepted_fraction"])


if __name__ == "__main__":
    main()
