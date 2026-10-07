"""Predictor/corrector loop shared by the classical (OpenMM) and QM (ASE) integrators.

One macro-step:

1. The predictor proposes (Δx, Δv) covering ``jump_time_ps`` of dynamics.
2. The proposal is checked (finite values, bond strain vs. the reference structure).
   Rejected proposals are redrawn (stochastic predictors) or halved (deterministic).
3. Optionally, with ``uq_samples > 1`` several proposals are drawn and their spread is
   used as an uncertainty estimate. Above ``uq_threshold`` the step *escalates*: the
   learned jump is discarded and the reference integrator runs the whole macro-step.
4. The corrector integrates ``corrector_steps`` micro-steps with real forces.

If every attempt is rejected the macro-step also falls back to the reference
integrator, so the frame spacing in time is always ``jump_time + corrector time`` and
the trajectory can be compared lag-for-lag with the baseline.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np
import torch

from .common import LOGGER, remove_com
from .geometry import bond_lengths, guess_bonds


@dataclass
class CorrectorResult:
    positions: np.ndarray
    velocities: np.ndarray
    ekin: float
    epot: float


class Corrector(Protocol):
    micro_dt_ps: float

    def run(self, positions_nm: np.ndarray, velocities: np.ndarray, steps: int) -> CorrectorResult:
        ...


class ZeroPredictor:
    """Control: no learned jump at all (the corrector alone)."""

    stochastic = False

    def predict(self, batch, generator=None, **_):
        return torch.zeros_like(batch["x_t"]), torch.zeros_like(batch["v_t"])


class BallisticPredictor:
    """Control: free flight, Δx = v·T, Δv = 0."""

    stochastic = False

    def __init__(self, jump_time_ps: float) -> None:
        self.jump_time_ps = jump_time_ps

    def predict(self, batch, generator=None, **_):
        return batch["v_t"] * self.jump_time_ps, torch.zeros_like(batch["v_t"])


class CudaGraphPredictor:
    """Replays :meth:`KStepModel.predict` from a captured CUDA graph.

    A hybrid rollout calls the model thousands of times with identical tensor shapes,
    and for small molecules the cost is almost entirely kernel-launch overhead
    (≈ 7 ms per call for the mean model, ≈ 100 ms for an 8-step Heun flow sampler).
    Capturing the whole sampler once per (atoms, replicas) shape removes that.
    """

    _KEYS = ("x_t", "v_t", "atom_types", "masses", "batch")

    def __init__(self, model) -> None:
        self.model = model
        self.cfg = model.cfg
        self.stochastic = getattr(model.cfg, "kind", None) == "flow"
        self._cache: Dict = {}

    def predict(self, batch, generator=None, noise_scale: float = 1.0, **kwargs):
        if batch["x_t"].device.type != "cuda":
            return self.model.predict(batch, generator=generator, noise_scale=noise_scale, **kwargs)
        key = (batch["x_t"].shape[0], int(batch["num_graphs"]), tuple(sorted(kwargs.items())))
        if key not in self._cache:
            self._cache[key] = self._capture(batch, kwargs)
        static, out, graph = self._cache[key]
        for k in self._KEYS:
            static[k].copy_(batch[k])
        if self.stochastic:
            static["nx"].normal_(generator=generator).mul_(noise_scale)
            static["nv"].normal_(generator=generator).mul_(noise_scale)
        graph.replay()
        return out[0].clone(), out[1].clone()

    def _capture(self, batch, kwargs):
        from .model import full_graph

        static = {k: batch[k].clone() for k in self._KEYS}
        static["num_graphs"] = int(batch["num_graphs"])
        static["nx"] = torch.randn_like(static["x_t"])
        static["nv"] = torch.randn_like(static["v_t"])
        edge_index = full_graph(static["batch"])
        noise = (static["nx"], static["nv"]) if self.stochastic else None

        def run():
            return self.model.predict(static, noise=noise, edge_index=edge_index, **kwargs)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        return static, out, graph


def is_stochastic(predictor) -> bool:
    if hasattr(predictor, "stochastic"):
        return bool(predictor.stochastic)
    cfg = getattr(predictor, "cfg", None)
    return getattr(cfg, "kind", None) == "flow"


@dataclass
class HybridOptions:
    jump_time_ps: float
    corrector_steps: int
    delta_scale: float = 1.0
    max_attempts: int = 3
    max_bond_strain: float = 0.25     # reject if any bond changes length by more than this fraction
    max_delta_pos: float = 0.0        # optional hard cap on |Δx| per atom (nm), 0 = off
    max_delta_vel: float = 0.0        # optional hard cap on |Δv| per atom (nm/ps), 0 = off
    uq_samples: int = 1
    uq_threshold: float = math.inf    # nm; RMS per-atom spread of Δx across samples
    sampler_kwargs: Dict = field(default_factory=dict)
    log_every: int = 0


def _cap(delta: torch.Tensor, limit: float) -> torch.Tensor:
    if limit <= 0:
        return delta
    norms = delta.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return delta * torch.clamp(limit / norms, max=1.0)


def _make_batch(pos, vel, z, masses, device, replicas: int = 1) -> Dict[str, torch.Tensor]:
    n = len(z)
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    batch = {
        "x_t": t(pos).repeat(replicas, 1),
        "v_t": t(vel).repeat(replicas, 1),
        "atom_types": t(z, torch.long).repeat(replicas),
        "masses": t(masses).repeat(replicas),
        "batch": torch.arange(replicas, device=device).repeat_interleave(n),
        "num_graphs": replicas,
    }
    return batch


def run_hybrid(
    predictor,
    corrector: Corrector,
    positions_nm: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    atom_types: np.ndarray,
    n_macro: int,
    opts: HybridOptions,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Dict:
    device = torch.device(device)
    masses = np.asarray(masses, dtype=np.float64)
    z = np.asarray(atom_types, dtype=np.int64)
    pos, vel = remove_com(np.asarray(positions_nm, np.float64), np.asarray(velocities, np.float64), masses)

    bonds = guess_bonds(pos, z)
    ref_len = bond_lengths(pos, bonds) if len(bonds) else np.zeros(0)
    stochastic = is_stochastic(predictor)
    micro_dt = corrector.micro_dt_ps
    full_steps = int(round(opts.jump_time_ps / micro_dt)) + opts.corrector_steps
    replicas = max(1, opts.uq_samples) if stochastic else 1

    T = n_macro + 1
    out_pos = np.zeros((T, len(z), 3), np.float32)
    out_vel = np.zeros_like(out_pos)
    ekin = np.full(T, np.nan)
    epot = np.full(T, np.nan)
    out_pos[0], out_vel[0] = pos, vel
    status = np.zeros(n_macro, dtype=np.int8)       # 0 accepted, 1 escalated (UQ), 2 fallback (rejected)
    attempts_used = np.zeros(n_macro, dtype=np.int16)
    spread = np.full(n_macro, np.nan)
    force_calls = 0
    t_model = 0.0
    t_corr = 0.0
    t_start = time.perf_counter()
    overhead = int(getattr(corrector, "extra_force_calls_per_run", 0))
    n_done = n_macro

    def acceptable(p: np.ndarray, v: np.ndarray) -> bool:
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
            return False
        if len(bonds) and opts.max_bond_strain > 0:
            strain = np.abs(bond_lengths(p, bonds) / ref_len - 1.0)
            if float(strain.max()) > opts.max_bond_strain:
                return False
        return True

    for step in range(n_macro):
        proposal = None
        attempt = 0
        scale = opts.delta_scale
        while attempt < max(1, opts.max_attempts):
            t0 = time.perf_counter()
            batch = _make_batch(pos, vel, z, masses, device, replicas)
            dx, dv = predictor.predict(batch, generator=generator, **opts.sampler_kwargs)
            dx = _cap(dx.float(), opts.max_delta_pos).view(replicas, len(z), 3)
            dv = _cap(dv.float(), opts.max_delta_vel).view(replicas, len(z), 3)
            if replicas > 1 and attempt == 0:
                spread[step] = float(dx.std(dim=0).norm(dim=-1).pow(2).mean().sqrt())
            dx0 = dx[0].double().cpu().numpy()
            dv0 = dv[0].double().cpu().numpy()
            t_model += time.perf_counter() - t0
            if replicas > 1 and spread[step] > opts.uq_threshold:
                break
            cand_p, cand_v = pos + scale * dx0, vel + scale * dv0
            attempt += 1
            if acceptable(cand_p, cand_v):
                proposal = (cand_p, cand_v)
                break
            if not stochastic:
                scale *= 0.5
        attempts_used[step] = attempt

        t0 = time.perf_counter()
        result = None
        if proposal is not None:
            result = corrector.run(proposal[0], proposal[1], opts.corrector_steps)
            force_calls += opts.corrector_steps + overhead
            if not acceptable(result.positions, result.velocities):
                result = None
        if result is None:
            status[step] = 1 if (replicas > 1 and spread[step] > opts.uq_threshold) else 2
            result = corrector.run(pos, vel, full_steps)
            force_calls += full_steps + overhead
            if not (np.all(np.isfinite(result.positions)) and np.all(np.isfinite(result.velocities))):
                LOGGER.error("Reference integrator failed at macro-step %d; truncating trajectory", step + 1)
                n_done = step
                break
        t_corr += time.perf_counter() - t0

        pos, vel = remove_com(np.asarray(result.positions, np.float64), np.asarray(result.velocities, np.float64), masses)
        out_pos[step + 1], out_vel[step + 1] = pos, vel
        ekin[step + 1], epot[step + 1] = result.ekin, result.epot
        if opts.log_every and (step + 1) % opts.log_every == 0:
            LOGGER.info("macro-step %d/%d | escalated %d | fallback %d", step + 1, n_macro,
                        int((status[:step + 1] == 1).sum()), int((status[:step + 1] == 2).sum()))

    wall = time.perf_counter() - t_start
    macro_ps = full_steps * micro_dt
    if n_done < n_macro:
        out_pos, out_vel = out_pos[:n_done + 1], out_vel[:n_done + 1]
        ekin, epot = ekin[:n_done + 1], epot[:n_done + 1]
        status, attempts_used, spread = status[:n_done], attempts_used[:n_done], spread[:n_done]
        T, n_macro = n_done + 1, n_done
    return {
        "pos": out_pos,
        "vel": out_vel,
        "Ekin": ekin,
        "Epot": epot,
        "Etot": ekin + epot,
        "time_ps": np.arange(T) * macro_ps,
        "status": status,
        "attempts": attempts_used,
        "uq_spread": spread,
        "summary": {
            "macro_steps": n_macro,
            "macro_step_ps": macro_ps,
            "simulated_ps": n_macro * macro_ps,
            "force_calls": int(force_calls),
            "force_calls_reference": int(n_macro * full_steps),
            "force_call_savings": float(n_macro * full_steps / max(force_calls, 1)),
            "accepted_fraction": float((status == 0).mean()) if n_macro else float("nan"),
            "escalated_fraction": float((status == 1).mean()) if n_macro else float("nan"),
            "fallback_fraction": float((status == 2).mean()) if n_macro else float("nan"),
            "mean_attempts": float(attempts_used.mean()) if n_macro else float("nan"),
            "wall_clock_s": wall,
            "model_time_s": t_model,
            "corrector_time_s": t_corr,
            "stochastic_predictor": stochastic,
            "uq_samples": replicas,
        },
    }
