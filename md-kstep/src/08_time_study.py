"""Time study comparing baseline MD, EGNN hybrid, and Transformer hybrid.

For each molecule:
- Measure baseline OpenMM wall-clock via a short run, estimate full-run time
- Read hybrid runs (EGNN + Transformer) and their wall-clock from metadata
- Compute per-step bond-length RMSE vs baseline for EGNN and Transformer

Outputs
- JSON summary with times and speedups per molecule
- Bar chart of wall-clock per method per molecule (single figure)
- Line chart of accuracy (bond RMSE) per time step for EGNN vs Transformer
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import configure_logging, ensure_dir, load_yaml, remove_com, LOGGER
from openmm import (LangevinIntegrator, Platform, XmlSerializer, unit)
from openmm.app import PDBFile, Simulation


class MDConfig:
    def __init__(self, cfg: Dict):
        self.dt_fs = cfg.get("dt_fs", 2.0)
        self.temperature_K = cfg.get("temperature_K", 300.0)
        self.friction_per_ps = cfg.get("friction_per_ps", 1.0)
        self.save_interval_steps = int(cfg.get("save_interval_steps", 50))
        self.platform = cfg.get("platform", "CUDA")
        self.constraints = cfg.get("constraints", "HBonds")
        self.random_seed = int(cfg.get("random_seed", 42))
        self.nve_window_ps = float(cfg.get("nve_window_ps", 0.0))
        self.nve_every_ps = float(cfg.get("nve_every_ps", 0.0))


def _load_openmm_bundle(molecule_dir: Path):
    pdb_path = molecule_dir / "structure.pdb"
    xml_path = molecule_dir / "forcefield.xml"
    pdb = PDBFile(str(pdb_path))
    with xml_path.open("r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    return pdb, system


def load_openmm_sim(molecule_dir: Path, md_cfg: MDConfig) -> Simulation:
    pdb, system = _load_openmm_bundle(molecule_dir)
    integrator = LangevinIntegrator(
        md_cfg.temperature_K * unit.kelvin,
        md_cfg.friction_per_ps / unit.picosecond,
        md_cfg.dt_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(md_cfg.random_seed)
    platform = Platform.getPlatformByName(md_cfg.platform)
    return Simulation(pdb.topology, system, integrator, platform)


def apply_corrector(simulation: Simulation, positions_nm: np.ndarray, velocities_nm_per_ps: np.ndarray, steps: int) -> Dict[str, np.ndarray]:
    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.setVelocities(velocities_nm_per_ps * (unit.nanometer / unit.picosecond))
    simulation.context.applyConstraints(1e-6)
    simulation.context.applyVelocityConstraints(1e-6)
    micro_steps = max(1, int(steps))
    simulation.step(micro_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    corrected_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    corrected_vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return {
        "positions": corrected_pos.astype(np.float32),
        "velocities": corrected_vel.astype(np.float32),
        "Ekin": kinetic,
        "Epot": potential,
    }


def load_npz(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    nve_windows = json.loads(str(data.get("nve_windows", "[]")))
    return {
        "pos": data["pos"],
        "vel": data["vel"],
        "time_ps": data.get("time_ps"),
        "atom_types": data["atom_types"],
        "masses": data["masses"],
        "metadata": metadata,
        "nve_windows": nve_windows,
    }


def guess_bonds(positions_nm: np.ndarray, atom_types: np.ndarray, tol: float = 0.1) -> List[Tuple[int, int]]:
    # Å radii
    radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02}
    coords = positions_nm * 10.0
    n = coords.shape[0]
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ri = radii.get(int(atom_types[i]), 0.75)
            rj = radii.get(int(atom_types[j]), 0.75)
            cutoff = ri + rj + tol
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                bonds.append((i, j))
    return bonds


def measure_bonds(positions_nm: np.ndarray, bonds: List[Tuple[int, int]]) -> np.ndarray:
    coords = positions_nm * 10.0
    return np.asarray([np.linalg.norm(coords[i] - coords[j]) for i, j in bonds], dtype=np.float32)


def bond_rmse_per_step(baseline_pos: np.ndarray, hybrid_pos: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
    # Build bonds from initial baseline frame
    bonds = guess_bonds(baseline_pos[0], atom_types)
    T = min(len(baseline_pos), len(hybrid_pos))
    rmses: List[float] = []
    for t in range(T):
        b_bonds = measure_bonds(baseline_pos[t], bonds) if bonds else np.array([])
        h_bonds = measure_bonds(hybrid_pos[t], bonds) if bonds else np.array([])
        if b_bonds.size == 0:
            rmses.append(np.nan)
        else:
            rmses.append(float(np.sqrt(np.mean((b_bonds - h_bonds) ** 2))))
    return np.asarray(rmses, dtype=np.float32)


def estimate_baseline_time(molecule_dir: Path, md_cfg: Path, initial_npz: Path, steps: int, k_steps: int, device: str) -> Tuple[float, Dict]:
    # Short baseline run to estimate time per micro-step, then scale
    # Load md.yaml
    md_data = load_yaml(md_cfg)
    cfg = MDConfig(md_data)
    # baseline micro-steps per macro-step
    baseline_steps = max(1, int(k_steps * cfg.save_interval_steps))
    short_macros = max(2, min(5, steps))

    # Minimal loader for initial state
    data = np.load(initial_npz, allow_pickle=True)
    positions = data["pos"][0]
    velocities = data["vel"][0]
    masses = data["masses"]
    state = {"positions": positions, "velocities": velocities, "masses": masses}
    masses_np = state["masses"].astype(np.float32)
    positions_np, velocities_np = remove_com(state["positions"], state["velocities"], masses_np)

    sim = load_openmm_sim(molecule_dir, cfg)
    # Set CUDA or CPU via platform in md config; device arg unused here, but kept for symmetry

    pos = positions_np
    vel = velocities_np
    start = time.perf_counter()
    total_calls = 0
    for _ in range(short_macros):
        corr = apply_corrector(sim, pos, vel, baseline_steps)
        total_calls += baseline_steps
        pos, vel = remove_com(corr["positions"], corr["velocities"], masses_np)
    short_wall = time.perf_counter() - start
    per_call = short_wall / max(total_calls, 1)
    est_full = per_call * (steps * baseline_steps)
    meta = {
        "short_macros": short_macros,
        "baseline_steps": baseline_steps,
        "short_wall_s": short_wall,
        "per_micro_step_s": per_call,
        "est_full_wall_s": est_full,
        "total_micro_steps_full": steps * baseline_steps,
    }
    return est_full, meta


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, default=Path("data/md"))
    p.add_argument("--egnn-root", type=Path, default=Path("outputs/hybrid_cf05_long"))
    p.add_argument("--transformer-root", type=Path, default=Path("outputs/hybrid_transformer_cf05_long"))
    p.add_argument("--md-config", type=Path, default=Path("configs/md.yaml"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--k-steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/time_study"))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    ensure_dir(args.out_dir)

    # Collect molecules (names) from baseline root
    molecules = [d.name for d in sorted(args.baseline_root.iterdir()) if (d / "trajectory.npz").exists()]
    LOGGER.info("Found %d molecules", len(molecules))

    summary: Dict[str, Dict] = {}
    egnn_curves: List[np.ndarray] = []
    tr_curves: List[np.ndarray] = []

    for mol in molecules:
        LOGGER.info("Processing %s", mol)
        base_traj_path = args.baseline_root / mol / "trajectory.npz"
        egnn_path = args.egnn_root / f"{mol}.npz"
        tr_path = args.transformer_root / f"{mol}.npz"

        if not egnn_path.exists() or not tr_path.exists():
            LOGGER.warning("Missing hybrids for %s; skipping", mol)
            continue

        # Estimate baseline wall-clock via short run
        # OpenMM system lives under data/raw/<mol>
        est_time, baseline_meta = estimate_baseline_time(Path("data/raw") / mol, args.md_config, base_traj_path, args.steps, args.k_steps, args.device)

        egnn = load_npz(egnn_path)
        tr = load_npz(tr_path)
        egnn_time = float(egnn["metadata"].get("wall_clock_s", np.nan))
        tr_time = float(tr["metadata"].get("wall_clock_s", np.nan))

        base = load_npz(base_traj_path)
        atom_types = base["atom_types"]
        # Align lengths
        T = min(len(base["pos"]), len(egnn["pos"]))
        egnn_curve = bond_rmse_per_step(base["pos"][:T], egnn["pos"][:T], atom_types)
        T2 = min(len(base["pos"]), len(tr["pos"]))
        tr_curve = bond_rmse_per_step(base["pos"][:T2], tr["pos"][:T2], atom_types)

        egnn_curves.append(egnn_curve)
        tr_curves.append(tr_curve)

        summary[mol] = {
            "time": {
                "baseline_est_s": est_time,
                "egnn_s": egnn_time,
                "transformer_s": tr_time,
                "speedup_egnn": float(est_time / max(egnn_time, 1e-6)),
                "speedup_transformer": float(est_time / max(tr_time, 1e-6)),
            },
            "accuracy": {
                "bond_rmse_per_step_egnn": egnn_curve.tolist(),
                "bond_rmse_per_step_transformer": tr_curve.tolist(),
            },
            "baseline_meta": baseline_meta,
        }

    # Save summary JSON
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: time bar chart per molecule
    mols = list(summary.keys())
    if mols:
        x = np.arange(len(mols))
        width = 0.25
        base_times = np.array([summary[m]["time"]["baseline_est_s"] for m in mols], dtype=float)
        e_times = np.array([summary[m]["time"]["egnn_s"] for m in mols], dtype=float)
        t_times = np.array([summary[m]["time"]["transformer_s"] for m in mols], dtype=float)

        plt.figure(figsize=(max(10, len(mols) * 0.7), 5))
        plt.bar(x - width, base_times, width=width, label="Baseline (est)")
        plt.bar(x, e_times, width=width, label="EGNN")
        plt.bar(x + width, t_times, width=width, label="Transformer")
        plt.xticks(x, mols, rotation=60, ha="right")
        plt.ylabel("Wall-clock time (s)")
        plt.title("Per-molecule wall-clock times (50 macro-steps, k=4, 100 ps windows)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(args.out_dir / "time_bar_methods.png", dpi=200)
        plt.close()

        # Plot: accuracy per step (bond RMSE) averaged across molecules
        if egnn_curves and tr_curves:
            # Pad curves to common length with NaN and compute mean over valid entries
            max_T = max(cur.shape[0] for cur in egnn_curves)
            def pad_to(arrs: List[np.ndarray], T: int) -> np.ndarray:
                padded = np.full((len(arrs), T), np.nan, dtype=np.float32)
                for i, a in enumerate(arrs):
                    padded[i, : a.shape[0]] = a
                return padded

            e_stack = pad_to(egnn_curves, max_T)
            t_stack = pad_to(tr_curves, max_T)
            e_mean = np.nanmean(e_stack, axis=0)
            e_std = np.nanstd(e_stack, axis=0)
            t_mean = np.nanmean(t_stack, axis=0)
            t_std = np.nanstd(t_stack, axis=0)

            steps_axis = np.arange(max_T)
            plt.figure(figsize=(8, 4))
            plt.plot(steps_axis, e_mean, label="EGNN")
            plt.fill_between(steps_axis, e_mean - e_std, e_mean + e_std, alpha=0.2)
            plt.plot(steps_axis, t_mean, label="Transformer")
            plt.fill_between(steps_axis, t_mean - t_std, t_mean + t_std, alpha=0.2)
            plt.xlabel("Macro-step index")
            plt.ylabel("Bond RMSE (Å)")
            plt.title("Accuracy per time step vs baseline (mean ± std across molecules)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.legend()
            plt.savefig(args.out_dir / "accuracy_per_step.png", dpi=200)
            plt.close()

    LOGGER.info("Saved time study to %s", args.out_dir)


if __name__ == "__main__":
    main()
