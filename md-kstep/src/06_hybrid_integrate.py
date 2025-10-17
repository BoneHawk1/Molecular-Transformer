"""Hybrid integrator combining learned k-step jumps with a one-step OpenMM corrector."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from openmm import (Platform, VerletIntegrator, XmlSerializer, unit)
from openmm.app import PDBFile, Simulation

from utils import configure_logging, ensure_dir, load_yaml, set_seed, LOGGER

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "03_model.py"
_spec = importlib.util.spec_from_file_location("model_module", MODEL_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _module  # ensure dataclasses resolve module references
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]
build_model_from_config = _module.build_model_from_config  # type: ignore[attr-defined]


@dataclass
class MDConfig:
    dt_fs: float
    temperature_K: float
    platform: str
    constraints: str
    random_seed: int
    nve_window_ps: float = 0.0
    nve_every_ps: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path) -> "MDConfig":
        cfg = load_yaml(path)
        return cls(
            dt_fs=cfg.get("dt_fs", 2.0),
            temperature_K=cfg.get("temperature_K", 300.0),
            platform=cfg.get("platform", "CUDA"),
            constraints=cfg.get("constraints", "HBonds"),
            random_seed=cfg.get("random_seed", 42),
            nve_window_ps=cfg.get("nve_window_ps", 0.0),
            nve_every_ps=cfg.get("nve_every_ps", 0.0),
        )


def load_model(checkpoint: Path, model_cfg: Dict, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(model_cfg)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data.get("model", checkpoint_data)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_initial_state(md_npz: Path, frame: int) -> Dict[str, np.ndarray]:
    data = np.load(md_npz, allow_pickle=True)
    positions = data["pos"][frame]
    velocities = data["vel"][frame]
    masses = data["masses"]
    atom_types = data["atom_types"]
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    time_ps = data["time_ps"] if "time_ps" in data else None
    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "atom_types": atom_types,
        "metadata": metadata,
        "time_ps": time_ps,
    }


def load_openmm_sim(molecule_dir: Path, md_cfg: MDConfig) -> Simulation:
    pdb_path = molecule_dir / "structure.pdb"
    xml_path = molecule_dir / "forcefield.xml"
    pdb = PDBFile(str(pdb_path))
    with xml_path.open("r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    integrator = VerletIntegrator(md_cfg.dt_fs * unit.femtosecond)
    platform = Platform.getPlatformByName(md_cfg.platform)
    sim = Simulation(pdb.topology, system, integrator, platform)
    return sim


def _load_openmm_bundle(molecule_dir: Path):
    pdb_path = molecule_dir / "structure.pdb"
    xml_path = molecule_dir / "forcefield.xml"
    pdb = PDBFile(str(pdb_path))
    with xml_path.open("r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    return pdb, system


def apply_corrector(simulation: Simulation, positions_nm: np.ndarray, velocities_nm_per_ps: np.ndarray) -> Dict[str, np.ndarray]:
    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.setVelocities(velocities_nm_per_ps * (unit.nanometer / unit.picosecond))
    simulation.step(1)
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


def _run_nve_window(molecule_dir: Path, md_cfg: MDConfig,
                    positions_nm: np.ndarray, velocities_nm_per_ps: np.ndarray,
                    steps: int) -> Dict[str, np.ndarray]:
    pdb, system = _load_openmm_bundle(molecule_dir)
    integrator = VerletIntegrator(md_cfg.dt_fs * unit.femtosecond)
    platform = Platform.getPlatformByName(md_cfg.platform)
    sim = Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(positions_nm * unit.nanometer)
    sim.context.setVelocities(velocities_nm_per_ps * (unit.nanometer / unit.picosecond))

    energies = []
    for _ in range(steps):
        state = sim.context.getState(getEnergy=True)
        energies.append([
            state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        ])
        sim.step(1)
    energies = np.asarray(energies, dtype=np.float64)
    return {
        "kinetic": energies[:, 0],
        "potential": energies[:, 1],
        "total": energies.sum(axis=1),
    }


def run_hybrid(model: torch.nn.Module, md_cfg: MDConfig, molecule_dir: Path, md_npz: Path, frame: int,
               steps: int, k_steps: int, device: torch.device) -> Dict[str, np.ndarray]:
    state = load_initial_state(md_npz, frame)
    positions = torch.from_numpy(state["positions"]).to(device)
    velocities = torch.from_numpy(state["velocities"]).to(device)
    masses = torch.from_numpy(state["masses"]).to(device)
    atom_types = torch.from_numpy(state["atom_types"]).long().to(device)

    simulation = load_openmm_sim(molecule_dir, md_cfg)
    total_nodes = positions.shape[0]
    batch_index = torch.zeros(total_nodes, dtype=torch.long, device=device)

    pos_records = [state["positions"].astype(np.float32)]
    vel_records = [state["velocities"].astype(np.float32)]
    Ekin = [np.nan]
    Epot = [np.nan]

    force_calls = 0
    nve_windows = []
    nve_window_steps = int((md_cfg.nve_window_ps * 1000) / md_cfg.dt_fs) if md_cfg.nve_window_ps > 0 else 0
    nve_every_steps = int((md_cfg.nve_every_ps * 1000) / md_cfg.dt_fs) if md_cfg.nve_every_ps > 0 else 0

    for step_idx in range(steps):
        with torch.no_grad():
            batch = {
                "x_t": positions,
                "v_t": velocities,
                "atom_types": atom_types,
                "masses": masses,
                "batch": batch_index,
            }
            outputs = model(batch)
            delta_pos = outputs["delta_pos"]
            delta_vel = outputs["delta_vel"]
            pred_pos = positions + delta_pos
            pred_vel = velocities + delta_vel

        correction = apply_corrector(simulation, pred_pos.detach().cpu().numpy(), pred_vel.detach().cpu().numpy())
        positions = torch.from_numpy(correction["positions"]).to(device)
        velocities = torch.from_numpy(correction["velocities"]).to(device)

        pos_records.append(correction["positions"])
        vel_records.append(correction["velocities"])
        Ekin.append(correction["Ekin"])
        Epot.append(correction["Epot"])
        force_calls += 1
        LOGGER.info("Step %d/%d complete", step_idx + 1, steps)

        # Optionally run an NVE window at configured cadence (measured in base steps)
        if nve_window_steps > 0 and nve_every_steps > 0:
            base_steps_elapsed = (step_idx + 1) * k_steps
            if base_steps_elapsed % nve_every_steps == 0 or step_idx == 0:
                win = _run_nve_window(
                    molecule_dir,
                    md_cfg,
                    correction["positions"],
                    correction["velocities"],
                    nve_window_steps,
                )
                nve_windows.append({k: v.tolist() for k, v in win.items()})

    Etot = [kin + pot if not np.isnan(kin) else np.nan for kin, pot in zip(Ekin, Epot)]

    baseline_time = state.get("time_ps")
    if baseline_time is not None and len(baseline_time) >= frame + steps + 1:
        time_ps = np.asarray(baseline_time[frame:frame + steps + 1], dtype=np.float32)
    else:
        dt_ps = md_cfg.dt_fs * k_steps * 1e-3
        time_ps = np.arange(steps + 1, dtype=np.float32) * dt_ps

    return {
        "pos": np.stack(pos_records),
        "vel": np.stack(vel_records),
        "Ekin": np.asarray(Ekin),
        "Epot": np.asarray(Epot),
        "Etot": np.asarray(Etot),
        "force_calls": force_calls,
        "atom_types": state["atom_types"],
        "masses": state["masses"],
        "metadata": state["metadata"],
        "time_ps": time_ps,
        "nve_windows": nve_windows,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--model-config", type=Path, required=True, help="Model YAML config")
    parser.add_argument("--md-config", type=Path, required=True, help="MD YAML config")
    parser.add_argument("--molecule", type=Path, required=True, help="Path to molecule raw directory (PDB/XML)")
    parser.add_argument("--initial-md", type=Path, required=True, help="Baseline trajectory NPZ for initial state")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--steps", type=int, default=100, help="Number of macro-steps to integrate")
    parser.add_argument("--k-steps", type=int, default=8, help="Size of the learned macro-step in base steps")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--out", type=Path, required=True, help="Output NPZ path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    ensure_dir(args.out.parent)
    set_seed(args.seed)

    device = torch.device(args.device)
    model_cfg = load_yaml(args.model_config)
    model = load_model(args.checkpoint, model_cfg, device)
    md_cfg = MDConfig.from_yaml(args.md_config)

    start = time.perf_counter()
    results = run_hybrid(model, md_cfg, args.molecule, args.initial_md, args.frame, args.steps, args.k_steps, device)
    wall_clock = time.perf_counter() - start

    baseline_force_calls = args.steps * args.k_steps
    savings = baseline_force_calls / max(results["force_calls"], 1)

    metadata = {
        "k_steps": args.k_steps,
        "steps": args.steps,
        "dt_fs": md_cfg.dt_fs,
        "wall_clock_s": wall_clock,
        "force_calls_hybrid": results["force_calls"],
        "force_calls_baseline": baseline_force_calls,
        "force_call_savings": savings,
        "initial_metadata": results["metadata"],
    }

    np.savez(
        args.out,
        pos=results["pos"],
        vel=results["vel"],
        Ekin=results["Ekin"],
        Epot=results["Epot"],
        Etot=results["Etot"],
        atom_types=results["atom_types"],
        masses=results["masses"],
        time_ps=results["time_ps"],
        metadata=json.dumps(metadata),
        nve_windows=json.dumps(results["nve_windows"]),
    )
    LOGGER.info("Hybrid rollout saved to %s (wall-clock %.2fs, savings %.2fx)", args.out, wall_clock, savings)


if __name__ == "__main__":
    main()
