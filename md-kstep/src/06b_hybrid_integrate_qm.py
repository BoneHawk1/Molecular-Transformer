"""Hybrid QM integrator combining learned k-step jumps with a one-step xTB corrector."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

try:
    from ase import Atoms, units
    from ase.calculators.xtb import XTB
    from ase.md.velocityverlet import VelocityVerlet
    from ase.md.langevin import Langevin
except ImportError as e:
    raise ImportError("QM hybrid integrator requires: pip install ase; conda install -c conda-forge xtb") from e

from utils import configure_logging, ensure_dir, load_yaml, remove_com, set_seed, LOGGER

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "03_model.py"
_spec = importlib.util.spec_from_file_location("model_module", MODEL_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]
build_model_from_config = _module.build_model_from_config  # type: ignore[attr-defined]


@dataclass
class QMConfig:
    dt_fs: float
    temperature_K: float
    friction_per_ps: float
    save_interval_steps: int
    method: str  # xTB method
    charge: int
    random_seed: int
    nve_window_ps: float = 0.0
    nve_every_ps: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path) -> "QMConfig":
        cfg = load_yaml(path)
        return cls(
            dt_fs=cfg.get("dt_fs", 0.25),
            temperature_K=cfg.get("temperature_K", 300.0),
            friction_per_ps=cfg.get("friction_per_ps", 0.002),
            save_interval_steps=int(cfg.get("save_interval_steps", 1)),
            method=cfg.get("method", "GFN2-xTB"),
            charge=cfg.get("charge", 0),
            random_seed=cfg.get("random_seed", 42),
            nve_window_ps=cfg.get("nve_window_ps", 0.0),
            nve_every_ps=cfg.get("nve_every_ps", 0.0),
        )


def load_model(checkpoint: Path, model_cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Load trained k-step model."""
    model = build_model_from_config(model_cfg)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data.get("model", checkpoint_data)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_initial_state(qm_npz: Path, frame: int) -> Dict[str, np.ndarray]:
    """Load initial state from QM trajectory."""
    data = np.load(qm_npz, allow_pickle=True)
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
        "atomic_numbers": atom_types,  # For QM, atom_types are atomic numbers
        "metadata": metadata,
        "time_ps": time_ps,
    }


def apply_qm_corrector(
    positions_nm: np.ndarray,
    velocities_nm_per_ps: np.ndarray,
    atomic_numbers: np.ndarray,
    qm_cfg: QMConfig,
    steps: int = 1,
) -> Dict[str, np.ndarray]:
    """Apply xTB one-step corrector."""
    # Convert units: nm -> Angstrom, nm/ps -> Angstrom/fs
    positions_A = positions_nm * 10.0
    velocities_A_per_fs = velocities_nm_per_ps * 0.01
    
    # Create ASE Atoms object
    atoms = Atoms(numbers=atomic_numbers, positions=positions_A)
    atoms.calc = XTB(method=qm_cfg.method, charge=qm_cfg.charge)
    atoms.set_velocities(velocities_A_per_fs)
    
    # One corrector step using Velocity Verlet
    dyn = VelocityVerlet(atoms, timestep=qm_cfg.dt_fs * units.fs)
    
    try:
        dyn.run(steps)
        
        # Extract corrected state
        corrected_pos_nm = atoms.get_positions() / 10.0  # Angstrom -> nm
        corrected_vel_nm_per_ps = atoms.get_velocities() * 100.0  # Angstrom/fs -> nm/ps
        
        # Get energies
        kinetic = atoms.get_kinetic_energy() * 96.4853  # eV -> kJ/mol
        potential = atoms.get_potential_energy() * 96.4853  # eV -> kJ/mol
        
        return {
            "positions": corrected_pos_nm.astype(np.float32),
            "velocities": corrected_vel_nm_per_ps.astype(np.float32),
            "Ekin": kinetic,
            "Epot": potential,
        }
    except Exception as e:
        LOGGER.error("QM corrector step failed: %s", e)
        # Return input state if corrector fails
        return {
            "positions": positions_nm.astype(np.float32),
            "velocities": velocities_nm_per_ps.astype(np.float32),
            "Ekin": float('nan'),
            "Epot": float('nan'),
        }


def _run_nve_window(
    positions_nm: np.ndarray,
    velocities_nm_per_ps: np.ndarray,
    atomic_numbers: np.ndarray,
    qm_cfg: QMConfig,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Run NVE window for energy conservation analysis."""
    positions_A = positions_nm * 10.0
    velocities_A_per_fs = velocities_nm_per_ps * 0.01
    
    atoms = Atoms(numbers=atomic_numbers, positions=positions_A)
    atoms.calc = XTB(method=qm_cfg.method, charge=qm_cfg.charge)
    atoms.set_velocities(velocities_A_per_fs)
    
    dyn = VelocityVerlet(atoms, timestep=qm_cfg.dt_fs * units.fs)
    
    energies = []
    for _ in range(steps):
        try:
            kinetic = atoms.get_kinetic_energy() * 96.4853
            potential = atoms.get_potential_energy() * 96.4853
            energies.append([kinetic, potential])
            dyn.run(1)
        except Exception as e:
            LOGGER.warning("NVE step failed: %s", e)
            break
    
    if not energies:
        return {
            "kinetic": np.array([]),
            "potential": np.array([]),
            "total": np.array([]),
        }
    
    energies = np.asarray(energies, dtype=np.float64)
    return {
        "kinetic": energies[:, 0],
        "potential": energies[:, 1],
        "total": energies.sum(axis=1),
    }


def _limit_vector_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Limit per-atom vector norm for stability."""
    norms = torch.linalg.norm(vec, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-12), max=1.0)
    return vec * scale


def _is_stable(pos: np.ndarray, vel: np.ndarray, pos_threshold: float, vel_threshold: float) -> bool:
    """Check if state is physically reasonable."""
    if np.isnan(pos).any() or np.isnan(vel).any():
        return False
    pos_max = np.abs(pos).max()
    vel_max = np.abs(vel).max()
    return pos_max < pos_threshold and vel_max < vel_threshold


def run_hybrid_qm(
    model: torch.nn.Module,
    qm_cfg: QMConfig,
    qm_npz: Path,
    frame: int,
    steps: int,
    k_steps: int,
    device: torch.device,
    max_delta_pos: float,
    max_delta_vel: float,
    delta_scale: float,
    max_attempts: int,
    pos_threshold: float,
    vel_threshold: float,
) -> Dict[str, np.ndarray]:
    """Run hybrid QM integration: k-step ML jump + xTB corrector."""
    
    # Load initial state
    state = load_initial_state(qm_npz, frame)
    positions_np = state["positions"]
    velocities_np = state["velocities"]
    masses_np = state["masses"]
    atomic_numbers = state["atomic_numbers"]
    
    # Center COM
    positions_np, velocities_np = remove_com(positions_np, velocities_np, masses_np)
    
    # Prepare for model
    positions = torch.from_numpy(positions_np).float().to(device)
    velocities = torch.from_numpy(velocities_np).float().to(device)
    atom_types = torch.from_numpy(atomic_numbers).long().to(device)
    masses = torch.from_numpy(masses_np).float().to(device)
    batch_index = torch.zeros(len(positions), dtype=torch.long, device=device)
    
    # Storage
    n_atoms = len(positions_np)
    trajectory_pos = np.zeros((steps + 1, n_atoms, 3), dtype=np.float32)
    trajectory_vel = np.zeros_like(trajectory_pos)
    trajectory_Ekin = np.zeros(steps + 1, dtype=np.float64)
    trajectory_Epot = np.zeros_like(trajectory_Ekin)
    
    trajectory_pos[0] = positions_np
    trajectory_vel[0] = velocities_np
    
    # Initial energy
    try:
        atoms_init = Atoms(numbers=atomic_numbers, positions=positions_np * 10.0)
        atoms_init.calc = XTB(method=qm_cfg.method, charge=qm_cfg.charge)
        atoms_init.set_velocities(velocities_np * 0.01)
        trajectory_Ekin[0] = atoms_init.get_kinetic_energy() * 96.4853
        trajectory_Epot[0] = atoms_init.get_potential_energy() * 96.4853
    except Exception:
        trajectory_Ekin[0] = float('nan')
        trajectory_Epot[0] = float('nan')
    
    force_calls = 0
    failed_steps = 0
    
    LOGGER.info("Running hybrid QM integration: %d steps, k=%d", steps, k_steps)
    start_time = time.time()
    
    for step_idx in range(steps):
        # ML k-step prediction
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
            
            # Limit displacements for stability
            delta_pos = _limit_vector_norm(delta_pos, max_delta_pos)
            delta_vel = _limit_vector_norm(delta_vel, max_delta_vel)
        
        delta_pos_np = delta_pos.detach().cpu().numpy()
        delta_vel_np = delta_vel.detach().cpu().numpy()
        current_pos_np = positions.detach().cpu().numpy()
        current_vel_np = velocities.detach().cpu().numpy()
        
        # Adaptive scaling with retries
        attempts = 0
        scale = delta_scale
        success = False
        correction = None
        centered_pos = None
        centered_vel = None
        
        while attempts < max_attempts:
            scaled_pos = current_pos_np + delta_pos_np * scale
            scaled_vel = current_vel_np + delta_vel_np * scale
            
            # Apply xTB corrector
            corr = apply_qm_corrector(
                scaled_pos,
                scaled_vel,
                atomic_numbers,
                qm_cfg,
                steps=1,
            )
            force_calls += 1
            
            # Center COM
            centered_pos_tmp, centered_vel_tmp = remove_com(
                corr["positions"], corr["velocities"], masses_np
            )
            
            if _is_stable(centered_pos_tmp, centered_vel_tmp, pos_threshold, vel_threshold):
                success = True
                correction = corr
                centered_pos = centered_pos_tmp
                centered_vel = centered_vel_tmp
                break
            
            scale *= 0.5
            attempts += 1
        
        if not success:
            LOGGER.warning("Step %d failed after %d attempts, using last valid state", step_idx, max_attempts)
            failed_steps += 1
            # Use previous state
            centered_pos = current_pos_np
            centered_vel = current_vel_np
            correction = {"Ekin": float('nan'), "Epot": float('nan')}
        
        # Update state
        positions = torch.from_numpy(centered_pos).float().to(device)
        velocities = torch.from_numpy(centered_vel).float().to(device)
        
        trajectory_pos[step_idx + 1] = centered_pos
        trajectory_vel[step_idx + 1] = centered_vel
        trajectory_Ekin[step_idx + 1] = correction["Ekin"]
        trajectory_Epot[step_idx + 1] = correction["Epot"]
        
        # Progress logging
        if (step_idx + 1) % max(1, steps // 10) == 0:
            elapsed = time.time() - start_time
            LOGGER.info("Progress: %d/%d steps (%.1f%%), %.1f s", 
                       step_idx + 1, steps, 100 * (step_idx + 1) / steps, elapsed)
    
    elapsed_total = time.time() - start_time
    LOGGER.info("Hybrid QM integration complete: %d steps, %d force calls, %d failures, %.2f s",
               steps, force_calls, failed_steps, elapsed_total)
    
    time_ps = np.arange(steps + 1) * (k_steps * qm_cfg.dt_fs / 1000.0)  # Convert fs to ps
    
    return {
        "pos": trajectory_pos,
        "vel": trajectory_vel,
        "Ekin": trajectory_Ekin,
        "Epot": trajectory_Epot,
        "Etot": trajectory_Ekin + trajectory_Epot,
        "time_ps": time_ps,
        "force_calls": force_calls,
        "failed_steps": failed_steps,
        "k_steps": k_steps,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained k-step model checkpoint")
    parser.add_argument("--model-config", type=Path, required=True, help="Model config YAML")
    parser.add_argument("--qm-config", type=Path, required=True, help="QM config YAML")
    parser.add_argument("--qm-traj", type=Path, required=True, help="QM trajectory npz (for initial state)")
    parser.add_argument("--out", type=Path, required=True, help="Output npz file")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame from QM trajectory")
    parser.add_argument("--steps", type=int, default=100, help="Number of hybrid integration steps")
    parser.add_argument("--k-steps", type=int, default=4, help="k-step horizon (must match training)")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--max-delta-pos", type=float, default=0.05, help="Max position delta per atom (nm)")
    parser.add_argument("--max-delta-vel", type=float, default=5.0, help="Max velocity delta per atom (nm/ps)")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="Initial scaling for ML predictions")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max retry attempts per step")
    parser.add_argument("--pos-threshold", type=float, default=2.0, help="Max position magnitude (nm)")
    parser.add_argument("--vel-threshold", type=float, default=50.0, help="Max velocity magnitude (nm/ps)")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    
    # Load configs
    model_cfg = load_yaml(args.model_config)
    qm_cfg = QMConfig.from_yaml(args.qm_config)
    set_seed(qm_cfg.random_seed)
    
    device = torch.device(args.device)
    
    # Load model
    LOGGER.info("Loading model from %s", args.checkpoint)
    model = load_model(args.checkpoint, model_cfg, device)
    
    # Run hybrid integration
    result = run_hybrid_qm(
        model=model,
        qm_cfg=qm_cfg,
        qm_npz=args.qm_traj,
        frame=args.frame,
        steps=args.steps,
        k_steps=args.k_steps,
        device=device,
        max_delta_pos=args.max_delta_pos,
        max_delta_vel=args.max_delta_vel,
        delta_scale=args.delta_scale,
        max_attempts=args.max_attempts,
        pos_threshold=args.pos_threshold,
        vel_threshold=args.vel_threshold,
    )
    
    # Save trajectory
    ensure_dir(Path(args.out).parent)
    
    # Load metadata from input
    input_data = np.load(args.qm_traj, allow_pickle=True)
    masses = input_data["masses"]
    atom_types = input_data["atom_types"]
    metadata_str = str(input_data["metadata"]) if "metadata" in input_data else "{}"
    metadata = json.loads(metadata_str)
    metadata["hybrid_qm"] = True
    metadata["k_steps"] = args.k_steps
    metadata["force_calls"] = int(result["force_calls"])
    metadata["failed_steps"] = int(result["failed_steps"])
    
    np.savez(
        args.out,
        pos=result["pos"],
        vel=result["vel"],
        Ekin=result["Ekin"],
        Epot=result["Epot"],
        Etot=result["Etot"],
        time_ps=result["time_ps"],
        masses=masses,
        atom_types=atom_types,
        metadata=json.dumps(metadata),
        k_steps=result["k_steps"],
    )
    
    LOGGER.info("Saved hybrid QM trajectory to %s", args.out)


if __name__ == "__main__":
    main()

