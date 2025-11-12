"""QM hybrid integrator combining learned k-step jumps with QM corrector."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from utils import configure_logging, ensure_dir, load_yaml, remove_com, set_seed, LOGGER
from qm_engine import QMEngine, QMConfig

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "03_model.py"
_spec = importlib.util.spec_from_file_location("model_module", MODEL_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
build_model_from_config = _module.build_model_from_config


def load_model(checkpoint: Path, model_cfg: Dict, device: torch.device) -> torch.nn.Module:
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
    positions = data["positions"][frame]
    velocities = data["velocities"][frame]
    masses = data["masses"]
    atom_types = data["atom_types"] if "atom_types" in data else None
    atom_symbols = data["atom_symbols"].tolist() if "atom_symbols" in data else []
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    time_ps = data["time_ps"] if "time_ps" in data else None
    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "atom_types": atom_types,
        "atom_symbols": atom_symbols,
        "metadata": metadata,
        "time_ps": time_ps,
    }


def apply_qm_corrector(
    qm_engine: QMEngine,
    positions_nm: np.ndarray,
    velocities_nm_per_ps: np.ndarray,
    masses: np.ndarray,
    dt_fs: float,
    steps: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Apply QM corrector using velocity Verlet integration.
    
    Args:
        qm_engine: QM calculation engine
        positions_nm: Current positions in nm [N, 3]
        velocities_nm_per_ps: Current velocities in nm/ps [N, 3]
        masses: Atomic masses in amu [N]
        dt_fs: Time step in femtoseconds
        steps: Number of correction steps
        
    Returns:
        Dictionary with corrected positions, velocities, and energy
    """
    positions = positions_nm.copy()
    velocities = velocities_nm_per_ps.copy()
    
    dt_ps = dt_fs * 1e-3  # femtoseconds to picoseconds
    
    # Conversion factor: kJ/mol/nm to nm/ps^2 per amu
    # F = ma -> a = F/m
    # 1 kJ/mol = 1.6605e-21 J per molecule
    # 1 amu = 1.6605e-27 kg
    # So F (kJ/mol/nm) / m (amu) * conversion = acceleration (nm/ps^2)
    conversion_factor = 1.0 / 1.6605e-6  # Approximate conversion
    
    for _ in range(steps):
        # Calculate forces
        result = qm_engine.calculate(positions)
        forces = result["forces"]  # kJ/mol/nm
        
        # Velocity Verlet step
        # v(t+dt/2) = v(t) + (dt/2) * a(t)
        accelerations = forces / masses[:, None] * conversion_factor
        velocities_half = velocities + 0.5 * dt_ps * accelerations
        
        # x(t+dt) = x(t) + dt * v(t+dt/2)
        positions = positions + dt_ps * velocities_half
        
        # Calculate new forces
        result = qm_engine.calculate(positions)
        forces_new = result["forces"]
        
        # v(t+dt) = v(t+dt/2) + (dt/2) * a(t+dt)
        accelerations_new = forces_new / masses[:, None] * conversion_factor
        velocities = velocities_half + 0.5 * dt_ps * accelerations_new
    
    return {
        "positions": positions.astype(np.float32),
        "velocities": velocities.astype(np.float32),
        "energy": result["energy"],
        "forces": result["forces"].astype(np.float32),
        "converged": result["converged"],
    }


def _limit_vector_norm(delta: torch.Tensor, limit: float) -> torch.Tensor:
    """Limit vector norm to maximum value."""
    if limit <= 0 or delta.numel() == 0:
        return delta
    norms = torch.linalg.norm(delta, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    scale = torch.clamp(limit / norms, max=1.0)
    return delta * scale


def _is_stable(pos: np.ndarray, vel: np.ndarray, pos_threshold: float, vel_threshold: float) -> bool:
    """Check if positions and velocities are within acceptable bounds."""
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        return False
    if pos.size and float(np.abs(pos).max()) > pos_threshold:
        return False
    if vel.size and float(np.abs(vel).max()) > vel_threshold:
        return False
    return True


def run_qm_hybrid(
    model: torch.nn.Module,
    qm_engine: QMEngine,
    qm_npz: Path,
    frame: int,
    steps: int,
    k_steps: int,
    device: torch.device,
    dt_fs: float,
    max_delta_pos: float,
    max_delta_vel: float,
    delta_scale: float,
    max_attempts: int,
    pos_threshold: float,
    vel_threshold: float,
    corrector_steps: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Run hybrid QM integrator with learned k-step jumps and QM corrector.
    
    Args:
        model: Trained k-step prediction model
        qm_engine: QM calculation engine
        qm_npz: Path to QM trajectory NPZ file
        frame: Starting frame index
        steps: Number of macro-steps
        k_steps: Size of learned macro-step
        device: Torch device
        dt_fs: Time step in femtoseconds
        max_delta_pos: Maximum position delta (nm)
        max_delta_vel: Maximum velocity delta (nm/ps)
        delta_scale: Scale factor for deltas
        max_attempts: Maximum attempts before fallback
        pos_threshold: Position stability threshold (nm)
        vel_threshold: Velocity stability threshold (nm/ps)
        corrector_steps: Number of QM correction steps
        
    Returns:
        Dictionary with trajectory data
    """
    state = load_initial_state(qm_npz, frame)
    masses_np = state["masses"].astype(np.float32)
    positions_np, velocities_np = remove_com(state["positions"], state["velocities"], masses_np)
    positions = torch.from_numpy(positions_np).to(device)
    velocities = torch.from_numpy(velocities_np).to(device)
    masses = torch.from_numpy(masses_np).to(device)
    atom_types = torch.from_numpy(state["atom_types"]).long().to(device) if state["atom_types"] is not None else None
    
    if atom_types is None:
        # Create dummy atom types if not available
        atom_types = torch.zeros(len(positions), dtype=torch.long, device=device)
    
    total_nodes = positions.shape[0]
    batch_index = torch.zeros(total_nodes, dtype=torch.long, device=device)
    
    pos_records = [positions_np.astype(np.float32)]
    vel_records = [velocities_np.astype(np.float32)]
    energy_records = [np.nan]
    force_calls = 0
    
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
            delta_pos = _limit_vector_norm(delta_pos, max_delta_pos)
            delta_vel = _limit_vector_norm(delta_vel, max_delta_vel)
        
        delta_pos_np = delta_pos.detach().cpu().numpy()
        delta_vel_np = delta_vel.detach().cpu().numpy()
        current_pos_np = positions.detach().cpu().numpy()
        current_vel_np = velocities.detach().cpu().numpy()
        
        attempts = 0
        scale = delta_scale
        success = False
        correction = None
        centered_pos = None
        centered_vel = None
        
        while attempts < max_attempts:
            scaled_pos = current_pos_np + delta_pos_np * scale
            scaled_vel = current_vel_np + delta_vel_np * scale
            
            try:
                corr = apply_qm_corrector(
                    qm_engine,
                    scaled_pos,
                    scaled_vel,
                    masses_np,
                    dt_fs,
                    corrector_steps,
                )
                force_calls += corrector_steps
                centered_pos_tmp, centered_vel_tmp = remove_com(
                    corr["positions"], corr["velocities"], masses_np
                )
                
                if _is_stable(centered_pos_tmp, centered_vel_tmp, pos_threshold, vel_threshold):
                    success = True
                    correction = corr
                    centered_pos = centered_pos_tmp
                    centered_vel = centered_vel_tmp
                    break
            except Exception as e:
                LOGGER.warning(f"QM corrector failed at attempt {attempts + 1}: {e}")
            
            scale *= 0.5
            attempts += 1
        
        if not success:
            LOGGER.warning(f"Macro-step {step_idx + 1} diverged; falling back to baseline")
            try:
                corr = apply_qm_corrector(
                    qm_engine,
                    current_pos_np,
                    current_vel_np,
                    masses_np,
                    dt_fs,
                    corrector_steps,
                )
                force_calls += corrector_steps
                centered_pos, centered_vel = remove_com(
                    corr["positions"], corr["velocities"], masses_np
                )
                correction = corr
            except Exception as e:
                LOGGER.error(f"Fallback QM corrector also failed: {e}")
                raise
        
        positions = torch.from_numpy(centered_pos).to(device)
        velocities = torch.from_numpy(centered_vel).to(device)
        
        assert centered_pos is not None and centered_vel is not None and correction is not None
        pos_records.append(centered_pos.astype(np.float32))
        vel_records.append(centered_vel.astype(np.float32))
        energy_records.append(correction["energy"])
        
        LOGGER.info(f"Step {step_idx + 1}/{steps} complete (energy: {correction['energy']:.6f} Ha)")
    
    # Create time array
    dt_ps = dt_fs * k_steps * corrector_steps * 1e-3
    time_ps = np.arange(steps + 1, dtype=np.float32) * dt_ps
    
    return {
        "pos": np.stack(pos_records),
        "vel": np.stack(vel_records),
        "energy": np.asarray(energy_records),
        "force_calls": force_calls,
        "atom_types": state["atom_types"] if state["atom_types"] is not None else np.zeros(len(masses_np), dtype=np.int64),
        "masses": masses_np,
        "atom_symbols": state["atom_symbols"],
        "metadata": state["metadata"],
        "time_ps": time_ps,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--model-config", type=Path, required=True, help="Model YAML config")
    parser.add_argument("--qm-config", type=Path, required=True, help="QM config YAML")
    parser.add_argument("--initial-qm", type=Path, required=True, help="Initial QM trajectory NPZ")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--steps", type=int, default=100, help="Number of macro-steps")
    parser.add_argument("--k-steps", type=int, default=8, help="Size of learned macro-step")
    parser.add_argument("--dt-fs", type=float, default=0.5, help="Time step in fs")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--out", type=Path, required=True, help="Output NPZ path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-delta-pos", type=float, default=0.1, help="Max Δx (nm)")
    parser.add_argument("--max-delta-vel", type=float, default=2.0, help="Max Δv (nm/ps)")
    parser.add_argument("--delta-scale", type=float, default=0.5, help="Delta scale factor")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts before fallback")
    parser.add_argument("--pos-threshold", type=float, default=1.0, help="Position threshold (nm)")
    parser.add_argument("--vel-threshold", type=float, default=5.0, help="Velocity threshold (nm/ps)")
    parser.add_argument("--corrector-steps", type=int, default=1, help="QM correction steps")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    ensure_dir(args.out.parent)
    set_seed(args.seed)
    
    device = torch.device(args.device)
    model_cfg = load_yaml(args.model_config)
    model = load_model(args.checkpoint, model_cfg, device)
    
    qm_cfg = QMConfig.from_yaml(args.qm_config)
    
    # Load atom symbols from initial state
    initial_state = load_initial_state(args.initial_qm, args.frame)
    atom_symbols = initial_state["atom_symbols"]
    
    if not atom_symbols:
        raise ValueError("Atom symbols not found in initial QM trajectory")
    
    qm_engine = QMEngine(qm_cfg, atom_symbols)
    
    start = time.perf_counter()
    results = run_qm_hybrid(
        model,
        qm_engine,
        args.initial_qm,
        args.frame,
        args.steps,
        args.k_steps,
        device,
        args.dt_fs,
        args.max_delta_pos,
        args.max_delta_vel,
        args.delta_scale,
        args.max_attempts,
        args.pos_threshold,
        args.vel_threshold,
        args.corrector_steps,
    )
    wall_clock = time.perf_counter() - start
    
    baseline_force_calls = args.steps * args.k_steps
    savings = baseline_force_calls / max(results["force_calls"], 1)
    
    metadata = {
        "k_steps": args.k_steps,
        "steps": args.steps,
        "dt_fs": args.dt_fs,
        "wall_clock_s": wall_clock,
        "force_calls_hybrid": results["force_calls"],
        "force_calls_baseline": baseline_force_calls,
        "force_call_savings": savings,
        "corrector_steps": args.corrector_steps,
        "initial_metadata": results["metadata"],
    }
    
    np.savez(
        args.out,
        pos=results["pos"],
        vel=results["vel"],
        energy=results["energy"],
        atom_types=results["atom_types"],
        masses=results["masses"],
        atom_symbols=results["atom_symbols"],
        time_ps=results["time_ps"],
        metadata=json.dumps(metadata),
    )
    LOGGER.info(f"QM hybrid rollout saved to {args.out} (wall-clock {wall_clock:.2f}s, savings {savings:.2f}x)")


if __name__ == "__main__":
    main()

