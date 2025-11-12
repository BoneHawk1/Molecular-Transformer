"""QM validation metrics for energy conservation and structural accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import configure_logging, load_yaml, LOGGER


def compute_energy_drift(energies: np.ndarray, time_ps: np.ndarray) -> Dict[str, float]:
    """
    Compute energy drift metrics.
    
    Args:
        energies: Energy values in Hartree [T]
        time_ps: Time points in picoseconds [T]
        
    Returns:
        Dictionary with drift metrics
    """
    if len(energies) < 2:
        return {"drift_per_ps": np.nan, "total_drift": np.nan, "relative_drift": np.nan}
    
    # Convert to kcal/mol for reporting
    energies_kcal = energies * 627.509  # Hartree to kcal/mol
    
    total_time = time_ps[-1] - time_ps[0]
    if total_time <= 0:
        return {"drift_per_ps": np.nan, "total_drift": np.nan, "relative_drift": np.nan}
    
    initial_energy = energies_kcal[0]
    final_energy = energies_kcal[-1]
    total_drift = final_energy - initial_energy
    drift_per_ps = total_drift / total_time
    
    # Relative drift as percentage
    if abs(initial_energy) > 1e-6:
        relative_drift = (total_drift / abs(initial_energy)) * 100.0
    else:
        relative_drift = np.nan
    
    return {
        "drift_per_ps": float(drift_per_ps),
        "total_drift": float(total_drift),
        "relative_drift": float(relative_drift),
        "initial_energy_kcal": float(initial_energy),
        "final_energy_kcal": float(final_energy),
    }


def compute_rmsd(trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
    """
    Compute RMSD between two trajectories.
    
    Args:
        trajectory1: [T, N, 3]
        trajectory2: [T, N, 3]
        
    Returns:
        RMSD per frame [T]
    """
    if trajectory1.shape != trajectory2.shape:
        raise ValueError(f"Shape mismatch: {trajectory1.shape} vs {trajectory2.shape}")
    
    diff = trajectory1 - trajectory2
    rmsd = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))  # [T]
    return rmsd


def compute_orbital_stability(orbital_energies: np.ndarray) -> Dict[str, float]:
    """
    Compute orbital energy stability metrics.
    
    Args:
        orbital_energies: [T, M] orbital energies in eV
        
    Returns:
        Dictionary with stability metrics
    """
    if orbital_energies.shape[0] < 2:
        return {"mean_std": np.nan, "max_deviation": np.nan}
    
    # Standard deviation across time for each orbital
    std_per_orbital = np.std(orbital_energies, axis=0)
    mean_std = float(np.mean(std_per_orbital))
    
    # Maximum deviation from initial
    initial = orbital_energies[0]
    deviations = np.abs(orbital_energies - initial)
    max_deviation = float(np.max(deviations))
    
    return {
        "mean_std": mean_std,
        "max_deviation": max_deviation,
    }


def compute_dipole_correlation(
    dipoles_pred: np.ndarray,
    dipoles_ref: np.ndarray,
) -> Dict[str, float]:
    """
    Compute dipole moment correlation.
    
    Args:
        dipoles_pred: [T, 3] predicted dipole moments
        dipoles_ref: [T, 3] reference dipole moments
        
    Returns:
        Dictionary with correlation metrics
    """
    if dipoles_pred.shape != dipoles_ref.shape:
        raise ValueError("Dipole shape mismatch")
    
    # Magnitudes
    mag_pred = np.linalg.norm(dipoles_pred, axis=1)
    mag_ref = np.linalg.norm(dipoles_ref, axis=1)
    
    # Correlation coefficient
    if len(mag_pred) > 1:
        correlation = float(np.corrcoef(mag_pred, mag_ref)[0, 1])
    else:
        correlation = np.nan
    
    # Mean absolute error
    mae = float(np.mean(np.abs(mag_pred - mag_ref)))
    
    return {
        "correlation": correlation,
        "mae_debye": mae * 0.2081943,  # Convert to Debye
    }


def validate_qm_trajectory(
    hybrid_npz: Path,
    reference_npz: Path,
) -> Dict[str, float]:
    """
    Validate QM hybrid trajectory against reference.
    
    Args:
        hybrid_npz: Path to hybrid trajectory NPZ
        reference_npz: Path to reference QM trajectory NPZ
        
    Returns:
        Dictionary with validation metrics
    """
    hybrid_data = np.load(hybrid_npz, allow_pickle=True)
    ref_data = np.load(reference_npz, allow_pickle=True)
    
    hybrid_pos = hybrid_data["pos"]
    hybrid_vel = hybrid_data["vel"]
    hybrid_energy = hybrid_data["energy"]
    hybrid_time = hybrid_data["time_ps"]
    
    ref_pos = ref_data["positions"]
    ref_vel = ref_data["velocities"]
    ref_energy = ref_data["energies"]
    ref_time = ref_data["time_ps"]
    
    # Align trajectories (assume same length or interpolate)
    min_len = min(len(hybrid_pos), len(ref_pos))
    hybrid_pos = hybrid_pos[:min_len]
    hybrid_vel = hybrid_vel[:min_len]
    hybrid_energy = hybrid_energy[:min_len]
    ref_pos = ref_pos[:min_len]
    ref_vel = ref_vel[:min_len]
    ref_energy = ref_energy[:min_len]
    
    metrics = {}
    
    # Energy drift
    hybrid_drift = compute_energy_drift(hybrid_energy, hybrid_time[:min_len])
    ref_drift = compute_energy_drift(ref_energy, ref_time[:min_len])
    
    metrics.update({
        f"hybrid_{k}": v for k, v in hybrid_drift.items()
    })
    metrics.update({
        f"reference_{k}": v for k, v in ref_drift.items()
    })
    
    # Structural RMSD
    rmsd = compute_rmsd(hybrid_pos, ref_pos)
    metrics["mean_rmsd_nm"] = float(np.mean(rmsd))
    metrics["max_rmsd_nm"] = float(np.max(rmsd))
    
    # Velocity RMSD
    vel_rmsd = compute_rmsd(hybrid_vel, ref_vel)
    metrics["mean_vel_rmsd_nm_per_ps"] = float(np.mean(vel_rmsd))
    
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, required=True, help="Hybrid trajectory NPZ")
    parser.add_argument("--reference", type=Path, required=True, help="Reference QM trajectory NPZ")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON with metrics")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    
    metrics = validate_qm_trajectory(args.hybrid, args.reference)
    
    # Save metrics
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    LOGGER.info("Validation metrics:")
    for key, value in metrics.items():
        LOGGER.info(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()

