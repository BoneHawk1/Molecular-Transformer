"""Evaluate QM hybrid integrator: energy conservation, structure, and stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import configure_logging, ensure_dir, LOGGER


def load_trajectory(path: Path) -> Dict:
    """Load trajectory from npz file."""
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    return {
        "pos": data["pos"],
        "vel": data["vel"],
        "Ekin": data["Ekin"],
        "Epot": data["Epot"],
        "Etot": data["Etot"] if "Etot" in data else data["Ekin"] + data["Epot"],
        "time_ps": data["time_ps"] if "time_ps" in data else None,
        "masses": data["masses"] if "masses" in data else None,
        "atom_types": data["atom_types"] if "atom_types" in data else None,
        "metadata": metadata,
    }


def compute_energy_drift(energies: np.ndarray, time_ps: np.ndarray) -> Dict[str, float]:
    """Compute energy drift statistics."""
    # Remove NaN values
    valid = ~np.isnan(energies)
    if valid.sum() < 2:
        return {
            "drift_per_ps": float('nan'),
            "drift_total": float('nan'),
            "std": float('nan'),
        }
    
    energies_valid = energies[valid]
    time_valid = time_ps[valid]
    
    # Linear fit
    if len(time_valid) > 1 and time_valid[-1] > time_valid[0]:
        slope = (energies_valid[-1] - energies_valid[0]) / (time_valid[-1] - time_valid[0])
        drift_total = energies_valid[-1] - energies_valid[0]
        std = np.std(energies_valid)
    else:
        slope = float('nan')
        drift_total = float('nan')
        std = float('nan')
    
    return {
        "drift_per_ps": slope,
        "drift_total": drift_total,
        "std": std,
    }


def compute_bond_lengths(pos: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
    """Compute all pairwise bond lengths (for small molecules)."""
    n_atoms = pos.shape[0]
    if n_atoms > 50:  # Skip for large molecules
        return np.array([])
    
    bonds = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(pos[i] - pos[j])
            bonds.append(dist * 10.0)  # nm -> Angstrom
    
    return np.array(bonds)


def compute_rmsd(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute RMSD between two structures (Angstrom)."""
    if pos1.shape != pos2.shape:
        return float('nan')
    
    diff = (pos1 - pos2) * 10.0  # nm -> Angstrom
    rmsd = np.sqrt(np.mean(diff ** 2))
    return rmsd


def evaluate_trajectory(name: str, baseline: Dict, hybrid: Dict) -> Dict:
    """Evaluate hybrid trajectory against baseline."""
    LOGGER.info("Evaluating %s", name)
    
    # Energy conservation
    baseline_drift = compute_energy_drift(baseline["Etot"], baseline["time_ps"])
    hybrid_drift = compute_energy_drift(hybrid["Etot"], hybrid["time_ps"])
    
    # Structural comparison (at matching timepoints)
    min_frames = min(len(baseline["pos"]), len(hybrid["pos"]))
    
    rmsd_trajectory = []
    for i in range(min_frames):
        rmsd = compute_rmsd(baseline["pos"][i], hybrid["pos"][i])
        rmsd_trajectory.append(rmsd)
    
    rmsd_mean = np.nanmean(rmsd_trajectory)
    rmsd_max = np.nanmax(rmsd_trajectory)
    rmsd_final = rmsd_trajectory[-1] if rmsd_trajectory else float('nan')
    
    # Bond length statistics (first and last frames)
    baseline_bonds_0 = compute_bond_lengths(baseline["pos"][0], baseline.get("atom_types", np.array([])))
    hybrid_bonds_0 = compute_bond_lengths(hybrid["pos"][0], hybrid.get("atom_types", np.array([])))
    
    if len(baseline["pos"]) > 1 and len(hybrid["pos"]) > 1:
        baseline_bonds_f = compute_bond_lengths(baseline["pos"][-1], baseline.get("atom_types", np.array([])))
        hybrid_bonds_f = compute_bond_lengths(hybrid["pos"][-1], hybrid.get("atom_types", np.array([])))
    else:
        baseline_bonds_f = baseline_bonds_0
        hybrid_bonds_f = hybrid_bonds_0
    
    # Compare bond distributions
    if len(baseline_bonds_0) > 0 and len(hybrid_bonds_0) > 0:
        bond_diff_0 = np.abs(baseline_bonds_0 - hybrid_bonds_0)
        bond_rmse_0 = np.sqrt(np.mean(bond_diff_0 ** 2))
    else:
        bond_rmse_0 = float('nan')
    
    if len(baseline_bonds_f) > 0 and len(hybrid_bonds_f) > 0:
        bond_diff_f = np.abs(baseline_bonds_f - hybrid_bonds_f)
        bond_rmse_f = np.sqrt(np.mean(bond_diff_f ** 2))
    else:
        bond_rmse_f = float('nan')
    
    # Force calls (from metadata)
    force_calls = hybrid["metadata"].get("force_calls", 0)
    failed_steps = hybrid["metadata"].get("failed_steps", 0)
    k_steps = hybrid["metadata"].get("k_steps", 4)
    
    return {
        "molecule": name,
        "baseline_energy_drift_per_ps": baseline_drift["drift_per_ps"],
        "hybrid_energy_drift_per_ps": hybrid_drift["drift_per_ps"],
        "baseline_energy_std": baseline_drift["std"],
        "hybrid_energy_std": hybrid_drift["std"],
        "rmsd_mean": rmsd_mean,
        "rmsd_max": rmsd_max,
        "rmsd_final": rmsd_final,
        "bond_rmse_initial": bond_rmse_0,
        "bond_rmse_final": bond_rmse_f,
        "force_calls": force_calls,
        "failed_steps": failed_steps,
        "k_steps": k_steps,
        "num_frames": len(hybrid["pos"]),
    }


def plot_energy_comparison(metrics: List[Dict], out_dir: Path) -> None:
    """Plot energy drift comparison."""
    molecules = [m["molecule"] for m in metrics]
    baseline_drift = [m["baseline_energy_drift_per_ps"] for m in metrics]
    hybrid_drift = [m["hybrid_energy_drift_per_ps"] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(molecules))
    width = 0.35
    
    ax.bar(x - width/2, baseline_drift, width, label='Baseline QM', alpha=0.8)
    ax.bar(x + width/2, hybrid_drift, width, label='Hybrid QM', alpha=0.8)
    
    ax.set_xlabel('Molecule')
    ax.set_ylabel('Energy Drift (kJ/mol/ps)')
    ax.set_title('QM Energy Conservation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "energy_drift_comparison.png", dpi=150)
    plt.close()
    
    LOGGER.info("Saved energy drift comparison plot")


def plot_rmsd_trajectory(metrics: List[Dict], out_dir: Path) -> None:
    """Plot RMSD statistics."""
    molecules = [m["molecule"] for m in metrics]
    rmsd_mean = [m["rmsd_mean"] for m in metrics]
    rmsd_max = [m["rmsd_max"] for m in metrics]
    rmsd_final = [m["rmsd_final"] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(molecules))
    
    ax.bar(x, rmsd_mean, alpha=0.6, label='Mean RMSD')
    ax.scatter(x, rmsd_max, color='red', s=50, label='Max RMSD', zorder=3)
    ax.scatter(x, rmsd_final, color='green', s=50, label='Final RMSD', zorder=3)
    
    ax.set_xlabel('Molecule')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('Structural Deviation: Hybrid vs Baseline QM')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "rmsd_comparison.png", dpi=150)
    plt.close()
    
    LOGGER.info("Saved RMSD comparison plot")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline QM trajectory directory")
    parser.add_argument("--hybrid", type=Path, required=True, help="Hybrid QM trajectory directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics and plots")
    parser.add_argument("--pattern", type=str, default="*_hybrid_k*.npz", help="Hybrid file pattern")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    ensure_dir(args.out_dir)
    
    metrics: List[Dict] = []
    
    # Find all hybrid trajectories
    hybrid_files = sorted(args.hybrid.glob(args.pattern))
    
    if not hybrid_files:
        LOGGER.error("No hybrid trajectories found in %s", args.hybrid)
        return
    
    for hybrid_path in hybrid_files:
        # Extract molecule name from filename
        name = hybrid_path.stem.replace("_hybrid_k4", "").replace("_hybrid_k8", "").replace("_hybrid_k12", "")
        
        # Find corresponding baseline
        baseline_path = args.baseline / name / "trajectory.npz"
        if not baseline_path.exists():
            LOGGER.warning("Baseline trajectory missing for %s", name)
            continue
        
        try:
            baseline = load_trajectory(baseline_path)
            hybrid = load_trajectory(hybrid_path)
            metrics.append(evaluate_trajectory(name, baseline, hybrid))
        except Exception as e:
            LOGGER.error("Failed to evaluate %s: %s", name, e)
            continue
    
    if not metrics:
        LOGGER.error("No valid trajectories evaluated")
        return
    
    # Save metrics
    metrics_path = args.out_dir / "qm_metrics.json"
    summary = {
        "molecules": metrics,
        "mean_energy_drift_baseline": float(np.nanmean([m["baseline_energy_drift_per_ps"] for m in metrics])),
        "mean_energy_drift_hybrid": float(np.nanmean([m["hybrid_energy_drift_per_ps"] for m in metrics])),
        "mean_rmsd": float(np.nanmean([m["rmsd_mean"] for m in metrics])),
        "mean_rmsd_final": float(np.nanmean([m["rmsd_final"] for m in metrics])),
        "mean_bond_rmse_final": float(np.nanmean([m["bond_rmse_final"] for m in metrics])),
        "total_failed_steps": int(np.sum([m["failed_steps"] for m in metrics])),
        "total_force_calls": int(np.sum([m["force_calls"] for m in metrics])),
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    LOGGER.info("Wrote metrics to %s", metrics_path)
    
    # Generate plots
    plot_energy_comparison(metrics, args.out_dir)
    plot_rmsd_trajectory(metrics, args.out_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("QM HYBRID INTEGRATOR EVALUATION SUMMARY")
    print("="*60)
    print(f"Molecules evaluated: {len(metrics)}")
    print(f"Mean energy drift (baseline): {summary['mean_energy_drift_baseline']:.4f} kJ/mol/ps")
    print(f"Mean energy drift (hybrid):   {summary['mean_energy_drift_hybrid']:.4f} kJ/mol/ps")
    print(f"Mean RMSD (trajectory):       {summary['mean_rmsd']:.3f} Å")
    print(f"Mean RMSD (final):            {summary['mean_rmsd_final']:.3f} Å")
    print(f"Mean bond RMSE (final):       {summary['mean_bond_rmse_final']:.4f} Å")
    print(f"Total failed steps:           {summary['total_failed_steps']}")
    print(f"Total force calls:            {summary['total_force_calls']}")
    print("="*60)


if __name__ == "__main__":
    main()

