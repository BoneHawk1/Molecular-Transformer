"""QM-specific dataset handling for k-step QM models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils import configure_logging, ensure_dir, remove_com, LOGGER


def load_qm_trajectory(path: Path) -> Dict:
    """Load QM trajectory from NPZ file."""
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    
    result = {
        "pos": data["positions"],  # [T, N, 3] in nm
        "vel": data["velocities"],  # [T, N, 3] in nm/ps
        "energies": data["energies"],  # [T] in Hartree
        "forces": data["forces"],  # [T, N, 3] in kJ/mol/nm
        "masses": data["masses"],
        "atom_symbols": data["atom_symbols"].tolist() if "atom_symbols" in data else [],
        "time_ps": data["time_ps"] if "time_ps" in data else None,
        "metadata": metadata,
    }
    
    # Convert atom symbols to atom types (simple mapping)
    if "atom_types" not in data and result["atom_symbols"]:
        atom_type_map = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
            "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
            "S": 16, "Cl": 17, "Ar": 18,
        }
        result["atom_types"] = np.array([
            atom_type_map.get(sym, 0) for sym in result["atom_symbols"]
        ], dtype=np.int64)
    elif "atom_types" in data:
        result["atom_types"] = data["atom_types"]
    else:
        raise ValueError("No atom types or symbols found in trajectory")
    
    return result


def process_qm_window(
    traj: Dict,
    idx: int,
    k: int,
    include_electronic: bool = False,
) -> Dict:
    """
    Process a k-step window from QM trajectory.
    
    Args:
        traj: QM trajectory dictionary
        idx: Starting frame index
        k: Number of steps to skip
        include_electronic: Whether to include electronic structure data
        
    Returns:
        Dictionary with window data
    """
    pos = traj["pos"]
    vel = traj["vel"]
    masses = traj["masses"]
    energies = traj.get("energies", None)
    forces = traj.get("forces", None)
    
    x_t = pos[idx]
    v_t = vel[idx]
    x_tk = pos[idx + k]
    v_tk = vel[idx + k]
    
    # Remove COM motion
    x_t_centered, v_t_centered = remove_com(x_t, v_t, masses)
    x_tk_centered, v_tk_centered = remove_com(x_tk, v_tk, masses)
    
    result = {
        "x_t": x_t_centered.astype(np.float32),
        "v_t": v_t_centered.astype(np.float32),
        "x_tk": x_tk_centered.astype(np.float32),
        "v_tk": v_tk_centered.astype(np.float32),
        "masses": masses.astype(np.float32),
        "atom_types": traj["atom_types"].astype(np.int64),
    }
    
    # Include energy and force information if available
    if energies is not None:
        result["energy_t"] = float(energies[idx])
        result["energy_tk"] = float(energies[idx + k])
        result["delta_energy"] = float(energies[idx + k] - energies[idx])
    
    if forces is not None:
        result["forces_t"] = forces[idx].astype(np.float32)
        result["forces_tk"] = forces[idx + k].astype(np.float32)
    
    return result


def create_qm_dataset(
    trajectory_paths: List[Path],
    output_path: Path,
    k_steps: int,
    stride: int = 10,
    max_samples_per_mol: int = 10000,
    include_electronic: bool = False,
) -> None:
    """
    Create QM dataset from multiple trajectories.
    
    Args:
        trajectory_paths: List of paths to QM trajectory NPZ files
        output_path: Output NPZ file path
        k_steps: Number of steps to skip
        stride: Frame stride when sampling windows
        max_samples_per_mol: Maximum samples per molecule
        include_electronic: Whether to include electronic structure data
    """
    samples: List[Dict] = []
    molecule_ids: List[str] = []
    
    for traj_path in trajectory_paths:
        mol_name = traj_path.parent.name
        LOGGER.info(f"Processing {mol_name}")
        
        traj = load_qm_trajectory(traj_path)
        num_frames = traj["pos"].shape[0]
        
        # Enumerate windows
        window_indices = list(range(0, num_frames - k_steps, stride))
        if max_samples_per_mol > 0 and len(window_indices) > max_samples_per_mol:
            indices = np.random.choice(window_indices, max_samples_per_mol, replace=False)
            window_indices = sorted(indices.tolist())
        
        for idx in window_indices:
            sample = process_qm_window(traj, idx, k_steps, include_electronic)
            samples.append(sample)
            molecule_ids.append(mol_name)
    
    # Save dataset
    ensure_dir(output_path.parent)
    arrays = {}
    for key in ["x_t", "v_t", "x_tk", "v_tk", "masses", "atom_types"]:
        arrays[key] = np.array([s[key] for s in samples], dtype=object)
    
    # Optional arrays
    if "energy_t" in samples[0]:
        arrays["energy_t"] = np.array([s["energy_t"] for s in samples], dtype=np.float32)
        arrays["energy_tk"] = np.array([s["energy_tk"] for s in samples], dtype=np.float32)
        arrays["delta_energy"] = np.array([s["delta_energy"] for s in samples], dtype=np.float32)
    
    if "forces_t" in samples[0]:
        arrays["forces_t"] = np.array([s["forces_t"] for s in samples], dtype=object)
        arrays["forces_tk"] = np.array([s["forces_tk"] for s in samples], dtype=object)
    
    arrays["molecule"] = np.array(molecule_ids)
    arrays["k_steps"] = k_steps
    
    np.savez(output_path, **arrays)
    LOGGER.info(f"Created QM dataset with {len(samples)} samples at {output_path}")

