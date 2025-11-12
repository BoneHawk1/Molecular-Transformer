"""Generate QM reference data for training k-step QM models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import configure_logging, ensure_dir, load_yaml, set_seed, LOGGER
from qm_engine import QMEngine, QMConfig

# Atomic masses in amu (for reference)
ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
}


def get_atom_symbols_from_smiles(smiles: str) -> List[str]:
    """Extract atom symbols from SMILES string (simplified)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    return symbols


def get_initial_geometry(smiles: str) -> np.ndarray:
    """Get initial 3D geometry from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    # Convert from Angstrom to nm
    return positions / 10.0


def run_qm_md_trajectory(
    qm_engine: QMEngine,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    masses: np.ndarray,
    dt_fs: float,
    num_steps: int,
    temperature: float = 300.0,
) -> Dict:
    """
    Run QM MD trajectory using velocity Verlet (simplified, no thermostat).
    
    Note: This is a simplified MD integrator. For production, use a proper
    QM MD package or integrate with existing MD frameworks.
    """
    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    
    # Convert masses from amu to kg, then to internal units
    # For simplicity, use atomic mass units directly
    masses_amu = masses
    
    pos_history = [positions.copy()]
    vel_history = [velocities.copy()]
    energy_history = []
    force_history = []
    
    dt_ps = dt_fs * 1e-3  # femtoseconds to picoseconds
    dt_s = dt_fs * 1e-15  # femtoseconds to seconds
    
    # Initial force calculation
    result = qm_engine.calculate(positions)
    forces = result["forces"]
    energy = result["energy"]
    
    energy_history.append(energy)
    force_history.append(forces.copy())
    
    # Convert forces from kJ/mol/nm to internal units
    # Force in kJ/mol/nm, need acceleration in nm/ps^2
    # F = ma -> a = F/m
    # Convert: 1 kJ/mol = 1.6605e-21 J per molecule
    # 1 amu = 1.6605e-27 kg
    # So F (kJ/mol/nm) / m (amu) gives acceleration in nm/ps^2 (with conversion factor)
    conversion_factor = 1.0 / (1.6605e-6)  # Approximate conversion
    
    for step in range(num_steps):
        # Velocity Verlet step
        # v(t+dt/2) = v(t) + (dt/2) * a(t)
        accelerations = forces / masses_amu[:, None] * conversion_factor
        velocities_half = velocities + 0.5 * dt_ps * accelerations
        
        # x(t+dt) = x(t) + dt * v(t+dt/2)
        positions = positions + dt_ps * velocities_half
        
        # Calculate new forces
        result = qm_engine.calculate(positions)
        forces_new = result["forces"]
        energy_new = result["energy"]
        
        # v(t+dt) = v(t+dt/2) + (dt/2) * a(t+dt)
        accelerations_new = forces_new / masses_amu[:, None] * conversion_factor
        velocities = velocities_half + 0.5 * dt_ps * accelerations_new
        
        forces = forces_new
        energy = energy_new
        
        pos_history.append(positions.copy())
        vel_history.append(velocities.copy())
        energy_history.append(energy)
        force_history.append(forces.copy())
        
        if (step + 1) % 100 == 0:
            LOGGER.info(f"Completed {step + 1}/{num_steps} QM MD steps")
    
    return {
        "positions": np.array(pos_history, dtype=np.float32),  # [T, N, 3] in nm
        "velocities": np.array(vel_history, dtype=np.float32),  # [T, N, 3] in nm/ps
        "energies": np.array(energy_history, dtype=np.float32),  # [T] in Hartree
        "forces": np.array(force_history, dtype=np.float32),  # [T, N, 3] in kJ/mol/nm
        "masses": masses.astype(np.float32),
    }


def generate_qm_trajectory(
    smiles: str,
    qm_config: QMConfig,
    output_dir: Path,
    dt_fs: float = 0.5,
    num_steps: int = 1000,
    save_interval: int = 10,
    temperature: float = 300.0,
    seed: int = 42,
) -> None:
    """Generate QM trajectory for a molecule."""
    ensure_dir(output_dir)
    
    # Get atom symbols and initial geometry
    atom_symbols = get_atom_symbols_from_smiles(smiles)
    initial_positions = get_initial_geometry(smiles)
    
    # Get atomic masses
    masses = np.array([ATOMIC_MASSES.get(sym, 12.0) for sym in atom_symbols], dtype=np.float32)
    
    # Initialize velocities from Maxwell-Boltzmann distribution
    np.random.seed(seed)
    k_B = 0.008314462618  # kJ/(mol·K)
    sigma_v = np.sqrt(k_B * temperature / masses[:, None])  # nm/ps
    initial_velocities = np.random.normal(0, sigma_v, size=initial_positions.shape).astype(np.float32)
    
    # Remove COM motion
    com_vel = np.sum(initial_velocities * masses[:, None], axis=0) / np.sum(masses)
    initial_velocities -= com_vel
    
    # Create QM engine
    qm_engine = QMEngine(qm_config, atom_symbols)
    
    LOGGER.info(f"Generating QM trajectory for {smiles} ({len(atom_symbols)} atoms)")
    LOGGER.info(f"Method: {qm_config.method}, Basis: {qm_config.basis}")
    
    # Run trajectory
    trajectory = run_qm_md_trajectory(
        qm_engine,
        initial_positions,
        initial_velocities,
        masses,
        dt_fs,
        num_steps,
        temperature,
    )
    
    # Subsample according to save_interval
    indices = np.arange(0, len(trajectory["positions"]), save_interval)
    trajectory_subsampled = {
        "positions": trajectory["positions"][indices],
        "velocities": trajectory["velocities"][indices],
        "energies": trajectory["energies"][indices],
        "forces": trajectory["forces"][indices],
        "masses": trajectory["masses"],
        "atom_symbols": atom_symbols,
        "time_ps": np.arange(len(indices)) * dt_fs * save_interval * 1e-3,
    }
    
    # Save trajectory
    output_path = output_dir / "trajectory.npz"
    metadata = {
        "smiles": smiles,
        "method": qm_config.method,
        "basis": qm_config.basis,
        "dt_fs": dt_fs,
        "num_steps": num_steps,
        "save_interval": save_interval,
        "temperature": temperature,
        "num_atoms": len(atom_symbols),
    }
    
    np.savez(
        output_path,
        **trajectory_subsampled,
        metadata=json.dumps(metadata),
    )
    
    LOGGER.info(f"Saved QM trajectory to {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string")
    parser.add_argument("--qm-config", type=Path, required=True, help="QM config YAML")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--dt-fs", type=float, default=0.5, help="Time step in fs")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of MD steps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save every N steps")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    set_seed(args.seed)
    
    qm_config = QMConfig.from_yaml(args.qm_config)
    generate_qm_trajectory(
        args.smiles,
        qm_config,
        args.output_dir,
        args.dt_fs,
        args.num_steps,
        args.save_interval,
        args.temperature,
        args.seed,
    )


if __name__ == "__main__":
    main()

