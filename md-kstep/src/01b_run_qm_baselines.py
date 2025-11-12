"""Run semi-empirical QM (xTB) trajectories and store results as NumPy archives."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from ase import Atoms, units
    from ase.calculators.xtb import XTB
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    raise ImportError("QM baseline requires: pip install ase rdkit-pypi; conda install -c conda-forge xtb") from e

from utils import configure_logging, ensure_dir, load_yaml, set_seed, compute_time_grid, LOGGER


@dataclass
class QMConfig:
    temperature_K: float
    friction_per_ps: float
    dt_fs: float
    length_ps: float  # Note: ps not ns (QM is expensive)
    save_interval_steps: int
    method: str  # xTB method: GFN2-xTB, GFN1-xTB, GFN-FF
    random_seed: int
    charge: int
    spin_multiplicity: int
    nve_window_ps: float
    nve_every_ps: float
    equilibration_ps: float  # Pre-production equilibration

    @classmethod
    def from_dict(cls, data: Dict) -> "QMConfig":
        return cls(
            temperature_K=data.get("temperature_K", 300.0),
            friction_per_ps=data.get("friction_per_ps", 0.002),  # Lower friction for QM
            dt_fs=data.get("dt_fs", 0.25),  # Much smaller than MM!
            length_ps=data.get("length_ps", 20.0),  # Shorter trajectories
            save_interval_steps=data.get("save_interval_steps", 1),  # Save every step
            method=data.get("method", "GFN2-xTB"),
            random_seed=data.get("random_seed", 42),
            charge=data.get("charge", 0),
            spin_multiplicity=data.get("spin_multiplicity", 1),
            nve_window_ps=data.get("nve_window_ps", 0.0),
            nve_every_ps=data.get("nve_every_ps", 0.0),
            equilibration_ps=data.get("equilibration_ps", 5.0),
        )


def _smiles_to_atoms(smiles: str, charge: int = 0) -> Atoms:
    """Convert SMILES to ASE Atoms with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates with RDKit
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"Failed to generate 3D coordinates for {smiles}")
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract coordinates and atomic numbers
    conf = mol.GetConformer()
    positions = []
    atomic_numbers = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
        atomic_numbers.append(mol.GetAtomWithIdx(i).GetAtomicNum())
    
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    return atoms


def _get_masses_ase(atoms: Atoms) -> np.ndarray:
    """Get atomic masses in Daltons."""
    return atoms.get_masses()


def _collect_frame_ase(atoms: Atoms) -> Dict[str, np.ndarray]:
    """Collect state from ASE Atoms object."""
    positions = atoms.get_positions() / 10.0  # Angstrom -> nm
    velocities = atoms.get_velocities() * 100.0  # Angstrom/fs -> nm/ps
    
    try:
        potential = atoms.get_potential_energy() * 96.4853  # eV -> kJ/mol
        kinetic = atoms.get_kinetic_energy() * 96.4853
        forces = atoms.get_forces() * 96.4853 / 10.0  # eV/Angstrom -> kJ/mol/nm
    except Exception:
        # If calculator fails, return NaN
        potential = float('nan')
        kinetic = float('nan')
        forces = np.full_like(positions, float('nan'))
    
    total = kinetic + potential
    
    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "forces": np.asarray(forces, dtype=np.float32),
        "kinetic": kinetic,
        "potential": potential,
        "total": total,
    }


def _run_nve_window_ase(atoms: Atoms, config: QMConfig, steps: int) -> Dict[str, np.ndarray]:
    """Run NVE (microcanonical) window for energy conservation analysis."""
    from ase.md.verlet import VelocityVerlet
    
    # Create fresh calculator for NVE
    atoms.calc = XTB(method=config.method, charge=config.charge)
    
    dyn = VelocityVerlet(atoms, timestep=config.dt_fs * units.fs)
    
    energies = []
    for _ in range(steps):
        try:
            kinetic = atoms.get_kinetic_energy() * 96.4853  # eV -> kJ/mol
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


def run_qm(smiles: str, name: str, out_dir: Path, config: QMConfig) -> Path:
    """Run QM trajectory for a single molecule."""
    LOGGER.info("Setting up QM trajectory for %s (SMILES: %s)", name, smiles)
    
    # Build initial structure
    atoms = _smiles_to_atoms(smiles, config.charge)
    n_atoms = len(atoms)
    masses = _get_masses_ase(atoms)
    atomic_numbers = atoms.get_atomic_numbers()
    
    LOGGER.info("Molecule has %d atoms", n_atoms)
    
    # Attach xTB calculator
    atoms.calc = XTB(method=config.method, charge=config.charge)
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K, rng=np.random.RandomState(config.random_seed))
    
    # Equilibration phase (not saved)
    if config.equilibration_ps > 0:
        eq_steps = int(config.equilibration_ps * 1000 / config.dt_fs)
        LOGGER.info("Equilibrating for %d steps (%.1f ps)", eq_steps, config.equilibration_ps)
        
        dyn_eq = Langevin(
            atoms,
            timestep=config.dt_fs * units.fs,
            temperature_K=config.temperature_K,
            friction=config.friction_per_ps / (1000.0 / units.fs),  # Convert to ASE units
        )
        
        try:
            dyn_eq.run(eq_steps)
        except Exception as e:
            LOGGER.error("Equilibration failed: %s", e)
            raise
    
    # Production run
    total_steps = int(config.length_ps * 1000 / config.dt_fs)
    save_interval = config.save_interval_steps
    num_frames = total_steps // save_interval + 1
    
    LOGGER.info("Running %d steps (%.1f ps) with save interval %d", total_steps, config.length_ps, save_interval)
    
    # Allocate storage
    positions = np.zeros((num_frames, n_atoms, 3), dtype=np.float32)
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)
    kinetic = np.zeros(num_frames, dtype=np.float64)
    potential = np.zeros_like(kinetic)
    total = np.zeros_like(kinetic)
    
    # Setup dynamics
    dyn = Langevin(
        atoms,
        timestep=config.dt_fs * units.fs,
        temperature_K=config.temperature_K,
        friction=config.friction_per_ps / (1000.0 / units.fs),
    )
    
    save_idx = 0
    nve_payload: List[Dict[str, np.ndarray]] = []
    nve_window_steps = int((config.nve_window_ps * 1000) / config.dt_fs) if config.nve_window_ps > 0 else 0
    nve_every_steps = int((config.nve_every_ps * 1000) / config.dt_fs) if config.nve_every_ps > 0 else 0
    
    for step in range(0, total_steps + 1):
        if step % save_interval == 0:
            frame = _collect_frame_ase(atoms)
            positions[save_idx] = frame["positions"]
            velocities[save_idx] = frame["velocities"]
            forces[save_idx] = frame["forces"]
            kinetic[save_idx] = frame["kinetic"]
            potential[save_idx] = frame["potential"]
            total[save_idx] = frame["total"]
            
            # Optional NVE windows
            if nve_window_steps > 0 and nve_every_steps > 0 and step % nve_every_steps == 0:
                LOGGER.info("Running NVE window (%d steps) from step %d", nve_window_steps, step)
                # Save current state
                pos_save = atoms.get_positions().copy()
                vel_save = atoms.get_velocities().copy()
                
                nve_payload.append(_run_nve_window_ase(atoms, config, nve_window_steps))
                
                # Restore state and reinitialize Langevin
                atoms.set_positions(pos_save)
                atoms.set_velocities(vel_save)
                atoms.calc = XTB(method=config.method, charge=config.charge)
                dyn = Langevin(
                    atoms,
                    timestep=config.dt_fs * units.fs,
                    temperature_K=config.temperature_K,
                    friction=config.friction_per_ps / (1000.0 / units.fs),
                )
            
            save_idx += 1
            
            # Progress logging
            if step % (total_steps // 10) == 0:
                LOGGER.info("Progress: %d/%d steps (%.1f%%)", step, total_steps, 100 * step / total_steps)
        
        if step < total_steps:
            try:
                dyn.run(1)
            except Exception as e:
                LOGGER.error("QM step failed at step %d: %s", step, e)
                # Truncate arrays to saved frames
                positions = positions[:save_idx]
                velocities = velocities[:save_idx]
                forces = forces[:save_idx]
                kinetic = kinetic[:save_idx]
                potential = potential[:save_idx]
                total = total[:save_idx]
                break
    
    time_ps = compute_time_grid(len(positions), config.dt_fs, save_interval)
    metadata = {
        "smiles": smiles,
        "name": name,
        "config": config.__dict__,
        "num_atoms": int(n_atoms),
        "qm_method": config.method,
    }
    
    out_dir = out_dir / name
    ensure_dir(out_dir)
    npz_path = out_dir / "trajectory.npz"
    
    nve_serialised = json.dumps([
        {key: value.tolist() for key, value in window.items()} for window in nve_payload
    ])
    
    np.savez(
        npz_path,
        pos=positions,
        vel=velocities,
        forces=forces,
        Ekin=kinetic,
        Epot=potential,
        Etot=total,
        box=np.tile(np.eye(3, dtype=np.float32), (len(positions), 1, 1)),
        masses=masses.astype(np.float32),
        atom_types=atomic_numbers.astype(np.int32),
        time_ps=time_ps,
        metadata=json.dumps(metadata),
        nve_windows=nve_serialised,
    )
    
    LOGGER.info("Saved QM trajectory to %s", npz_path)
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-file", type=Path, required=True, help="File with SMILES and names")
    parser.add_argument("--config", type=Path, required=True, help="QM configuration YAML")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for trajectories")
    parser.add_argument("--molecule", type=str, default=None, help="Run specific molecule only (by name)")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    config = QMConfig.from_dict(load_yaml(args.config))
    set_seed(config.random_seed)
    
    # Parse SMILES file
    molecules = []
    with open(args.smiles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                smiles, name = parts
                molecules.append((smiles, name))
    
    LOGGER.info("Found %d molecules in %s", len(molecules), args.smiles_file)
    
    # Filter if specific molecule requested
    if args.molecule:
        molecules = [(s, n) for s, n in molecules if n == args.molecule]
        if not molecules:
            LOGGER.error("Molecule %s not found in SMILES file", args.molecule)
            return
    
    # Run trajectories
    for smiles, name in molecules:
        try:
            run_qm(smiles, name, args.out, config)
        except Exception as e:
            LOGGER.error("Failed to run QM for %s: %s", name, e)
            continue


if __name__ == "__main__":
    main()

