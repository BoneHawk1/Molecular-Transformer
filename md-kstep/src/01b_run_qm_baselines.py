"""Run semi-empirical QM (xTB) trajectories and store results as NumPy archives."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from ase import Atoms, units
    from xtb.ase.calculator import XTB
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
    total_energy = np.zeros_like(kinetic)

    # Setup dynamics
    dyn = Langevin(
        atoms,
        timestep=config.dt_fs * units.fs,
        temperature_K=config.temperature_K,
        friction=config.friction_per_ps / (1000.0 / units.fs),
    )

    # Observer closure to collect data efficiently during MD run
    save_idx = [0]  # Use list to allow mutation in closure
    nve_payload: List[Dict[str, np.ndarray]] = []
    nve_window_steps = int((config.nve_window_ps * 1000) / config.dt_fs) if config.nve_window_ps > 0 else 0
    nve_every_steps = int((config.nve_every_ps * 1000) / config.dt_fs) if config.nve_every_ps > 0 else 0
    failed = [False]  # Track if simulation failed
    start_time = [time.time()]  # Track start time for ETA calculation

    def observer():
        """Called by ASE dynamics at each timestep."""
        step = dyn.nsteps

        # Save data at specified interval
        if step % save_interval == 0:
            try:
                frame = _collect_frame_ase(atoms)
                idx = save_idx[0]
                if idx < num_frames:
                    positions[idx] = frame["positions"]
                    velocities[idx] = frame["velocities"]
                    forces[idx] = frame["forces"]
                    kinetic[idx] = frame["kinetic"]
                    potential[idx] = frame["potential"]
                    total_energy[idx] = frame["total"]
                    save_idx[0] += 1

                # Optional NVE windows
                if nve_window_steps > 0 and nve_every_steps > 0 and step % nve_every_steps == 0 and step > 0:
                    LOGGER.info("Running NVE window (%d steps) from step %d", nve_window_steps, step)
                    # Save current state
                    pos_save = atoms.get_positions().copy()
                    vel_save = atoms.get_velocities().copy()

                    nve_payload.append(_run_nve_window_ase(atoms, config, nve_window_steps))

                    # Restore state and reinitialize Langevin
                    atoms.set_positions(pos_save)
                    atoms.set_velocities(vel_save)
                    atoms.calc = XTB(method=config.method, charge=config.charge)
                    # Note: Cannot recreate dyn object here, but state is restored

                # Progress logging with ETA
                if step > 0 and step % (total_steps // 10) == 0:
                    elapsed_sec = time.time() - start_time[0]
                    progress_frac = step / total_steps
                    estimated_total_sec = elapsed_sec / progress_frac
                    eta_sec = estimated_total_sec - elapsed_sec

                    elapsed_min = elapsed_sec / 60
                    eta_min = eta_sec / 60

                    LOGGER.info("Progress: %d/%d steps (%.1f%%) | Elapsed: %.1f min | ETA: %.1f min",
                               step, total_steps, 100 * progress_frac, elapsed_min, eta_min)

            except Exception as e:
                LOGGER.error("QM observer failed at step %d: %s", step, e)
                failed[0] = True
                raise  # Stop dynamics on error

    # Attach observer and run full trajectory in one call
    dyn.attach(observer, interval=1)

    try:
        # ✅ KEY FIX: Run entire trajectory in single call
        # This allows ASE/xTB to optimize internally without Python loop overhead
        dyn.run(total_steps)
    except Exception as e:
        LOGGER.error("QM trajectory failed: %s", e)
        # Truncate arrays to saved frames
        idx = save_idx[0]
        positions = positions[:idx]
        velocities = velocities[:idx]
        forces = forces[:idx]
        kinetic = kinetic[:idx]
        potential = potential[:idx]
        total_energy = total_energy[:idx]
    
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
        Etot=total_energy,
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
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--omp-num-threads", type=int, default=None, help="Set OMP_NUM_THREADS for xTB per process")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    config = QMConfig.from_dict(load_yaml(args.config))
    set_seed(config.random_seed)

    # Configure OpenMP threading for xTB if requested.
    if args.omp_num_threads is not None:
        try:
            threads = int(args.omp_num_threads)
            if threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(threads)
                os.environ.setdefault("OMP_STACKSIZE", "4G")
            else:
                LOGGER.warning("Ignoring non-positive --omp-num-threads=%d", threads)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid --omp-num-threads=%r; leaving OMP_NUM_THREADS unchanged", args.omp_num_threads)
    elif int(args.num_workers) > 1:
        # Default to 1 thread per worker to avoid oversubscription when parallelizing.
        if os.environ.get("OMP_NUM_THREADS") is None:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ.setdefault("OMP_STACKSIZE", "4G")
            LOGGER.info("Setting default OMP_NUM_THREADS=1 for parallel workers (override with --omp-num-threads)")
    
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
    
    # Run trajectories (parallel if requested)
    num_workers = max(1, int(args.num_workers))
    if num_workers > 1 and len(molecules) > 1:
        LOGGER.info("Running %d molecules with %d workers (OMP_NUM_THREADS=%s)", len(molecules), num_workers, os.environ.get("OMP_NUM_THREADS", "default"))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_name = {
                executor.submit(run_qm, smiles, name, args.out, config): name
                for smiles, name in molecules
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    npz_path = future.result()
                    LOGGER.info("Completed %s -> %s", name, npz_path)
                except Exception as e:
                    LOGGER.error("Failed to run QM for %s: %s", name, e)
                    continue
    else:
        LOGGER.info("Running sequentially (num-workers=%d)", num_workers)
        for smiles, name in molecules:
            try:
                run_qm(smiles, name, args.out, config)
            except Exception as e:
                LOGGER.error("Failed to run QM for %s: %s", name, e)
                continue


if __name__ == "__main__":
    main()

