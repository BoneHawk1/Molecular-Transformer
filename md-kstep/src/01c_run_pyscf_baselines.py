"""Run ab initio QM (PySCF) trajectories with optional GPU acceleration."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from rdkit import Chem
from rdkit.Chem import AllChem

from pyscf_gpu_calculator import PySCFGPUCalculator
from utils import (
    LOGGER,
    compute_time_grid,
    configure_logging,
    ensure_dir,
    load_yaml,
    set_seed,
)


@dataclass
class PySCFConfig:
    temperature_K: float
    friction_per_ps: float
    dt_fs: float
    length_ps: float
    save_interval_steps: int
    method: str
    basis: str
    xc: str
    random_seed: int
    charge: int
    spin: int
    equilibration_ps: float
    conv_tol: float
    max_cycle: int
    use_gpu: bool
    nve_window_ps: float
    nve_every_ps: float

    @classmethod
    def from_dict(cls, data: Dict) -> "PySCFConfig":
        return cls(
            temperature_K=data.get("temperature_K", 300.0),
            friction_per_ps=data.get("friction_per_ps", 0.01),
            dt_fs=data.get("dt_fs", 0.5),
            length_ps=data.get("length_ps", 5.0),
            save_interval_steps=data.get("save_interval_steps", 5),
            method=data.get("method", "rhf"),
            basis=data.get("basis", "sto-3g"),
            xc=data.get("xc", "pbe"),
            random_seed=data.get("random_seed", 42),
            charge=data.get("charge", 0),
            spin=data.get("spin", 0),
            equilibration_ps=data.get("equilibration_ps", 0.5),
            conv_tol=data.get("conv_tol", 1e-8),
            max_cycle=data.get("max_cycle", 100),
            use_gpu=data.get("use_gpu", True),
            nve_window_ps=data.get("nve_window_ps", 0.0),
            nve_every_ps=data.get("nve_every_ps", 0.0),
        )


def _smiles_to_atoms(smiles: str, charge: int = 0) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"Failed to generate 3D coordinates for {smiles}")

    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    positions = []
    atomic_numbers = []
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        positions.append([pos.x, pos.y, pos.z])
        atomic_numbers.append(mol.GetAtomWithIdx(idx).GetAtomicNum())

    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    atoms.set_initial_charges([0.0] * len(atomic_numbers))
    return atoms


def _collect_frame_ase(atoms: Atoms) -> Dict[str, np.ndarray]:
    positions = atoms.get_positions() / 10.0  # Angstrom -> nm
    velocities = atoms.get_velocities() * 100.0  # Angstrom/fs -> nm/ps

    try:
        potential = atoms.get_potential_energy() * 96.4853  # eV -> kJ/mol
        kinetic = atoms.get_kinetic_energy() * 96.4853
        forces = atoms.get_forces() * 96.4853 / 10.0  # eV/Angstrom -> kJ/mol/nm
    except Exception:
        potential = float("nan")
        kinetic = float("nan")
        forces = np.full_like(positions, float("nan"))

    total = kinetic + potential

    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "forces": np.asarray(forces, dtype=np.float32),
        "kinetic": kinetic,
        "potential": potential,
        "total": total,
    }


def _run_nve_window_ase(atoms: Atoms, config: PySCFConfig, steps: int) -> Dict[str, np.ndarray]:
    atoms.calc = PySCFGPUCalculator(
        method=config.method,
        basis=config.basis,
        xc=config.xc,
        charge=config.charge,
        spin=config.spin,
        use_gpu=config.use_gpu,
        conv_tol=config.conv_tol,
        max_cycle=config.max_cycle,
    )

    dyn = VelocityVerlet(atoms, timestep=config.dt_fs * units.fs)

    energies = []
    for _ in range(steps):
        try:
            kinetic = atoms.get_kinetic_energy() * 96.4853
            potential = atoms.get_potential_energy() * 96.4853
            energies.append([kinetic, potential])
            dyn.run(1)
        except Exception as exc:
            LOGGER.warning("NVE step failed: %s", exc)
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


def _validate_npz_schema(npz_path: Path) -> None:
    """Confirm saved NPZ aligns with downstream dataset expectations."""
    required_keys = {
        "pos",
        "vel",
        "forces",
        "Ekin",
        "Epot",
        "Etot",
        "box",
        "masses",
        "atom_types",
        "time_ps",
        "metadata",
        "nve_windows",
    }
    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.keys())
        missing = required_keys - keys
        if missing:
            raise ValueError(f"PySCF NPZ missing keys: {missing}")
        if data["pos"].dtype != np.float32 or data["vel"].dtype != np.float32:
            raise ValueError("Positions and velocities must be float32 for dataset compatibility")


def run_pyscf(smiles: str, name: str, out_dir: Path, config: PySCFConfig) -> Path:
    LOGGER.info("Setting up PySCF trajectory for %s (SMILES: %s)", name, smiles)

    atoms = _smiles_to_atoms(smiles, config.charge)
    n_atoms = len(atoms)
    masses = atoms.get_masses()
    atomic_numbers = atoms.get_atomic_numbers()

    LOGGER.info("Molecule has %d atoms", n_atoms)

    atoms.calc = PySCFGPUCalculator(
        method=config.method,
        basis=config.basis,
        xc=config.xc,
        charge=config.charge,
        spin=config.spin,
        use_gpu=config.use_gpu,
        conv_tol=config.conv_tol,
        max_cycle=config.max_cycle,
    )

    rng = np.random.RandomState(config.random_seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K, rng=rng)

    if config.equilibration_ps > 0:
        eq_steps = int(config.equilibration_ps * 1000 / config.dt_fs)
        LOGGER.info("Equilibrating for %d steps (%.2f ps)", eq_steps, config.equilibration_ps)
        dyn_eq = Langevin(
            atoms,
            timestep=config.dt_fs * units.fs,
            temperature_K=config.temperature_K,
            friction=config.friction_per_ps / (1000.0 / units.fs),
        )
        try:
            dyn_eq.run(eq_steps)
        except Exception as exc:
            LOGGER.error("Equilibration failed for %s: %s", name, exc)
            raise

    total_steps = int(config.length_ps * 1000 / config.dt_fs)
    save_interval = max(1, config.save_interval_steps)
    num_frames = total_steps // save_interval + 1

    LOGGER.info(
        "Running %d steps (%.2f ps) with save interval %d",
        total_steps,
        config.length_ps,
        save_interval,
    )

    positions = np.zeros((num_frames, n_atoms, 3), dtype=np.float32)
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)
    kinetic = np.zeros(num_frames, dtype=np.float64)
    potential = np.zeros_like(kinetic)
    total = np.zeros_like(kinetic)

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

            if nve_window_steps > 0 and nve_every_steps > 0 and step % nve_every_steps == 0:
                LOGGER.info("Running NVE window (%d steps) from step %d", nve_window_steps, step)
                pos_snapshot = atoms.get_positions().copy()
                vel_snapshot = atoms.get_velocities().copy()

                nve_payload.append(_run_nve_window_ase(atoms, config, nve_window_steps))

                atoms.set_positions(pos_snapshot)
                atoms.set_velocities(vel_snapshot)
                atoms.calc = PySCFGPUCalculator(
                    method=config.method,
                    basis=config.basis,
                    xc=config.xc,
                    charge=config.charge,
                    spin=config.spin,
                    use_gpu=config.use_gpu,
                    conv_tol=config.conv_tol,
                    max_cycle=config.max_cycle,
                )
                dyn = Langevin(
                    atoms,
                    timestep=config.dt_fs * units.fs,
                    temperature_K=config.temperature_K,
                    friction=config.friction_per_ps / (1000.0 / units.fs),
                )

            save_idx += 1

            if total_steps > 0 and step % max(1, total_steps // 10) == 0:
                LOGGER.info("Progress: %d/%d steps (%.1f%%)", step, total_steps, 100 * step / total_steps)

        if step < total_steps:
            try:
                dyn.run(1)
            except Exception as exc:
                LOGGER.error("PySCF MD failed at step %d: %s", step, exc)
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
        "method": config.method,
        "basis": config.basis,
        "xc": config.xc,
        "gpu_enabled": bool(config.use_gpu),
    }

    out_dir = out_dir / name
    ensure_dir(out_dir)
    npz_path = out_dir / "trajectory.npz"

    nve_serialised = json.dumps([{key: value.tolist() for key, value in window.items()} for window in nve_payload])

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

    LOGGER.info("Saved PySCF trajectory to %s", npz_path)
    _validate_npz_schema(npz_path)
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-file", type=Path, required=True, help="File with SMILES and names")
    parser.add_argument("--config", type=Path, required=True, help="PySCF configuration YAML")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for trajectories")
    parser.add_argument("--molecule", type=str, default=None, help="Run specific molecule only (by name)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--omp-num-threads", type=int, default=None, help="Set OMP_NUM_THREADS for PySCF per process")
    return parser


def _configure_threading(num_workers: int, omp_threads: Optional[int]) -> None:
    if omp_threads is not None:
        try:
            threads = int(omp_threads)
            if threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(threads)
                os.environ.setdefault("OMP_STACKSIZE", "4G")
            else:
                LOGGER.warning("Ignoring non-positive --omp-num-threads=%d", threads)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid --omp-num-threads=%r; leaving OMP_NUM_THREADS unchanged", omp_threads)
    elif num_workers > 1 and os.environ.get("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ.setdefault("OMP_STACKSIZE", "4G")
        LOGGER.info("Setting default OMP_NUM_THREADS=1 for parallel PySCF workers")


def _parse_smiles(smiles_file: Path) -> List[Tuple[str, str]]:
    molecules: List[Tuple[str, str]] = []
    with open(smiles_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                smiles, name = parts
                molecules.append((smiles, name))
    return molecules


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    _configure_threading(args.num_workers, args.omp_num_threads)

    config = PySCFConfig.from_dict(load_yaml(args.config))
    set_seed(config.random_seed)

    molecules = _parse_smiles(args.smiles_file)
    LOGGER.info("Found %d molecules in %s", len(molecules), args.smiles_file)

    if args.molecule:
        molecules = [(s, n) for s, n in molecules if n == args.molecule]
        if not molecules:
            LOGGER.error("Molecule %s not found in SMILES file", args.molecule)
            return

    num_workers = max(1, int(args.num_workers))
    if num_workers > 1 and len(molecules) > 1:
        LOGGER.info(
            "Running %d molecules with %d workers (OMP_NUM_THREADS=%s)",
            len(molecules),
            num_workers,
            os.environ.get("OMP_NUM_THREADS", "default"),
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(run_pyscf, smiles, name, args.out, config): name
                for smiles, name in molecules
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    npz_path = future.result()
                    LOGGER.info("Completed %s -> %s", name, npz_path)
                except Exception as exc:
                    LOGGER.error("Failed to run PySCF for %s: %s", name, exc)
    else:
        LOGGER.info("Running sequentially (num-workers=%d)", num_workers)
        for smiles, name in molecules:
            try:
                run_pyscf(smiles, name, args.out, config)
            except Exception as exc:
                LOGGER.error("Failed to run PySCF for %s: %s", name, exc)


if __name__ == "__main__":
    main()


