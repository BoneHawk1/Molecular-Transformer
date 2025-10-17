"""Run implicit-solvent OpenMM MD trajectories and store results as NumPy archives."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from openmm import (LangevinIntegrator, Platform, VerletIntegrator, XmlSerializer,
                    unit)
from openmm.app import PDBFile, Simulation

from utils import configure_logging, ensure_dir, load_yaml, set_seed, compute_time_grid, LOGGER


@dataclass
class MDConfig:
    temperature_K: float
    friction_per_ps: float
    dt_fs: float
    length_ns: float
    save_interval_steps: int
    constraints: str
    implicit_solvent: str
    nonbonded_cutoff_nm: float
    nve_window_ps: float
    nve_every_ps: float
    platform: str
    random_seed: int
    reporter_stride: int
    store_forces: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "MDConfig":
        return cls(
            temperature_K=data.get("temperature_K", 300.0),
            friction_per_ps=data.get("friction_per_ps", 1.0),
            dt_fs=data.get("dt_fs", 2.0),
            length_ns=data.get("length_ns", 1.0),
            save_interval_steps=data.get("save_interval_steps", 50),
            constraints=data.get("constraints", "HBonds"),
            implicit_solvent=data.get("implicit_solvent", "OBC2"),
            nonbonded_cutoff_nm=data.get("nonbonded_cutoff_nm", 1.0),
            nve_window_ps=data.get("nve_window_ps", 0.0),
            nve_every_ps=data.get("nve_every_ps", 0.0),
            platform=data.get("platform", "CUDA"),
            random_seed=data.get("random_seed", 42),
            reporter_stride=data.get("reporter_stride", 10),
            store_forces=data.get("store_forces", False),
        )


def _load_openmm_bundle(molecule_dir: Path):
    pdb_path = molecule_dir / "structure.pdb"
    xml_path = molecule_dir / "forcefield.xml"
    if not pdb_path.exists() or not xml_path.exists():
        raise FileNotFoundError(f"Expected PDB/XML in {molecule_dir}")
    pdb = PDBFile(str(pdb_path))
    with xml_path.open("r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    return pdb, system


def _create_simulation(pdb: PDBFile, system, integrator, platform_name: str) -> Simulation:
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    return simulation


def _initialise_simulation(simulation: Simulation, pdb: PDBFile, temperature_K: float, seed: int) -> None:
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)


def _get_masses(system) -> np.ndarray:
    masses = []
    for idx in range(system.getNumParticles()):
        mass = system.getParticleMass(idx).value_in_unit(unit.dalton)
        masses.append(mass)
    return np.asarray(masses, dtype=np.float32)


def _collect_frame(simulation: Simulation, store_forces: bool) -> Dict[str, np.ndarray]:
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=store_forces)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    total = kinetic + potential
    frame = {
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "kinetic": kinetic,
        "potential": potential,
        "total": total,
    }
    if store_forces:
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        frame["forces"] = np.asarray(forces, dtype=np.float32)
    return frame


def _run_nve_window(system, topology, positions, velocities, config: MDConfig, platform_name: str, steps: int) -> Dict[str, np.ndarray]:
    integrator = VerletIntegrator(config.dt_fs * unit.femtosecond)
    simulation = Simulation(topology, system, integrator, Platform.getPlatformByName(platform_name))
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    energies = []
    for _ in range(steps):
        state = simulation.context.getState(getEnergy=True)
        energies.append([
            state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        ])
        simulation.step(1)

    energies = np.asarray(energies, dtype=np.float64)
    return {
        "kinetic": energies[:, 0],
        "potential": energies[:, 1],
        "total": energies.sum(axis=1),
    }


def run_md(molecule_dir: Path, out_dir: Path, config: MDConfig) -> Path:
    pdb, system = _load_openmm_bundle(molecule_dir)

    integrator = LangevinIntegrator(
        config.temperature_K * unit.kelvin,
        config.friction_per_ps / unit.picosecond,
        config.dt_fs * unit.femtosecond,
    )

    simulation = _create_simulation(pdb, system, integrator, config.platform)
    _initialise_simulation(simulation, pdb, config.temperature_K, config.random_seed)

    total_steps = int(config.length_ns * 1e6 / config.dt_fs)
    save_interval = config.save_interval_steps
    num_frames = total_steps // save_interval + 1

    LOGGER.info("Running %d steps (%.1f ns) with save interval %d", total_steps, config.length_ns, save_interval)

    masses = _get_masses(system)
    n_atoms = len(masses)
    positions = np.zeros((num_frames, n_atoms, 3), dtype=np.float32)
    velocities = np.zeros_like(positions)
    kinetic = np.zeros(num_frames, dtype=np.float64)
    potential = np.zeros_like(kinetic)
    total = np.zeros_like(kinetic)
    forces = np.zeros_like(positions) if config.store_forces else None

    save_idx = 0
    nve_payload: List[Dict[str, np.ndarray]] = []
    nve_window_steps = int((config.nve_window_ps * 1000) / (config.dt_fs)) if config.nve_window_ps > 0 else 0
    nve_every_steps = int((config.nve_every_ps * 1000) / (config.dt_fs)) if config.nve_every_ps > 0 else 0

    for step in range(0, total_steps + 1):
        if step % save_interval == 0:
            frame = _collect_frame(simulation, config.store_forces)
            positions[save_idx] = frame["positions"]
            velocities[save_idx] = frame["velocities"]
            kinetic[save_idx] = frame["kinetic"]
            potential[save_idx] = frame["potential"]
            total[save_idx] = frame["total"]
            if config.store_forces:
                assert forces is not None
                forces[save_idx] = frame["forces"]

            if nve_window_steps > 0 and nve_every_steps > 0 and step % nve_every_steps == 0:
                LOGGER.info("Running NVE window (%d steps) from step %d", nve_window_steps, step)
                state = simulation.context.getState(getPositions=True, getVelocities=True)
                nve_payload.append(
                    _run_nve_window(
                        system,
                        pdb.topology,
                        state.getPositions(asNumpy=True),
                        state.getVelocities(asNumpy=True),
                        config,
                        config.platform,
                        nve_window_steps,
                    )
                )

            save_idx += 1
        if step < total_steps:
            simulation.step(1)

    time_ps = compute_time_grid(num_frames, config.dt_fs, save_interval)
    metadata = {
        "smiles": molecule_dir.name,
        "config": config.__dict__,
        "num_atoms": int(n_atoms),
    }

    out_dir = out_dir / molecule_dir.name
    ensure_dir(out_dir)
    npz_path = out_dir / "trajectory.npz"
    nve_serialised = json.dumps([
        {key: value.tolist() for key, value in window.items()} for window in nve_payload
    ])
    np.savez(
        npz_path,
        pos=positions,
        vel=velocities,
        Ekin=kinetic,
        Epot=potential,
        Etot=total,
        forces=forces if config.store_forces else None,
        box=np.tile(np.eye(3, dtype=np.float32), (num_frames, 1, 1)),
        masses=masses,
        atom_types=np.array([atom.element.atomic_number for atom in pdb.topology.atoms()], dtype=np.int32),
        time_ps=time_ps,
        metadata=json.dumps(metadata),
        nve_windows=nve_serialised,
    )

    LOGGER.info("Saved trajectory to %s", npz_path)
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=Path, required=True, help="Path to molecule directory containing PDB/XML")
    parser.add_argument("--config", type=Path, required=True, help="MD configuration YAML")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for trajectories")
    parser.add_argument("--store-forces", action="store_true", help="Store instantaneous forces in the NPZ")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    config_dict = load_yaml(args.config)
    config_dict["store_forces"] = args.store_forces or config_dict.get("store_forces", False)
    config = MDConfig.from_dict(config_dict)
    set_seed(config.random_seed)

    run_md(args.molecule, args.out, config)


if __name__ == "__main__":
    main()
