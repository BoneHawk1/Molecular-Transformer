from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from utils import ensure_dir, get_logger, load_yaml


def build_simulation(pdb_path: Path, conf: Dict) -> Tuple["app.Simulation", int]:
    from openmm import unit
    from openmm import app, Platform

    # Forcefield with GBSA OBC2 and constraints
    ff_files = [
        "amber14-all.xml",
        "amber14/tip3pfb.xml",  # irrelevant for implicit, but harmless
        "amber14/protein.ff14SB.xml",
        "amber14/implicit/gbsa-obc2.xml",
    ]
    forcefield = app.ForceField(*ff_files)

    pdb = app.PDBFile(str(pdb_path))
    modeller = app.Modeller(pdb.topology, pdb.positions)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=conf.get("nonbonded_cutoff_nm", 1.0) * unit.nanometer,
        constraints=app.HBonds if conf.get("constraints", "HBonds") == "HBonds" else None,
    )

    temperature = conf.get("temperature_K", 300) * unit.kelvin
    friction = conf.get("friction_per_ps", 1.0) / unit.picosecond
    dt = conf.get("dt_fs", 2.0) * unit.femtosecond

    integrator = app.LangevinIntegrator(temperature, friction, dt)
    platform = Platform.getPlatformByName("CPU")
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature)

    num_atoms = modeller.topology.getNumAtoms()
    return sim, num_atoms


def run_md_for_molecule(pdb_path: Path, out_npz: Path, conf: Dict) -> None:
    from openmm import unit

    logger = get_logger()
    ensure_dir(out_npz.parent)
    sim, num_atoms = build_simulation(pdb_path, conf)

    steps_total = int(conf.get("length_ns", 1.0) * 1e6 / conf.get("dt_fs", 2.0))
    save_interval = int(conf.get("save_interval_steps", 50))
    n_frames = steps_total // save_interval

    pos = np.zeros((n_frames, num_atoms, 3), dtype=np.float32)
    vel = np.zeros((n_frames, num_atoms, 3), dtype=np.float32)
    epot = np.zeros((n_frames,), dtype=np.float64)
    ekin = np.zeros((n_frames,), dtype=np.float64)
    etot = np.zeros((n_frames,), dtype=np.float64)

    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    pos[0] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel[0] = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    epot[0] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ekin[0] = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    etot[0] = epot[0] + ekin[0]

    frame = 1
    for step in range(1, steps_total + 1):
        sim.step(1)
        if step % save_interval == 0 and frame < n_frames:
            st = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            pos[frame] = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            vel[frame] = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
            epot[frame] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ekin[frame] = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            etot[frame] = epot[frame] + ekin[frame]
            frame += 1

    np.savez_compressed(
        out_npz,
        pos=pos,
        vel=vel,
        box=np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        Epot=epot,
        Ekin=ekin,
        Etot=etot,
    )
    logger.info(f"Saved {out_npz}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline MD rollouts (implicit solvent)")
    parser.add_argument("--conf", type=str, required=True, help="Path to MD YAML config")
    parser.add_argument("--raw", type=str, required=True, help="Directory with prepared PDBs (from 00)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for MD npz")
    args = parser.parse_args()

    conf = load_yaml(args.conf)
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    logger = get_logger()
    for pdb_path in sorted(raw_dir.glob("*.pdb")):
        name = pdb_path.stem
        out_npz = out_dir / f"{name}_traj.npz"
        logger.info(f"Running MD for {name}")
        run_md_for_molecule(pdb_path, out_npz, conf)


if __name__ == "__main__":
    main()

