"""OpenMM corrector for the classical hybrid integrator (requires ``openmm``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from openmm import LangevinIntegrator, LocalEnergyMinimizer, Platform, XmlSerializer, unit
from openmm.app import PDBFile, Simulation

from .hybrid import CorrectorResult


class OpenMMCorrector:
    """Runs a few Langevin micro-steps with the real force field."""

    def __init__(self, molecule_dir: Path, md_cfg: dict, seed: int, platform: str | None = None) -> None:
        pdb = PDBFile(str(molecule_dir / "structure.pdb"))
        system = XmlSerializer.deserialize((molecule_dir / "forcefield.xml").read_text())
        dt_fs = float(md_cfg.get("dt_fs", 2.0))
        integrator = LangevinIntegrator(
            float(md_cfg.get("temperature_K", 300.0)) * unit.kelvin,
            float(md_cfg.get("friction_per_ps", 1.0)) / unit.picosecond,
            dt_fs * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(seed)
        self.sim = Simulation(pdb.topology, system, integrator,
                              Platform.getPlatformByName(platform or md_cfg.get("platform", "CUDA")))
        self.micro_dt_ps = dt_fs * 1e-3

    def run(self, positions_nm, velocities, steps: int) -> CorrectorResult:
        ctx = self.sim.context
        ctx.setPositions(np.asarray(positions_nm) * unit.nanometer)
        ctx.setVelocities(np.asarray(velocities) * (unit.nanometer / unit.picosecond))
        ctx.applyConstraints(1e-6)
        ctx.applyVelocityConstraints(1e-6)
        self.sim.step(int(steps))
        state = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
        return CorrectorResult(
            positions=state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            velocities=state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
            ekin=state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
            epot=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        )

    def energy(self, positions_nm) -> float:
        ctx = self.sim.context
        ctx.setPositions(np.asarray(positions_nm) * unit.nanometer)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def quench(self, positions_nm, iterations: int, target_epot: float | None = None, chunk: int = 2):
        """Relax bond/angle strain left by a learned jump with a few L-BFGS iterations.

        With ``target_epot`` the minimiser runs in chunks of ``chunk`` iterations and stops
        as soon as the potential energy is at or below the target (a *thermal* quench: it
        removes excess strain without freezing out thermal bond/angle fluctuations).
        Returns (positions, force_evaluations); OpenMM does not report exact evaluation
        counts, so 2 per iteration (an upper bound for L-BFGS with line search) plus one per
        energy check are counted.
        """
        ctx = self.sim.context
        ctx.setPositions(np.asarray(positions_nm) * unit.nanometer)
        evals = 0
        if target_epot is None:
            LocalEnergyMinimizer.minimize(ctx, 1.0, int(iterations))
            evals = 2 * int(iterations)
        else:
            done = 0
            while True:
                e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                evals += 1
                if e <= target_epot or done >= iterations:
                    break
                LocalEnergyMinimizer.minimize(ctx, 1.0, chunk)
                done += chunk
                evals += 2 * chunk
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return np.asarray(pos), evals
