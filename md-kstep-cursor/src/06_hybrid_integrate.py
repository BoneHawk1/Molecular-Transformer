from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import numpy as np
import torch

from utils import ensure_dir, get_logger, load_yaml


def load_model(checkpoint: Path, device: str):
    # Dynamic import build_model
    model_py = Path(__file__).with_name("03_model.py")
    spec = importlib.util.spec_from_file_location("kstep_model", model_py)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build_model = mod.build_model

    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def one_step_corrector(x_nm, v_nmps, pdb_path: Path, md_conf):
    # Perform one velocity-Verlet step in OpenMM as corrector
    from openmm import unit
    from openmm import app, Platform

    ff_files = ["amber14-all.xml", "amber14/implicit/gbsa-obc2.xml"]
    forcefield = app.ForceField(*ff_files)
    pdb = app.PDBFile(str(pdb_path))
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=md_conf.get("nonbonded_cutoff_nm", 1.0) * unit.nanometer,
        constraints=app.HBonds if md_conf.get("constraints", "HBonds") == "HBonds" else None,
    )
    integrator = app.VerletIntegrator(md_conf.get("dt_fs", 2.0) * unit.femtosecond)
    platform = Platform.getPlatformByName("CPU")
    sim = app.Simulation(pdb.topology, system, integrator, platform)

    # Load state
    sim.context.setPositions(x_nm * unit.nanometer)
    sim.context.setVelocities(v_nmps * unit.nanometer / unit.picosecond)
    sim.step(1)  # one step as projection
    st = sim.context.getState(getPositions=True, getVelocities=True)
    x_star = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    v_star = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    return x_star.astype(np.float32), v_star.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Hybrid integration: learned jump + one-step corrector")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--md-conf", type=str, required=True)
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    logger = get_logger()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), device)
    md_conf = load_yaml(args.md_conf)

    # For demo, run hybrid over each prepared PDB with initial state from its first MD frame if available
    ensure_dir(Path(args.out))
    for pdb_path in sorted(Path(args.raw).glob("*.pdb")):
        name = pdb_path.stem
        # Load an initial state from matching MD, else embed zeros
        traj_file = Path("data/md") / f"{name}_traj.npz"
        if traj_file.exists():
            with np.load(traj_file) as d:
                x0 = d["pos"][0]
                v0 = d["vel"][0]
        else:
            # fallback
            from openmm import app
            pdb = app.PDBFile(str(pdb_path))
            N = pdb.topology.getNumAtoms()
            x0 = np.zeros((N, 3), dtype=np.float32)
            v0 = np.zeros((N, 3), dtype=np.float32)

        x = torch.from_numpy(x0[None]).to(device)
        v = torch.from_numpy(v0[None]).to(device)
        # Dummy atom types and mask
        N = x.shape[1]
        atom_types = torch.zeros((1, N), dtype=torch.long, device=device)
        mask = torch.ones((1, N), dtype=torch.bool, device=device)

        # Learned jump
        with torch.no_grad():
            out = model(x, atom_types, mask)
            x_hat = (x + out["delta_x"]).cpu().numpy()[0]
            v_hat = (v + out["delta_v"]).cpu().numpy()[0]

        # One-step corrector
        x_star, v_star = one_step_corrector(x_hat, v_hat, pdb_path, md_conf)
        np.savez_compressed(Path(args.out) / f"{name}_hybrid_step.npz", x=x_star, v=v_star)
        logger.info(f"Saved hybrid step for {name}")


if __name__ == "__main__":
    main()

