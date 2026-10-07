"""Cut baseline trajectories into (x_t, v_t) -> (x_{t+k}, v_{t+k}) training windows.

Writes one ``dataset_k{k}.npz`` per horizon in the flat ``kstep-dataset-v2`` format
(see :mod:`kstep.data`) and molecule-disjoint train/val/test splits. Existing split
files are kept unless ``--resplit`` is given, so retraining does not silently move
molecules between splits.

Rotation augmentation is no longer needed (the models are exactly equivariant), and
the old ``--augment-rotations`` flag has been removed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from kstep.common import LOGGER, configure_logging, load_trajectory, write_json
from kstep.data import build_windows, save_dataset


def build_splits(molecules: Sequence[str], splits_dir: Path, seed: int, resplit: bool) -> None:
    if not resplit and all((splits_dir / f"{s}.json").exists() for s in ("train", "val", "test")):
        LOGGER.info("Keeping existing splits in %s", splits_dir)
        return
    mols = list(molecules)
    np.random.default_rng(seed).shuffle(mols)
    n_train = int(0.7 * len(mols))
    n_val = max(1, int(0.15 * len(mols)))
    split = {"train": mols[:n_train], "val": mols[n_train:n_train + n_val], "test": mols[n_train + n_val:]}
    for name, items in split.items():
        write_json({"molecules": items}, splits_dir / f"{name}.json")
        LOGGER.info("%s split: %d molecules", name, len(items))


def frame_dt_ps(traj: dict) -> float:
    t = np.asarray(traj.get("time_ps", []), dtype=np.float64)
    if t.size > 1:
        return float(np.median(np.diff(t)))
    cfg = traj["metadata"].get("config", {})
    return float(cfg.get("dt_fs", 2.0) * cfg.get("save_interval_steps", 1) * 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-root", type=Path, required=True, help="Directory of per-molecule folders with trajectory.npz")
    parser.add_argument("--out-root", type=Path, required=True, help="Where to write dataset_k*.npz")
    parser.add_argument("--splits-dir", type=Path, required=True, help="Directory for train/val/test JSON")
    parser.add_argument("--ks", nargs="+", type=int, default=[4, 8, 12], help="k-step horizons (in stored frames)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride between window starts")
    parser.add_argument("--max-samples-per-mol", type=int, default=0, help="Cap windows per molecule (0 = no cap)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Discard this many initial frames (equilibration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resplit", action="store_true", help="Regenerate splits even if they exist")
    args = parser.parse_args()
    configure_logging()

    mol_dirs = sorted(p for p in args.md_root.iterdir() if (p / "trajectory.npz").exists())
    if not mol_dirs:
        raise RuntimeError(f"No */trajectory.npz under {args.md_root}")
    build_splits([p.name for p in mol_dirs], args.splits_dir, args.seed, args.resplit)

    trajs = {}
    for d in mol_dirs:
        traj = load_trajectory(d / "trajectory.npz")
        traj["pos"] = traj["pos"][args.skip_frames:]
        traj["vel"] = traj["vel"][args.skip_frames:]
        trajs[d.name] = traj
    dts = {name: frame_dt_ps(t) for name, t in trajs.items()}
    if len(set(np.round(list(dts.values()), 9))) > 1:
        LOGGER.warning("Trajectories have different frame spacings: %s", dts)
    rng = np.random.default_rng(args.seed)
    for k in args.ks:
        parts, names = [], []
        for name, traj in trajs.items():
            parts.append(build_windows(traj, k, args.stride, args.max_samples_per_mol, rng))
            names.append(name)
        save_dataset(args.out_root / f"dataset_k{k}.npz", parts, names, k, dts[names[0]])


if __name__ == "__main__":
    main()
