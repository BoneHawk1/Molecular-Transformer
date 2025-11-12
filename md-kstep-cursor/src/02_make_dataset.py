from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from utils import ensure_dir, get_logger, remove_com_translation, remove_com_velocity, save_npz


def load_traj(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as d:
        return {k: d[k] for k in d.files}


def make_windows(
    traj: Dict[str, np.ndarray],
    k: int,
    stride: int,
    masses: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    pos = traj["pos"]  # [T,N,3] nm
    vel = traj["vel"]  # [T,N,3] nm/ps
    T, N, _ = pos.shape

    # indices
    idx_t = np.arange(0, T - k, stride, dtype=np.int64)
    idx_tk = idx_t + k

    x_t = pos[idx_t]
    v_t = vel[idx_t]
    x_tk = pos[idx_tk]
    v_tk = vel[idx_tk]

    if masses is None:
        masses = np.ones((N,), dtype=np.float32)

    # remove COM at both ends
    x_t = remove_com_translation(x_t, masses)
    v_t = remove_com_velocity(v_t, masses)
    x_tk = remove_com_translation(x_tk, masses)
    v_tk = remove_com_velocity(v_tk, masses)

    atom_types = np.zeros((N,), dtype=np.int64)  # placeholder; can be set from topology

    return {
        "x_t": x_t,
        "v_t": v_t,
        "x_tk": x_tk,
        "v_tk": v_tk,
        "atom_types": atom_types,
    }


def main():
    parser = argparse.ArgumentParser(description="Pack k-step windows from MD trajectories")
    parser.add_argument("--md", type=str, required=True, help="Directory with *_traj.npz baseline files")
    parser.add_argument("--splits", type=str, required=True, help="Directory to write train/val/test JSON")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", type=str, required=True, help="Output dataset npz path")
    parser.add_argument("--split_ratio", type=str, default="8,1,1", help="train,val,test ratios by molecules")
    args = parser.parse_args()

    md_dir = Path(args.md)
    ensure_dir(Path(args.splits))
    ensure_dir(Path(args.out).parent)
    logger = get_logger()

    files = sorted(md_dir.glob("*_traj.npz"))
    assert files, f"No trajectories found in {md_dir}"

    mol_windows: List[Dict[str, np.ndarray]] = []
    mol_ids: List[int] = []

    for mid, f in enumerate(files):
        traj = load_traj(f)
        w = make_windows(traj, k=args.k, stride=args.stride)
        n = w["x_t"].shape[0]
        logger.info(f"{f.name}: {n} windows")
        mol_windows.append(w)
        mol_ids.extend([mid] * n)

    # concatenate across molecules
    x_t = np.concatenate([w["x_t"] for w in mol_windows], axis=0)
    v_t = np.concatenate([w["v_t"] for w in mol_windows], axis=0)
    x_tk = np.concatenate([w["x_tk"] for w in mol_windows], axis=0)
    v_tk = np.concatenate([w["v_tk"] for w in mol_windows], axis=0)
    atom_types = mol_windows[0]["atom_types"]  # assume same N and types across windows of a molecule
    molecule_id = np.array(mol_ids, dtype=np.int64)

    save_npz(args.out, x_t=x_t, v_t=v_t, x_tk=x_tk, v_tk=v_tk, atom_types=atom_types, molecule_id=molecule_id)
    logger.info(f"Saved dataset: {args.out} ({x_t.shape[0]} samples)")

    # Write splits by molecules
    ratios = list(map(int, args.split_ratio.split(',')))
    total = sum(ratios)
    num_mols = len(files)
    n_train = int(ratios[0] / total * num_mols)
    n_val = int(ratios[1] / total * num_mols)
    indices = np.arange(num_mols)
    np.random.seed(42)
    np.random.shuffle(indices)
    splits = {
        "train": indices[:n_train].tolist(),
        "val": indices[n_train:n_train+n_val].tolist(),
        "test": indices[n_train+n_val:].tolist(),
    }
    with open(Path(args.splits) / "train.json", "w") as f:
        json.dump(splits["train"], f)
    with open(Path(args.splits) / "val.json", "w") as f:
        json.dump(splits["val"], f)
    with open(Path(args.splits) / "test.json", "w") as f:
        json.dump(splits["test"], f)
    logger.info(f"Wrote splits to {args.splits}")


if __name__ == "__main__":
    main()

