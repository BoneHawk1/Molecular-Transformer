"""Create supervised (x_t, v_t) -> (x_{t+k}, v_{t+k}) datasets from MD trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from utils import (configure_logging, ensure_dir, remove_com, set_seed,
                   write_json, LOGGER)


def _load_trajectory(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    return {
        "pos": data["pos"],
        "vel": data["vel"],
        "box": data["box"],
        "masses": data["masses"],
        "atom_types": data["atom_types"],
        "time_ps": data["time_ps"],
        "metadata": metadata,
    }


def _enumerate_windows(num_frames: int, k_steps: int, stride: int) -> Iterable[int]:
    max_start = num_frames - k_steps
    for start in range(0, max_start, stride):
        yield start


def _process_window(traj: Dict, idx: int, k: int) -> Dict:
    pos = traj["pos"]
    vel = traj["vel"]
    masses = traj["masses"]

    x_t = pos[idx]
    v_t = vel[idx]
    x_tk = pos[idx + k]
    v_tk = vel[idx + k]

    x_t_centered, v_t_centered = remove_com(x_t, v_t, masses)
    x_tk_centered, v_tk_centered = remove_com(x_tk, v_tk, masses)

    return {
        "x_t": x_t_centered.astype(np.float32),
        "v_t": v_t_centered.astype(np.float32),
        "x_tk": x_tk_centered.astype(np.float32),
        "v_tk": v_tk_centered.astype(np.float32),
        "masses": masses.astype(np.float32),
        "atom_types": traj["atom_types"].astype(np.int64),
    }


def _random_rotation_matrix() -> np.ndarray:
    """Sample a random 3D rotation matrix using uniformly distributed quaternions."""
    u1, u2, u3 = np.random.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], dtype=np.float64)
    x, y, z, w = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return rotation


def _apply_rotation(sample: Dict[str, np.ndarray], rotation: np.ndarray) -> Dict[str, np.ndarray]:
    """Rotate positions and velocities for an existing sample."""
    return {
        "x_t": sample["x_t"] @ rotation.T,
        "v_t": sample["v_t"] @ rotation.T,
        "x_tk": sample["x_tk"] @ rotation.T,
        "v_tk": sample["v_tk"] @ rotation.T,
        "masses": sample["masses"],
        "atom_types": sample["atom_types"],
    }


def _save_dataset(samples: List[Dict], out_path: Path, molecule_ids: List[str], k: int) -> None:
    ensure_dir(out_path.parent)
    arrays = {
        key: np.array([sample[key] for sample in samples], dtype=object)
        for key in ["x_t", "v_t", "x_tk", "v_tk", "masses", "atom_types"]
    }
    np.savez(out_path, **arrays, molecule=np.array(molecule_ids), k_steps=k)
    LOGGER.info("Wrote %d samples to %s", len(samples), out_path)


def _build_splits(molecules: Sequence[str], splits_dir: Path, seed: int) -> None:
    ensure_dir(splits_dir)
    mols = list(molecules)
    rng = np.random.default_rng(seed)
    rng.shuffle(mols)
    n = len(mols)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    split_data = {
        "train": mols[:n_train],
        "val": mols[n_train:n_train + n_val],
        "test": mols[n_train + n_val:],
    }
    for split, items in split_data.items():
        write_json({"molecules": items}, splits_dir / f"{split}.json")
        LOGGER.info("%s split: %d molecules", split, len(items))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-root", type=Path, required=True, help="Directory containing per-molecule trajectory folders")
    parser.add_argument("--out-root", type=Path, required=True, help="Directory to write dataset npz files")
    parser.add_argument("--splits-dir", type=Path, required=True, help="Directory for JSON splits")
    parser.add_argument("--ks", nargs="+", type=int, default=[4, 8, 12], help="List of k-step horizons")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride when sampling windows")
    parser.add_argument("--max-samples-per-mol", type=int, default=10000, help="Cap samples per molecule per k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--augment-rotations",
        type=int,
        default=0,
        help="Number of random rotations to apply per sampled window (data augmentation)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    set_seed(args.seed)

    trajectory_dirs = sorted([p for p in args.md_root.iterdir() if p.is_dir()])
    if not trajectory_dirs:
        raise RuntimeError(f"No trajectory directories found in {args.md_root}")

    molecule_names = [p.name for p in trajectory_dirs]
    _build_splits(molecule_names, args.splits_dir, args.seed)

    for k in args.ks:
        samples: List[Dict] = []
        molecule_ids: List[str] = []
        for mol_dir in trajectory_dirs:
            npz_path = mol_dir / "trajectory.npz"
            if not npz_path.exists():
                LOGGER.warning("Missing trajectory for %s", mol_dir.name)
                continue
            traj = _load_trajectory(npz_path)
            num_frames = traj["pos"].shape[0]
            window_indices = list(_enumerate_windows(num_frames, k, args.stride))
            if args.max_samples_per_mol > 0 and len(window_indices) > args.max_samples_per_mol:
                window_indices = list(np.random.choice(window_indices, args.max_samples_per_mol, replace=False))
            for idx in window_indices:
                samples.append(_process_window(traj, idx, k))
                molecule_ids.append(mol_dir.name)
                if args.augment_rotations > 0:
                    base_sample = samples[-1]
                    for _ in range(args.augment_rotations):
                        rotation = _random_rotation_matrix()
                        samples.append(_apply_rotation(base_sample, rotation))
                        molecule_ids.append(mol_dir.name)
        out_path = args.out_root / f"dataset_k{k}.npz"
        _save_dataset(samples, out_path, molecule_ids, k)


if __name__ == "__main__":
    main()
