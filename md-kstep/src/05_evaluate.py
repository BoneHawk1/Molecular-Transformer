"""Evaluate energy drift, structural statistics, and efficiency of hybrid integrator runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from utils import configure_logging, ensure_dir, LOGGER

COVALENT_RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}


def _parse_json_field(raw: Any, expected: Union[type, Tuple[type, ...]], default: Any) -> Any:
    """Decode JSON-serialised NPZ fields, falling back to Python objects."""
    if raw is None:
        return default
    if isinstance(raw, np.ndarray):
        # np.load wraps pickled objects/strings in 0-d arrays
        if raw.shape == ():
            return _parse_json_field(raw.item(), expected, default)
        raw = raw.tolist()
    if isinstance(raw, expected):
        return raw
    if isinstance(raw, (str, bytes)):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return default
        return loaded if isinstance(loaded, expected) else default
    return default


def load_npz(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    metadata = _parse_json_field(data.get("metadata"), dict, {})
    nve_windows = _parse_json_field(data.get("nve_windows"), list, [])
    trajectory = {
        "pos": data["pos"],
        "vel": data["vel"],
        "Etot": data["Etot"],
        "time_ps": data["time_ps"],
        "atom_types": data["atom_types"],
        "metadata": metadata,
        "nve_windows": nve_windows,
    }
    return trajectory


def guess_bonds(positions_nm: np.ndarray, atom_types: np.ndarray, tol: float = 0.1) -> List[Tuple[int, int]]:
    coords = positions_nm * 10.0  # convert to Å
    num_atoms = coords.shape[0]
    bonds: List[Tuple[int, int]] = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            ri = COVALENT_RADII.get(int(atom_types[i]), 0.75)
            rj = COVALENT_RADII.get(int(atom_types[j]), 0.75)
            cutoff = ri + rj + tol
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= cutoff:
                bonds.append((i, j))
    return bonds


def build_angles(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    adjacency: Dict[int, List[int]] = {}
    for i, j in bonds:
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    angles: List[Tuple[int, int, int]] = []
    for center, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        for idx in range(len(neighbors)):
            for jdx in range(idx + 1, len(neighbors)):
                angles.append((neighbors[idx], center, neighbors[jdx]))
    return angles


def build_dihedrals(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    adjacency: Dict[int, List[int]] = {}
    for i, j in bonds:
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    dihedrals: List[Tuple[int, int, int, int]] = []
    for bond in bonds:
        i, j = bond
        for k in adjacency.get(j, []):
            if k == i:
                continue
            for l in adjacency.get(k, []):
                if l in (i, j):
                    continue
                dihedrals.append((i, j, k, l))
    # remove duplicates by normalizing tuples
    unique = set()
    ordered: List[Tuple[int, int, int, int]] = []
    for a, b, c, d in dihedrals:
        key = (a, b, c, d)
        rev_key = (d, c, b, a)
        if key in unique or rev_key in unique:
            continue
        unique.add(key)
        ordered.append(key)
    return ordered


def measure_bonds(positions_nm: np.ndarray, bonds: List[Tuple[int, int]]) -> np.ndarray:
    coords = positions_nm * 10.0
    lengths = [np.linalg.norm(coords[i] - coords[j]) for i, j in bonds]
    return np.asarray(lengths, dtype=np.float32)


def measure_angles(positions_nm: np.ndarray, triples: List[Tuple[int, int, int]]) -> np.ndarray:
    coords = positions_nm * 10.0
    values: List[float] = []
    for i, j, k in triples:
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        v1 /= np.linalg.norm(v1) + 1e-9
        v2 /= np.linalg.norm(v2) + 1e-9
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        values.append(np.degrees(np.arccos(cos_theta)))
    return np.asarray(values, dtype=np.float32)


def measure_dihedrals(positions_nm: np.ndarray, quads: List[Tuple[int, int, int, int]]) -> np.ndarray:
    coords = positions_nm * 10.0
    values: List[float] = []
    for i, j, k, l in quads:
        b0 = coords[j] - coords[i]
        b1 = coords[k] - coords[j]
        b2 = coords[l] - coords[k]
        b1 /= np.linalg.norm(b1) + 1e-9
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.linalg.norm(np.cross(v, b1)) * np.linalg.norm(w)
        angle = np.degrees(np.arctan2(y, x + 1e-9))
        values.append(angle)
    return np.asarray(values, dtype=np.float32)


def compute_rdf(positions_nm: np.ndarray, atom_types: np.ndarray, bins: np.ndarray) -> np.ndarray:
    coords = positions_nm * 10.0
    heavy_idx = np.where(atom_types > 1)[0]
    coords = coords[heavy_idx]
    if coords.shape[0] < 2:
        return np.zeros(len(bins) - 1, dtype=np.float32)
    dists = []
    for i in range(coords.shape[0]):
        diff = coords[i + 1:] - coords[i]
        d = np.linalg.norm(diff, axis=1)
        dists.append(d)
    if not dists:
        return np.zeros(len(bins) - 1, dtype=np.float32)
    d_all = np.concatenate(dists)
    hist, _ = np.histogram(d_all, bins=bins, density=True)
    return hist.astype(np.float32)


def energy_drift_per_ps(window: Dict, dt_fs: float) -> float:
    total = np.asarray(window["total"], dtype=np.float64)
    length_ps = len(total) * (dt_fs * 0.001)
    return float((total[-1] - total[0]) / max(length_ps, 1e-3))


def summarize_drift(windows: List[Dict], dt_fs: float) -> Dict[str, float]:
    if not windows:
        return {"median": float("nan"), "iqr": float("nan")}
    drifts = np.array([energy_drift_per_ps(window, dt_fs) for window in windows])
    return {
        "median": float(np.median(drifts)),
        "iqr": float(np.subtract(*np.percentile(drifts, [75, 25])))
    }


def evaluate_molecule(mol: str, baseline_traj: Dict, hybrid_traj: Dict, bins: np.ndarray) -> Dict:
    atom_types = baseline_traj["atom_types"]
    bonds = guess_bonds(baseline_traj["pos"][0], atom_types)
    angles = build_angles(bonds)
    dihedrals = build_dihedrals(bonds)

    # Multi-frame structural metrics
    T = min(len(baseline_traj["pos"]), len(hybrid_traj["pos"]))
    bond_sq = []
    angle_sq = []
    dihedral_sq = []
    rdf_l1 = []
    for t in range(T):
        b_bonds = measure_bonds(baseline_traj["pos"][t], bonds) if len(bonds) else np.array([])
        h_bonds = measure_bonds(hybrid_traj["pos"][t], bonds) if len(bonds) else np.array([])
        if b_bonds.size:
            bond_sq.append(np.mean((b_bonds - h_bonds) ** 2))

        b_ang = measure_angles(baseline_traj["pos"][t], angles) if len(angles) else np.array([])
        h_ang = measure_angles(hybrid_traj["pos"][t], angles) if len(angles) else np.array([])
        if b_ang.size:
            angle_sq.append(np.mean((b_ang - h_ang) ** 2))

        b_dih = measure_dihedrals(baseline_traj["pos"][t], dihedrals) if len(dihedrals) else np.array([])
        h_dih = measure_dihedrals(hybrid_traj["pos"][t], dihedrals) if len(dihedrals) else np.array([])
        if b_dih.size:
            dihedral_sq.append(np.mean((b_dih - h_dih) ** 2))

        b_rdf = compute_rdf(baseline_traj["pos"][t], atom_types, bins)
        h_rdf = compute_rdf(hybrid_traj["pos"][t], atom_types, bins)
        rdf_l1.append(np.mean(np.abs(b_rdf - h_rdf)))

    bond_rmse = float(np.sqrt(np.mean(bond_sq))) if bond_sq else float("nan")
    angle_rmse = float(np.sqrt(np.mean(angle_sq))) if angle_sq else float("nan")
    dihedral_rmse = float(np.sqrt(np.mean(dihedral_sq))) if dihedral_sq else float("nan")
    rdf_l1_mean = float(np.mean(rdf_l1)) if rdf_l1 else float("nan")

    dt_fs = baseline_traj["metadata"].get("config", {}).get("dt_fs", 2.0)
    return {
        "molecule": mol,
        "nve_drift_baseline": summarize_drift(baseline_traj["nve_windows"], dt_fs),
        "nve_drift_hybrid": summarize_drift(hybrid_traj["nve_windows"], dt_fs),
        "bond_diff_rmse": bond_rmse,
        "angle_diff_rmse": angle_rmse,
        "dihedral_diff_rmse": dihedral_rmse,
        "rdf_l1": rdf_l1_mean,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline trajectory root")
    parser.add_argument("--hybrid-runs", type=Path, required=True, help="Hybrid trajectory root")
    parser.add_argument("--hybrid-pattern", default="{molecule}.npz", help="Filename pattern for hybrid runs")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics JSON")
    parser.add_argument("--rdf-max", type=float, default=12.0, help="RDF upper bound in Å")
    parser.add_argument("--rdf-bins", type=int, default=60, help="Number of RDF bins")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    ensure_dir(args.out_dir)

    bins = np.linspace(0, args.rdf_max, args.rdf_bins + 1)
    metrics: List[Dict] = []

    for mol_dir in sorted(args.baseline.iterdir()):
        traj_path = mol_dir / "trajectory.npz"
        if not traj_path.exists():
            continue
        baseline_traj = load_npz(traj_path)
        hybrid_path = args.hybrid_runs / args.hybrid_pattern.format(molecule=mol_dir.name)
        if not hybrid_path.exists():
            LOGGER.warning("Hybrid trajectory missing for %s", mol_dir.name)
            continue
        hybrid_traj = load_npz(hybrid_path)
        metrics.append(evaluate_molecule(mol_dir.name, baseline_traj, hybrid_traj, bins))

    summary_path = args.out_dir / "metrics.json"
    summary = {
        "molecules": metrics,
        "mean_bond_rmse": float(np.nanmean([m["bond_diff_rmse"] for m in metrics])) if metrics else float("nan"),
        "mean_angle_rmse": float(np.nanmean([m["angle_diff_rmse"] for m in metrics])) if metrics else float("nan"),
        "mean_dihedral_rmse": float(np.nanmean([m["dihedral_diff_rmse"] for m in metrics])) if metrics else float("nan"),
        "mean_rdf_l1": float(np.nanmean([m["rdf_l1"] for m in metrics])) if metrics else float("nan"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", summary_path)


if __name__ == "__main__":
    main()
