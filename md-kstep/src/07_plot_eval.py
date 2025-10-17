"""Generate plots from evaluation metrics and raw trajectories.

Produces:
- drift_median_bar.png: baseline vs. hybrid median energy drift per molecule
- drift_median_scatter.png: scatter of baseline vs. hybrid median drift
- structural_bar_*.png: per-molecule bars for bond/angle/dihedral/RDF metrics
- structural_hist_*.png: histograms across molecules for the above metrics
- rdf_overlay_<mol>.png: baseline vs. hybrid RDF overlays per molecule
- rdf_overlay_mean.png: mean RDF across molecules with std bands
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import configure_logging, ensure_dir, LOGGER


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _load_metrics(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    md = json.loads(str(data["metadata"])) if "metadata" in data else {}
    nve_windows = json.loads(str(data.get("nve_windows", "[]")))
    traj = {
        "pos": data["pos"],
        "atom_types": data["atom_types"],
        "time_ps": data.get("time_ps"),
        "metadata": md,
        "nve_windows": nve_windows,
    }
    return traj


def _guess_bonds(positions_nm: np.ndarray, atom_types: np.ndarray, tol: float = 0.1) -> List[Tuple[int, int]]:
    COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02}
    coords = positions_nm * 10.0
    n = coords.shape[0]
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_RADII.get(int(atom_types[i]), 0.75)
            rj = COVALENT_RADII.get(int(atom_types[j]), 0.75)
            cutoff = ri + rj + tol
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                bonds.append((i, j))
    return bonds


def _compute_rdf(positions_nm: np.ndarray, atom_types: np.ndarray, bins: np.ndarray) -> np.ndarray:
    coords = positions_nm * 10.0
    heavy_idx = np.where(atom_types > 1)[0]
    coords = coords[heavy_idx]
    if coords.shape[0] < 2:
        return np.zeros(len(bins) - 1, dtype=np.float32)
    dists = []
    for i in range(coords.shape[0]):
        diff = coords[i + 1:] - coords[i]
        d = np.linalg.norm(diff, axis=1)
        if d.size:
            dists.append(d)
    if not dists:
        return np.zeros(len(bins) - 1, dtype=np.float32)
    d_all = np.concatenate(dists)
    hist, _ = np.histogram(d_all, bins=bins, density=True)
    return hist.astype(np.float32)


def plot_drift(metrics: Dict, out_dir: Path) -> None:
    mols = [m["molecule"] for m in metrics["molecules"]]
    base = np.array([m["nve_drift_baseline"]["median"] for m in metrics["molecules"]], dtype=float)
    hybr = np.array([m["nve_drift_hybrid"]["median"] for m in metrics["molecules"]], dtype=float)

    x = np.arange(len(mols))
    w = 0.4
    plt.figure(figsize=(max(8, len(mols) * 0.6), 4))
    plt.bar(x - w / 2, base, width=w, label="baseline")
    plt.bar(x + w / 2, hybr, width=w, label="hybrid")
    plt.xticks(x, mols, rotation=60, ha="right")
    plt.ylabel("Median drift (kJ/mol/ps)")
    plt.title("NVE energy drift per molecule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "drift_median_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    lim = np.nanmax(np.abs(np.concatenate([base, hybr])))
    lim = 1.05 * lim if np.isfinite(lim) and lim > 0 else 1.0
    plt.scatter(base, hybr)
    plt.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1)
    plt.xlabel("Baseline median drift")
    plt.ylabel("Hybrid median drift")
    plt.title("Drift: hybrid vs. baseline (median)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "drift_median_scatter.png", dpi=200)
    plt.close()


def plot_structural(metrics: Dict, out_dir: Path) -> None:
    mols = [m["molecule"] for m in metrics["molecules"]]
    series = {
        "bond": np.array([m["bond_diff_rmse"] for m in metrics["molecules"]], dtype=float),
        "angle": np.array([m["angle_diff_rmse"] for m in metrics["molecules"]], dtype=float),
        "dihedral": np.array([m["dihedral_diff_rmse"] for m in metrics["molecules"]], dtype=float),
        "rdf": np.array([m["rdf_l1"] for m in metrics["molecules"]], dtype=float),
    }
    ylabels = {
        "bond": "RMSE (nm)",
        "angle": "RMSE (deg)",
        "dihedral": "RMSE (deg)",
        "rdf": "L1 distance",
    }

    for key, arr in series.items():
        plt.figure(figsize=(max(8, len(mols) * 0.6), 4))
        plt.bar(np.arange(len(mols)), arr)
        plt.xticks(range(len(mols)), mols, rotation=60, ha="right")
        plt.ylabel(ylabels[key])
        plt.title(f"{key.capitalize()} per molecule")
        plt.tight_layout()
        plt.savefig(out_dir / f"structural_bar_{key}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(5, 4))
        sns.histplot(arr[~np.isnan(arr)], bins=12, kde=True)
        plt.xlabel(ylabels[key])
        plt.title(f"{key.capitalize()} distribution across molecules")
        plt.tight_layout()
        plt.savefig(out_dir / f"structural_hist_{key}.png", dpi=200)
        plt.close()


def plot_rdf_overlays(baseline_root: Path, hybrid_root: Path, out_dir: Path, rdf_max: float, rdf_bins: int) -> None:
    bins = np.linspace(0, rdf_max, rdf_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    base_curves = []
    hybrid_curves = []
    mols: List[str] = []
    for mol_dir in sorted(baseline_root.iterdir()):
        traj_path = mol_dir / "trajectory.npz"
        if not traj_path.exists():
            continue
        hyb_path = hybrid_root / f"{mol_dir.name}.npz"
        if not hyb_path.exists():
            continue
        b = _load_npz(traj_path)
        h = _load_npz(hyb_path)
        atom_types = b["atom_types"]
        # average RDF across frames
        b_rdfs = []
        h_rdfs = []
        T = min(len(b["pos"]), len(h["pos"]))
        for t in range(T):
            b_rdfs.append(_compute_rdf(b["pos"][t], atom_types, bins))
            h_rdfs.append(_compute_rdf(h["pos"][t], atom_types, bins))
        if not b_rdfs:
            continue
        base_mean = np.mean(np.stack(b_rdfs), axis=0)
        hyb_mean = np.mean(np.stack(h_rdfs), axis=0)
        base_curves.append(base_mean)
        hybrid_curves.append(hyb_mean)
        mols.append(mol_dir.name)

        plt.figure(figsize=(6, 4))
        plt.plot(centers, base_mean, label="baseline")
        plt.plot(centers, hyb_mean, label="hybrid")
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title(f"RDF overlay: {mol_dir.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"rdf_overlay_{_sanitize(mol_dir.name)}.png", dpi=200)
        plt.close()

    if base_curves:
        base_arr = np.stack(base_curves)
        hyb_arr = np.stack(hybrid_curves)
        plt.figure(figsize=(6, 4))
        b_mean = base_arr.mean(axis=0)
        b_std = base_arr.std(axis=0)
        h_mean = hyb_arr.mean(axis=0)
        h_std = hyb_arr.std(axis=0)
        plt.plot(centers, b_mean, label="baseline")
        plt.fill_between(centers, b_mean - b_std, b_mean + b_std, alpha=0.2)
        plt.plot(centers, h_mean, label="hybrid")
        plt.fill_between(centers, h_mean - h_std, h_mean + h_std, alpha=0.2)
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("RDF overlay: mean ± std across molecules")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rdf_overlay_mean.png", dpi=200)
        plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=Path("outputs/eval/metrics.json"))
    p.add_argument("--baseline", type=Path, default=Path("data/md"))
    p.add_argument("--hybrid", type=Path, default=Path("outputs/hybrid"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/plots"))
    p.add_argument("--rdf-max", type=float, default=12.0)
    p.add_argument("--rdf-bins", type=int, default=60)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging()
    ensure_dir(args.out_dir)

    metrics = _load_metrics(args.metrics)
    LOGGER.info("Plotting drift metrics...")
    plot_drift(metrics, args.out_dir)

    LOGGER.info("Plotting structural metrics...")
    plot_structural(metrics, args.out_dir)

    LOGGER.info("Plotting RDF overlays...")
    plot_rdf_overlays(args.baseline, args.hybrid, args.out_dir, args.rdf_max, args.rdf_bins)

    LOGGER.info("Saved plots to %s", args.out_dir)


if __name__ == "__main__":
    main()

