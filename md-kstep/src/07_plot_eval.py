"""Plots for an evaluation produced by ``src/05_evaluate.py``.

Produces in ``--out-dir``:
  metrics_overview.png   headline metrics per condition (test molecules), with the
                         seed-to-seed noise floor shaded
  torsions_<mol>.png     torsion-angle distributions, baseline vs. each condition
  lag_rmsd_<mol>.png     how far the molecule moves per macro-step (Kabsch RMSD between
                         consecutive frames) – the corrector-only control sits near 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kstep.common import configure_logging, load_trajectory  # noqa: E402
from kstep.geometry import dihedrals_deg, kabsch_rmsd  # noqa: E402
from kstep.metrics import Topology  # noqa: E402

PANELS = [
    ("lag_rmsd_A_ratio", "lag-RMSD ratio\n(1 = MD)", 1.0),
    ("torsion_transitions_per_ps_ratio", "torsion flip-rate ratio\n(1 = MD)", 1.0),
    ("torsion_js", "torsion JS divergence\n(0 = MD)", 0.0),
    ("angle_js", "angle JS divergence\n(0 = MD)", 0.0),
    ("ekin_ratio", "<KE> ratio\n(1 = MD)", 1.0),
]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def overview(metrics: Dict, labels: List[str], mols: List[str], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.0 * len(PANELS), 3.4))
    for ax, (key, name, target) in zip(axes, PANELS):
        vals = [[metrics[m][lb]["metrics"].get(key, np.nan) for m in mols if lb in metrics[m]] for lb in labels]
        means = [np.nanmean(v) if v else np.nan for v in vals]
        ax.bar(range(len(labels)), means, color=PALETTE[: len(labels)])
        for i, v in enumerate(vals):
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=8, color="k", zorder=3)
        floor = [metrics[m][lb]["noise_floor"][key] for m in mols for lb in labels[:1]
                 if lb in metrics[m] and "noise_floor" in metrics[m][lb]]
        if floor:
            lo, hi = np.nanmin(floor), np.nanmax(floor)
            ax.axhspan(lo, hi, color="0.85", zorder=0, label="noise floor")
        ax.axhline(target, color="0.3", lw=0.8, ls="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def torsion_plot(mol: str, base: Dict, runs: Dict[str, Dict], lag: int, out: Path) -> None:
    topo = Topology(base["pos"][0], base["atom_types"])
    if not len(topo.torsions):
        return
    n_show = min(4, len(topo.torsions))
    fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 2.8), squeeze=False)
    bins = np.linspace(-180, 180, 37)
    ref = dihedrals_deg(base["pos"][::lag], topo.torsions[:n_show])
    for t, ax in enumerate(axes[0]):
        ax.hist(ref[:, t], bins=bins, density=True, color="0.75", label="baseline MD")
        for (label, traj), col in zip(runs.items(), PALETTE):
            phi = dihedrals_deg(traj["pos"], topo.torsions[t:t + 1])[:, 0]
            h, _ = np.histogram(phi, bins=bins, density=True)
            ax.step(bins[:-1], h, where="post", color=col, label=label)
        ax.set_title("torsion " + "-".join(map(str, topo.torsions[t])), fontsize=8)
        ax.set_xlabel("φ (deg)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(mol, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def lag_plot(mol: str, base: Dict, runs: Dict[str, Dict], lag: int, out: Path, n: int = 300) -> None:
    def series(pos):
        idx = np.arange(min(n, len(pos) - 1))
        return np.array([kabsch_rmsd(pos[i], pos[i + 1], base["masses"]) for i in idx]) * 10

    fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.linspace(0, 3.0, 61)
    ax.hist(series(base["pos"][::lag]), bins=bins, density=True, color="0.75", label="baseline MD")
    for (label, traj), col in zip(runs.items(), PALETTE):
        h, _ = np.histogram(series(traj["pos"]), bins=bins, density=True)
        ax.step(bins[:-1], h, where="post", color=col, label=label)
    ax.set_xlabel("RMSD between consecutive frames (Å)")
    ax.set_title(mol, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, required=True, help="Directory with metrics.json from 05_evaluate.py")
    p.add_argument("--baseline", type=Path, default=Path("data/md"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--molecules", nargs="*", default=None, help="Molecules for per-molecule plots (default: test split)")
    args = p.parse_args()
    configure_logging()
    out = args.out_dir or args.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.eval_dir / "metrics.json").read_text())
    conditions: Dict[str, str] = data["conditions"]
    metrics = data["molecules"]
    labels = list(conditions)
    test = [m for m, r in metrics.items() if r.get("split") == "test"] or list(metrics)
    overview(metrics, labels, test, out / "metrics_overview.png", f"test molecules ({len(test)})")
    overview(metrics, labels, list(metrics), out / "metrics_overview_all.png", f"all molecules ({len(metrics)})")
    for mol in args.molecules or test:
        base = load_trajectory(args.baseline / mol / "trajectory.npz")
        runs = {lb: load_trajectory(Path(d) / f"{mol}.npz") for lb, d in conditions.items() if (Path(d) / f"{mol}.npz").exists()}
        lag = next(iter(metrics[mol][lb]["lag_frames"] for lb in labels if lb in metrics[mol]))
        torsion_plot(mol, base, runs, lag, out / f"torsions_{_safe(mol)}.png")
        lag_plot(mol, base, runs, lag, out / f"lag_rmsd_{_safe(mol)}.png")


if __name__ == "__main__":
    main()
