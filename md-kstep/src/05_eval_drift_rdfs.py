from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_rdf, ensure_dir, get_logger


def energy_drift(Etot: np.ndarray, dt_fs: float, window_ps: float) -> float:
    # Linear drift per ps over a window (simple slope)
    T = len(Etot)
    steps_per_ps = int(1000.0 / dt_fs)
    win_steps = int(window_ps * steps_per_ps)
    if win_steps < 2:
        return 0.0
    x = np.arange(win_steps)
    drifts = []
    for start in range(0, T - win_steps, win_steps):
        y = Etot[start:start+win_steps]
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # convert slope per step to per ps
        drifts.append(m * steps_per_ps)
    return float(np.median(drifts)) if drifts else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate drift and RDFs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--md", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dt_fs", type=float, default=2.0)
    parser.add_argument("--rdf_rmax_nm", type=float, default=1.2)
    parser.add_argument("--rdf_bin_nm", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    logger = get_logger()

    # Energy drift per molecule
    drifts = []
    for f in sorted(Path(args.md).glob("*_traj.npz")):
        with np.load(f) as d:
            Etot = d["Etot"]
            drifts.append(energy_drift(Etot, args.dt_fs, window_ps=100.0))
    if drifts:
        plt.figure()
        plt.boxplot(drifts)
        plt.ylabel("Energy drift (kJ/mol/ps)")
        plt.savefig(out_dir / "energy_drift_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close()

    # RDF using first trajectory as example
    first = sorted(Path(args.md).glob("*_traj.npz"))
    if first:
        with np.load(first[0]) as d:
            r, g = compute_rdf(d["pos"], None, args.rdf_rmax_nm, args.rdf_bin_nm)
        np.savez(out_dir / "rdf.npz", r=r, g=g)
        plt.figure()
        plt.plot(r, g)
        plt.xlabel("r (nm)")
        plt.ylabel("g(r)")
        plt.savefig(out_dir / "rdf.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()

