"""Evaluate hybrid rollouts against the baseline MD (distributions + dynamics).

For each condition (``--hybrid label=dir``) and molecule, the hybrid trajectory
``dir/<molecule>.npz`` is compared with every k-th frame of the baseline, so both
series have (nearly) the same time between frames. If ``--reference2`` points at a
second baseline run with a different seed, the same comparison between the two
baselines is reported as ``noise_floor``: any hybrid number within that range is
indistinguishable from MD.

Outputs ``metrics.json`` (per molecule) and ``summary.md`` (means per split).

Example::

    python src/05_eval_drift_rdfs.py --baseline data/md --reference2 data/md_seed2 \\
        --hybrid zero=outputs/v2/hybrid/zero_k4 flow=outputs/v2/hybrid/flow_k4 \\
        --splits-dir data/splits --out-dir outputs/v2/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from kstep.common import LOGGER, configure_logging, load_trajectory, write_json
from kstep.metrics import Topology, compare

HEADLINE = [
    ("lag_rmsd_A_ratio", "lag-RMSD ratio", "1 = moves as far per step as MD"),
    ("torsion_step_deg_ratio", "torsion-step ratio", "1 = MD"),
    ("torsion_transitions_per_ps_ratio", "torsion-flip-rate ratio", "1 = MD"),
    ("torsion_js", "torsion JS", "0 = identical distribution"),
    ("angle_js", "angle JS", ""),
    ("bond_js", "bond JS", ""),
    ("rdf_l1", "pair-dist L1", ""),
    ("ekin_ratio", "<KE> ratio", "1 = correct temperature"),
]
RUN_FIELDS = ["force_call_savings", "accepted_fraction", "escalated_fraction", "fallback_fraction", "wall_clock_s",
              "model_time_s", "corrector_time_s", "simulated_ps"]


def frame_ps(traj: Dict) -> float:
    t = np.asarray(traj["time_ps"], dtype=np.float64)
    return float(np.median(np.diff(t)))


def load_splits(splits_dir: Path | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if splits_dir is None:
        return out
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.json"
        if path.exists():
            for mol in json.loads(path.read_text())["molecules"]:
                out[mol] = split
    return out


def evaluate_condition(base: Dict, hyb: Dict, ref2: Dict | None, topo: Topology) -> Dict:
    base_dt = frame_ps(base)
    meta = hyb["metadata"]
    hyb_dt = float(meta.get("macro_step_ps", frame_ps(hyb)))
    lag = max(1, int(round(hyb_dt / base_dt)))
    n = min(len(hyb["pos"]), len(base["pos"][::lag]))
    ref = {"pos": base["pos"][::lag][:n], "vel": base["vel"][::lag][:n]}
    test = {"pos": hyb["pos"][:n], "vel": hyb["vel"][:n]}
    masses = np.asarray(base["masses"], dtype=np.float64)
    out = {"metrics": compare(ref, test, topo, lag * base_dt, hyb_dt, masses), "frames": n, "lag_frames": lag,
           "run": {k: meta.get(k) for k in RUN_FIELDS}}
    if ref2 is not None:
        r2 = {"pos": ref2["pos"][::lag][:n], "vel": ref2["vel"][::lag][:n]}
        out["noise_floor"] = compare(ref, r2, topo, lag * base_dt, lag * base_dt, masses)
    return out


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.3f}" if abs(x) < 100 else f"{x:.0f}"


def summarise(results: Dict, splits: Dict[str, str], labels: List[str]) -> str:
    lines = ["# Hybrid evaluation summary", "",
             "Ratio columns are medians over molecules (ratios blow up for molecules where MD barely moves); "
             "other columns are means. *noise floor* = second baseline seed vs baseline, at the same frame spacing.", ""]
    groups = {"test": [m for m in results if splits.get(m) == "test"],
              "val": [m for m in results if splits.get(m) == "val"],
              "train": [m for m in results if splits.get(m) == "train"],
              "all": list(results)}
    for group, mols in groups.items():
        if not mols:
            continue
        lines += [f"## {group} molecules ({len(mols)}): {', '.join(sorted(mols))}", ""]
        header = "| condition | " + " | ".join(h[1] for h in HEADLINE) + " | force-call savings | accepted | wall s / ps |"
        lines += [header, "|" + "---|" * (len(HEADLINE) + 4)]
        conds = labels + ["noise_floor"]
        for cond in conds:
            vals, run_vals = [], {}
            for key, _, _ in HEADLINE:
                xs = []
                for m in mols:
                    if cond == "noise_floor":
                        any_label = next((lb for lb in labels if lb in results[m] and "noise_floor" in results[m][lb]), None)
                        if any_label:
                            xs.append(results[m][any_label]["noise_floor"].get(key, np.nan))
                    elif cond in results[m]:
                        xs.append(results[m][cond]["metrics"].get(key, np.nan))
                agg = np.nanmedian if key.endswith("_ratio") else np.nanmean
                vals.append(float(agg(xs)) if xs and np.isfinite(xs).any() else None)
            if cond != "noise_floor":
                runs = [results[m][cond]["run"] for m in mols if cond in results[m]]
                for f in ("force_call_savings", "accepted_fraction"):
                    v = [r.get(f) for r in runs if r.get(f) is not None]
                    run_vals[f] = float(np.mean(v)) if v else None
                spp = [r["wall_clock_s"] / r["simulated_ps"] for r in runs if r.get("simulated_ps")]
                run_vals["sps"] = float(np.mean(spp)) if spp else None
            extra = [fmt(run_vals.get("force_call_savings")), fmt(run_vals.get("accepted_fraction")), fmt(run_vals.get("sps"))] \
                if cond != "noise_floor" else ["–", "–", "–"]
            name = f"*{cond}*" if cond == "noise_floor" else cond
            lines.append(f"| {name} | " + " | ".join(fmt(v) for v in vals) + " | " + " | ".join(extra) + " |")
        lines.append("")
    lines += ["Metric notes: " + "; ".join(f"**{h[1]}** – {h[2]}" for h in HEADLINE if h[2]), ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True, help="data/md (per-molecule trajectory.npz)")
    p.add_argument("--reference2", type=Path, default=None,
                   help="Second baseline run (different seed) for the noise floor, or 'half' to split the baseline")
    p.add_argument("--hybrid", nargs="+", required=True, help="label=dir pairs; dir holds <molecule>.npz")
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    configure_logging()

    conditions = dict(item.split("=", 1) if "=" in item else (Path(item).name, item) for item in args.hybrid)
    splits = load_splits(args.splits_dir)
    results: Dict[str, Dict] = {}
    for mol_dir in sorted(d for d in args.baseline.iterdir() if (d / "trajectory.npz").exists()):
        mol = mol_dir.name
        runs = {lb: Path(d) / f"{mol}.npz" for lb, d in conditions.items() if (Path(d) / f"{mol}.npz").exists()}
        if not runs:
            continue
        base = load_trajectory(mol_dir / "trajectory.npz")
        ref2 = None
        if args.reference2 is not None and str(args.reference2) == "half":
            # No independent repeat (e.g. expensive QM): split the baseline in two halves
            half = len(base["pos"]) // 2
            ref2 = {**base, "pos": base["pos"][half:], "vel": base["vel"][half:]}
            base = {**base, "pos": base["pos"][:half], "vel": base["vel"][:half]}
        elif args.reference2 is not None and (args.reference2 / mol / "trajectory.npz").exists():
            ref2 = load_trajectory(args.reference2 / mol / "trajectory.npz")
        topo = Topology(base["pos"][0], base["atom_types"])
        results[mol] = {"split": splits.get(mol, "?")}
        for label, path in runs.items():
            results[mol][label] = evaluate_condition(base, load_trajectory(path), ref2, topo)
            m = results[mol][label]["metrics"]
            LOGGER.info("%-24s %-14s lagRMSD x%.2f  torsionJS %.3f  flips x%.2f", mol, label,
                        m.get("lag_rmsd_A_ratio", np.nan), m.get("torsion_js", np.nan),
                        m.get("torsion_transitions_per_ps_ratio", np.nan))
    if not results:
        raise SystemExit("No hybrid trajectories matched the baseline molecules")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"conditions": conditions, "molecules": results}, args.out_dir / "metrics.json")
    summary = summarise(results, splits, list(conditions))
    (args.out_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
