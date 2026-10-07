"""Wall-clock study: baseline OpenMM vs. hybrid predictors, measured back-to-back.

The 2025 study extrapolated the baseline from a few un-warmed macro-steps. Here every
condition is warmed up (CUDA graph capture, OpenMM kernel compilation) and then timed
over the same number of macro-steps on an otherwise idle machine. The baseline is
timed on every requested OpenMM platform and the hybrid corrector uses the fastest
one, so the comparison is against the best available reference.

Example::

    python src/08_time_study.py --md-config configs/md.yaml \\
        --condition zero=zero mean=outputs/v2/mean_k4/best.pt flow=outputs/v2/flow_k4/best.pt \\
        --out-dir outputs/v2/time_study

Run it with nothing else on the GPU; contention distorts small-kernel timings badly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from kstep.common import LOGGER, configure_logging, load_trajectory, load_yaml, remove_com, write_json
from kstep.hybrid import CudaGraphPredictor, HybridOptions, ZeroPredictor, run_hybrid
from kstep.model import KStepModel, load_predictor
from kstep.openmm_corrector import OpenMMCorrector


def time_baseline(corrector: OpenMMCorrector, pos, vel, masses, full_steps: int, n_macro: int, warmup: int) -> float:
    """Seconds per simulated ps for plain MD that stores one frame per macro-step."""
    for _ in range(warmup):
        r = corrector.run(pos, vel, full_steps)
        pos, vel = remove_com(r.positions, r.velocities, masses)
    t0 = time.perf_counter()
    for _ in range(n_macro):
        r = corrector.run(pos, vel, full_steps)
        pos, vel = remove_com(r.positions, r.velocities, masses)
    return (time.perf_counter() - t0) / (n_macro * full_steps * corrector.micro_dt_ps)


def make_predictor(spec: str, device: torch.device, flow_steps: int | None):
    if spec == "zero":
        return ZeroPredictor()
    if ":" in spec and not Path(spec).exists():  # legacy checkpoint:yaml
        ckpt, cfg = spec.split(":", 1)
        return load_predictor(Path(ckpt), device, legacy_model_config=load_yaml(Path(cfg)))
    model = load_predictor(Path(spec), device)
    if flow_steps:
        model.cfg.flow_steps = flow_steps
    return CudaGraphPredictor(model) if isinstance(model, KStepModel) and device.type == "cuda" else model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md-config", type=Path, default=Path("configs/md.yaml"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--md-root", type=Path, default=Path("data/md"))
    p.add_argument("--molecules", nargs="*", default=None, help="Default: test split")
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--condition", nargs="+", required=True,
                   help="label=spec; spec is 'zero', a v2 checkpoint, or legacy 'ckpt.pt:model.yaml'")
    p.add_argument("--flow-steps", type=int, default=None)
    p.add_argument("--k-steps", type=int, default=4, help="Jump length for predictors without a stored k (zero)")
    p.add_argument("--quench-iterations", type=int, default=20)
    p.add_argument("--quench-mode", choices=["full", "thermal"], default="thermal")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--no-thermostat", dest="thermostat", action="store_false")
    p.add_argument("--corrector-fraction", type=float, default=0.05)
    p.add_argument("--macro-steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--platforms", nargs="+", default=["CUDA", "CPU"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    configure_logging()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    md_cfg = load_yaml(args.md_config)
    S = int(md_cfg.get("save_interval_steps", 50))
    jump_steps = args.k_steps * S
    corr_steps = max(1, int(round(args.corrector_fraction * jump_steps)))
    full_steps = jump_steps + corr_steps  # baseline frame cadence used for the MD timing
    mols = args.molecules or json.loads((args.splits_dir / "test.json").read_text())["molecules"]
    conditions = dict(c.split("=", 1) for c in args.condition)

    results: Dict[str, Dict] = {}
    for mol in mols:
        init = load_trajectory(args.md_root / mol / "trajectory.npz")
        masses = np.asarray(init["masses"], np.float64)
        pos, vel = remove_com(init["pos"][0].astype(np.float64), init["vel"][0].astype(np.float64), masses)
        res: Dict[str, Dict] = {"baseline": {}}
        for plat in args.platforms:
            try:
                corr = OpenMMCorrector(args.raw_root / mol, md_cfg, seed=1, platform=plat)
                res["baseline"][plat] = time_baseline(corr, pos, vel, masses, full_steps, args.macro_steps, args.warmup)
                LOGGER.info("%-24s baseline %-5s %.4f s/ps", mol, plat, res["baseline"][plat])
            except Exception as exc:  # platform unavailable
                LOGGER.warning("Platform %s failed: %s", plat, exc)
        best_plat = min(res["baseline"], key=res["baseline"].get)
        res["best_platform"] = best_plat
        res["baseline_s_per_ps"] = res["baseline"][best_plat]
        res["micro_step_s"] = res["baseline_s_per_ps"] * float(md_cfg.get("dt_fs", 2.0)) * 1e-3
        for label, spec in conditions.items():
            predictor = make_predictor(spec, device, args.flow_steps)
            k = getattr(getattr(predictor, "model", predictor), "k_steps", None) or args.k_steps
            jump_k = int(k) * S
            corr = OpenMMCorrector(args.raw_root / mol, md_cfg, seed=1, platform=best_plat)
            corr_k = max(1, int(round(args.corrector_fraction * jump_k)))
            opts = HybridOptions(jump_time_ps=jump_k * corr.micro_dt_ps, corrector_steps=corr_k,
                                 max_attempts=args.max_attempts, quench_iterations=args.quench_iterations,
                                 quench_mode=args.quench_mode,
                                 thermostat_K=float(md_cfg.get("temperature_K", 300.0)) if args.thermostat else 0.0,
                                 temperature_K=float(md_cfg.get("temperature_K", 300.0)))
            gen = torch.Generator(device=device).manual_seed(0)
            run_hybrid(predictor, corr, pos, vel, masses, init["atom_types"], args.warmup, opts, device, gen)
            if device.type == "cuda":
                torch.cuda.synchronize()
            s = run_hybrid(predictor, corr, pos, vel, masses, init["atom_types"], args.macro_steps, opts, device,
                           gen)["summary"]
            sim_ps = s["simulated_ps"]
            res[label] = {
                "k_steps": int(k),
                "force_call_savings": s["force_call_savings"],
                "accepted_fraction": s["accepted_fraction"],
                "s_per_ps": s["wall_clock_s"] / sim_ps,
                "model_s_per_ps": s["model_time_s"] / sim_ps,
                "corrector_s_per_ps": s["corrector_time_s"] / sim_ps,
                "model_latency_ms": 1e3 * s["model_time_s"] / args.macro_steps,
                "speedup_vs_baseline": res["baseline_s_per_ps"] / (s["wall_clock_s"] / sim_ps),
            }
            LOGGER.info("%-24s %-12s %.4f s/ps (model %.2f ms/step) -> speedup %.2fx", mol, label,
                        res[label]["s_per_ps"], res[label]["model_latency_ms"], res[label]["speedup_vs_baseline"])
            del predictor
        results[mol] = res

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"k_steps": args.k_steps, "save_interval": S, "corrector_steps": corr_steps, "macro_steps": args.macro_steps,
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"}
    write_json({"meta": meta, "molecules": results}, args.out_dir / "time_study.json")

    labels: List[str] = list(conditions)
    lines = [f"# Wall-clock study (k={args.k_steps}, {corr_steps} corrector steps, {meta['device']})", "",
             "Seconds of wall-clock per simulated ps (lower is better); speedup = baseline / hybrid. Hybrid runs use the evaluation settings (proposal checks, quench, thermostat); rejected steps fall back to MD and are included.", "",
             "| molecule | baseline (best platform) | " + " | ".join(f"{lb} s/ps (speedup, model ms/step)" for lb in labels) + " |",
             "|---|---|" + "---|" * len(labels)]
    for mol, r in results.items():
        cells = [f"{r[lb]['s_per_ps']:.4f} ({r[lb]['speedup_vs_baseline']:.2f}x, {r[lb]['model_latency_ms']:.2f})" for lb in labels]
        lines.append(f"| {mol} | {r['baseline_s_per_ps']:.4f} ({r['best_platform']}) | " + " | ".join(cells) + " |")
    mean_cells = [f"{np.mean([r[lb]['speedup_vs_baseline'] for r in results.values()]):.2f}x" for lb in labels]
    lines += ["| **mean speedup** | | " + " | ".join(mean_cells) + " |", ""]
    (args.out_dir / "time_study.md").write_text("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * (len(labels) + 1), 3.6))
        names = ["baseline"] + labels
        model_part = [0.0] + [np.mean([r[lb]["model_s_per_ps"] for r in results.values()]) for lb in labels]
        corr_part = [np.mean([r["baseline_s_per_ps"] for r in results.values()])] + \
                    [np.mean([r[lb]["corrector_s_per_ps"] for r in results.values()]) for lb in labels]
        total = [np.mean([r["baseline_s_per_ps"] for r in results.values()])] + \
                [np.mean([r[lb]["s_per_ps"] for r in results.values()]) for lb in labels]
        other = [max(t - m - c, 0.0) for t, m, c in zip(total, model_part, corr_part)]
        ax.bar(names, corr_part, color="#4C72B0", label="MD / corrector")
        ax.bar(names, model_part, bottom=corr_part, color="#DD8452", label="model")
        ax.bar(names, other, bottom=np.add(corr_part, model_part), color="#BBBBBB", label="other")
        ax.set_ylabel("wall-clock s per simulated ps")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.out_dir / "time_study.png", dpi=160)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
