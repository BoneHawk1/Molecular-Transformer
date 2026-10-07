# Hybrid Integrator Findings — 2025-11-12

This document summarizes the hybrid integrator experiments, evaluations, and runtime study performed across EGNN and Transformer models on the small‑molecule benchmark in this repository. It also captures practical guidance on when hybrids can deliver actual speedups and changes made to the codebase to improve inference performance.

## Setup

- Baseline MD: OpenMM (CUDA), `configs/md.yaml`, save interval 50 steps, long NVE windows (100 ps every 200 ps) for apples‑to‑apples drift.
- Datasets: `data/md/*/trajectory.npz` (12 small molecules), initial states from frame 0.
- Models and checkpoints:
  - EGNN: `outputs/checkpoints_aug_wide/best.pt`
  - Transformer‑EGNN: `outputs/checkpoints_transformer_aug_wide/best.pt`
- Hybrid rollouts (macro‑step k=4, i.e., S = 4 × 50 = 200 baseline micro‑steps per macro‑step):
  - Corrector fractions tested: 0.025 (~40× fewer force calls), 0.05 (~20×), 0.10 (~10×)
  - CUDA device used for both model and OpenMM; “long” runs include 100 ps NVE windows.

## Evaluation Results (means across 12 molecules)

Lower is better for structural metrics; drift reported separately below. All runs use long NVE windows for fair drift comparison.

- EGNN (0.05 fraction, ~20× force‑call savings)
  - mean_bond_rmse ≈ 0.03898, mean_angle_rmse ≈ 4.2346, mean_dihedral_rmse ≈ 8.2133, mean_rdf_l1 ≈ 0.03368
  - Artifacts: `outputs/hybrid_cf05_long`, `outputs/eval_cf05_long/metrics.json`

- Transformer (0.05 fraction, ~20×)
  - mean_bond_rmse ≈ 0.03563, mean_angle_rmse ≈ 3.8217, mean_dihedral_rmse ≈ 7.8785, mean_rdf_l1 ≈ 0.03379
  - Artifacts: `outputs/hybrid_transformer_cf05_long`, `outputs/eval_transformer_cf05_long/metrics.json`

- Transformer (0.10 fraction, ~10×)
  - mean_bond_rmse ≈ 0.03214, mean_angle_rmse ≈ 3.8211, mean_dihedral_rmse ≈ 7.9544, mean_rdf_l1 ≈ 0.03472
  - Artifacts: `outputs/hybrid_transformer_cf10_long`, `outputs/eval_transformer_cf10_long/metrics.json`

- Transformer (0.025 fraction, ~40×)
  - mean_bond_rmse ≈ 0.04778, mean_angle_rmse ≈ 3.5743, mean_dihedral_rmse ≈ 7.9255, mean_rdf_l1 ≈ 0.03275
  - Artifacts: `outputs/hybrid_transformer_cf0025_long`, `outputs/eval_transformer_cf0025_long/metrics.json`

### Drift (100 ps windows)

- With long windows, hybrid median drift per molecule remains small for 0.05 and 0.10 and is consistent across molecules. At 0.025, medians remain small but can shift slightly more. Using matched 100 ps windows avoids the noise seen with very short windows.

## Time Study (actual wall‑clock)

Method: For each molecule, measured baseline micro‑step time via a short OpenMM run and extrapolated to 50 macro‑steps (k=4). Used NPZ metadata wall‑clock for hybrid EGNN/Transformer 0.05 runs. Plots and JSON produced to `outputs/time_study`.

- Mean speedup (baseline / hybrid wall‑clock) across molecules:
  - EGNN 0.05: ≈ 0.053 (hybrid ~19× slower than baseline)
  - Transformer 0.05: ≈ 0.045 (hybrid ~22× slower than baseline)
- Takeaway: On these small molecules, model inference dominates; hybrid does not beat baseline wall‑clock despite fewer force calls.
- Artifacts: `outputs/time_study/summary.json`, `time_bar_methods.png`, `accuracy_per_step.png`

## When Hybrids Win (Break‑Even Intuition)

Per macro‑step times: `T_base = S · t_force`, `T_hybrid = t_model + f · S · t_force` where S = k · save_interval and f is the corrector fraction.

- Break‑even condition: `t_model < (1 − f) · S · t_force`
- With measurements here (k=4 → S=200, f=0.05, t_force ≈ 1.03e−4 s), RHS ≈ 0.0196 s, while measured model time per macro‑step ≫ RHS. Increasing k to 8 doubles RHS, still not enough on this setup.
- Scenarios that can flip the inequality:
  - Expensive forces (QM/QM/MM or CPU baselines): larger `t_force`
  - Larger k (validated stability): larger S
  - Cheaper models and optimized inference (mixed precision, compile, compact architectures): smaller `t_model`

## Code Changes (Performance + Controls)

- Corrector cadence controls (fraction or fixed micro‑steps): `src/06_hybrid_integrate.py`
  - Flags: `--corrector-fraction`, `--corrector-steps`
- Mixed‑precision and optional model compilation for inference:
  - Flags: `--amp`, `--compile`
  - Autocast for CUDA during inference and optional `torch.compile` (first run may autotune).
- Plotting utilities for drift/structure/RDF overlays: `src/07_plot_eval.py`
- Time study script (baseline estimate + hybrid times + accuracy per step): `src/08_time_study.py`

## Recommendations

1) For near‑term speedups, prioritize QM/MM (or other expensive force models). The existing hybrid already saves 90–97.5% of force calls; when each call is costly, hybrids can deliver large wall‑clock gains.

2) In parallel, train a compact EGNN for classical MD:
   - Fewer layers (3–4), smaller hidden dim (64–96), mixed‑precision and compiled inference.
   - Validate larger k (e.g., 8–16) with small corrector fractions (0.05–0.1) while monitoring drift with long windows.

3) Operational tips:
   - Warm up compiled models (run a short rollout before timing) to amortize autotune.
   - Consider batching opportunities (predict multiple macro‑steps or molecules per call) to amortize Python/launch overhead.
   - Cache/reuse neighborhoods where valid to cut graph‑building cost.

## Reproduction (examples)

- Transformer 0.05 long (CUDA):
  ```
  conda run -n kstep python src/06_hybrid_integrate.py \
    --checkpoint outputs/checkpoints_transformer_aug_wide/best.pt \
    --model-config configs/model.yaml --md-config configs/md.yaml \
    --molecule data/raw/CC(C)CO --initial-md data/md/CC(C)CO/trajectory.npz \
    --frame 0 --steps 50 --k-steps 4 --corrector-fraction 0.05 \
    --device cuda --out outputs/hybrid_transformer_cf05_long/CC(C)CO.npz
  ```

- EGNN 0.05 long (CUDA):
  ```
  conda run -n kstep python src/06_hybrid_integrate.py \
    --checkpoint outputs/checkpoints_aug_wide/best.pt \
    --model-config configs/model_egnn.yaml --md-config configs/md.yaml \
    --molecule data/raw/CC(C)CO --initial-md data/md/CC(C)CO/trajectory.npz \
    --frame 0 --steps 50 --k-steps 4 --corrector-fraction 0.05 \
    --device cuda --out outputs/hybrid_cf05_long/CC(C)CO.npz
  ```

- Enable mixed‑precision + compile (first run pays autotune):
  ```
  --amp --compile
  ```

---

Date: 2025‑11‑12

