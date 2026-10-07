# CLAUDE.md

Guidance for Claude Code (and other agents) working in `md-kstep/`.

## What this is

A hybrid MD integrator: a neural network predicts a k-frame jump (Δx, Δv), then a few
real MD steps (OpenMM, or xTB/PySCF via ASE) correct it. Read `README.md` for the
workflow and `docs/REVISIT_2026.md` for the design decisions and current results.
`docs/archive/` describes the **2025** models and is out of date.

## Layout

- `src/kstep/` holds the library; put new logic here, not in the numbered scripts.
  - `model.py`: `KStepModel` (PaiNN backbone, `kind: mean | flow`), checkpoint I/O, `load_predictor`
  - `data.py`: flat `kstep-dataset-v2` format, `KStepData.gather` (GPU-side batching), normalisation tables
  - `hybrid.py`: `run_hybrid` predictor/corrector loop, controls, `CudaGraphPredictor`
  - `openmm_corrector.py`: OpenMM Langevin corrector; `06b` defines the ASE corrector
  - `metrics.py`, `geometry.py`: evaluation (distributions + dynamics), internal coordinates
  - `common.py`: COM removal, loaders, **ASE unit conversions**
  - `legacy_model.py`: frozen 2025 EGNN/Transformer-EGNN, only for loading old checkpoints
- `src/0N_*.py` are thin CLIs. They run with `src/` on `sys.path` (`python src/04_train.py ...`).
- `tests/`: pytest, CPU-only; run `python -m pytest` from `md-kstep/`.

## Invariants to keep

- Models must stay exactly E(3)-equivariant. Vector features may only be mixed by bias-free
  linear maps over channels, combined with invariant scalars, or contracted by dot products.
  Never apply per-Cartesian-component nonlinearities (e.g. `tanh` on Δx); use `soft_clamp_norm`.
  `tests/test_model.py` checks rotation, reflection, translation, permutation and momentum.
- Predictions are projected to zero COM displacement and zero net momentum (`project_zero_com`).
- Units: nm, nm/ps, g/mol, kJ/mol, ps. ASE uses Å and its own time unit; always go through
  `kstep.common.{ase_velocity_to_nm_per_ps, nm_per_ps_to_ase_velocity, friction_per_ps_to_ase}`.
- Hybrid frame spacing is `(k·save_interval + corrector_steps)·dt`; rejected or escalated steps
  fall back to the reference integrator for the whole macro-step, so spacing stays uniform.
- Evaluate against the noise floor (`--reference2`, a second baseline seed) and always include
  the `--predictor zero` control. Equilibrium metrics alone cannot tell a model from no model.

## Running on the home cluster

Data, checkpoints and the GPU env live on phalanx WSL (`~/Molecular-Transformer/md-kstep`,
env `~/miniconda/envs/kstep`). Background jobs must be started outside the SSH session's
job object, or Windows kills them on disconnect. For example:
`powershell -Command "Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='wsl.exe -e bash job.sh'}"`.
Small-kernel timings are meaningless while other GPU jobs run; run `08_time_study.py` alone.
