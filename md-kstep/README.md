# md-kstep: learned k-step jumps + physics correction for molecular dynamics

md-kstep trains a neural network to jump a molecule `k` stored frames ahead in one
step, (x_t, v_t) → (x_{t+k}, v_{t+k}). It then runs a few real MD steps (the *corrector*)
to put the system back on the force field's manifold. If the jump is good, one macro-step
replaces `k × save_interval` force evaluations with one network call and a handful of
force evaluations.

The project started as a CHEM 495 final project (Fall 2025; report in
[`Research Report/`](Research%20Report/)). It was overhauled in October 2026; see
[`docs/REVISIT_2026.md`](docs/REVISIT_2026.md) for what changed, why, and the new results.

## Current status (Oct 2026)

Trained on 101 small molecules (classical MD, 300 K) and tested on 24 held-out ones, the
stochastic flow-matching model at k=8 (an 800 fs jump):

* accepts 97 % of its learned jumps and saves **11×** force evaluations;
* matches MD's per-step motion within a few percent;
* reproduces bond, angle and pair-distance distributions within ~2–3× of the
  seed-to-seed noise floor;
* gave about a 2.2× wall-clock speedup over OpenMM in a clean timing run.

The main open problem is **kinetics**. The learned jumps cross torsional and ring
barriers too often (1.4× MD's flip rate). Examples are cyclohexane chair flips and
alanine-dipeptide φ basins. Fixing this probably needs a likelihood-based
Metropolis–Hastings step. The QM (xTB, 10 fs jump) model is not accurate yet.

Details, tables and plots: [`docs/REVISIT_2026.md`](docs/REVISIT_2026.md).

## What is in the box

| Stage | Script | Notes |
|---|---|---|
| Prepare molecules | `src/00_prep_mols.py` | SMILES → 3D → OpenFF (SMIRNOFF) system XML |
| Classical baselines | `src/01_run_md_baselines.py` | OpenMM Langevin, implicit solvent, `--seed` for independent repeats |
| QM baselines | `src/01b_run_qm_baselines.py`, `src/01c_run_pyscf_baselines.py` | xTB / PySCF through ASE |
| Datasets | `src/02_make_dataset.py` | flat `kstep-dataset-v2` windows, molecule-disjoint splits |
| Models | `src/kstep/model.py` (`src/03_model.py` checks a config) | exactly E(3)-equivariant PaiNN backbone; `mean` (deterministic) or `flow` (stochastic, flow matching) |
| Training | `src/04_train.py` | GPU-resident data, EMA, cosine LR, self-contained checkpoints |
| Evaluation | `src/05_evaluate.py`, `src/07_plot_eval.py` | distribution + dynamics metrics vs. a seed-to-seed noise floor |
| Hybrid integrators | `src/06_hybrid_integrate.py` (OpenMM), `src/06b_hybrid_integrate_qm.py` (xTB/PySCF) | shared loop in `src/kstep/hybrid.py`; controls `--predictor zero/ballistic`; uncertainty-triggered escalation |
| Wall-clock | `src/08_time_study.py` | baseline vs. hybrid measured back-to-back with warmup |

## Install

```bash
conda create -n kstep python=3.11 -y && conda activate kstep
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA
pip install -r requirements.txt
# QM extras: conda install -c conda-forge xtb-python ase ; pip install pyscf
```

## Classical-MD workflow (run from `md-kstep/`)

```bash
python src/00_prep_mols.py --smiles data/molecules.smi --out-dir data/raw
for m in data/raw/*/; do python src/01_run_md_baselines.py --molecule "$m" --config configs/md.yaml --out data/md; done
# independent second run per molecule: the noise floor for every metric
for m in data/raw/*/; do python src/01_run_md_baselines.py --molecule "$m" --config configs/md.yaml --out data/md_seed2 --seed 1234 --no-nve; done
python src/02_make_dataset.py --md-root data/md --out-root data/md --splits-dir data/splits --ks 4 8 12

KIND=flow K=4 bash scripts/train.sh          # -> outputs/v2/flow_k4/best.pt
KIND=mean K=4 bash scripts/train.sh          # deterministic comparison

scripts/run_hybrid_suite.sh flow_k8 --checkpoint outputs/v2/flow_k8/best.pt --quench-iterations 20   # thermal quench (default mode)
scripts/run_hybrid_suite.sh zero_k4 --predictor zero --k-steps 4      # corrector-only control
python src/05_evaluate.py --baseline data/md --reference2 data/md_seed2 --splits-dir data/splits \
    --hybrid zero=outputs/v2/hybrid/zero_k4 flow=outputs/v2/hybrid/flow_k4 --out-dir outputs/v2/eval
python src/07_plot_eval.py --eval-dir outputs/v2/eval
python src/08_time_study.py --condition zero=zero flow=outputs/v2/flow_k4/best.pt --out-dir outputs/v2/time_study
```

## QM workflow

```bash
bash scripts/run_qm_trajectories.sh                     # xTB trajectories -> data/qm
K_STEPS="4 40" bash scripts/make_qm_dataset.sh          # stored every 0.25 fs: k=40 is a 10 fs jump
KIND=flow K=40 bash scripts/train_qm.sh
CHECKPOINT=outputs/v2/qm_flow_k40/best.pt bash scripts/run_hybrid_qm.sh
bash scripts/eval_qm.sh
```

QM trajectories written before October 2026 stored velocities in the wrong units
(ASE's internal time unit was treated as femtoseconds, so velocities came out about 10.2× too large). `kstep.common.load_trajectory`
detects these files and rescales them when it loads them.

## How to read the evaluation

Comparing a hybrid trajectory frame by frame with a reference trajectory says little.
After about 100 fs two MD runs are effectively independent samples. A corrector-only
control (Δ = 0 plus a few Langevin steps) also samples the right equilibrium
distribution. `05_evaluate.py` therefore reports both:

* **equilibrium**: Jensen–Shannon divergence of bond, angle and torsion distributions, pair-distance L1, ⟨KE⟩ ratio;
* **dynamics**: how far the molecule moves per macro-step, and how often torsions flip, as ratios to the baseline at the same frame spacing. These ratios are what separate a useful jump model from no model.

Every number comes with the same metric computed between two independent baseline runs (the *noise floor*).

## Tests

```bash
python -m pytest            # CPU-only; equivariance, momentum conservation, data, metrics, hybrid loop, units
```

CI runs the same suite on every push (`.github/workflows/tests.yml`).

## Layout

```
src/kstep/        library: model, data, hybrid loop, metrics, geometry, OpenMM corrector, legacy model
src/NN_*.py       pipeline CLIs (see table above)
configs/          md/qm/model/train YAML; configs/legacy/ holds the 2025 model configs
scripts/          shell drivers; scripts/dev/ has 2025 threading-debug scratch scripts
tests/            pytest suite
docs/             REVISIT_2026.md (current); docs/archive/ (2025 documentation, describes the old models)
Research Report/  CHEM 495 final report and figures (Dec 2025)
```

The 2025 EGNN and Transformer-EGNN checkpoints can still be run for comparison. Pass
`--model-config configs/legacy/model_transformer_aug_wide.yaml` (or `model_egnn.yaml`)
to the hybrid scripts. These models are **not** rotation-equivariant; see
`docs/REVISIT_2026.md`.
