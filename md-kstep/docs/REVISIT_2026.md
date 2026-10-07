# Revisiting md-kstep (October 2026)

This note records what was wrong with the December 2025 version, what changed, and what
the rebuilt pipeline shows. Results are in [§3](#3-results); everything was run on phalanx
(RTX 5070 Ti, WSL2) on 2026-10-07.

## 1. Problems found in the 2025 version

| # | Problem | Consequence |
|---|---|---|
| 1 | **The "equivariant" models were not equivariant.** Δx/Δv came from `Linear(h)` on node features. Those features included raw velocity vectors (`vel_proj(vel)`), the EGNN coordinate updates were discarded, and the `tanh` clamp acted per Cartesian component. | Rotating the input changed the prediction by **137 %** (Δx) and 43 % (Δv) relative to the correctly rotated answer (Transformer-EGNN `checkpoints_transformer_aug_wide`). Random-rotation augmentation could not fix this. |
| 2 | **No control and no noise floor.** The default `--delta-scale 0.5` halved every jump, and the result then got 10 Langevin steps, which repair the structure on their own. Metrics were frame-by-frame RMSEs between two trajectories that decorrelate within ~100 fs. | The reported structural accuracy could not separate the model from "no model". Dihedral RMSEs of ~25° on flexible molecules are just two independent samples. |
| 3 | **Deterministic regression on stochastic data.** Targets were 400 fs ahead along Langevin trajectories. | An MSE model can only learn the conditional mean, which is not a valid dynamics sample. |
| 4 | **QM claims and units.** The xTB k=4 jump was only 1 fs. In the ethanol run 4,915 of 5,000 steps failed. The PySCF "2.8× faster" compared 100 hybrid calls with ~600 baseline calls, equilibration included. ASE velocities were treated as Å/fs (they are Å per ASE time unit ≈ 10.18 fs), so stored QM velocities were ~10.2× too large. Langevin friction was ~100× too small. | QM datasets had mislabelled Δv targets; timing comparisons were not like-for-like. |
| 5 | **Speed.** 8-layer, 256-wide transformer on ~15 atoms; neighbour graph rebuilt in a Python loop every call; the baseline time was extrapolated from 2–5 un-warmed macro-steps. | "Hybrid ~20× slower than MD", mostly from launch overhead and measurement. |
| 6 | **Hygiene and correctness.** No tests or CI; a 1,300-line trainer; the `lambda_force` "force loss" penalised `force_pred²` (pushing it to zero) instead of matching forces; config keys that did nothing (`attention_heads`, `use_cross_attention`); `configs/model.yaml` no longer matched the EGNN checkpoint; checkpoint directories silently renamed for transformers; pickled object-array datasets; a duplicate `md-kstep-cursor/` tree; 11 overlapping docs. | Hard to trust, hard to change. |

## 2. What changed

**Model (`src/kstep/model.py`).** The new backbone is a PaiNN-style message-passing
network on the fully connected intra-molecular graph, with a smooth cosine cutoff.
Vector inputs (velocity, and noisy Δ for the flow) enter only through bias-free channel
mixing and dot products, and outputs are built from vector features. The model is
therefore exactly E(3)-equivariant, translation-invariant and permutation-equivariant
(error ~1e-7). Inputs and targets are normalised per element with tables fitted on the
training split and stored in the checkpoint. Predictions are projected so that the COM
displacement and net momentum are exactly zero. There are two heads:

* `kind: mean`: deterministic regression (the old objective, now equivariant).
* `kind: flow`: **conditional flow matching**. The network learns a velocity field
  that carries isotropic Gaussian noise to samples of p(Δx, Δv | x_t, v_t). Sampling uses
  8 Heun steps by default. Isotropic noise plus an equivariant field makes the sampled
  distribution equivariant too. Rollouts are stochastic, like the Langevin reference.

**Training (`src/04_train.py`, ~200 lines).** The data lives on the GPU and batches are
gathered with vectorised indexing. AdamW, warmup + cosine LR, EMA, gradient clipping;
validation uses fixed noise so losses are comparable. Checkpoints are self-describing
(config + normalisation + k).

**Hybrid loop (`src/kstep/hybrid.py`), shared by OpenMM and xTB/PySCF.**
* A proposal is checked before the corrector runs: values must be finite and no bond may
  be strained by more than 25 %. Rejected proposals are redrawn (flow) or halved (mean).
* After `max_attempts` rejections the macro-step falls back to the reference integrator
  for its full length, so every frame is the same time apart and lag-matched comparison
  with the baseline is exact.
* **Uncertainty-triggered escalation:** with `--uq-samples M` the flow draws M proposals
  in one batched call. If their spread exceeds `--uq-threshold`, the step is handed to the
  reference integrator. This is a first, concrete version of the "UQ-triggered
  escalation" idea.
* Controls: `--predictor zero` (corrector only) and `--predictor ballistic`.
* `CudaGraphPredictor` captures the whole sampler once per molecule size.
* Force calls are counted honestly, including the extra gradient ASE needs at each new
  geometry.

**Evaluation (`src/05_evaluate.py`, `src/kstep/metrics.py`).** The hybrid is compared with
every k-th baseline frame:

* *Equilibrium*: JS divergence of bond, angle and torsion distributions, pair-distance L1, ⟨KE⟩ ratio.
* *Dynamics*: per-step lag RMSD, torsion step size and torsion flip rate, as ratios to the
  baseline.
* *Noise floor*: the same numbers between two independent baseline seeds (`data/md_seed2`),
  or between the two halves of the trajectory for QM.
* Results are broken down by train / val / test molecules.

**QM fixes.** Correct ASE velocity and friction conversions in `01b`, `01c` and
`qmmm_example`; new files carry `velocity_units`. Older files are rescaled on load by
`kstep.common.load_trajectory`. `06b` uses the shared loop and can run plain Verlet as a
timing reference (`--predictor zero --k-steps 0`). Training at k=40 (a 10 fs jump) is now
the default instead of k=4 (1 fs).

**Other changes.**
* `01_run_md_baselines.py` steps in chunks (one Python call per frame instead of per step),
  seeds the Langevin integrator, and takes `--seed`/`--no-nve`.
* `08_time_study.py` warms up and times baseline and hybrid back to back on the fastest
  OpenMM platform.
* Tests (`tests/`, 24 cases) and a GitHub Actions workflow.
* Old configs moved to `configs/legacy/`; the 2025 models are frozen in
  `kstep/legacy_model.py` so old checkpoints still load.
* Repo cleanup: scratch scripts moved to `scripts/dev/`, docs archived, the duplicate tree
  and stray logs removed.

**Stabilizing rollouts.** These were found while running the new models and are on by
default in `06_hybrid_integrate.py`.

* *Canonical kinetic-energy resampling* (`--thermostat csvr`). Small errors in the sampled
  Δv compound into runaway heating: isobutanol went from 84 to 1,279 kJ/mol of kinetic
  energy in 8 steps. Twenty femtoseconds of 1/ps Langevin friction cannot remove that. After
  each learned step the kinetic energy is redrawn from its canonical distribution, keeping
  the predicted velocity directions (Bussi CSVR with τ → 0). For an energy-conserving QM
  reference, use `--energy-rescale` instead.
* *Minimizer quench* (`--quench-iterations 10`). After a 400 fs jump, bonds and angles are
  strained: the post-corrector potential energy was about 200 kJ/mol against 61 ± 11 at
  equilibrium. Those fast modes have fully decorrelated over the jump; the model's real job
  is the slow (torsional) ones. Ten L-BFGS iterations remove the strain and change torsions
  by under ~20°. They are counted as 2 force evaluations per iteration, an upper bound.
* *Potential-energy guard.* A learned step is rejected when its corrected potential energy
  rises more than (dof/2 + 8·√(dof/2))·kT above the start. This catches clashes between
  non-bonded atoms that the bond-strain check misses; one such clash had produced an
  explosion.

## 3. Results

Setup:

* 12 molecules, 1 ns OpenMM Langevin runs at 300 K, frames every 100 fs. Split by molecule:
  8 train, 1 val (CC(C)NCCO), 3 test (aspirin, benzamide, phenol; all fairly rigid).
* Models: 4-layer, 128-wide, about 0.86 M parameters each.
* Training: k=4 runs for 40k steps (~45 min each); flow k=8 for 25k steps.
* Hybrid runs: 2,500 macro-steps from frame 0. The corrector does 5 % of the jump length in
  Langevin steps (10 steps at k=4, 20 at k=8), plus the quench for model runs.
* Every number is a mean over molecules.
* Full tables: [`results_2026/summary.md`](results_2026/summary.md).
* Figures: [`results_2026/`](results_2026/).

### 3.1 Classical MD (all 12 molecules)

| condition | lag-RMSD ratio | torsion-step ratio | torsion flip-rate ratio | torsion JS | angle JS | bond JS | ⟨KE⟩ ratio | force calls saved | learned steps accepted |
|---|---|---|---|---|---|---|---|---|---|
| **noise floor** (MD seed 2) | 1.00 | 1.00 | 1.01 | 0.062 | 0.006 | 0.006 | 1.00 | – | – |
| corrector only (Δ=0) | 0.59 | 0.74 | 0.84 | 0.141 | 0.007 | 0.009 | 1.00 | 21× | 100 % |
| corrector only + quench | 0.16 | 0.20 | 0.69 | 0.300 | 0.362 | 0.380 | 1.00 | 7× | 100 % |
| 2025 Transformer-EGNN (as published: Δ×0.5, caps) | 0.32 | 0.34 | 0.71 | 0.203 | 0.103 | 0.081 | **0.74** | 18× | 99 % |
| new deterministic (mean), k=4 | 0.28 | 0.29 | 0.64 | 0.226 | 0.267 | 0.273 | 1.00 | 5.2× | 93 % |
| **new stochastic (flow), k=4** | 0.85 | 0.87 | 0.89 | 0.076 | 0.045 | 0.114 | 1.00 | 6.4× | 93 % |
| **new stochastic (flow), k=8** | **0.94** | **1.00** | **0.94** | 0.086 | 0.036 | 0.065 | 0.99 | **8.4×** | 90 % |

![metrics overview](results_2026/metrics_overview_all.png)

How to read the table:

* The **dynamics columns** (the first three ratios) show how far the molecule moves per
  macro-step and how often torsions flip, relative to MD. Only the stochastic models get
  close to 1.
* The corrector-only controls, the 2025 model and the deterministic model all move the
  molecule too little. Typically they sit in one torsional well for a long time; see
  [`torsions_CC_C_NCCO.png`](results_2026/torsions_CC_C_NCCO.png), where `mean_k4` and
  `zero_k4_quench` collapse onto single spikes.
* The 2025 model also ran **cold**, at 0.74× the reference kinetic energy. It behaved like
  "no model" with extra noise, which the original frame-wise metrics could not reveal.
* **Torsion distributions.** For the flow models (JS 0.076–0.086) they are close to the
  seed-to-seed floor (0.062), which is itself large because 1 ns is short for the flexible
  molecules.
* **Not solved: bond and angle distributions.** These are still about 6–20× above the noise
  floor for the flow models. The quench leaves the fast modes slightly too cold and narrow
  at the stored frames. Without the quench the structures are strained instead (about
  200 kJ/mol), so a better treatment of fast modes is the main open problem (see §4).
* **Deterministic vs. stochastic.** The deterministic model is worse on every dynamics
  metric. Its conditional-mean jumps shrink motion and leave strained geometries, as
  predicted.
* **Held-out molecules.**
  * Test molecules: the flow models score torsion JS 0.025–0.044 against a floor of 0.003.
    The floor is tiny because these molecules are rigid. Only 67–74 % of learned steps are
    accepted, which pulls the test-set force-call saving down to 5–6×.
  * Val molecule CC(C)NCCO: flow k=8 reaches dynamics ratios of 1.06 / 1.26 / 0.92 and
    bond/angle JS 0.031/0.028, the closest of any condition.

### 3.2 Wall-clock (RTX 5070 Ti, each condition timed back-to-back with nothing else on the GPU)

| | baseline MD | corrector only | mean k=4 | flow k=4 | flow k=8 | 2025 transformer |
|---|---|---|---|---|---|---|
| mean speedup over MD (4 molecules) | 1× (0.053–0.064 s/ps) | 2.3× | 1.9× | 1.4× | **2.2×** | 0.9× |
| model latency per macro-step | – | 0.8 ms | 2.9 ms | 8.2 ms | 10.4 ms | 16 ms |

![time study](results_2026/time_study.png)

Read with care:

* *Corrector only* is fast because it does very little; it is also wrong (§3.1).
* For these 15–25-atom molecules one learned macro-step (≈10 ms for 16 network evaluations
  with CUDA graphs) costs about as much as the ~200 MD steps it replaces (0.06 ms each), so
  the honest wall-clock gain is about 2× at k=8.
* The 2025 claim that hybrids are "~20× slower" came from an un-warmed, extrapolated
  baseline and an un-captured transformer. The 2025 model still does not beat MD.
* Bigger gains need more expensive forces (QM, large systems) or larger k.

### 3.3 QM (xTB, 7 test molecules, k=40 = 10 fs jump + 2 Verlet steps, 10.5 ps runs)

| condition | lag-RMSD ratio | torsion JS | bond JS | ⟨KE⟩ ratio | xTB gradients saved | accepted |
|---|---|---|---|---|---|---|
| noise floor (two halves of the reference) | 1.00 | 0.083 | 0.057 | 0.97 | – | – |
| corrector only | 0.07 | 0.080 | 0.073 | 0.82 | 14× | 100 % |
| flow k=40 | 1.60 | 0.189 | 0.238 | **3.07** | 9.0× | 90 % |
| flow k=40 + energy rescale + energy guard | 1.12 | 0.253 | 0.187 | 1.17 | 9.3× | 81 % |

What this shows:

* The jump does move the molecule like xTB dynamics: the lag ratio is about 1.1, against
  0.07 for "no model".
* But the sampled structures are distorted (bond JS 3× the floor), and the model heats the
  system unless energy is held fixed.
* The QM flow model **overfit early**. Validation loss bottomed at step 6k (1.02) and rose to
  1.82 by step 30k, because training windows from one trajectory are highly correlated.
  `best.pt` is the early checkpoint.
* QM is the regime where the speedup argument works (each xTB gradient is ≈1.5–3 ms here),
  but the model is not accurate enough yet.
* Wall times in this run are only indicative: the 7 molecules ran in parallel on a shared
  CPU.

## 4. What to do next

1. **Fast modes.** Replace the quench with something that thermalizes bonds and angles
   properly. Options:
   * a short constrained or SHAKE-style projection, followed by enough Langevin steps to
     re-equilibrate (~50 fs);
   * have the model predict only slow/internal coordinates and resample the fast ones from
     their known Boltzmann distribution;
   * train with HBond constraints. The force-field XML has none, although `md.yaml` claims
     `constraints: HBonds`; `01` should apply them.
2. **More molecules.** Eight training molecules cause overfitting: the mean model's
   validation loss is best at 4k steps, the flow's at 11k, and the QM flow's at 6k. Generate
   more diverse baselines, subsample windows more sparsely, and add dropout.
3. **Larger k on flexible molecules.** k=8 was better than k=4 on dynamics and cost. Try
   k=12–20 now that the rollouts are stable.
4. **Uncertainty-triggered escalation is implemented** (`--uq-samples`, `--uq-threshold`,
   tested) but not tuned. The flow models' spread could not yet separate good from bad
   jumps at a useful threshold. With better-calibrated models this becomes the
   accuracy/cost dial.
5. **QM.** Retrain with fewer overlapping windows (stride ≥ 40) and early stopping. Then
   rerun `06b` with `--energy-rescale` against the k=40 baselines.

## 5. Reproducing

```bash
# on phalanx (WSL), env ~/miniconda/envs/kstep, repo ~/Molecular-Transformer/md-kstep
python src/02_make_dataset.py --md-root data/md --out-root data/md --splits-dir data/splits --ks 4 8 12
for m in data/raw/*/; do python src/01_run_md_baselines.py --molecule "$m" --config configs/md.yaml --out data/md_seed2 --seed 1234 --no-nve; done
KIND=mean K=4 bash scripts/train.sh; KIND=flow K=4 bash scripts/train.sh; KIND=flow K=8 bash scripts/train.sh --max-steps 25000
scripts/run_hybrid_suite.sh zero_k4 --predictor zero --k-steps 4
scripts/run_hybrid_suite.sh flow_k8 --checkpoint outputs/v2/flow_k8/best.pt --quench-iterations 10
#   (likewise flow_k4, mean_k4, zero_k4_quench; legacy: --checkpoint outputs/checkpoints_transformer_aug_wide/best.pt
#    --model-config configs/legacy/model_transformer_aug_wide.yaml --k-steps 4 --delta-scale 0.5
#    --max-delta-pos 0.2 --max-delta-vel 4.5 --thermostat none)
python src/05_evaluate.py --baseline data/md --reference2 data/md_seed2 --splits-dir data/splits \
    --hybrid zero_k4=... flow_k8=... --out-dir outputs/v2/eval
python src/07_plot_eval.py --eval-dir outputs/v2/eval
python src/08_time_study.py --condition zero=zero flow_k8=outputs/v2/flow_k8/best.pt ... --out-dir outputs/v2/time_study
```

The legacy-transformer runs used the code before the thermostat existed; the rerun
command above adds `--thermostat none` to reproduce them.
