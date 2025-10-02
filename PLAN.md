---

# 0) What you’ll have at the end

* A small equivariant model that predicts **k-step** state jumps ((\Delta x,\Delta v)).
* A **one-step physics corrector** (one true force eval) to stabilize the jump.
* Scripts to: generate MD data, train the model, run the hybrid integrator, and evaluate **(a)** energy drift, **(b)** structural stats (bonds/angles/dihedrals, RDFs), **(c)** **force-call savings** and wall-clock.

---

# 1) Project structure (create this skeleton)

```
md-kstep/
├─ env/                         # conda env file(s) (optional)
├─ data/
│  ├─ molecules.smi             # list of SMILES you choose
│  ├─ raw/                      # raw molecules/PDB/topologies
│  ├─ md/                       # baseline MD trajs (npz) + metadata
│  └─ splits/                   # train/val/test JSON indexes
├─ src/
│  ├─ 00_prep_mols.py           # build 3D coords + parameters
│  ├─ 01_run_md_baselines.py    # generate MD rollouts (OpenMM)
│  ├─ 02_make_dataset.py        # window & pack (x_t,v_t)->(x_{t+k},v_{t+k})
│  ├─ 03_model.py               # equivariant encoder + Δx,Δv heads
│  ├─ 04_train.py               # training loop
│  ├─ 05_eval_drift_rdfs.py     # energy drift & structure metrics
│  ├─ 06_hybrid_integrate.py    # learned jump + one-step corrector
│  └─ utils.py                  # units, graphing, seeding, logging
├─ configs/
│  ├─ md.yaml                   # MD settings
│  ├─ model.yaml                # model hyperparams
│  └─ train.yaml                # train hparams
└─ README.md
```

---

# 2) Environment setup (one-GPU friendly)

**Assumptions:** Linux/macOS, CUDA-capable GPU (your 5070 Ti), Conda (or mamba).

```bash
# Create & activate env
conda create -n kstep python=3.11 -y
conda activate kstep

# Install PyTorch matching your CUDA (check pytorch.org for the exact command)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core packages
pip install numpy scipy pandas pyyaml tqdm hydra-core
pip install mdtraj MDAnalysis matplotlib seaborn wandb

# Chem/MD helpers
pip install rdkit-pypi openff-toolkit openmm==8.* openmmforcefields
# Equivariant / graph stack (choose one path; the first is simpler to get running)
pip install torchmd-net e3nn torch_geometric

# (Optional, but handy) line-profiler + rich logging
pip install line_profiler rich
```

**Gotchas:**

* Make sure PyTorch and CUDA versions match.
* If `torch_geometric` wheels complain, consult their install page; worst case, you can start with **TorchMD-Net** which bundles neighbor finding.

---

# 3) Choose systems (keep it tiny & diverse)

Create `data/molecules.smi` with ~10–12 small, neutral, drug-like molecules (≤30 heavy atoms). Mix aromatic/aliphatic, H-bond donors/acceptors. Example entries:

```
c1ccccc1O
CC(=O)Oc1ccccc1C(=O)O
CCN(CC)CCO
c1ccncc1
CC(C)CO
C1=CC(=O)NC(=O)N1
```

---

# 4) Build 3D molecules + parameters (OpenFF, implicit solvent)

Script: `src/00_prep_mols.py`

* Read SMILES → RDKit → add Hs → embed 3D (ETKDG) → MMFF minimize (quick).
* Parameterize with **OpenFF** (handles GAFF-like params).
* Save: `PDB`, `XML`/`FF` objects for OpenMM.

Pseudocode:

```python
# 00_prep_mols.py
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

ff = ForceField("openff_unconstrained-2.2.1.offxml")  # okay to use current stable

for smi in smiles_list:
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(rdmol)
    offmol = Molecule.from_rdkit(rdmol)
    # write PDB; parameters applied later in OpenMM before simulation
```

---

# 5) Generate baseline MD trajectories (OpenMM)

Script: `src/01_run_md_baselines.py`

**Settings (implicit solvent for speed):**

* Thermostat: **Langevin** 300 K, friction 1/ps.
* **Time step:** 2 fs.
* **Constraints:** HBonds (standard).
* **Nonbonded:** cutoffs ~1.0 nm; implicit solvent **GBSA OBC2** (no PME).
* **Length:** per molecule 1–2 ns total; save frame every 50–100 steps.
* **Also collect** short **NVE** windows (e.g., switch off thermostat for 50–100 ps every ~200 ps) for drift evaluation.

Implementation outline:

```python
# build OpenMM system with GBSA
from simtk import unit
from openmm import app, LangevinIntegrator, Platform
...
integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
# GBSA: app.GBSAOBCForce or via forcefield XML
# Save positions, velocities, potential energy, forces (optionally) to NPZ
```

**Output format:** For each molecule, write `npz` with arrays:

* `pos` [T, N, 3] in **nanometers**
* `vel` [T, N, 3] in **nm/ps**
* `box` [T, 3, 3] (identity for implicit)
* `Epot`, `Ekin`, `Etot` [T]
* optional `forces` [T, N, 3] (can be sparse to save time)

---

# 6) Make supervised dataset of k-step windows

Script: `src/02_make_dataset.py`

**Goal:** Pack pairs ((x_t,v_t)\to(x_{t+k},v_{t+k})) for k ∈ {4, 8, 12, 16}. Use stride (s) (e.g., sample every 10th saved frame to reduce correlation).

Steps:

1. For each traj, pick random `t` indices so that `t+k` is valid.
2. **Remove COM translation** from positions and **COM velocity** from velocities at both `t` and `t+k` (keeps supervision centered).
3. Save samples to `data/md/dataset_k{K}.npz` with arrays:

   * `x_t`, `v_t`, `x_tk`, `v_tk`, `atom_types`, `molecule_id`.
4. Create `data/splits/{train,val,test}.json` grouping by molecule (no leakage across splits).

**Tip:** Keep **units consistent** (nm, nm/ps). Store atom masses if you’ll compute momentum loss.

---

# 7) Model: small equivariant encoder + vector heads

Script: `src/03_model.py`

**Pick a simple but solid baseline:** TorchMD-Net **ET** (Equivariant Transformer) or MACE-mini. Start with ET; it’s turnkey.

**Inputs:**

* Atom types (embedding).
* Neighbor graph within cutoff (6–8 Å).
* Relative geometry (you get this from ET).

**Outputs:**

* **Vector head** for (\Delta x) and (\Delta v) (predict in nm and nm/ps).
* (Optional) **Force head** (\widehat{F}(x_t)) for aux loss.

**Recommended tiny config (fits easily):**

* Layers: 4
* Hidden channels: 128
* Heads: 4
* Cutoff: 0.6–0.8 nm
* Radial basis size: 8–16
* Activation: SiLU
* **Head squashing:** scale `tanh` on outputs to cap max step (stability)

---

# 8) Training loop (pure supervised to start)

Script: `src/04_train.py`

**Losses (start simple, add as needed):**

* ( \mathcal{L}*{\text{pos}} = |\hat{x}*{t+k} - x_{t+k}|_2^2 )
* ( \mathcal{L}*{\text{vel}} = \lambda_v |\hat{v}*{t+k} - v_{t+k}|_2^2 )  (e.g., (\lambda_v=0.5))
* COM/momentum reg: penalize predicted COM shift and net momentum.
* (Optional) Force-matching: (\lambda_F|\hat{F}(x_t)-F(x_t)|_2^2)

**Important stability tricks:**

* **Predict deltas, not absolutes.** ( \hat{x}_{t+k} = x_t + \Delta x )
* **Head clamp:** `Δx = max_disp * tanh(raw_Δx)` (start `max_disp = 0.02 nm` per k=8)
* **Grad clip:** `clip_grad_norm_(model.parameters(), 1.0)`
* **Mixed precision** (fp16/bf16) + gradient checkpointing.

**Hparams (good first pass):**

* Optimizer: AdamW (lr 2e-4 → cosine decay to 2e-5), weight decay 0.01
* Batch: start 16–32 (use grad accumulation to reach effective 128)
* Epochs/steps: 100k–200k steps usually suffice
* Early stop on **val** (\mathcal{L}*{\text{pos}}+\mathcal{L}*{\text{vel}})
* Log with W&B: losses, learning rate, histograms of Δx magnitudes

**Validation during training:**

* Report **rollout** metric: chain your predictor for 2–3 jumps (no corrector) and compute RMSD vs ground truth at `t+2k`, `t+3k` (catches drift early).

---

# 9) One-step physics corrector (OpenMM)

Script: `src/06_hybrid_integrate.py`

**Loop for each macro-step (size = k * base Δt):**

1. **Learned jump:** from ((x_t, v_t)) → ((\hat{x}*{t+k}, \hat{v}*{t+k})).
2. **Corrector using one true force eval:**

   * Load ((\hat{x}, \hat{v})) into an OpenMM **velocity-Verlet** (or Langevin with tiny dt just for projection).
   * Compute **one** force/energy step:

     * ( v_{t+1/2} = \hat{v} + \frac{\Delta t}{2m} F(\hat{x}) )
     * ( x^\star = \hat{x} + \Delta t , v_{t+1/2} )
     * Re-compute (F(x^\star)), then finalize ( v^\star = v_{t+1/2} + \frac{\Delta t}{2m} F(x^\star))
   * Apply **constraints** (HBonds) so bond lengths stay physical.
   * Use the **base Δt** here (2 fs) only as a corrector “nudge” (not k*Δt).
3. Set state to ((x^\star, v^\star)), proceed to next macro-step.

**Why base Δt for corrector?** It mirrors a real integrator step and projects onto the physical manifold (constraints, forces) without paying for k steps.

**Bookkeeping:**

* Count force calls: baseline uses **k** per macro-window; hybrid uses **1** → nominal k× reduction.
* Time CPU/GPU wall-clock too (Python overhead can eat some gains; keep OpenMM context alive).

---

# 10) Evaluation: physics fidelity & efficiency

Script: `src/05_eval_drift_rdfs.py`

**(A) Energy drift (NVE windows):**

* For each molecule, compare baseline vs hybrid over 50–100 ps **NVE**:

  * Total energy drift per ps (median & IQR across windows).
  * Plot violin/box plots.

**(B) Structural statistics:**

* **Bond/angle/dihedral** histograms vs baseline (MDAnalysis has helpers).
* **RDFs (g(r))** for heavy-atom pairs (bin 0.02 nm up to 1.0–1.2 nm).
* **Distributional match**: compute KL divergence between histograms.

**(C) Efficiency:**

* **Force-call savings** (nominal = k/1).
* **Wall-clock** speedup (report per molecule; Python overhead matters).

**(D) Time-to-basin (optional but nice):**

* From randomized initial states, measure time to reach conformers within (\varepsilon) kcal/mol of baseline’s best minimum or within RMSD ≤ 0.75 Å.

---

# 11) Default hyperparameter grid (run these in order)

1. **k = 4**, `max_disp = 0.01 nm`, no force head → sanity check.
2. **k = 8**, `max_disp = 0.02 nm`, add momentum/COM reg (λ=1e-2).
3. **k = 8**, **+ force head** (λ_F = 0.1) if rollouts drift.
4. **k = 12**, `max_disp = 0.03 nm`.
5. **k = 16** only if (2)–(4) are stable.

**Acceptance rule:** advance k only if NVE drift and RDFs are comparable to baseline (no statistically significant degradation over 50–100 ps).

---

# 12) Reproducibility & logging checklist

* Fix seeds (`torch`, `numpy`, Python `random`), and OpenMM platform determinism where possible.
* Save:

  * `configs/` used
  * model ckpts (best val)
  * dataset indices & hashes
  * MD seeds and thermostat settings
* Log W&B runs with tags: molecule-set, k, model version, loss ablations.

---

# 13) Troubleshooting (most common failures)

**Explosion / huge Δx:**

* Lower `max_disp`; add `tanh` scaling on heads.
* Increase λ on COM/momentum penalty.
* Add tiny Gaussian noise to inputs during training (robustness).

**Looks accurate for 1 jump but drifts on 2–3 chained jumps:**

* Add auxiliary **force-matching** head (stabilizes latent geometry).
* Increase training horizon diversity: sample windows at different temperatures (e.g., 280/300/320 K).
* Try predicting **Δv** from **accelerations** (F/m) features.

**Hybrid runs slower than expected:**

* Keep an **OpenMM Context** alive between steps; avoid re-creating Simulation.
* Batch force queries when possible (vectorize across molecules in one context is hard; instead, minimize Python overhead).
* Profile with `line_profiler`.

**Constraints violated (bonds too long/short):**

* Make sure the corrector step uses **constraints on HBonds**.
* Optionally, project predicted positions onto constrained manifold before corrector using OpenMM’s `LocalEnergyMinimizer` for just a few iterations (fast).

---

# 14) Example configs (drop in `configs/`)

`model.yaml`

```yaml
arch: torchmdnet_et
hidden_channels: 128
num_layers: 4
num_heads: 4
cutoff_nm: 0.7
rbf_size: 16
vec_channels: 16
predict_delta: true
max_disp_nm: 0.02      # tune per k
use_force_head: false  # set true in ablation 3
tanh_head: true
```

`train.yaml`

```yaml
seed: 42
k_steps: 8              # 4, 8, 12 ...
batch_size: 32
grad_accum: 4
lr: 2.0e-4
lr_min: 2.0e-5
weight_decay: 1.0e-2
epochs: 50
amp: true
grad_clip: 1.0
lambda_vel: 0.5
lambda_com: 1.0e-2
lambda_force: 0.0      # 0.1 when use_force_head = true
val_every_steps: 1000
checkpoint_every_steps: 2000
```

`md.yaml`

```yaml
temperature_K: 300
friction_per_ps: 1.0
dt_fs: 2.0
length_ns: 1.0
save_interval_steps: 50
constraints: HBonds
implicit_solvent: OBC2
nonbonded_cutoff_nm: 1.0
nve_window_ps: 100
```

---

# 15) Stretch (only if everything else is confirmed working)

* **Curriculum k:** train k=4 first, then fine-tune to k=8 on same weights.
* **Temperature conditioning:** add temperature as a feature to improve robustness.
* **Tiny explicit solvent** box test (short): confirms method survives long-range interactions.

---

