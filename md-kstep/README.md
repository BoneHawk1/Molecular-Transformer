md-kstep: Hybrid k-Step Molecular Integrator
============================================

This repository contains a lightweight workflow for training an equivariant neural network to predict coarse k-step molecular dynamics (MD) state updates and correcting those predictions with a single physics-based force call. The end goal is to reduce the number of expensive force evaluations while retaining baseline MD fidelity.

Pipeline Overview
-----------------
- **Data prep** (`src/00_prep_mols.py`): Embed 3D conformers from SMILES, minimize, and export OpenMM-ready topology/parameter files using OpenFF.
- **Baseline MD** (`src/01_run_md_baselines.py`): Run implicit-solvent OpenMM trajectories, including optional NVE windows for drift analysis, and store trajectories in NumPy format.
- **Dataset assembly** (`src/02_make_dataset.py`): Convert trajectories into supervised k-step windows, center by the center of mass (COM), and create molecule-disjoint splits.
- **Model** (`src/03_model.py`): Small EGNN-style equivariant network that predicts position/velocity deltas with displacement clamping and optional auxiliary heads.
- **Training** (`src/04_train.py`): Supervised training loop with mixed precision, gradient clipping, and rollout validation.
- **Evaluation** (`src/05_eval_drift_rdfs.py`): Compare baseline vs. hybrid integrator energy drift, structural histograms, and efficiency metrics.
- **Hybrid integration** (`src/06_hybrid_integrate.py`): Compose learned k-step jumps with a single OpenMM velocity-Verlet corrector step.

Environment
-----------
Create and activate the conda environment (assumes CUDA 12.1 wheels; adjust as needed):

```bash
conda create -n kstep python=3.11 -y
conda activate kstep

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas pyyaml tqdm hydra-core
pip install mdtraj MDAnalysis matplotlib seaborn wandb
pip install rdkit-pypi openff-toolkit openmm==8.* openmmforcefields
pip install torchmd-net e3nn torch_geometric torch_scatter torch_sparse
pip install line_profiler rich
```

Project Layout
--------------
```
md-kstep/
├─ env/                         # optional lock files / export scripts
├─ data/
│  ├─ molecules.smi             # curated SMILES list
│  ├─ raw/                      # PDB + XML per molecule
│  ├─ md/                       # trajectory npz files + metadata
│  └─ splits/                   # JSON molecule-level splits
├─ src/
│  ├─ 00_prep_mols.py           # build 3D coordinates + parameters
│  ├─ 01_run_md_baselines.py    # generate OpenMM trajectories
│  ├─ 02_make_dataset.py        # pack k-step supervision windows
│  ├─ 03_model.py               # equivariant Δx/Δv predictor
│  ├─ 04_train.py               # supervised training loop
│  ├─ 05_eval_drift_rdfs.py     # energy + structure evaluation
│  ├─ 06_hybrid_integrate.py    # learned jump + 1-step corrector
│  └─ utils.py                  # shared utilities
├─ configs/
│  ├─ md.yaml                   # OpenMM baseline settings
│  ├─ model.yaml                # model hyperparameters
│  └─ train.yaml                # training hyperparameters
└─ README.md
```

Typical Workflow
----------------
1. **Select molecules**: Edit `data/molecules.smi` with ~10 neutral, drug-like SMILES.
2. **Prep molecules**:
   ```bash
   python src/00_prep_mols.py --smiles data/molecules.smi --out-dir data/raw
   ```
3. **Run baseline MD** (generates `data/md/<mol>.npz`):
   ```bash
   python src/01_run_md_baselines.py --molecule data/raw/aspirin \
       --config configs/md.yaml --out data/md
   ```
4. **Dataset windows**:
   ```bash
   python src/02_make_dataset.py --md-root data/md --out-root data/md \
       --splits-dir data/splits --ks 4 8 12
   ```
5. **Train model** (loads configs from `configs/`):
   ```bash
   python src/04_train.py --dataset data/md/dataset_k8.npz \
       --model-config configs/model.yaml \
       --train-config configs/train.yaml \
       --splits data/splits/train.json data/splits/val.json
   ```
6. **Evaluate drift/structure**:
   ```bash
   python src/05_eval_drift_rdfs.py --baseline data/md \
       --hybrid-runs outputs/hybrid --out-dir outputs/eval
   ```
7. **Hybrid integration**:
   ```bash
   python src/06_hybrid_integrate.py --checkpoint outputs/checkpoints/best.pt \
       --md-config configs/md.yaml --molecule data/raw/aspirin \
       --out outputs/hybrid/aspirin_k8.npz
   ```

Key Outputs
-----------
- `data/md/*.npz`: Baseline trajectories, metadata, optional NVE windows.
- `data/md/dataset_k{K}.npz`: Supervised training samples.
- `outputs/checkpoints/`: Model checkpoints (set in `train.yaml`).
- `outputs/hybrid/`: Hybrid rollout trajectories for evaluation.
- `outputs/eval/`: Drift and structural comparison plots/metrics.

Reproducibility Notes
---------------------
- Seeds are set via `src/utils.py:set_seed`.
- Config files are version-controlled; scripts optionally log Hydra/W&B runs.
- Force-field definitions are serialized per molecule for exact reproducibility.
- Stochastic components (OpenMM Langevin, dataset sampling) expose configurable seeds.

Next Steps
----------
- Curriculum fine-tuning from `k=4` → `k=8`.
- Optional force head and temperature conditioning (see `configs/model.yaml` and `train.yaml`).
- Profiling hybrid integrator overhead with `line_profiler`.
