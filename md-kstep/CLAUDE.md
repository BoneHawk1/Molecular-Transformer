# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**md-kstep** is a hybrid k-step molecular dynamics integrator that uses equivariant neural networks to predict coarse k-step position/velocity updates, then corrects them with a single physics-based force evaluation. The goal is to reduce expensive force calculations while maintaining MD fidelity.

The project supports two modes:
- **MM (Molecular Mechanics)**: Uses OpenMM for force evaluation (2 fs timesteps)
- **QM (Quantum Mechanics)**: Uses xTB/PySCF for force evaluation (0.25 fs timesteps) with transfer learning from MM models

## Architecture

### Core Pipeline Flow

1. **Molecule Preparation** (`src/00_prep_mols.py`): SMILES → 3D conformers → OpenFF topology/parameters
2. **Baseline Trajectories**: Generate reference MD data
   - MM: `src/01_run_md_baselines.py` (OpenMM with Langevin dynamics)
   - QM: `src/01b_run_qm_baselines.py` (xTB via ASE) or `src/01c_run_pyscf_baselines.py` (PySCF GPU/CPU)
3. **Dataset Creation** (`src/02_make_dataset.py`): Extract k-step supervision windows with COM centering
4. **Model Architecture** (`src/03_model.py`):
   - Base: EGNN (Equivariant Graph Neural Network)
   - Advanced: Transformer-EGNN hybrid with attention mechanisms
   - Outputs: Δx and Δv predictions with displacement clamping
   - Optional: Force head for auxiliary supervision (critical for QM)
5. **Training** (`src/04_train.py`): Supervised learning with structural regularization
6. **Hybrid Integration**:
   - MM: `src/06_hybrid_integrate.py` (learned jump + 1 OpenMM step)
   - QM: `src/06b_hybrid_integrate_qm.py` (learned jump + 1 xTB step)
7. **Evaluation**:
   - MM: `src/05_eval_drift_rdfs.py` (energy drift + RDF analysis)
   - QM: `src/05b_eval_qm.py` (QM-specific metrics)

### Model Architecture Details

The model uses graph neural networks that respect SE(3) equivariance:
- **EGNN layers**: Message passing on radius graphs (cutoff typically 0.7 nm)
- **Transformer-EGNN**: Adds multi-head attention, learned positional encoding, and cross-attention
- **Key invariants**: COM momentum conservation, rotational equivariance
- **Outputs**: Clamped position/velocity deltas (tighter bounds for QM vs MM)
- **Structural regularization**: Penalizes bond/angle/dihedral deviations in predicted structures

### Transfer Learning Strategy (MM → QM)

1. Pretrain on abundant MM data (10-20 ns, fast to generate)
2. Fine-tune on limited QM data (200-1500 ps, expensive to generate)
3. Lower learning rate (5e-5 vs 1e-4) and stronger structural penalties for QM
4. Optional layer freezing with `--freeze-layers N`

## Common Commands

### Environment Setup
```bash
conda create -n kstep python=3.11 -y
conda activate kstep
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas pyyaml tqdm hydra-core
pip install mdtraj MDAnalysis matplotlib seaborn wandb
pip install rdkit-pypi openff-toolkit openmm==8.* openmmforcefields
pip install torchmd-net e3nn torch_geometric torch_scatter torch_sparse
pip install line_profiler rich

# For QM support
conda install -c conda-forge xtb ase pyscf
pip install gpu4pyscf cupy-cuda12x  # GPU PySCF (optional)
```

### MM Workflow
```bash
# 1. Prep molecules from SMILES
python src/00_prep_mols.py --smiles data/molecules.smi --out-dir data/raw

# 2. Generate baseline MD trajectories
python src/01_run_md_baselines.py \
    --molecule data/raw/aspirin \
    --config configs/md.yaml \
    --out data/md

# 3. Create k-step dataset
python src/02_make_dataset.py \
    --md-root data/md \
    --out-root data/md \
    --splits-dir data/splits \
    --ks 4 8 12

# 4. Train model
python src/04_train.py \
    --dataset data/md/dataset_k8.npz \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --splits data/splits/train.json data/splits/val.json \
    --device cuda

# 5. Run hybrid integration
python src/06_hybrid_integrate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --md-config configs/md.yaml \
    --molecule data/raw/aspirin \
    --out outputs/hybrid/aspirin_k8.npz

# 6. Evaluate
python src/05_eval_drift_rdfs.py \
    --baseline data/md \
    --hybrid-runs outputs/hybrid \
    --out-dir outputs/eval
```

### QM Workflow
```bash
# 1. Generate QM trajectories (xTB baseline, ~1-10 hours)
bash run_qm_trajectories.sh  # Uses src/01b_run_qm_baselines.py

# 2. Create QM dataset
bash make_qm_dataset.sh

# 3. Train with transfer learning from MM model (~1-3 hours)
bash train_qm.sh  # Loads pretrained MM checkpoint from outputs/checkpoints_transformer_aug_wide/best.pt

# 4. Run hybrid QM integration
bash run_hybrid_qm.sh

# 5. Evaluate QM results
bash eval_qm.sh
```

### PySCF GPU Workflow (Optional - for ab initio baselines)
```bash
# Generate small-molecule trajectories with PySCF GPU
python src/01c_run_pyscf_baselines.py \
    --smiles-file data/qm_small_molecules.smi \
    --config configs/qm_pyscf.yaml \
    --out data/qm_pyscf \
    --num-workers 4 \
    --omp-num-threads 2
```

### Testing & Development
```bash
# Single test (if using pytest)
pytest tests/test_specific.py -v

# Profile code
python -m line_profiler script.py

# Check GPU utilization during training
watch -n 0.5 nvidia-smi
```

## Key Configuration Parameters

### Model Configs
- **MM** (`configs/model.yaml` or `configs/model_transformer.yaml`):
  - `max_disp_nm: 0.25` (loose bounds for 2 fs × k timesteps)
  - `use_force_head: false` (forces less critical for MM)
- **QM** (`configs/model_qm.yaml`):
  - `max_disp_nm: 0.01` (tight bounds for 0.25 fs × k timesteps)
  - `use_force_head: true` (force matching critical for QM)
  - `arch: transformer_egnn` (more expressive architecture)

### Training Configs
- **MM** (`configs/train.yaml`):
  - `k_steps: 8-12`, `lr: 1e-4`, `batch_size: 256`
- **QM** (`configs/train_qm.yaml`):
  - `k_steps: 4`, `lr: 5e-5` (fine-tuning), `batch_size: 128`
  - Stronger structural penalties (2-5x increase in bond/angle/dihedral lambdas)
  - `random_rotate_mode: per_graph` (more data augmentation)

### Trajectory Configs
- **MM** (`configs/md.yaml`): `dt_fs: 2.0`, `friction_per_ps: 0.1`
- **QM** (`configs/qm.yaml`): `dt_fs: 0.25`, `friction_per_ps: 0.002`, `method: GFN2-xTB`
- **PySCF** (`configs/qm_pyscf.yaml`): `dt_fs: 0.25`, `method: RHF`, `basis: sto-3g`

## Critical Implementation Details

### Data Format
All trajectories stored as NPZ files with keys:
- `positions`: (n_frames, n_atoms, 3) in nm
- `velocities`: (n_frames, n_atoms, 3) in nm/ps
- `forces`: (n_frames, n_atoms, 3) in kJ/(mol·nm) (QM only)
- `energies`: (n_frames,) in kJ/mol
- `temperature`: (n_frames,) in K
- `atom_types`: (n_atoms,) atomic numbers
- `masses`: (n_atoms,) in amu

### COM Centering
All training samples are COM-centered to enforce translational invariance. See `src/utils.py:center_by_com()`.

### Structural Regularization
Training loss includes penalties for predicted geometries that violate bond/angle/dihedral distributions from equilibrium MD. Indices are cached per molecule in `STRUCT_INDEXES` global dict.

### Random Rotation Augmentation
- `random_rotate_mode: batch`: Single rotation applied to entire batch (faster)
- `random_rotate_mode: per_graph`: Different rotation per molecule (better for small datasets like QM)

### Displacement Clamping
Model outputs are clamped via `torch.tanh` to prevent catastrophic predictions:
- `Δx` clamped to `[-max_disp_nm, max_disp_nm]`
- `Δv` clamped to `[-max_dvel_nm_per_ps, max_dvel_nm_per_ps]`

### Hybrid Integration Corrector Step
After model prediction:
1. Apply learned Δx and Δv
2. Run single physics-based integration step (OpenMM velocity-Verlet for MM, xTB Verlet for QM)
3. Project trajectory back onto physical manifold

### PySCF GPU Integration
- Uses `pyscf_gpu_calculator.py` (ASE calculator wrapper)
- Automatically detects GPU4PySCF; falls back to CPU PySCF
- Thread-safe: `omp_num_threads` + `max_cores` managed per worker
- Outputs same NPZ schema as xTB (interchangeable)

## File Organization

```
md-kstep/
├── src/                      # All Python source
│   ├── 00_prep_mols.py       # SMILES → 3D + OpenFF
│   ├── 01_run_md_baselines.py      # MM trajectories (OpenMM)
│   ├── 01b_run_qm_baselines.py     # QM trajectories (xTB)
│   ├── 01c_run_pyscf_baselines.py  # QM trajectories (PySCF GPU/CPU)
│   ├── 02_make_dataset.py    # k-step dataset assembly
│   ├── 03_model.py           # EGNN/Transformer-EGNN architectures
│   ├── 04_train.py           # Training loop
│   ├── 05_eval_drift_rdfs.py # MM evaluation
│   ├── 05b_eval_qm.py        # QM evaluation
│   ├── 06_hybrid_integrate.py      # MM hybrid integrator
│   ├── 06b_hybrid_integrate_qm.py  # QM hybrid integrator
│   ├── pyscf_gpu_calculator.py     # PySCF ASE calculator
│   ├── qmmm_example.py       # QM/MM proof-of-concept
│   └── utils.py              # Shared utilities
├── configs/                  # YAML configurations
│   ├── md*.yaml             # MM trajectory settings
│   ├── qm*.yaml             # QM trajectory settings
│   ├── model*.yaml          # Model architectures
│   └── train*.yaml          # Training hyperparameters
├── data/
│   ├── molecules.smi        # MM molecule list
│   ├── qm_molecules.smi     # QM molecule list (22 molecules)
│   ├── qm_small_molecules.smi  # Tiny molecules for PySCF (3-5 atoms)
│   ├── raw/                 # PDB + XML topology files
│   ├── md/                  # MM trajectories + datasets
│   ├── qm/                  # QM trajectories + datasets
│   ├── qm_pyscf/            # PySCF trajectories (optional)
│   ├── splits/              # MM train/val/test JSON
│   └── qm_splits/           # QM train/val/test JSON
├── outputs/
│   ├── checkpoints/         # MM model checkpoints
│   ├── checkpoints_qm/      # QM model checkpoints
│   ├── hybrid/              # MM hybrid rollouts
│   ├── hybrid_qm/           # QM hybrid rollouts
│   ├── eval/                # MM evaluation outputs
│   └── eval_qm/             # QM evaluation outputs
└── *.sh                     # Batch scripts
```

## Debugging & Common Issues

### Training Loss Not Decreasing
1. Check data quality: Inspect trajectories for NaN or exploding values
2. Reduce learning rate or increase warmup
3. Verify model architecture matches dataset (e.g., k_steps in config vs dataset)
4. Check that structural regularization weights aren't too high

### Hybrid Integration Unstable
1. Reduce `--delta-scale` (default 1.0 → try 0.5)
2. Increase `--max-attempts` (default 5 → try 10)
3. Tighten `max_disp_nm` in model config
4. Check for bad initial conditions in molecule prep

### Transfer Learning Not Helping
1. Verify MM checkpoint path and architecture compatibility
2. Check learning rate (should be lower for fine-tuning: 5e-5 vs 1e-4)
3. Try unfreezing more layers (reduce `--freeze-layers`)
4. Ensure QM data quality (stable trajectories, no drift)

### Out of Memory (OOM) During Training
1. Reduce `batch_size` and increase `grad_accum` to maintain effective batch size
2. Reduce model size: `hidden_dim`, `num_layers`, or `feedforward_dim`
3. Disable structural regularization temporarily (set all `lambda_struct_*` to 0)
4. Use mixed precision (`amp: true` in train config)

### xTB/PySCF Calculator Fails
1. Verify installation: `python -c "from xtb.ase.calculators import XTB"`
2. Check charge/multiplicity settings in config
3. Try different xTB method: `GFN1-xTB` or `GFN-FF` instead of `GFN2-xTB`
4. For PySCF GPU: Ensure CUDA + CuPy versions match, check `gpu4pyscf` installation

## Reproducibility

- Seeds controlled via `utils.set_seed()` and config files (`random_seed`, `seed`)
- Force fields serialized per molecule (XML files in `data/raw/`)
- Config files version-controlled; scripts log all hyperparameters
- W&B integration optional for experiment tracking (`configs/train*.yaml`)

## Performance Expectations

### MM Model (Reference)
- **Speedup**: 4-12x force call reduction (k=4 to k=12)
- **Accuracy**: Bond RMSE ~0.04 Å, Angle RMSE ~4°
- **Training**: 10-20 ns data, ~10-20 GPU hours

### QM Model (Target)
- **Speedup**: 4-8x force call reduction (k=4 to k=8)
- **Accuracy**: Bond RMSE <0.02 Å (2x better than MM), Angle RMSE <2°
- **Training**: 200-1500 ps QM data (~1-50 GPU hours to generate), 1-3 hours to train with transfer learning
- **Why more accurate**: Stronger forces, smaller molecules, tighter geometries, transfer learning head start

## References

Key papers and tools:
- xTB: Grimme et al., J. Chem. Theory Comput. 2017, 13, 1989
- ASE: Larsen et al., J. Phys.: Condens. Matter 2017, 29, 273002
- EGNN: Satorras et al., ICML 2021
- OpenMM: Eastman et al., PLoS Comput. Biol. 2017, 13, e1005659
- OpenFF: Wagner et al., J. Chem. Theory Comput. 2022, 18, 4, 2559
