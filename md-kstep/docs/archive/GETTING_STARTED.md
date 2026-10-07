# Getting Started with md-kstep

This guide will walk you through setting up md-kstep and running your first hybrid molecular dynamics simulation from scratch. By the end, you'll have trained a neural network to accelerate MD simulations and evaluated its performance.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Test Run](#quick-test-run)
4. [Full MM Workflow](#full-mm-workflow)
5. [QM Workflow (Advanced)](#qm-workflow-advanced)
6. [Understanding the Outputs](#understanding-the-outputs)
7. [Next Steps](#next-steps)

## Prerequisites

### System Requirements

- **Operating System**: Linux (recommended), macOS, or Windows (WSL2)
- **Python**: 3.11
- **GPU**: NVIDIA GPU with CUDA 12.1+ (recommended for training)
  - Can run on CPU, but training will be much slower
- **RAM**: 16 GB minimum, 32 GB recommended
- **Disk Space**: 50 GB for full MM workflow, 100 GB for QM

### Knowledge Prerequisites

- Basic command line usage
- Basic Python programming (helpful but not required)
- Familiarity with conda/pip package management
- Understanding of molecular structures (helpful)

## Installation

### Step 1: Create Conda Environment

```bash
# Create a fresh conda environment
conda create -n kstep python=3.11 -y

# Activate it
conda activate kstep
```

**Note**: Always activate this environment before working with md-kstep:
```bash
conda activate kstep
```

### Step 2: Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version if needed):

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower training)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.1.0+cu121
CUDA available: True
```

### Step 3: Install Core Dependencies

```bash
# Scientific computing
pip install numpy scipy pandas pyyaml tqdm hydra-core

# Molecular dynamics and analysis
pip install mdtraj MDAnalysis matplotlib seaborn

# Chemistry tools
pip install rdkit-pypi openff-toolkit openmm==8.* openmmforcefields

# Graph neural network libraries
pip install torchmd-net e3nn torch_geometric torch_scatter torch_sparse

# Utilities
pip install line_profiler rich

# Optional: Weights & Biases for experiment tracking
pip install wandb
```

**Troubleshooting**:
- If `torch_scatter` or `torch_sparse` fail, try installing them from PyG wheels:
  ```bash
  pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
  ```

### Step 4: Install QM Dependencies (Optional)

Only needed if you plan to run quantum mechanics simulations:

```bash
# Install xTB and ASE via conda
conda install -c conda-forge xtb ase -y

# Optional: PySCF for ab initio calculations
pip install pyscf

# Optional: GPU-accelerated PySCF (requires CUDA)
pip install gpu4pyscf cupy-cuda12x
```

**Verify QM installation**:
```bash
python -c "from xtb.ase.calculator import XTB; print('xTB installed successfully')"
```

### Step 5: Clone Repository

```bash
# Navigate to your projects directory
cd ~/projects  # or wherever you keep your code

# Clone the repository (or download and extract)
git clone https://github.com/yourusername/md-kstep.git
cd md-kstep
```

### Step 6: Verify Installation

```bash
# Check that all imports work
python -c "
import torch
import numpy as np
import mdtraj
import openmm
from rdkit import Chem
from openff.toolkit.topology import Molecule
print('All core dependencies installed successfully!')
"
```

## Quick Test Run

Let's run a minimal example to verify everything works. This takes about 30-60 minutes on a GPU.

### 1. Prepare a Single Molecule

Create a test SMILES file:
```bash
echo "CC(=O)Oc1ccccc1C(=O)O aspirin" > data/test_molecule.smi
```

Generate 3D structure and parameters:
```bash
python src/00_prep_mols.py \
    --smiles data/test_molecule.smi \
    --out-dir data/raw \
    --log-level INFO
```

**What this does**: Converts the SMILES string to a 3D structure, assigns force field parameters, and saves PDB and XML files.

**Expected output**:
```
[INFO] Processing 1 molecules...
[INFO] aspirin: Generated conformer, minimized, saved to data/raw/aspirin/
[INFO] Completed successfully
```

### 2. Generate Short Baseline Trajectory

```bash
python src/01_run_md_baselines.py \
    --molecule data/raw/aspirin \
    --config configs/md_demo.yaml \
    --out data/md_test \
    --log-level INFO
```

**What this does**: Runs a short MD simulation (1 ns) to generate training data.

**Expected runtime**: 5-10 minutes

**Expected output**:
```
[INFO] Running MD for aspirin...
[INFO] Equilibration: 5000 steps (10 ps)
[INFO] Production: 500000 steps (1000 ps)
[INFO] Saved trajectory to data/md_test/aspirin.npz
```

### 3. Create Training Dataset

```bash
python src/02_make_dataset.py \
    --md-root data/md_test \
    --out-root data/md_test \
    --splits-dir data/splits_test \
    --ks 4 \
    --train-frac 0.8 \
    --log-level INFO
```

**What this does**: Extracts k-step windows from the trajectory and creates train/val splits.

**Expected output**:
```
[INFO] Found 1 trajectories
[INFO] Creating dataset for k=4...
[INFO] Total samples: 4990 (train: 3992, val: 998)
[INFO] Saved to data/md_test/dataset_k4.npz
```

### 4. Train Model (Quick)

```bash
python src/04_train.py \
    --dataset data/md_test/dataset_k4.npz \
    --model-config configs/model_egnn.yaml \
    --train-config configs/train_debug.yaml \
    --splits data/splits_test/train.json data/splits_test/val.json \
    --device cuda \
    --log-level INFO
```

**What this does**: Trains a small EGNN model for 10 epochs (quick test).

**Expected runtime**: 10-20 minutes on GPU, 1-2 hours on CPU

**Expected output**:
```
[INFO] Epoch 1/10: train_loss=0.0234, val_loss=0.0198
[INFO] Epoch 5/10: train_loss=0.0045, val_loss=0.0052
[INFO] Epoch 10/10: train_loss=0.0021, val_loss=0.0028
[INFO] Best checkpoint saved to outputs/checkpoints/best.pt
```

### 5. Run Hybrid Integration

```bash
python src/06_hybrid_integrate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --md-config configs/md_demo.yaml \
    --molecule data/raw/aspirin \
    --out outputs/hybrid_test/aspirin_k4.npz \
    --n-steps 500 \
    --log-level INFO
```

**What this does**: Runs 500 steps using the trained model + physics correction.

**Expected runtime**: 2-5 minutes

**Expected output**:
```
[INFO] Running hybrid integration for 500 steps...
[INFO] Completed 500/500 steps (0 failures)
[INFO] Average energy: 145.3 kJ/mol, drift: 0.023 kJ/mol/ps
[INFO] Saved trajectory to outputs/hybrid_test/aspirin_k4.npz
```

### 6. Quick Evaluation

```bash
python src/05_eval_drift_rdfs.py \
    --baseline data/md_test \
    --hybrid-runs outputs/hybrid_test \
    --out-dir outputs/eval_test \
    --log-level INFO
```

**What this does**: Compares baseline vs. hybrid energy and structure.

**Expected output**:
```
[INFO] Evaluating aspirin...
[INFO] Baseline energy drift: 0.018 kJ/mol/ps
[INFO] Hybrid energy drift: 0.023 kJ/mol/ps
[INFO] Bond RMSE: 0.035 Å
[INFO] Plots saved to outputs/eval_test/
```

**Check the results**:
```bash
ls outputs/eval_test/
# Should see: energy_drift.png, bond_distributions.png, summary.json
```

If you made it here successfully, congratulations! Your installation is working correctly.

## Full MM Workflow

Now let's run the full molecular mechanics workflow with multiple molecules and longer trajectories.

### 1. Select Molecules

Edit `data/molecules.smi` or create your own SMILES file with 5-10 small, drug-like molecules:

```bash
cat data/molecules.smi
# Example content:
# CC(=O)Oc1ccccc1C(=O)O aspirin
# CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen
# CN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine
# ...
```

**Tips for molecule selection**:
- Start with neutral, stable molecules
- Avoid very flexible molecules initially
- 10-30 heavy atoms is a good range
- Check that RDKit can parse the SMILES

### 2. Prepare All Molecules

```bash
python src/00_prep_mols.py \
    --smiles data/molecules.smi \
    --out-dir data/raw \
    --n-conformers 1 \
    --log-level INFO
```

**Expected runtime**: 2-5 minutes per molecule

### 3. Generate Baseline Trajectories

Run for each molecule (or use a batch script):

```bash
# For a single molecule
python src/01_run_md_baselines.py \
    --molecule data/raw/aspirin \
    --config configs/md.yaml \
    --out data/md \
    --log-level INFO

# For all molecules, use a loop:
for mol_dir in data/raw/*/; do
    mol_name=$(basename "$mol_dir")
    echo "Running MD for $mol_name..."
    python src/01_run_md_baselines.py \
        --molecule "$mol_dir" \
        --config configs/md.yaml \
        --out data/md \
        --log-level INFO
done
```

**Expected runtime**:
- 2-4 hours per molecule (10-20 ns trajectory)
- Can run in parallel if you have multiple GPUs

**Monitor progress**:
```bash
# Check what trajectories have been generated
ls data/md/*.npz

# Check trajectory info
python -c "
import numpy as np
data = np.load('data/md/aspirin.npz')
print(f'Frames: {len(data[\"positions\"])}')
print(f'Atoms: {len(data[\"atom_types\"])}')
print(f'Duration: {len(data[\"positions\"]) * 0.002} ns')  # 2 fs timesteps
"
```

### 4. Create Training Dataset

```bash
python src/02_make_dataset.py \
    --md-root data/md \
    --out-root data/md \
    --splits-dir data/splits \
    --ks 4 8 12 \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --log-level INFO
```

**What this does**:
- Creates datasets for k=4, 8, and 12
- Splits molecules into train (70%), val (15%), test (15%)
- COM-centers all frames
- Applies rotation augmentation (3x data)

**Expected output**:
```
[INFO] Found 10 trajectories
[INFO] Creating dataset for k=4...
[INFO] Total samples: 119700 (train: 83790, val: 17955, test: 17955)
[INFO] Saved to data/md/dataset_k4.npz
[INFO] Creating dataset for k=8...
...
```

### 5. Train Model

Start with k=8 (good balance of speed and accuracy):

```bash
python src/04_train.py \
    --dataset data/md/dataset_k8.npz \
    --model-config configs/model_transformer.yaml \
    --train-config configs/train.yaml \
    --splits data/splits/train.json data/splits/val.json \
    --device cuda \
    --out-dir outputs/checkpoints_k8 \
    --log-level INFO
```

**Expected runtime**: 10-20 hours on a modern GPU

**Monitor training**:
```bash
# Watch the log
tail -f outputs/checkpoints_k8/training.log

# If using W&B, check dashboard
# https://wandb.ai/your-username/md-kstep
```

**What to look for**:
- Train loss should decrease steadily
- Val loss should track train loss (within 2x)
- Structural losses (bond, angle) should decrease
- Best checkpoint saved when val loss is lowest

**Typical loss progression**:
```
Epoch 1:   train=0.0450, val=0.0398
Epoch 10:  train=0.0089, val=0.0095
Epoch 50:  train=0.0021, val=0.0028
Epoch 100: train=0.0012, val=0.0019 (best)
Epoch 150: train=0.0008, val=0.0021 (early stopping)
```

### 6. Run Hybrid Integration

Test on a held-out test molecule:

```bash
# Pick a test molecule
python src/06_hybrid_integrate.py \
    --checkpoint outputs/checkpoints_k8/best.pt \
    --md-config configs/md.yaml \
    --molecule data/raw/test_molecule \
    --out outputs/hybrid/test_molecule_k8.npz \
    --n-steps 5000 \
    --temperature 300.0 \
    --log-level INFO
```

**Run for all test molecules**:
```bash
# Get test molecules from splits
test_mols=$(python -c "
import json
with open('data/splits/test.json') as f:
    print(' '.join(json.load(f)))
")

# Run hybrid integration for each
for mol in $test_mols; do
    echo "Running hybrid integration for $mol..."
    python src/06_hybrid_integrate.py \
        --checkpoint outputs/checkpoints_k8/best.pt \
        --md-config configs/md.yaml \
        --molecule data/raw/$mol \
        --out outputs/hybrid/${mol}_k8.npz \
        --n-steps 5000
done
```

### 7. Evaluate Performance

```bash
python src/05_eval_drift_rdfs.py \
    --baseline data/md \
    --hybrid-runs outputs/hybrid \
    --out-dir outputs/eval \
    --log-level INFO
```

**What this generates**:
- `energy_drift_comparison.png`: Energy conservation comparison
- `rdf_comparison.png`: Structural accuracy (radial distribution functions)
- `bond_distributions.png`: Bond length accuracy
- `summary.json`: Numerical metrics
- `per_molecule_metrics.csv`: Detailed per-molecule results

**Check results**:
```bash
# View summary
cat outputs/eval/summary.json

# View detailed metrics
head -20 outputs/eval/per_molecule_metrics.csv

# Open plots (Linux with display)
xdg-open outputs/eval/energy_drift_comparison.png
```

**Success criteria**:
- Energy drift < 0.05 kJ/mol/ps (similar to baseline)
- Bond RMSE < 0.05 Å
- Angle RMSE < 5°
- RDFs match baseline closely
- <5% failed steps

## QM Workflow (Advanced)

For quantum mechanics simulations with higher accuracy. See [QM_WORKFLOW.md](QM_WORKFLOW.md) for full details.

### Quick QM Setup

```bash
# 1. Install QM dependencies
conda install -c conda-forge xtb ase -y

# 2. Generate QM trajectories (uses smaller molecules, shorter time)
bash run_qm_trajectories.sh

# 3. Create QM dataset
bash make_qm_dataset.sh

# 4. Train with transfer learning (loads MM checkpoint)
bash train_qm.sh

# 5. Run hybrid QM integration
bash run_hybrid_qm.sh

# 6. Evaluate
bash eval_qm.sh
```

**Key differences from MM**:
- Uses 0.25 fs timesteps (8x smaller)
- Smaller k values (k=4 typical)
- Tighter displacement bounds
- Force head enabled
- Transfer learning from MM model

## Understanding the Outputs

### Data Files

**`.npz` trajectory files** contain:
```python
import numpy as np
data = np.load('data/md/molecule.npz')

# Available arrays:
data['positions']      # (n_frames, n_atoms, 3) - positions in nm
data['velocities']     # (n_frames, n_atoms, 3) - velocities in nm/ps
data['forces']         # (n_frames, n_atoms, 3) - forces in kJ/(mol·nm) [QM only]
data['energies']       # (n_frames,) - potential energy in kJ/mol
data['temperature']    # (n_frames,) - temperature in K
data['atom_types']     # (n_atoms,) - atomic numbers
data['masses']         # (n_atoms,) - masses in amu
```

**`.pt` checkpoint files** contain:
```python
import torch
ckpt = torch.load('outputs/checkpoints/best.pt')

# Available keys:
ckpt['model_state_dict']    # Trained model weights
ckpt['config']              # Model configuration
ckpt['epoch']               # Training epoch
ckpt['train_loss']          # Training loss
ckpt['val_loss']            # Validation loss
```

### Evaluation Metrics

**Energy drift**: Rate of energy change over time
- Good: < 0.05 kJ/mol/ps
- Acceptable: 0.05-0.1 kJ/mol/ps
- Poor: > 0.1 kJ/mol/ps (check model/config)

**Bond RMSE**: Root mean square error in bond lengths
- Excellent: < 0.02 Å (QM)
- Good: 0.02-0.05 Å (MM)
- Acceptable: 0.05-0.1 Å
- Poor: > 0.1 Å (retrain needed)

**Speedup**: Reduction in force evaluations
- k=4: 3-4x speedup
- k=8: 6-8x speedup
- k=12: 10-12x speedup

## Next Steps

### Experiment with Hyperparameters

**Model size**:
```yaml
# configs/model_custom.yaml
hidden_dim: 256        # Try 128, 256, 512
num_layers: 6          # Try 4, 6, 8
attention_heads: 8     # Try 4, 8, 16
```

**Training settings**:
```yaml
# configs/train_custom.yaml
lr: 5e-5              # Try 1e-4, 5e-5, 1e-5
batch_size: 512       # Adjust based on GPU memory
k_steps: 8            # Try 4, 8, 12, 16
```

**Structural regularization**:
```yaml
lambda_struct_bond: 5.0      # Increase for tighter bonds
lambda_struct_angle: 2.0     # Increase for better angles
lambda_struct_dihedral: 1.0  # Increase for better dihedrals
```

### Curriculum Learning

Train with increasing k:

```bash
# 1. Train k=4 (converges fastest)
python src/04_train.py --dataset data/md/dataset_k4.npz ...

# 2. Fine-tune to k=8 (load k=4 checkpoint)
python src/04_train.py --dataset data/md/dataset_k8.npz \
    --pretrained outputs/checkpoints_k4/best.pt ...

# 3. Fine-tune to k=12 (load k=8 checkpoint)
python src/04_train.py --dataset data/md/dataset_k12.npz \
    --pretrained outputs/checkpoints_k8/best.pt ...
```

### Scale to Larger Molecules

- Try molecules with 30-50 atoms
- May need to increase model capacity
- Consider larger cutoff radius (0.9-1.0 nm)

### QM/MM Hybrid Systems

Combine QM accuracy in active regions with MM speed:

See [QMMM_INTEGRATION.md](QMMM_INTEGRATION.md) for implementation guide.

### Production Deployment

For large-scale simulations:

1. **Profile performance**: Use `line_profiler` to find bottlenecks
2. **Optimize inference**: Export to TorchScript or ONNX
3. **Batch predictions**: Run multiple replicas in parallel
4. **Monitor stability**: Track energy drift continuously

## Troubleshooting

### Installation Issues

**PyTorch/CUDA mismatch**:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**torch_geometric build errors**:
```bash
# Use pre-built wheels
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Training Issues

**Loss not decreasing**:
- Check data: `python -c "import numpy as np; d=np.load('data/md/dataset_k8.npz'); print({k: v.shape for k, v in d.items()})"`
- Reduce learning rate: `lr: 5e-5` instead of `1e-4`
- Check for NaN: Add `torch.autograd.set_detect_anomaly(True)` in train script

**Out of memory**:
- Reduce batch size: `batch_size: 128` instead of `512`
- Use gradient accumulation: `grad_accum: 4`
- Reduce model size: `hidden_dim: 128` instead of `256`

**Val loss >> train loss**:
- Overfitting - add dropout: `dropout: 0.1`
- Reduce model capacity
- Get more diverse training data

### Runtime Issues

**Hybrid integration failing**:
- Check model outputs: Add `--debug` flag
- Reduce delta scale: `--delta-scale 0.5`
- Increase retries: `--max-attempts 10`
- Check initial structure quality

**Slow performance**:
- Verify GPU usage: `watch -n 0.5 nvidia-smi`
- Profile code: `python -m line_profiler src/04_train.py`
- Reduce data augmentation temporarily

## Getting Help

- **Documentation**: Check other markdown files in this repo
- **Issues**: Search existing issues or open a new one
- **Examples**: Look at `configs/` for working configurations
- **Code**: Read docstrings in `src/` modules

## Summary Checklist

After completing this guide, you should have:

- [ ] Installed all dependencies successfully
- [ ] Run quick test with single molecule
- [ ] Trained a model on full MM dataset
- [ ] Evaluated hybrid integrator performance
- [ ] Understood key metrics and outputs
- [ ] (Optional) Completed QM workflow

Congratulations! You're now ready to use md-kstep for your own molecular dynamics research.
