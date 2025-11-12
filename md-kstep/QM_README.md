# QM Transfer: k-Step Integrator for Quantum Mechanics

This directory contains the implementation of the k-step integrator technology transferred from molecular mechanics to quantum mechanics calculations, enabling efficient QM/MM dynamics simulations.

## Overview

The QM transfer extends the existing molecular mechanics k-step integrator to work with quantum mechanical calculations, providing significant computational savings for QM/MM simulations where QM force evaluations are 100-1000x more expensive than MM.

## Key Components

### 1. QM Engine (`src/qm_engine.py`)

Python interface for QM calculations using PySCF:
- Supports HF, DFT, and other methods
- Configurable basis sets and SCF convergence
- Caching for expensive calculations
- QM/MM boundary handling with electrostatic embedding

**Usage:**
```python
from qm_engine import QMEngine, QMConfig

config = QMConfig.from_yaml("configs/qm.yaml")
engine = QMEngine(config, atom_symbols=["H", "H", "O"])
result = engine.calculate(positions_nm)  # positions in nm
```

### 2. QM Dataset Generation (`scripts/generate_qm_data.py`)

Generate QM reference trajectories for training:
- Small molecule QM MD trajectories
- Energy, force, and wavefunction data
- Configurable basis sets and methods

**Usage:**
```bash
python scripts/generate_qm_data.py \
    --smiles "O" \
    --qm-config configs/qm.yaml \
    --output-dir data/qm/water \
    --dt-fs 0.5 \
    --num-steps 1000
```

### 3. QM-Enhanced Model (`src/03_model.py`)

Extended EGNN model with QM features:
- Electronic structure encoder
- QM-aware cross-attention mechanism
- Optional electronic variable prediction
- Orbital orthogonality constraints

**Configuration:**
Enable QM features in `configs/model_qm.yaml`:
```yaml
use_qm_features: true
electronic_dim: 64
predict_electronic: true
lambda_electronic: 0.1
```

### 4. QM Hybrid Integrator (`src/06_qm_hybrid_integrate.py`)

Hybrid integrator combining learned k-step jumps with QM corrector:
- Learned k-step position/velocity updates
- Single QM force evaluation for correction
- Automatic fallback on divergence
- Energy conservation monitoring

**Usage:**
```bash
python src/06_qm_hybrid_integrate.py \
    --checkpoint checkpoints/model.pt \
    --model-config configs/model_qm.yaml \
    --qm-config configs/qm.yaml \
    --initial-qm data/qm/water/trajectory.npz \
    --steps 100 \
    --k-steps 8 \
    --out outputs/qm_hybrid.npz
```

### 5. QM Validation (`src/qm_validation.py`)

Validation metrics for QM trajectories:
- Energy drift analysis
- Structural RMSD vs reference
- Orbital stability metrics
- Dipole moment correlation

**Usage:**
```bash
python src/qm_validation.py \
    --hybrid outputs/qm_hybrid.npz \
    --reference data/qm/water/trajectory.npz \
    --out outputs/validation.json
```

## Configuration Files

### `configs/qm.yaml`
QM calculation settings (method, basis, SCF parameters)

### `configs/qmmm.yaml`
QM/MM boundary configuration (QM region, MM charges, embedding)

### `configs/model_qm.yaml`
QM-enhanced model architecture settings

## Performance Optimization

### Caching
QM calculations are automatically cached when `use_cache: true` is set in the QM config. Cache keys are based on positions and calculation parameters.

### Parallelization
For batch QM calculations, use multiple processes:
```python
from multiprocessing import Pool
# Distribute QM calculations across processes
```

### GPU Acceleration
PySCF supports GPU acceleration for certain operations. Configure via PySCF environment variables.

## Expected Performance

- **Force-call reduction**: 20-100x depending on QM cost
- **Energy conservation**: <0.1 kcal/mol per 100 steps
- **Structural accuracy**: <0.1 Å RMSD vs full QM
- **Computational speedup**: 10-50x wall-time for QM/MM systems

## QM/MM Integration

The QM/MM engine (`QMMMEngine` in `qm_engine.py`) handles:
- Electrostatic embedding of MM environment
- QM/MM boundary interactions
- Force combination for hybrid systems

**Example:**
```python
from qm_engine import QMMMEngine, QMEngine, QMConfig

qm_config = QMConfig.from_yaml("configs/qm.yaml")
qm_engine = QMEngine(qm_config, qm_atom_symbols)

mm_charges = np.array([...])  # MM point charges
mm_positions = np.array([...])  # MM positions in nm

qmmm_engine = QMMMEngine(qm_engine, mm_charges, mm_positions)
result = qmmm_engine.calculate(qm_positions)
```

## Training QM Models

1. **Generate QM data:**
   ```bash
   python scripts/generate_qm_data.py --smiles "H2O" --qm-config configs/qm.yaml --output-dir data/qm/water
   ```

2. **Create dataset:**
   ```python
   from qm_dataset import create_qm_dataset
   create_qm_dataset(trajectory_paths, output_path, k_steps=8)
   ```

3. **Train model:**
   ```bash
   python src/04_train.py \
       --dataset data/qm/dataset_k8.npz \
       --model-config configs/model_qm.yaml \
       --train-config configs/train.yaml \
       --splits data/splits/train.json data/splits/val.json
   ```

## Troubleshooting

### SCF Convergence Failures
- Increase `scf_max_cycles` in QM config
- Adjust `scf_level_shift` or `scf_damp_factor`
- Use smaller time steps or k values

### Energy Drift
- Reduce `max_disp_nm` in model config
- Increase corrector steps
- Check QM method/basis appropriateness

### Memory Issues
- Reduce batch size during training
- Use smaller basis sets for initial testing
- Enable gradient checkpointing

## Dependencies

- PySCF: `pip install pyscf`
- RDKit: `pip install rdkit-pypi`
- PyTorch: `pip install torch`
- NumPy, SciPy: Standard scientific Python stack

## References

See `PLAN.md` for the original transfer plan and implementation details.

