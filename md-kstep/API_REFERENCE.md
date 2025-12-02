# API Reference

This document provides detailed documentation for all modules, classes, and functions in the md-kstep codebase.

## Table of Contents

1. [Core Utilities](#core-utilities-utilspy)
2. [Molecule Preparation](#molecule-preparation-00_prep_molspy)
3. [Trajectory Generation](#trajectory-generation)
4. [Dataset Creation](#dataset-creation-02_make_datasetpy)
5. [Model Architecture](#model-architecture-03_modelpy)
6. [Training](#training-04_trainpy)
7. [Evaluation](#evaluation)
8. [Hybrid Integration](#hybrid-integration)
9. [Configuration Files](#configuration-files)

---

## Core Utilities (`utils.py`)

Common utility functions used across all scripts.

### Logging

#### `configure_logging(level: int = logging.INFO) -> None`

Configure logging with a consistent format for CLI scripts.

**Parameters:**
- `level`: Logging level (default: `logging.INFO`)

**Example:**
```python
from utils import configure_logging
configure_logging(level=logging.DEBUG)
```

#### `LOGGER`

Global logger instance for all md-kstep modules.

**Example:**
```python
from utils import LOGGER
LOGGER.info("Processing molecule...")
LOGGER.warning("High energy detected")
```

### Random Seeds

#### `set_seed(seed: int) -> None`

Set random seeds for Python, NumPy, and PyTorch for reproducibility.

**Parameters:**
- `seed`: Random seed value

**Example:**
```python
from utils import set_seed
set_seed(42)
```

### File I/O

#### `load_yaml(path: Path) -> Dict`

Load a YAML configuration file.

**Parameters:**
- `path`: Path to YAML file

**Returns:**
- Dictionary containing configuration

**Example:**
```python
from utils import load_yaml
config = load_yaml(Path('configs/model.yaml'))
print(config['hidden_dim'])
```

#### `write_json(data: Dict, path: Path) -> None`

Write dictionary to JSON file with pretty formatting.

**Parameters:**
- `data`: Dictionary to save
- `path`: Output path

**Example:**
```python
from utils import write_json
write_json({'train': ['mol1', 'mol2']}, Path('splits/train.json'))
```

#### `read_smiles(path: Path) -> List[str]`

Read SMILES file, skipping blank and commented lines.

**Parameters:**
- `path`: Path to SMILES file

**Returns:**
- List of SMILES strings

**Example:**
```python
from utils import read_smiles
smiles = read_smiles(Path('data/molecules.smi'))
# ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', ...]
```

### Molecular Operations

#### `center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray`

Compute center of mass for a single frame.

**Parameters:**
- `positions`: Atom positions, shape `(n_atoms, 3)`, units: nm
- `masses`: Atom masses, shape `(n_atoms,)`, units: amu

**Returns:**
- Center of mass coordinates, shape `(3,)`, units: nm

**Example:**
```python
import numpy as np
from utils import center_of_mass

positions = np.random.rand(10, 3)  # 10 atoms
masses = np.array([12.0, 1.0, 1.0, 16.0, ...])  # 10 masses
com = center_of_mass(positions, masses)
# array([0.5, 0.5, 0.5])
```

#### `center_by_com(positions: np.ndarray, masses: np.ndarray) -> np.ndarray`

Center positions by subtracting center of mass.

**Parameters:**
- `positions`: Atom positions, shape `(n_atoms, 3)`
- `masses`: Atom masses, shape `(n_atoms,)`

**Returns:**
- COM-centered positions, shape `(n_atoms, 3)`

**Example:**
```python
from utils import center_by_com

positions_centered = center_by_com(positions, masses)
# Now sum(positions_centered * masses[:, None]) ≈ 0
```

### Context Managers

#### `numpy_seed(seed: Optional[int]) -> Iterator[None]`

Temporarily set NumPy random seed within a context.

**Parameters:**
- `seed`: Seed value (None to skip)

**Example:**
```python
from utils import numpy_seed
import numpy as np

with numpy_seed(42):
    random_values = np.random.rand(10)
# NumPy random state restored after context
```

---

## Molecule Preparation (`00_prep_mols.py`)

Convert SMILES strings to 3D structures with force field parameters.

### Command-Line Interface

```bash
python src/00_prep_mols.py \
    --smiles data/molecules.smi \
    --out-dir data/raw \
    --n-conformers 1 \
    --log-level INFO
```

**Arguments:**
- `--smiles`: Path to SMILES file
- `--out-dir`: Output directory for molecule files
- `--n-conformers`: Number of conformers to generate (default: 1)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Output Structure

For each molecule, creates:
```
data/raw/
└── aspirin/
    ├── mol.pdb          # 3D structure
    ├── system.xml       # OpenMM System (force field parameters)
    └── state.xml        # Initial state (positions, velocities)
```

### Key Functions

#### `prepare_molecule(smiles: str, name: str, out_dir: Path) -> None`

Full pipeline: SMILES → 3D → minimized → force field.

**Parameters:**
- `smiles`: SMILES string
- `name`: Molecule name
- `out_dir`: Output directory

**Steps:**
1. Parse SMILES with RDKit
2. Generate 3D conformer (UFF embedding)
3. Minimize energy (UFF)
4. Assign OpenFF force field
5. Export PDB and XML files

---

## Trajectory Generation

### MM Trajectories (`01_run_md_baselines.py`)

Generate molecular mechanics trajectories using OpenMM.

#### Command-Line Interface

```bash
python src/01_run_md_baselines.py \
    --molecule data/raw/aspirin \
    --config configs/md.yaml \
    --out data/md \
    --log-level INFO
```

**Arguments:**
- `--molecule`: Path to molecule directory
- `--config`: MD configuration YAML
- `--out`: Output directory
- `--log-level`: Logging level

#### Configuration (`configs/md.yaml`)

```yaml
dt_fs: 2.0                  # Timestep (femtoseconds)
temperature_K: 300.0        # Temperature (Kelvin)
friction_per_ps: 0.1        # Friction coefficient (1/ps)
equil_steps: 5000           # Equilibration steps
prod_steps: 5000000         # Production steps (10 ns @ 2 fs)
save_interval: 50           # Save every N steps
nve_after_steps: 4900000    # Start NVE window
platform: 'CUDA'            # OpenMM platform (CUDA, CPU, OpenCL)
```

#### Output Format

NPZ file with keys:
- `positions`: `(n_frames, n_atoms, 3)` - nm
- `velocities`: `(n_frames, n_atoms, 3)` - nm/ps
- `energies`: `(n_frames,)` - kJ/mol
- `temperature`: `(n_frames,)` - K
- `atom_types`: `(n_atoms,)` - atomic numbers
- `masses`: `(n_atoms,)` - amu

**Example:**
```python
import numpy as np

data = np.load('data/md/aspirin.npz')
print(data['positions'].shape)  # (100000, 21, 3)
print(data['energies'].mean())  # 145.3 kJ/mol
```

### QM Trajectories (`01b_run_qm_baselines.py`)

Generate quantum mechanics trajectories using xTB.

#### Command-Line Interface

```bash
python src/01b_run_qm_baselines.py \
    --smiles-file data/qm_molecules.smi \
    --config configs/qm.yaml \
    --out data/qm \
    --num-workers 4 \
    --log-level INFO
```

**Arguments:**
- `--smiles-file`: Path to SMILES file
- `--config`: QM configuration YAML
- `--out`: Output directory
- `--num-workers`: Parallel workers
- `--omp-num-threads`: OpenMP threads per worker

#### Configuration (`configs/qm.yaml`)

```yaml
dt_fs: 0.25                 # Timestep (smaller for QM)
temperature_K: 300.0
friction_per_ps: 0.002      # Lower friction for QM
equil_ps: 5.0               # Equilibration time
prod_ps: 30.0               # Production time
save_interval: 20
method: 'GFN2-xTB'          # xTB method
accuracy: 1.0e-6            # SCF convergence
```

#### Output Format

Same as MM, plus:
- `forces`: `(n_frames, n_atoms, 3)` - kJ/(mol·nm)

**Example:**
```python
data = np.load('data/qm/water.npz')
print(data['forces'].shape)  # (6000, 3, 3)
```

### PySCF Trajectories (`01c_run_pyscf_baselines.py`)

Generate ab initio trajectories using PySCF (CPU or GPU).

#### Command-Line Interface

```bash
python src/01c_run_pyscf_baselines.py \
    --smiles-file data/qm_small_molecules.smi \
    --config configs/qm_pyscf.yaml \
    --out data/qm_pyscf \
    --num-workers 2 \
    --log-level INFO
```

**Arguments:**
- `--smiles-file`: Path to SMILES file
- `--config`: PySCF configuration YAML
- `--out`: Output directory
- `--num-workers`: Parallel workers
- `--omp-num-threads`: OpenMP threads per worker

#### Configuration (`configs/qm_pyscf.yaml`)

```yaml
dt_fs: 0.25
temperature_K: 300.0
friction_per_ps: 0.002
equil_ps: 2.0
prod_ps: 10.0
save_interval: 20
method: 'RHF'               # Hartree-Fock
basis: 'sto-3g'             # Basis set
use_gpu: true               # Use GPU4PySCF if available
```

---

## Dataset Creation (`02_make_dataset.py`)

Convert trajectories into k-step training samples.

### Command-Line Interface

```bash
python src/02_make_dataset.py \
    --md-root data/md \
    --out-root data/md \
    --splits-dir data/splits \
    --ks 4 8 12 \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --stride 1 \
    --rotation-augment 3 \
    --log-level INFO
```

**Arguments:**
- `--md-root`: Directory containing trajectory NPZ files
- `--out-root`: Output directory for datasets
- `--splits-dir`: Output directory for molecule splits
- `--ks`: k values to generate datasets for
- `--train-frac`: Fraction of molecules for training
- `--val-frac`: Fraction for validation
- `--stride`: Stride for sampling (1 = dense, 10 = sparse)
- `--rotation-augment`: Number of random rotations per sample

### Output

For each k:
- `data/md/dataset_k{k}.npz`: Training samples
- `data/splits/train.json`: List of training molecules
- `data/splits/val.json`: List of validation molecules
- `data/splits/test.json`: List of test molecules

### Dataset Format

NPZ file with keys:
- `x_t`: `(n_samples, n_atoms, 3)` - Initial positions (COM-centered)
- `v_t`: `(n_samples, n_atoms, 3)` - Initial velocities
- `x_tk`: `(n_samples, n_atoms, 3)` - Target positions after k steps
- `v_tk`: `(n_samples, n_atoms, 3)` - Target velocities after k steps
- `atom_types`: `(n_samples, n_atoms)` - Atomic numbers
- `masses`: `(n_samples, n_atoms)` - Atomic masses
- `molecule`: `(n_samples,)` - Molecule names (for splits)
- `k_steps`: Scalar - k value

**Example:**
```python
data = np.load('data/md/dataset_k8.npz')
print(f"Samples: {len(data['x_t'])}")  # 119700
print(f"k-steps: {data['k_steps']}")   # 8
```

---

## Model Architecture (`03_model.py`)

Equivariant neural networks for k-step prediction.

### Configuration

#### `ModelConfig` (dataclass)

```python
@dataclass
class ModelConfig:
    arch: str = "egnn"                      # "egnn" or "transformer_egnn"
    hidden_dim: int = 128                   # Hidden layer size
    num_layers: int = 4                     # Number of EGNN layers
    cutoff_nm: float = 0.7                  # Graph cutoff radius (nm)
    activation: str = "silu"                # Activation function
    predict_delta: bool = True              # Always True
    max_disp_nm: float = 0.02               # Max displacement (nm)
    max_dvel_nm_per_ps: float = 0.1         # Max velocity change (nm/ps)
    use_force_head: bool = False            # Add force prediction head
    force_head_weight: float = 0.1          # Force loss weight
    dropout: float = 0.0                    # Dropout rate
    embedding_dim: int = 64                 # Atom type embedding size
    layer_norm: bool = True                 # Use layer normalization
    # Transformer-specific
    attention_heads: int = 8                # Number of attention heads
    use_edge_attention: bool = True         # Use edge attention
    attention_dropout: float = 0.0          # Attention dropout
    positional_encoding: str = "none"       # "none" or "learned"
    feedforward_dim: Optional[int] = None   # FFN hidden size (4*hidden_dim)
```

**Load from YAML:**
```python
from utils import load_yaml
from src.model import ModelConfig

config_dict = load_yaml('configs/model.yaml')
config = ModelConfig(**config_dict)
```

### Building Models

#### `build_model_from_config(config: ModelConfig) -> nn.Module`

Create model from configuration.

**Parameters:**
- `config`: ModelConfig instance

**Returns:**
- PyTorch model (EGNN or TransformerEGNN)

**Example:**
```python
from src.model import build_model_from_config, ModelConfig

config = ModelConfig(
    arch='transformer_egnn',
    hidden_dim=256,
    num_layers=6,
    cutoff_nm=0.7
)
model = build_model_from_config(config)
print(model)
```

### Model Classes

#### `EGNN(nn.Module)`

Base equivariant graph neural network.

**Constructor:**
```python
EGNN(
    hidden_dim: int,
    num_layers: int,
    cutoff_nm: float,
    activation: nn.Module,
    max_disp_nm: float,
    max_dvel_nm_per_ps: float,
    use_force_head: bool,
    dropout: float,
    embedding_dim: int,
    layer_norm: bool
)
```

**Forward Pass:**
```python
def forward(
    self,
    x: torch.Tensor,          # Positions (batch, n_atoms, 3)
    v: torch.Tensor,          # Velocities (batch, n_atoms, 3)
    atom_types: torch.Tensor, # Atomic numbers (batch, n_atoms)
    batch_idx: torch.Tensor   # Batch indices (batch*n_atoms,)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
        delta_x: Position changes (batch, n_atoms, 3)
        delta_v: Velocity changes (batch, n_atoms, 3)
        forces: Optional force predictions (batch, n_atoms, 3)
    """
```

**Example:**
```python
import torch
from src.model import EGNN, _activation

model = EGNN(
    hidden_dim=128,
    num_layers=4,
    cutoff_nm=0.7,
    activation=_activation('silu'),
    max_disp_nm=0.02,
    max_dvel_nm_per_ps=0.1,
    use_force_head=False,
    dropout=0.0,
    embedding_dim=64,
    layer_norm=True
)

x = torch.randn(2, 10, 3)      # 2 molecules, 10 atoms each
v = torch.randn(2, 10, 3)
atom_types = torch.randint(1, 9, (2, 10))
batch_idx = torch.cat([torch.zeros(10), torch.ones(10)]).long()

delta_x, delta_v, forces = model(x, v, atom_types, batch_idx)
print(delta_x.shape)  # (2, 10, 3)
```

#### `TransformerEGNN(nn.Module)`

Transformer-enhanced EGNN with attention.

**Constructor:**
```python
TransformerEGNN(
    hidden_dim: int,
    num_layers: int,
    cutoff_nm: float,
    activation: nn.Module,
    max_disp_nm: float,
    max_dvel_nm_per_ps: float,
    use_force_head: bool,
    dropout: float,
    embedding_dim: int,
    layer_norm: bool,
    attention_heads: int,
    attention_dropout: float,
    feedforward_dim: int
)
```

**Usage:**
Same as EGNN, but with attention mechanisms for better expressiveness.

---

## Training (`04_train.py`)

Supervised training loop with structural regularization.

### Command-Line Interface

```bash
python src/04_train.py \
    --dataset data/md/dataset_k8.npz \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --splits data/splits/train.json data/splits/val.json \
    --device cuda \
    --out-dir outputs/checkpoints \
    --pretrained outputs/checkpoints_mm/best.pt \
    --freeze-layers 0 \
    --log-level INFO
```

**Arguments:**
- `--dataset`: Path to dataset NPZ
- `--model-config`: Model configuration YAML
- `--train-config`: Training configuration YAML
- `--splits`: Paths to train and val JSON files
- `--device`: Device (`cuda` or `cpu`)
- `--out-dir`: Checkpoint output directory
- `--pretrained`: Optional pretrained checkpoint for transfer learning
- `--freeze-layers`: Number of layers to freeze (0 = none)
- `--log-level`: Logging level

### Configuration

#### `TrainConfig` (dataclass)

```python
@dataclass
class TrainConfig:
    seed: int = 42
    k_steps: int = 8
    batch_size: int = 512
    grad_accum: int = 1
    num_workers: int = 4
    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-5
    max_epochs: int = 200
    steps_per_epoch: int = 500
    amp: bool = True                        # Mixed precision
    grad_clip: float = 1.0
    lambda_vel: float = 1.0                 # Velocity loss weight
    lambda_com: float = 0.1                 # COM penalty
    lambda_force: float = 0.1               # Force loss weight
    val_every_steps: int = 100
    checkpoint_every_steps: int = 500
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    resume: Optional[str] = None
    wandb: Dict = field(default_factory=dict)
    random_rotate: bool = True
    random_rotate_mode: str = "batch"       # "batch" or "per_graph"
    # Structural regularization
    lambda_struct_bond: float = 0.0
    lambda_struct_angle: float = 0.0
    lambda_struct_dihedral: float = 0.0
    struct_max_bonds: Optional[int] = None
    struct_max_angles: Optional[int] = None
    struct_max_dihedrals: Optional[int] = None
    warmup_ratio: float = 0.05
    ema_decay: float = 0.0
    use_uncertainty_weighting: bool = False
    curriculum_struct_epochs: int = 0
    max_val_batches: int = 0
```

### Key Classes

#### `KStepDataset(Dataset)`

PyTorch dataset for k-step samples.

**Constructor:**
```python
KStepDataset(
    dataset_path: Path,
    allowed_molecules: Optional[Iterable[str]] = None
)
```

**Attributes:**
- `x_t`, `v_t`: Initial state
- `x_tk`, `v_tk`: Target state
- `masses`, `atom_types`: Molecular properties
- `molecule`: Molecule names
- `k_steps`: k value
- `pos_std`, `vel_std`: Normalization statistics

**Example:**
```python
from src.train import KStepDataset

dataset = KStepDataset(
    Path('data/md/dataset_k8.npz'),
    allowed_molecules=['aspirin', 'caffeine']
)
print(len(dataset))  # Number of samples
sample = dataset[0]  # Dict with 'x_t', 'v_t', etc.
```

### Output

Training produces:
- `outputs/checkpoints/best.pt`: Best validation checkpoint
- `outputs/checkpoints/latest.pt`: Latest checkpoint
- `outputs/logs/training.log`: Training log

**Checkpoint format:**
```python
checkpoint = {
    'epoch': int,
    'step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scaler_state_dict': dict,
    'train_loss': float,
    'val_loss': float,
    'config': dict,
    'model_config': dict
}
```

---

## Evaluation

### MM Evaluation (`05_eval_drift_rdfs.py`)

Compare baseline vs. hybrid trajectories.

#### Command-Line Interface

```bash
python src/05_eval_drift_rdfs.py \
    --baseline data/md \
    --hybrid-runs outputs/hybrid \
    --out-dir outputs/eval \
    --log-level INFO
```

**Arguments:**
- `--baseline`: Directory with baseline trajectories
- `--hybrid-runs`: Directory with hybrid trajectories
- `--out-dir`: Output directory for plots/metrics

#### Output

- `energy_drift_comparison.png`: Energy drift bar chart
- `rdf_comparison.png`: Radial distribution functions
- `bond_distributions.png`: Bond length histograms
- `summary.json`: Overall metrics
- `per_molecule_metrics.csv`: Detailed metrics per molecule

**Metrics:**
- Energy drift (kJ/mol/ps)
- Bond RMSE (Å)
- Angle RMSE (degrees)
- RDF similarity

### QM Evaluation (`05b_eval_qm.py`)

QM-specific evaluation metrics.

#### Command-Line Interface

```bash
python src/05b_eval_qm.py \
    --baseline data/qm \
    --hybrid-runs outputs/hybrid_qm \
    --out-dir outputs/eval_qm \
    --log-level INFO
```

**Output:**
- Energy drift comparison
- RMSD trajectories
- Force call efficiency
- Failed step statistics

---

## Hybrid Integration

### MM Integration (`06_hybrid_integrate.py`)

Hybrid integrator with OpenMM corrector.

#### Command-Line Interface

```bash
python src/06_hybrid_integrate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --md-config configs/md.yaml \
    --molecule data/raw/aspirin \
    --out outputs/hybrid/aspirin_k8.npz \
    --n-steps 5000 \
    --temperature 300.0 \
    --delta-scale 1.0 \
    --max-attempts 5 \
    --log-level INFO
```

**Arguments:**
- `--checkpoint`: Trained model checkpoint
- `--md-config`: MD configuration (for corrector)
- `--molecule`: Molecule directory
- `--out`: Output trajectory path
- `--n-steps`: Number of hybrid steps
- `--temperature`: Temperature (K)
- `--delta-scale`: Scaling factor for model predictions
- `--max-attempts`: Retry attempts if step fails

### QM Integration (`06b_hybrid_integrate_qm.py`)

Hybrid integrator with xTB corrector.

#### Command-Line Interface

```bash
python src/06b_hybrid_integrate_qm.py \
    --checkpoint outputs/checkpoints_qm/best.pt \
    --qm-config configs/qm.yaml \
    --molecule-name water \
    --smiles "O" \
    --out outputs/hybrid_qm/water_k4.npz \
    --n-steps 1000 \
    --temperature 300.0 \
    --log-level INFO
```

**Arguments:**
- `--checkpoint`: Trained model checkpoint
- `--qm-config`: QM configuration (for xTB)
- `--molecule-name`: Molecule name
- `--smiles`: SMILES string
- `--out`: Output trajectory path
- `--n-steps`: Number of hybrid steps
- `--temperature`: Temperature (K)

---

## Configuration Files

### Model Configs

#### `configs/model.yaml` (Base EGNN)

```yaml
arch: egnn
hidden_dim: 128
num_layers: 4
cutoff_nm: 0.7
activation: silu
max_disp_nm: 0.25          # MM: larger displacements
max_dvel_nm_per_ps: 6.0
use_force_head: false      # MM: forces optional
dropout: 0.0
embedding_dim: 64
layer_norm: true
```

#### `configs/model_transformer.yaml` (Transformer-EGNN)

```yaml
arch: transformer_egnn
hidden_dim: 256
num_layers: 6
cutoff_nm: 0.7
activation: silu
max_disp_nm: 0.25
max_dvel_nm_per_ps: 6.0
use_force_head: false
dropout: 0.0
embedding_dim: 64
layer_norm: true
attention_heads: 8
use_edge_attention: true
attention_dropout: 0.0
positional_encoding: learned
feedforward_dim: 1024      # 4 * hidden_dim
```

#### `configs/model_qm.yaml` (QM Model)

```yaml
arch: transformer_egnn
hidden_dim: 256
num_layers: 6
cutoff_nm: 0.7
activation: silu
max_disp_nm: 0.01          # QM: tighter bounds
max_dvel_nm_per_ps: 2.0
use_force_head: true       # QM: force head critical
force_head_weight: 0.1
dropout: 0.0
embedding_dim: 64
layer_norm: true
attention_heads: 8
feedforward_dim: 1024
```

### Training Configs

#### `configs/train.yaml` (MM Training)

```yaml
seed: 42
k_steps: 8
batch_size: 512
grad_accum: 1
num_workers: 4
lr: 1.0e-4
lr_min: 1.0e-6
weight_decay: 1.0e-5
max_epochs: 200
steps_per_epoch: 500
amp: true
grad_clip: 1.0
lambda_vel: 1.0
lambda_com: 0.1
lambda_force: 0.1
val_every_steps: 100
checkpoint_every_steps: 500
checkpoint_dir: outputs/checkpoints
random_rotate: true
random_rotate_mode: batch
# Structural regularization
lambda_struct_bond: 1.0
lambda_struct_angle: 0.5
lambda_struct_dihedral: 0.2
warmup_ratio: 0.05
```

#### `configs/train_qm.yaml` (QM Training)

```yaml
seed: 42
k_steps: 4                 # QM: smaller k
batch_size: 128            # QM: smaller batches
grad_accum: 4
num_workers: 4
lr: 5.0e-5                 # QM: lower LR for fine-tuning
lr_min: 1.0e-6
weight_decay: 1.0e-5
max_epochs: 100
steps_per_epoch: 200
amp: true
grad_clip: 1.0
lambda_vel: 1.0
lambda_com: 0.1
lambda_force: 0.2          # QM: higher force weight
val_every_steps: 50
checkpoint_every_steps: 200
checkpoint_dir: outputs/checkpoints_qm
random_rotate: true
random_rotate_mode: per_graph  # QM: per-graph rotation
# Structural regularization (stronger for QM)
lambda_struct_bond: 5.0
lambda_struct_angle: 2.0
lambda_struct_dihedral: 1.0
warmup_ratio: 0.05
```

### Trajectory Configs

See earlier sections for `md.yaml`, `qm.yaml`, and `qm_pyscf.yaml`.

---

## Helper Scripts

### QM Training Monitor (`scripts/qm_train_tui.py`)

Interactive TUI for monitoring QM training progress.

```bash
python scripts/qm_train_tui.py \
    --checkpoint-dir outputs/checkpoints_qm \
    --log-file outputs/logs/training.log
```

**Features:**
- Real-time loss curves
- GPU utilization
- Training metrics
- Checkpoint status

---

## Common Usage Patterns

### 1. Load Model for Inference

```python
import torch
from src.model import build_model_from_config, ModelConfig

# Load checkpoint
ckpt = torch.load('outputs/checkpoints/best.pt', map_location='cpu')

# Rebuild model
config = ModelConfig(**ckpt['model_config'])
model = build_model_from_config(config)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    delta_x, delta_v, forces = model(x, v, atom_types, batch_idx)
```

### 2. Custom Training Loop

```python
from src.train import KStepDataset, TrainConfig
from torch.utils.data import DataLoader

# Load dataset
train_data = KStepDataset(
    Path('data/md/dataset_k8.npz'),
    allowed_molecules=train_molecules
)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# Training loop
for epoch in range(max_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        delta_x, delta_v, _ = model(batch['x_t'], batch['v_t'], ...)
        loss = compute_loss(delta_x, delta_v, batch)
        loss.backward()
        optimizer.step()
```

### 3. Evaluate Custom Trajectory

```python
import numpy as np
from src.eval import compute_energy_drift, compute_bond_rmse

# Load trajectories
baseline = np.load('data/md/molecule.npz')
hybrid = np.load('outputs/hybrid/molecule.npz')

# Compute metrics
drift_baseline = compute_energy_drift(baseline['energies'], baseline['times'])
drift_hybrid = compute_energy_drift(hybrid['energies'], hybrid['times'])

print(f"Baseline drift: {drift_baseline:.4f} kJ/mol/ps")
print(f"Hybrid drift: {drift_hybrid:.4f} kJ/mol/ps")
```

---

## Units Summary

All units used throughout the codebase:

| Quantity | Unit | Symbol |
|----------|------|--------|
| Distance | nanometer | nm |
| Velocity | nanometer/picosecond | nm/ps |
| Force | kilojoule/(mol·nanometer) | kJ/(mol·nm) |
| Energy | kilojoule/mole | kJ/mol |
| Mass | atomic mass unit | amu |
| Time | picosecond or femtosecond | ps, fs |
| Temperature | Kelvin | K |
| Friction | 1/picosecond | 1/ps |

**Conversions:**
- 1 nm = 10 Å (Angstrom)
- 1 nm/ps = 1000 m/s = 0.01 Å/fs
- 1 kJ/mol = 0.239006 kcal/mol
- 1 amu = 1.66054 × 10⁻²⁷ kg

---

This API reference provides the foundation for understanding and extending md-kstep. For architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md). For usage examples, see [GETTING_STARTED.md](GETTING_STARTED.md).
