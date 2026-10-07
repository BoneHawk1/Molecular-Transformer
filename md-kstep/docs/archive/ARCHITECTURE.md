# md-kstep Architecture

This document provides a detailed technical overview of the md-kstep hybrid molecular dynamics integrator, covering design decisions, implementation details, and the mathematical foundations.

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training System](#training-system)
5. [Hybrid Integration](#hybrid-integration)
6. [Evaluation Framework](#evaluation-framework)
7. [Design Decisions](#design-decisions)
8. [Performance Optimization](#performance-optimization)

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MD-KSTEP PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

Input: SMILES strings
  │
  ├─► [00_prep_mols.py] ──► 3D structures + force fields
  │
  ├─► [01_run_md_baselines.py] ──► MD trajectories (MM)
  │   [01b_run_qm_baselines.py] ──► MD trajectories (QM)
  │
  ├─► [02_make_dataset.py] ──► k-step training samples
  │
  ├─► [04_train.py] ──► Trained neural network
  │                      (EGNN or Transformer-EGNN)
  │
  ├─► [06_hybrid_integrate.py] ──► Hybrid trajectories
  │   [06b_hybrid_integrate_qm.py]
  │
  └─► [05_eval_drift_rdfs.py] ──► Evaluation metrics
      [05b_eval_qm.py]

Output: Fast, accurate MD simulations
```

### Core Concept: Hybrid Integration

The key innovation is combining learned multi-step predictions with physics-based corrections:

**Traditional MD** (Velocity-Verlet):
```
for i in range(N):
    forces = compute_forces(positions)  # EXPENSIVE
    positions, velocities = integrate(forces, dt)
```
**Cost**: N force evaluations for N steps

**Hybrid k-step MD**:
```
for i in range(N // k):
    # Fast: Neural network prediction
    delta_x, delta_v = model.predict(positions, velocities, k)
    positions += delta_x
    velocities += delta_v

    # Expensive: Single physics correction
    forces = compute_forces(positions)
    positions, velocities = corrector_step(forces, dt)
```
**Cost**: N/k force evaluations for N steps → **k-fold speedup**

### Mathematical Foundation

#### k-step Prediction Problem

Given molecular state at time t:
- Positions: **x**(t) ∈ ℝⁿˣ³
- Velocities: **v**(t) ∈ ℝⁿˣ³
- Atomic numbers: **Z** ∈ ℕⁿ
- Masses: **m** ∈ ℝⁿ

Predict state at time t + k·Δt:
- Δ**x** = **x**(t + k·Δt) - **x**(t)
- Δ**v** = **v**(t + k·Δt) - **v**(t)

#### Equivariance Requirements

The model must be SE(3)-equivariant:

1. **Translation equivariance**:
   ```
   f(x + c, v) = f(x, v) + c
   ```
   Enforced via COM centering

2. **Rotation equivariance**:
   ```
   f(Rx, Rv) = Rf(x, v)  for any rotation matrix R
   ```
   Enforced via EGNN architecture

3. **Permutation invariance**:
   ```
   f(Px, Pv, PZ) = Pf(x, v, Z)  for any permutation P
   ```
   Enforced via graph structure

## Data Pipeline

### Stage 1: Molecule Preparation (`00_prep_mols.py`)

**Input**: SMILES strings
**Output**: 3D structures + force field parameters

**Process**:
1. **Parse SMILES** → RDKit Mol object
2. **Generate 3D conformer** → UFF embedding
3. **Minimize energy** → RDKit UFF minimizer
4. **Assign force field** → OpenFF (SMIRNOFF)
5. **Export**:
   - PDB: atomic coordinates
   - XML: force field parameters (bonds, angles, torsions, vdW)

**Key code** (src/00_prep_mols.py:45):
```python
# Create OpenFF Molecule and assign force field
off_mol = Molecule.from_rdkit(rdkit_mol)
forcefield = ForceField('openff-2.0.0.offxml')
topology = off_mol.to_topology()
system = forcefield.create_openmm_system(topology)
```

**Why OpenFF?**
- Modern, extensible force field
- Better coverage of drug-like molecules than AMBER/CHARMM
- Programmatic access to parameters

### Stage 2: Trajectory Generation

#### MM Trajectories (`01_run_md_baselines.py`)

**Input**: Molecule directory (PDB + XML)
**Output**: NPZ trajectory file

**Process**:
1. **Load system** → OpenMM System from XML
2. **Setup simulation**:
   - Integrator: Langevin (NVT ensemble)
   - Temperature: 300 K (configurable)
   - Friction: 0.1 ps⁻¹
   - Timestep: 2 fs (MM-appropriate)
   - Platform: CUDA (GPU-accelerated)
3. **Equilibration**: 5-10 ps to reach thermal equilibrium
4. **Production**: 10-20 ns trajectory
5. **Optional NVE window**: For energy conservation analysis
6. **Save**: positions, velocities, energies, temperatures

**Key code** (src/01_run_md_baselines.py:78):
```python
integrator = LangevinMiddleIntegrator(
    temperature * kelvin,
    friction / picosecond,
    dt * femtoseconds
)
simulation = Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature * kelvin)
```

**Timestep choice**:
- 2 fs: Standard for bonded systems with constraints
- Too large → integration errors
- Too small → unnecessary computation

#### QM Trajectories (`01b_run_qm_baselines.py`)

**Input**: SMILES
**Output**: NPZ trajectory file with forces

**Process**:
1. **Create ASE Atoms** from RDKit conformer
2. **Attach xTB calculator**:
   - Method: GFN2-xTB (tight-binding DFT)
   - Accuracy: 1e-6 Ha
3. **Setup dynamics**:
   - Integrator: Langevin (ASE)
   - Temperature: 300 K
   - Timestep: 0.25 fs (QM-appropriate)
   - Friction: 0.002 fs⁻¹
4. **Run**: 30-200 ps (shorter than MM due to cost)
5. **Save**: positions, velocities, forces, energies

**Key code** (src/01b_run_qm_baselines.py:92):
```python
from xtb.ase.calculator import XTB

calc = XTB(method='GFN2-xTB', accuracy=1e-6)
atoms.calc = calc
dyn = Langevin(atoms, timestep=0.25*fs, temperature_K=300, friction=0.002)
```

**Why smaller timestep?**
- QM forces have steeper gradients
- Need finer resolution to capture dynamics
- 0.25 fs prevents bond breaking/making

### Stage 3: Dataset Creation (`02_make_dataset.py`)

**Input**: Trajectory NPZ files
**Output**: Dataset NPZ with k-step windows

**Process**:
1. **Load trajectories** for all molecules
2. **Extract k-step windows**:
   ```
   For each trajectory:
     For t in range(0, T - k*stride, stride):
       Sample = {x[t], v[t], x[t+k], v[t+k], atom_types, masses}
   ```
3. **COM-center frames**:
   ```
   com = sum(x_i * m_i) / sum(m_i)
   x_centered = x - com
   ```
4. **Rotation augmentation** (optional):
   - Generate random rotation matrix R ∈ SO(3)
   - Apply to positions and velocities: (Rx, Rv)
   - Repeat 3x per sample (data augmentation)
5. **Create molecule splits**:
   - Train: 70% of molecules
   - Val: 15% of molecules
   - Test: 15% of molecules
   - Ensures model generalizes to new molecules
6. **Save dataset**:
   - Keys: `x0`, `v0`, `x_target`, `v_target`, `atom_types`, `masses`, `mol_ids`

**Key code** (src/02_make_dataset.py:145):
```python
def center_by_com(positions, masses):
    """Center positions by center of mass."""
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    return positions - com

def random_rotation_matrix():
    """Generate random SO(3) rotation using QR decomposition."""
    M = np.random.randn(3, 3)
    Q, R = np.linalg.qr(M)
    return Q @ np.diag(np.sign(np.diag(R)))
```

**Why COM centering?**
- Removes translational degrees of freedom
- Enforces translational invariance
- Improves model convergence

## Model Architecture

### Base Architecture: EGNN (Equivariant Graph Neural Network)

#### Graph Construction

For each molecular state, construct a graph:
- **Nodes**: Atoms (n nodes)
- **Node features**: Atomic number embeddings + optional velocities
- **Edges**: All atom pairs within cutoff distance (typically 0.7 nm)
- **Edge features**: Squared distances

**Key code** (src/03_model.py:245):
```python
def radius_graph(positions, cutoff):
    """Build graph from spatial coordinates."""
    dists = torch.cdist(positions, positions)  # (n, n)
    edge_index = (dists < cutoff).nonzero().t()  # (2, E)
    return edge_index
```

#### EGNN Layer

Each layer performs:

1. **Edge message**:
   ```
   m_ij = MLP_edge([h_i, h_j, ||x_i - x_j||²])
   ```

2. **Coordinate update** (equivariant):
   ```
   Δx_i = Σ_j (x_i - x_j) · MLP_coord(m_ij)
   x_i ← x_i + Δx_i
   ```

3. **Node update**:
   ```
   Δh_i = MLP_node([h_i, Σ_j m_ij])
   h_i ← LayerNorm(h_i + Δh_i)
   ```

**Key code** (src/03_model.py:82):
```python
class EGNNLayer(nn.Module):
    def forward(self, x, h, edge_index):
        row, col = edge_index
        rel = x[row] - x[col]  # Relative positions
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)

        # Edge messages
        edge_input = torch.cat([h[row], h[col], dist2], dim=-1)
        edge_feat = self.edge_mlp(edge_input)

        # Coordinate update (equivariant)
        coord_gate = self.coord_mlp(edge_feat)
        coord_update = rel * coord_gate
        delta_x = scatter_mean(coord_update, row, x.size(0))
        x = x + delta_x

        # Node update
        agg = scatter_mean(edge_feat, row, h.size(0))
        delta_h = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + delta_h)

        return x, h
```

**Why this is equivariant**:
- Only uses relative positions (x_i - x_j)
- Coordinate update scales relative vector: r_ij · scalar
- Distances are rotation-invariant: ||Rx_i - Rx_j|| = ||x_i - x_j||

### Advanced Architecture: Transformer-EGNN

Adds attention mechanisms for enhanced expressiveness:

#### Components

1. **Multi-head attention**:
   ```
   Attn(Q, K, V) = softmax(QK^T / √d_k) V
   ```

2. **Learned positional encoding**:
   - Maps atomic numbers to position embeddings
   - Added to node features

3. **Feedforward network**:
   ```
   FFN(x) = Linear(GELU(Linear(x)))
   ```

4. **Cross-attention** (optional):
   - Attention between spatial and latent features
   - Currently experimental

**Key code** (src/03_model.py:198):
```python
class TransformerEGNNLayer(nn.Module):
    def forward(self, x, h, edge_index):
        # Self-attention on node features
        attn_out = self.attention(
            h.unsqueeze(0),  # Add batch dim
            h.unsqueeze(0),
            h.unsqueeze(0)
        )[0].squeeze(0)

        h = self.norm1(h + self.attn_dropout(attn_out))

        # EGNN coordinate + message passing
        x, h_egnn = self.egnn_layer(x, h, edge_index)

        # Feedforward
        ff_out = self.feedforward(h_egnn)
        h = self.norm2(h_egnn + self.ff_dropout(ff_out))

        return x, h
```

**When to use**:
- Transformer-EGNN: Better for QM (complex interactions)
- EGNN: Faster, good for MM (simpler interactions)

### Output Heads

#### Position/Velocity Head (Main)

```python
class PredictionHead(nn.Module):
    def forward(self, h, v_init, config):
        # Predict raw deltas
        raw_dx = self.pos_mlp(h)  # (n, 3)
        raw_dv = self.vel_mlp(h)  # (n, 3)

        # Clamp to prevent catastrophic predictions
        delta_x = config.max_disp_nm * torch.tanh(raw_dx / config.max_disp_nm)
        delta_v = config.max_dvel_nm_per_ps * torch.tanh(raw_dv / config.max_dvel_nm_per_ps)

        return delta_x, delta_v
```

**Why clamping?**
- Prevents model from predicting unphysical jumps
- tanh ensures outputs in [-max, +max]
- Critical for stability during early training

**Bounds**:
- MM: max_disp_nm = 0.25 (large k, 2 fs timesteps)
- QM: max_disp_nm = 0.01 (small k, 0.25 fs timesteps)

#### Force Head (Auxiliary, QM)

```python
class ForceHead(nn.Module):
    def forward(self, h):
        forces = self.force_mlp(h)  # (n, 3)
        return forces
```

**Why forces?**
- Provides stronger gradient signal for QM
- Forces are available from xTB/PySCF
- Helps model learn energy landscape

**Loss weighting**:
```
L_total = L_position + L_velocity + λ_force * L_force
```
- λ_force = 0.1 (don't dominate position/velocity)

## Training System

### Loss Function

**Base loss** (MSE on deltas):
```
L_base = ||Δx_pred - Δx_true||² + ||Δv_pred - Δv_true||²
```

**Structural regularization**:
```
L_struct = λ_bond * L_bond + λ_angle * L_angle + λ_dihedral * L_dihedral
```

Where:
```
L_bond = Σ_bonds |d(i,j)_pred - d(i,j)_equil|²
L_angle = Σ_angles |θ(i,j,k)_pred - θ(i,j,k)_equil|²
L_dihedral = Σ_dihedrals |ϕ(i,j,k,l)_pred - ϕ(i,j,k,l)_equil|²
```

**Total loss**:
```
L_total = L_base + L_struct + λ_force * L_force
```

**Key code** (src/04_train.py:215):
```python
def structural_loss(x_pred, topology_indices, equilibrium_values):
    """Compute structural regularization loss."""
    bond_loss = 0.0
    for (i, j), d_equil in zip(topology_indices['bonds'], equilibrium_values['bonds']):
        d_pred = torch.norm(x_pred[i] - x_pred[j])
        bond_loss += (d_pred - d_equil) ** 2

    # Similar for angles and dihedrals...
    return bond_loss + angle_loss + dihedral_loss
```

**Why structural regularization?**
- Prevents unphysical geometries
- Especially important when k is large
- Uses topology information from force field

### Optimization

**Optimizer**: AdamW with decoupled weight decay
```
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,  # 5e-5 for QM fine-tuning
    weight_decay=1e-5,
    betas=(0.9, 0.999)
)
```

**Learning rate schedule**: Cosine annealing with warmup
```
warmup: Linear ramp from 0 to lr over 5 epochs
main: Cosine decay to lr_min over remaining epochs
```

**Gradient clipping**: Prevents exploding gradients
```
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Mixed precision** (AMP): Faster training, less memory
```
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

### Training Loop

**Key code** (src/04_train.py:380):
```python
def train_epoch(model, dataloader, optimizer, config):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Random rotation augmentation
        if config.random_rotate_mode == 'per_graph':
            batch = rotate_batch_per_graph(batch)

        # Forward pass
        with torch.cuda.amp.autocast():
            delta_x, delta_v = model(batch)
            loss = compute_loss(delta_x, delta_v, batch, config)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Transfer Learning (MM → QM)

**Process**:
1. **Load MM checkpoint**:
   ```python
   ckpt = torch.load('outputs/checkpoints_mm/best.pt')
   model.load_state_dict(ckpt['model_state_dict'], strict=False)
   ```

2. **Optionally freeze layers**:
   ```python
   for layer in model.egnn_layers[:freeze_layers]:
       for param in layer.parameters():
           param.requires_grad = False
   ```

3. **Fine-tune on QM data**:
   - Lower learning rate (5e-5 vs 1e-4)
   - Stronger structural regularization
   - Smaller k (4 vs 8)

**Why it works**:
- MM model learns general molecular geometry
- QM data refines to quantum accuracy
- Data-efficient: needs less QM data

## Hybrid Integration

### Algorithm

```python
def hybrid_step(positions, velocities, model, corrector, k, dt):
    """
    Single hybrid k-step.

    Args:
        positions: Current positions (nm)
        velocities: Current velocities (nm/ps)
        model: Trained neural network
        corrector: Physics integrator (OpenMM or ASE)
        k: Number of timesteps to predict
        dt: Base timestep (fs)

    Returns:
        positions: Updated positions
        velocities: Updated velocities
    """
    # 1. ML Jump
    delta_x, delta_v = model.predict(positions, velocities, k)
    positions_pred = positions + delta_x
    velocities_pred = velocities + delta_v

    # 2. Physics Correction
    positions_final, velocities_final = corrector.step(
        positions_pred,
        velocities_pred,
        dt
    )

    return positions_final, velocities_final
```

### MM Corrector (OpenMM)

**Key code** (src/06_hybrid_integrate.py:125):
```python
def corrector_step_mm(simulation, x_pred, v_pred, dt):
    """
    Single OpenMM velocity-Verlet step.
    """
    # Set predicted state
    simulation.context.setPositions(x_pred * nanometers)
    simulation.context.setVelocities(v_pred * nanometers / picosecond)

    # Integrate one timestep
    simulation.step(1)

    # Extract corrected state
    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True
    )

    x_corr = state.getPositions(asNumpy=True).value_in_unit(nanometers)
    v_corr = state.getVelocities(asNumpy=True).value_in_unit(nanometers/picosecond)

    return x_corr, v_corr
```

### QM Corrector (xTB)

**Key code** (src/06b_hybrid_integrate_qm.py:142):
```python
def corrector_step_qm(atoms, x_pred, v_pred, dt):
    """
    Single ASE Velocity-Verlet step with xTB forces.
    """
    # Convert units: nm → Angstrom, nm/ps → Angstrom/fs
    x_ang = x_pred * 10.0
    v_ang_fs = v_pred * 0.01

    # Set state
    atoms.set_positions(x_ang)
    atoms.set_velocities(v_ang_fs)

    # Compute forces
    forces = atoms.get_forces()  # eV/Angstrom

    # Velocity-Verlet integration
    masses = atoms.get_masses()  # amu
    accel = forces / masses[:, None]  # Angstrom/fs²

    v_half = v_ang_fs + 0.5 * accel * dt
    x_new = x_ang + v_half * dt

    atoms.set_positions(x_new)
    forces_new = atoms.get_forces()
    accel_new = forces_new / masses[:, None]

    v_new = v_half + 0.5 * accel_new * dt

    # Convert back: Angstrom → nm, Angstrom/fs → nm/ps
    x_corr = x_new / 10.0
    v_corr = v_new * 100.0

    return x_corr, v_corr
```

### Stability Checks

**Adaptive scaling**:
```python
for attempt in range(max_attempts):
    scale = delta_scale * (0.8 ** attempt)  # Reduce if failing

    dx_scaled = delta_x * scale
    dv_scaled = delta_v * scale

    x_pred = x + dx_scaled
    v_pred = v + dv_scaled

    try:
        x_corr, v_corr = corrector_step(x_pred, v_pred)

        # Check for NaN or explosion
        if is_stable(x_corr, v_corr):
            return x_corr, v_corr
    except:
        continue

raise IntegrationError("Failed after max_attempts")
```

**Stability criteria**:
- No NaN values
- Bond lengths < 2x equilibrium
- Kinetic energy < 10x average
- Potential energy change < 1000 kJ/mol

## Evaluation Framework

### Energy Drift Analysis

**Energy conservation** (NVE ensemble):
```
drift = (E_final - E_initial) / (t_final - t_initial)
```

**Expected values**:
- Baseline: 0.01-0.02 kJ/mol/ps
- Hybrid (good): 0.02-0.05 kJ/mol/ps
- Hybrid (acceptable): 0.05-0.1 kJ/mol/ps

**Key code** (src/05_eval_drift_rdfs.py:78):
```python
def compute_energy_drift(energies, times):
    """Compute linear drift rate via least-squares fit."""
    coeffs = np.polyfit(times, energies, deg=1)
    drift = coeffs[0]  # kJ/mol/ps
    return drift
```

### Structural Metrics

**Bond RMSE**:
```
RMSE_bond = sqrt(mean((d_hybrid - d_baseline)²))
```

**Radial Distribution Functions** (RDF):
```
g(r) = (V / N²) Σ_i Σ_j δ(r - r_ij) / (4πr² Δr)
```

Compares spatial structure between baseline and hybrid.

**Key code** (src/05_eval_drift_rdfs.py:145):
```python
def compute_rdf(positions, r_range=(0.1, 1.5), bins=150):
    """Compute radial distribution function."""
    n_atoms = len(positions[0])
    hist = np.zeros(bins)

    for frame in positions:
        dists = pdist(frame)  # All pairwise distances
        hist += np.histogram(dists, bins=bins, range=r_range)[0]

    # Normalize by ideal gas
    r = np.linspace(r_range[0], r_range[1], bins)
    volume = (4/3) * np.pi * (r_range[1]**3 - r_range[0]**3)
    rho = n_atoms / volume
    norm = 4 * np.pi * r**2 * rho

    g_r = hist / (len(positions) * n_atoms * norm)
    return r, g_r
```

### Performance Metrics

**Speedup**:
```
speedup = force_calls_baseline / force_calls_hybrid
        ≈ k (in practice: 0.8k to 0.95k due to failures)
```

**Accuracy vs. Speed Tradeoff**:
- k=4: 3-4x speedup, excellent accuracy
- k=8: 6-8x speedup, good accuracy
- k=12: 10-12x speedup, acceptable accuracy

## Design Decisions

### Why Equivariant Networks?

**Alternative**: Standard MLP or CNN
- ❌ Would need to learn symmetries from data
- ❌ Requires rotation augmentation (expensive)
- ❌ Worse generalization

**EGNN**:
- ✅ Symmetries baked into architecture
- ✅ Better sample efficiency
- ✅ Generalizes to unseen orientations

### Why Hybrid Integration?

**Alternative**: Pure ML integrator
- ❌ Accumulates errors over time
- ❌ Can drift off physical manifold
- ❌ Requires long rollout training (expensive)

**Hybrid**:
- ✅ Physics correction prevents drift
- ✅ Only needs supervised k-step prediction
- ✅ Stable over long trajectories

### Why Transfer Learning?

**Alternative**: Train QM model from scratch
- ❌ Needs lots of expensive QM data
- ❌ Slower convergence
- ❌ Harder to tune hyperparameters

**Transfer**:
- ✅ Leverages abundant MM data
- ✅ Faster convergence (2-3x fewer epochs)
- ✅ Better generalization

### Why COM Centering?

**Alternative**: Use absolute coordinates
- ❌ Model must learn translation invariance
- ❌ Larger prediction space
- ❌ Harder to converge

**COM-centered**:
- ✅ Removes 3 DOF (translation)
- ✅ Enforces symmetry exactly
- ✅ Faster training

## Performance Optimization

### Computational Bottlenecks

**Profiling results** (typical MM training):
```
Function                          % Time
--------------------------------------------
Model forward pass                 40%
Loss computation                   25%
Data loading                       15%
Structural regularization          10%
Gradient computation               10%
```

### Optimization Strategies

**1. Efficient graph construction**:
```python
# Slow: Nested loops
edge_index = []
for i in range(n):
    for j in range(i+1, n):
        if dist[i, j] < cutoff:
            edge_index.append([i, j])

# Fast: Vectorized with torch_cluster
from torch_cluster import radius_graph
edge_index = radius_graph(positions, r=cutoff)
```

**2. Caching topology indices**:
```python
# Cache bond/angle/dihedral indices per molecule
STRUCT_INDEXES = {}

def get_structure_indices(mol_id, topology):
    if mol_id not in STRUCT_INDEXES:
        STRUCT_INDEXES[mol_id] = compute_indices(topology)
    return STRUCT_INDEXES[mol_id]
```

**3. Mixed precision training**:
- ~30% speedup
- ~50% memory reduction
- Minimal accuracy loss

**4. Gradient accumulation**:
```python
# Effective batch size = batch_size * grad_accum
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / grad_accum
    loss.backward()

    if (i + 1) % grad_accum == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Memory Optimization

**1. Checkpoint activations** (not currently used):
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Trade compute for memory
    return checkpoint(self.expensive_layer, x)
```

**2. Smaller batches + gradient accumulation**:
- Fits on smaller GPUs
- Maintains effective batch size

**3. Delete intermediate tensors**:
```python
loss = compute_loss(...)
loss.backward()
del loss  # Free memory immediately
```

### Inference Optimization (Future)

**1. TorchScript compilation**:
```python
model_scripted = torch.jit.script(model)
model_scripted.save('model_optimized.pt')
```

**2. ONNX export**:
```python
torch.onnx.export(model, example_input, 'model.onnx')
# Can use ONNX Runtime for faster inference
```

**3. Quantization**:
```python
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

## Future Directions

### Active Areas of Development

1. **Learned thermostat**: Implicit temperature control
2. **Multi-resolution**: Different k for different regions
3. **Uncertainty quantification**: Confidence estimates
4. **Long-range interactions**: Improved cutoff schemes
5. **QM/MM integration**: Hybrid quantum/classical regions

### Scalability

**Current limits**:
- Molecules: <50 atoms (GPU memory)
- Trajectory length: Limited by disk space
- Cutoff: 0.7-1.0 nm (neighbor list size)

**Future improvements**:
- Sparse representations
- Hierarchical graphs
- Distributed training

---

This architecture document provides the technical foundation for understanding and extending md-kstep. For implementation details, see the source code and API_REFERENCE.md.
