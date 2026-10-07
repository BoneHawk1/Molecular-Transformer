# QM/MM Integration Strategy

This document outlines the strategy for integrating the QM k-step integrator into a QM/MM pipeline.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│          Full System (Protein + Ligand)         │
│                                                  │
│    MM Region              QM Region             │
│  ┌────────────┐         ┌──────────┐           │
│  │  Protein   │◄───────►│  Active  │           │
│  │  Backbone  │ Boundary │   Site   │           │
│  │  & Solvent │  Atoms  │          │           │
│  └────────────┘         └──────────┘           │
└─────────────────────────────────────────────────┘
```

## Two Integration Approaches

### Option 1: ASE-Based QM/MM (Simpler)

**Advantages:**
- Built-in QM/MM support in ASE
- Simpler to implement
- Works with xTB and other QM codes

**Implementation:**
```python
from ase.calculators.qmmm import SimpleQMMM
from ase.calculators.xtb import XTB
from ase.calculators.amber import Amber  # or other MM

# Define QM region (atom indices)
qm_region = [0, 1, 2, 3, 4]  # Active site atoms

# Create QM/MM calculator
qm_calc = XTB(method='GFN2-xTB')
mm_calc = Amber(...)  # or OpenMM via custom calculator

qmmm_calc = SimpleQMMM(
    qm_indices=qm_region,
    qm_calculator=qm_calc,
    mm_calculator=mm_calc,
    vacuum=False,
)

atoms.calc = qmmm_calc
```

**Integration with k-step model:**
- Apply ML k-step prediction to QM region only
- Use standard MM integrator for MM region
- Apply QM/MM corrector after ML jump

**Limitations:**
- Less flexible than OpenMM-Torch
- May be slower for large systems
- Limited control over MM details

### Option 2: OpenMM-Torch Integration (More Control)

**Advantages:**
- Full control over MM dynamics
- Can use OpenMM's optimized MM code
- Better performance for large systems
- Can integrate ML model as custom force

**Implementation:**
```python
import torch
from openmm import CustomExternalForce

class MLQMForce(CustomExternalForce):
    """Custom OpenMM force using ML k-step predictions."""
    
    def __init__(self, model, qm_atoms, k_steps):
        super().__init__("0")  # Zero energy (forces handled separately)
        self.model = model
        self.qm_atoms = qm_atoms
        self.k_steps = k_steps
        self.step_count = 0
        
    def updateParametersInContext(self, context):
        # Every k steps, get ML prediction
        if self.step_count % self.k_steps == 0:
            # Get current QM region state
            state = context.getState(getPositions=True, getVelocities=True)
            qm_pos = state.getPositions()[self.qm_atoms]
            qm_vel = state.getVelocities()[self.qm_atoms]
            
            # ML prediction
            with torch.no_grad():
                delta_pos, delta_vel = self.model.predict(qm_pos, qm_vel)
            
            # Apply jump + corrector
            # ... (implementation details)
        
        self.step_count += 1
```

**Full system integration:**
1. Standard OpenMM for MM region
2. ML k-step integrator for QM region
3. QM/MM coupling via electrostatics + boundary terms
4. Single xTB corrector call per macro-step

**Challenges:**
- More complex implementation
- Need to handle QM/MM boundary correctly
- Requires careful force partitioning

## Recommended Integration Workflow

### Phase 1: Proof-of-Concept (ASE-based)

1. **Simple QM/MM system:**
   - Small molecule (10-20 atoms) as QM region
   - TIP3P water box as MM region
   - Validate energy conservation

2. **Implementation:**
   ```python
   # Pseudo-code for hybrid QM/MM integrator
   
   for step in range(num_steps):
       # Every k steps:
       if step % k == 0:
           # ML prediction for QM region
           qm_delta_pos, qm_delta_vel = model.predict(qm_state)
           
           # Apply jump to QM atoms
           qm_atoms.positions += qm_delta_pos
           qm_atoms.velocities += qm_delta_vel
           
           # One QM/MM corrector step
           qmmm_corrector_step(full_system, dt=base_dt)
       
       # Standard MM propagation for MM region
       mm_region.step(base_dt)
   ```

3. **Validation metrics:**
   - Total energy conservation
   - QM region structure preservation
   - MM region dynamics fidelity
   - No artifacts at QM/MM boundary

### Phase 2: Production (OpenMM-Torch or ASE)

1. **Test systems (increasing complexity):**
   - Solvated small molecule (20 QM + 500 MM atoms)
   - Peptide active site (30 QM + 1000 MM atoms)
   - Full enzyme (30 QM + 5000+ MM atoms)

2. **Performance benchmarks:**
   - Compare hybrid vs standard QM/MM wall time
   - Measure speedup factor (target: 10-50x for k=4-12)
   - Profile bottlenecks (ML inference vs QM corrector)

3. **Production features:**
   - Temperature coupling for both regions
   - Pressure control (if needed)
   - Constraints handling
   - Trajectory output/analysis

## QM/MM Boundary Treatment

### Link Atom Approach (Simpler)

- Add "link" hydrogen atoms at QM/MM bonds
- Scale MM charges near boundary
- ASE SimpleQMMM handles this automatically

### Electrostatic Embedding

- QM calculation includes MM point charges
- Polarizes QM electron density
- More accurate but requires QM code support

**Recommended for k-step model:**
Use electrostatic embedding if xTB supports it (check ASE docs), otherwise link atoms.

## Expected Performance

### Theoretical Speedup

For k=4 (QM k-step integrator):
- **Force call reduction:** 4x (4 QM calls → 1 per macro-step)
- **Overhead:** ML inference (~1ms) + data transfer
- **Net speedup:** 3-3.5x for small QM regions (10-30 atoms)

For k=8:
- **Force call reduction:** 8x
- **Net speedup:** 6-7x (with proper tuning)

### Realistic Expectations

- Small QM regions (10-30 atoms): 3-7x speedup
- Medium QM regions (30-100 atoms): 5-10x speedup (QM dominates)
- Large QM regions (100+ atoms): 8-15x speedup

## Implementation Checklist

### Phase 1: ASE-based PoC
- [ ] Implement ASE QM/MM calculator wrapper
- [ ] Integrate k-step model for QM region
- [ ] Add QM/MM corrector step
- [ ] Test on solvated small molecule
- [ ] Validate energy conservation
- [ ] Measure speedup vs standard QM/MM

### Phase 2: Optimization
- [ ] Profile and optimize bottlenecks
- [ ] Implement adaptive k-step (start small, increase if stable)
- [ ] Add error detection and recovery
- [ ] Test on enzyme active site

### Phase 3: Production (if needed)
- [ ] OpenMM-Torch integration (if ASE too slow)
- [ ] Production-ready error handling
- [ ] Extensive validation suite
- [ ] Documentation and examples

## Code Structure

```
md-kstep/
├── src/
│   ├── qmmm/
│   │   ├── __init__.py
│   │   ├── ase_integrator.py      # ASE-based QM/MM integrator
│   │   ├── openmm_torch.py        # OpenMM-Torch integrator (optional)
│   │   ├── boundary.py            # QM/MM boundary handling
│   │   └── utils.py               # Shared utilities
│   └── ...
├── examples/
│   ├── qmmm_water_box.py          # Simple QM/MM example
│   ├── qmmm_enzyme.py             # Enzyme active site example
│   └── ...
└── tests/
    ├── test_qmmm_energy.py        # Energy conservation tests
    ├── test_qmmm_boundary.py      # Boundary treatment tests
    └── ...
```

## References

- **ASE QM/MM:** https://wiki.fysik.dtu.dk/ase/ase/calculators/qmmm.html
- **OpenMM-Torch:** https://github.com/openmm/openmm-torch
- **xTB QM/MM:** Check xTB documentation for QM/MM support
- **Boundary treatments:** Senn & Thiel, Angew. Chem. Int. Ed. 2009

## Next Steps

1. Install ASE QM/MM dependencies
2. Implement simple QM/MM wrapper in `src/qmmm/ase_integrator.py`
3. Test on water box example
4. Validate energy conservation
5. Measure speedup
6. Iterate and optimize

## Contact / Support

For questions about:
- QM k-step model: See main README
- QM/MM implementation: This document
- ASE issues: ASE mailing list / GitLab
- OpenMM issues: OpenMM forums

