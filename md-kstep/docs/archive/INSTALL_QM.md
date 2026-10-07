# QM Environment Setup

## Install xTB and ASE

### Option 1: Conda (Recommended)
```bash
conda activate kstep  # or your environment name
conda install -c conda-forge xtb ase
```

### Option 2: Manual xTB Installation
If conda install fails:
```bash
# Download xTB from https://github.com/grimme-lab/xtb/releases
# Extract and add to PATH
export PATH=/path/to/xtb/bin:$PATH

# Install ASE via pip
pip install ase
```

## Verify Installation

```python
# Test xTB
from ase import Atoms
from ase.calculators.xtb import XTB

atoms = Atoms('H2O', positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0)])
atoms.calc = XTB(method='GFN2-xTB')
energy = atoms.get_potential_energy()
print(f"xTB working! H2O energy: {energy:.3f} eV")
```

## Additional Dependencies
Already in base environment:
- torch
- numpy
- scipy
- rdkit (for SMILES processing)

## GPU Support for xTB
xTB can use GPU acceleration (optional):
```bash
export OMP_NUM_THREADS=4  # CPU parallelization
export OMP_STACKSIZE=4G
```

Note: GPU support requires specific xTB builds. CPU is sufficient for this PoC.

