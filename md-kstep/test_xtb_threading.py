"""Test xTB respects OMP_NUM_THREADS=1,1 format."""
import os

# Set xTB threading limits BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1,1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_STACKSIZE"] = "4G"

import subprocess
import time
from ase import Atoms
from xtb.ase.calculator import XTB

# Create a simple molecule (water)
atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
atoms.calc = XTB(method="GFN2-xTB")

print(f"Process {os.getpid()}: Starting xTB calculation...")

# Check thread count before calculation
result_before = subprocess.run(['cat', f'/proc/{os.getpid()}/status'], capture_output=True, text=True)
for line in result_before.stdout.split('\n'):
    if line.startswith('Threads:'):
        threads_before = int(line.split()[1])
        print(f"Threads before xTB: {threads_before}")
        break

# Trigger xTB calculation
start = time.time()
energy = atoms.get_potential_energy()
elapsed = time.time() - start
print(f"Energy: {energy:.4f} eV (calculated in {elapsed:.2f}s)")

# Check thread count after calculation
result_after = subprocess.run(['cat', f'/proc/{os.getpid()}/status'], capture_output=True, text=True)
for line in result_after.stdout.split('\n'):
    if line.startswith('Threads:'):
        threads_after = int(line.split()[1])
        print(f"Threads after xTB: {threads_after}")
        break

if threads_after <= 2:
    print(f"✅ SUCCESS: xTB respects threading limits ({threads_after} threads)")
else:
    print(f"❌ FAILURE: xTB spawned too many threads ({threads_after} threads, expected ≤2)")
