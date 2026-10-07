#!/usr/bin/env python3
"""Test xTB thread spawning during actual calculation."""
import os
import time
import psutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def worker_init(threads):
    """Set threading environment variables."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["OMP_STACKSIZE"] = "4G"


def run_xtb_calc():
    """Run actual xTB calculation and check thread count."""
    from ase import Atoms
    from xtb.ase.calculator import XTB

    pid = os.getpid()
    proc = psutil.Process(pid)

    # Check threads before calculation
    threads_before = proc.num_threads()
    omp = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    print(f"Worker {pid} BEFORE calc: {threads_before} threads, OMP_NUM_THREADS={omp}")

    # Create simple molecule (water)
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.calc = XTB(method="GFN2-xTB")

    # Run calculation
    energy = atoms.get_potential_energy()

    # Check threads after calculation
    time.sleep(0.5)  # Let threads spawn
    threads_after = proc.num_threads()
    print(f"Worker {pid} AFTER calc: {threads_after} threads (energy={energy:.4f} eV)")

    return (threads_before, threads_after)


if __name__ == "__main__":
    print("Testing xTB thread spawning during calculation...")
    print(f"Setting OMP_NUM_THREADS=1 in worker initializer\n")

    init_func = partial(worker_init, 1)

    with ProcessPoolExecutor(max_workers=2, initializer=init_func) as executor:
        futures = [executor.submit(run_xtb_calc) for _ in range(2)]
        results = [f.result() for f in futures]

    print(f"\nResults: before/after threads")
    for i, (before, after) in enumerate(results):
        print(f"  Worker {i+1}: {before} → {after} threads")

    if any(after > 15 for _, after in results):
        print("\n❌ PROBLEM: xTB is spawning many threads despite OMP_NUM_THREADS=1")
    else:
        print("\n✅ SUCCESS: xTB respecting thread limits")
