#!/usr/bin/env python3
"""Test that lazy imports respect OMP_NUM_THREADS set by initializer."""
import os
import sys
import psutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def worker_init(threads):
    """Set threading before any imports."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    print(f"[Init {os.getpid()}] Set OMP_NUM_THREADS={threads}", flush=True)


def test_with_top_level_import():
    """BAD: Import at top means threads spawn before init."""
    # This import happens when module loads, BEFORE worker_init
    from xtb.ase.calculator import XTB
    from ase import Atoms

    pid = os.getpid()
    threads = psutil.Process(pid).num_threads()
    omp = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.calc = XTB()
    energy = atoms.get_potential_energy()

    print(f"[Worker {pid}] TOP LEVEL IMPORT: {threads} threads, OMP={omp}")
    return threads


def test_with_lazy_import():
    """GOOD: Import inside function, AFTER init."""
    # These imports happen AFTER worker_init has run
    from xtb.ase.calculator import XTB
    from ase import Atoms

    pid = os.getpid()
    threads = psutil.Process(pid).num_threads()
    omp = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.calc = XTB()
    energy = atoms.get_potential_energy()

    print(f"[Worker {pid}] LAZY IMPORT: {threads} threads, OMP={omp}")
    return threads


if __name__ == "__main__":
    test_type = sys.argv[1] if len(sys.argv) > 1 else "lazy"

    print(f"\nTesting: {test_type} imports")
    print("="*50)

    init_func = partial(worker_init, 1)

    if test_type == "top":
        # This will FAIL because imports already happened
        func = test_with_top_level_import
    else:
        # This should SUCCEED
        func = test_with_lazy_import

    with ProcessPoolExecutor(max_workers=2, initializer=init_func) as executor:
        futures = [executor.submit(func) for _ in range(2)]
        results = [f.result() for f in futures]

    print(f"\nThread counts: {results}")
    if all(t <= 15 for t in results):
        print("✅ SUCCESS: Thread limits respected")
    else:
        print(f"❌ FAIL: Still spawning {max(results)} threads (should be <15)")
