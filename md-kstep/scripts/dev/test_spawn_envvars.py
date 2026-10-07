"""Test that spawn + module-level env vars correctly limits threads."""
# Set environment variables at module level (before any imports)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def worker_init():
    """Worker initializer."""
    pid = os.getpid()
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"[Worker {pid}] OMP_NUM_THREADS={omp}", flush=True)


def check_threads():
    """Check thread count after importing numpy."""
    import numpy as np  # This should respect module-level env vars
    pid = os.getpid()

    result = subprocess.run(['cat', f'/proc/{pid}/status'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if line.startswith('Threads:'):
            thread_count = int(line.split()[1])
            print(f"[Worker {pid}] Thread count: {thread_count}", flush=True)
            return thread_count
    return -1


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    print(f"Main process OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    with ProcessPoolExecutor(max_workers=3, initializer=worker_init) as executor:
        futures = [executor.submit(check_threads) for _ in range(3)]
        thread_counts = [f.result() for f in futures]

    print(f"\nThread counts: {thread_counts}")
    if all(tc <= 2 for tc in thread_counts):  # Allow 1-2 threads (main + maybe one helper)
        print("✅ SUCCESS!")
    else:
        print(f"❌ FAILURE: Expected 1-2 threads each, got {thread_counts}")
