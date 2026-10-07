"""Test that spawn method correctly limits threads per worker."""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def worker_init(threads_per_worker):
    """Set threading limits before any library imports."""
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
    print(f"[Worker {os.getpid()}] Set OMP_NUM_THREADS={threads_per_worker}", flush=True)


def check_threads():
    """Import numpy and check actual thread count."""
    import numpy as np
    import subprocess

    pid = os.getpid()
    # Count threads for this process
    result = subprocess.run(['cat', f'/proc/{pid}/status'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if line.startswith('Threads:'):
            thread_count = int(line.split()[1])
            print(f"[Worker {pid}] Actual thread count: {thread_count}", flush=True)
            return thread_count
    return -1


if __name__ == "__main__":
    # Test with spawn method
    print("Testing with spawn method...")
    multiprocessing.set_start_method('spawn', force=True)

    threads_per_worker = 1
    with ProcessPoolExecutor(max_workers=3, initializer=partial(worker_init, threads_per_worker)) as executor:
        futures = [executor.submit(check_threads) for _ in range(3)]
        thread_counts = [f.result() for f in futures]

    print(f"\nThread counts: {thread_counts}")
    if all(tc == 1 for tc in thread_counts):
        print("✅ SUCCESS: All workers have exactly 1 thread!")
    else:
        print(f"❌ FAILURE: Workers have {thread_counts} threads (expected [1, 1, 1])")
