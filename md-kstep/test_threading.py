#!/usr/bin/env python3
"""Test if ProcessPoolExecutor initializer sets environment variables."""
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def worker_init(threads):
    """Set threading environment variables."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    print(f"Worker PID {os.getpid()}: Set OMP_NUM_THREADS={threads}")


def check_threads():
    """Check thread count and environment in worker."""
    import psutil
    pid = os.getpid()
    proc = psutil.Process(pid)
    thread_count = proc.num_threads()
    omp = os.environ.get("OMP_NUM_THREADS", "NOT SET")

    # Also check xTB if available
    try:
        from xtb.ase.calculator import XTB
        print(f"Worker {pid}: {thread_count} threads, OMP_NUM_THREADS={omp}, xTB loaded")
    except ImportError:
        print(f"Worker {pid}: {thread_count} threads, OMP_NUM_THREADS={omp}, no xTB")

    return thread_count


if __name__ == "__main__":
    print("Testing ProcessPoolExecutor with thread limiting...")
    print(f"Main process PID: {os.getpid()}")

    threads_per_worker = 1
    init_func = partial(worker_init, threads_per_worker)

    with ProcessPoolExecutor(max_workers=2, initializer=init_func) as executor:
        futures = [executor.submit(check_threads) for _ in range(2)]
        results = [f.result() for f in futures]

    print(f"\nThread counts: {results}")
    print("If thread counts are >10, environment variables aren't working!")
