# Wall-clock study (k=8, 20 corrector steps, NVIDIA GeForce RTX 5070 Ti)

Seconds of wall-clock per simulated ps (lower is better); speedup = baseline / hybrid. Hybrid runs use the evaluation settings (proposal checks, quench, thermostat); rejected steps fall back to MD and are included.

| molecule | baseline (best platform) | zero s/ps (speedup, model ms/step) | flow_k8_v2 s/ps (speedup, model ms/step) | flow_k8_v2_large s/ps (speedup, model ms/step) | flow_k12_v2 s/ps (speedup, model ms/step) |
|---|---|---|---|---|---|
| CC(C)NCCO | 0.2372 (CUDA) | 0.0377 (6.29x, 6.35) | 0.1533 (1.55x, 25.80) | 0.1407 (1.69x, 28.62) | 0.1668 (1.42x, 36.39) |
| CC(=O)NC(C)C(=O)NC | 0.2511 (CUDA) | 0.0540 (4.65x, 8.31) | 0.2817 (0.89x, 30.00) | 0.2743 (0.92x, 29.07) | 0.2999 (0.84x, 41.40) |
| CCCCCO | 0.5086 (CUDA) | 0.0632 (8.05x, 8.56) | 0.1612 (3.16x, 25.47) | 0.1488 (3.42x, 28.40) | 0.1709 (2.98x, 39.25) |
| CC(=O)Oc1ccccc1C(=O)O | 0.4997 (CUDA) | 0.0575 (8.70x, 9.34) | 0.1741 (2.87x, 26.62) | 0.1644 (3.04x, 33.22) | 0.1781 (2.81x, 33.39) |
| **mean speedup** | | 6.92x | 2.12x | 2.26x | 2.01x |
