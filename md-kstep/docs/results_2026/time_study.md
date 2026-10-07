# Wall-clock study (k=4, 10 corrector steps, NVIDIA GeForce RTX 5070 Ti)

Seconds of wall-clock per simulated ps (lower is better); speedup = baseline / hybrid. Hybrid runs use the evaluation settings (proposal checks, quench, thermostat); rejected steps fall back to MD and are included.

| molecule | baseline (best platform) | zero s/ps (speedup, model ms/step) | mean_k4 s/ps (speedup, model ms/step) | flow_k4 s/ps (speedup, model ms/step) | flow_k8 s/ps (speedup, model ms/step) | legacy_transformer s/ps (speedup, model ms/step) |
|---|---|---|---|---|---|---|
| CC(C)NCCO | 0.0602 (CUDA) | 0.0258 (2.33x, 0.80) | 0.0276 (2.18x, 2.50) | 0.0446 (1.35x, 8.41) | 0.0307 (1.96x, 11.57) | 0.0617 (0.98x, 15.20) |
| CCN(CC)CCO | 0.0533 (CUDA) | 0.0263 (2.03x, 0.79) | 0.0284 (1.88x, 2.83) | 0.0448 (1.19x, 8.57) | 0.0291 (1.83x, 10.88) | 0.0662 (0.80x, 16.38) |
| c1ccccc1O | 0.0630 (CUDA) | 0.0283 (2.22x, 0.80) | 0.0351 (1.80x, 2.93) | 0.0451 (1.40x, 8.03) | 0.0270 (2.33x, 9.69) | 0.0714 (0.88x, 16.70) |
| CC(C)CO | 0.0636 (CUDA) | 0.0261 (2.44x, 0.85) | 0.0345 (1.84x, 3.11) | 0.0421 (1.51x, 7.89) | 0.0252 (2.52x, 9.62) | 0.0643 (0.99x, 15.80) |
| **mean speedup** | | 2.26x | 1.92x | 1.36x | 2.16x | 0.91x |
