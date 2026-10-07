# QM Hybrid Timing Notes (session summary)

## Environment
- Repo: `md-kstep`
- Model: `outputs/checkpoints_transformer_qm_scratch/best.pt`
- Configs: `configs/model_qm.yaml`, `configs/qm_test.yaml` (xTB), `configs/qm_pyscf_short.yaml` (PySCF short)
- Hardware: CUDA visible (`NVIDIA GeForce RTX 5070 Ti`), but GPU4PySCF not usable (`cublasLtGetEnvironmentMode` missing) so PySCF ran on CPU.
- Threads (where noted): `OMP/MKL/OPENBLAS/NUMEXPR=15`, `OMP_STACKSIZE=4G`

## xTB vs Hybrid (ethanol)
- Baseline xTB (`01b_run_qm_baselines.py`, 5 ps, dt=0.25 fs, k=1):
  - `outputs/qm_timing_xtb/ethanol/trajectory.npz`
  - Time: **27.94s** (1 thread)
  - Force calls: ~24k prod + 4k equil (per-step xTB)
- Hybrid v1 (k=4, 5k steps, `--max-attempts 5`, CPU):
  - Time: **126.10s**, force calls **24,660**, failures **4,915**.
- Hybrid v2 (k=4, `--max-attempts 3`, CPU):
  - Time: **112.91s**, force calls **14,830**, failures **4,915**.
- Hybrid v3 (k=4, `--max-attempts 3`, GPU model):
  - Time: **129.80s**, force calls **14,830**, failures **4,915** (GPU slower since xTB dominates).
- Hybrid v4 (k=4, `--max-attempts 1`, 15 threads, GPU model):
  - `outputs/qm_timing_hybrid/ethanol_hybrid_k4_gpu_attempt1_threads15.npz`
  - Time: **70.32s** (log 64.8s), force calls **5,000**, failures **4,915**.
- Takeaway: Hybrid reduces xTB calls (5k vs 24k+), but xTB corrector cost dominates; even with fewer retries and more threads hybrid stayed >2× slower than xTB on ethanol.

## xTB vs Hybrid (ethylamine, no failures)
- Baseline xTB (5 ps, 15 threads):
  - `outputs/qm_timing_xtb/ethylamine/trajectory.npz`
  - Time: **24.52s**.
- Hybrid (k=4, 5k steps, `--max-attempts 1`, 15 threads, GPU model):
  - `outputs/qm_timing_hybrid/ethylamine_hybrid_k4_gpu_attempt1_threads15.npz`
  - Time: **64.17s**, force calls **5,000**, failures **0**.
- Takeaway: Stable molecule still shows hybrid slower (~2.6×) because each xTB call is expensive; model compute is negligible.

## PySCF backend experiment (CPU fallback due to GPU4PySCF failure)
- Config: `configs/qm_pyscf_short.yaml` (0.2 ps, dt=0.5 fs, save_every=1, equil=0.1 ps, `backend: pyscf`, `use_gpu: true` but GPU4PySCF missing symbol).
- Baseline PySCF (ethylamine):
  - `outputs/qm_pyscf_timing/ethylamine/trajectory.npz`
  - Time: **142.11s** (CPU SCF), ~400 prod steps + 200 equil.
- Hybrid PySCF corrector (k=4, 100 steps, `--max-attempts 1`, 15 threads, GPU model):
  - `outputs/qm_timing_hybrid_pyscf/ethylamine_hybrid_k4_pyscf.npz`
  - Time: **51.25s**, force calls **100**, failures **0**; backend recorded as `pyscf`.
- Takeaway: Even on CPU PySCF, hybrid was **~2.8× faster** than the PySCF baseline by cutting SCF calls (100 vs ~600).

## Notes
- The hybrid script now accepts `--qm-backend {xtb,pyscf}`; metadata includes `qm_backend`, `force_calls`, and `failed_steps`.
- GPU4PySCF is not currently usable; `gpu4pyscf` import fails with missing `cublasLtGetEnvironmentMode` in `libcublas.so.12`.
- Further speedups would require reducing per-call QM cost (fix GPU4PySCF, tune threading) or reducing retries/step size (`--delta-scale`) while managing stability.***
