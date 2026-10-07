#!/bin/bash
# Quick test with a single small molecule

set -e

# Set threading limits
export OMP_NUM_THREADS="1,1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export OMP_STACKSIZE="4G"
export OMP_MAX_ACTIVE_LEVELS="1"
export XTB_WORKER_THREADS="1"

echo "Testing QM with threading controls:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  num-workers=3 (testing with 3 workers)"
echo ""

# Test with 3 workers on a small molecule
python -u src/01b_run_qm_baselines.py \
    --smiles-file data/qm_molecules.smi \
    --config configs/qm.yaml \
    --out data/qm_test \
    --num-workers 3 \
    --molecule glycine

echo ""
echo "Test complete! Check CPU usage was ~3 cores (not 20)"
