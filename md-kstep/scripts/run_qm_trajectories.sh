#!/bin/bash
# Batch script to generate QM trajectories for all molecules
# Usage: bash run_qm_trajectories.sh

set -e

# ========================================
# CRITICAL: Set threading limits BEFORE Python starts
# xTB requires nested format "N,1" for OMP_NUM_THREADS
# ========================================
export OMP_NUM_THREADS="1,1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export OMP_STACKSIZE="4G"
export OMP_MAX_ACTIVE_LEVELS="1"
export OMP_DYNAMIC="FALSE"
export OMP_PROC_BIND="TRUE"

# Signal to Python script that env vars are already set
export XTB_WORKER_THREADS="1"

SMILES_FILE="data/qm_molecules.smi"
CONFIG="configs/qm.yaml"
OUT_DIR="data/qm"

echo "Generating QM trajectories for molecules in $SMILES_FILE"
echo "Output directory: $OUT_DIR"
echo "Config: $CONFIG"
echo "Threading: OMP_NUM_THREADS=$OMP_NUM_THREADS (xTB nested format)"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Run all molecules (or specify --molecule <name> to run just one)
# Use unbuffered output (-u) to see logs in real-time
python -u src/01b_run_qm_baselines.py \
    --smiles-file "$SMILES_FILE" \
    --config "$CONFIG" \
    --out "$OUT_DIR" \
    --num-workers 40 \

echo ""
echo "QM trajectory generation complete!"
echo "Results saved to $OUT_DIR"

