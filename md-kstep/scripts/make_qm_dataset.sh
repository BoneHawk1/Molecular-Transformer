#!/bin/bash
# Create QM k-step dataset from QM trajectories
# Usage: bash make_qm_dataset.sh

set -e

QM_ROOT="data/qm"
SPLITS_DIR="data/qm_splits"
K_STEPS="4"  # Start with k=4 for QM (1 fs macro-step)

echo "Creating QM dataset from trajectories in $QM_ROOT"
echo "k-step values: $K_STEPS"
echo "Output: $QM_ROOT/dataset_k*.npz"
echo "Splits: $SPLITS_DIR"
echo ""

mkdir -p "$SPLITS_DIR"

python src/02_make_dataset.py \
    --md-root "$QM_ROOT" \
    --out-root "$QM_ROOT" \
    --splits-dir "$SPLITS_DIR" \
    --ks $K_STEPS \
    --stride 1 \
    --max-samples-per-mol 50000 \
    --augment-rotations 3 \
    --seed 42

echo ""
echo "QM dataset creation complete!"
echo "Dataset: $QM_ROOT/dataset_k*.npz"
echo "Splits: $SPLITS_DIR/{train,val,test}.json"

