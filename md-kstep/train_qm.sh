#!/bin/bash
# Train QM k-step model from scratch
# Usage: bash train_qm.sh
# For a live terminal UI of training/validation loss, run:
#   python scripts/qm_train_tui.py
# in a separate terminal while this script is running.

set -e

# Paths
QM_DATASET="data/qm/dataset_k4.npz"
MODEL_CONFIG="configs/model_qm.yaml"
TRAIN_CONFIG="configs/train_qm.yaml"
TRAIN_SPLIT="data/qm_splits/train.json"
VAL_SPLIT="data/qm_splits/val.json"

echo "Training QM k-step model from scratch"
echo "Dataset: $QM_DATASET"
echo "Config: $TRAIN_CONFIG"
echo ""

python src/04_train.py \
    --dataset "$QM_DATASET" \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --splits "$TRAIN_SPLIT" "$VAL_SPLIT" \
    --device cuda

echo ""
echo "QM training complete!"
echo "Checkpoints saved to: outputs/checkpoints_qm_scratch/"
