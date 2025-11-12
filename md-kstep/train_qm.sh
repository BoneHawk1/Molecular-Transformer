#!/bin/bash
# Train QM k-step model with transfer learning from MM model
# Usage: bash train_qm.sh

set -e

# Paths
QM_DATASET="data/qm/dataset_k4.npz"
MODEL_CONFIG="configs/model_qm.yaml"
TRAIN_CONFIG="configs/train_qm.yaml"
TRAIN_SPLIT="data/qm_splits/train.json"
VAL_SPLIT="data/qm_splits/val.json"
PRETRAINED="outputs/checkpoints_transformer_aug_wide/best.pt"  # MM pretrained model

echo "Training QM k-step model with transfer learning"
echo "Dataset: $QM_DATASET"
echo "Pretrained: $PRETRAINED"
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED" ]; then
    echo "WARNING: Pretrained checkpoint not found at $PRETRAINED"
    echo "Training from scratch instead..."
    PRETRAINED_FLAG=""
else
    echo "Using pretrained MM model: $PRETRAINED"
    PRETRAINED_FLAG="--pretrained $PRETRAINED"
fi

python src/04_train.py \
    --dataset "$QM_DATASET" \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --splits "$TRAIN_SPLIT" "$VAL_SPLIT" \
    --device cuda \
    $PRETRAINED_FLAG \
    --freeze-layers 0

echo ""
echo "QM training complete!"
echo "Checkpoints saved to: outputs/checkpoints_qm/"

