#!/usr/bin/env bash
# Train a k-step model on the xTB trajectories (run from md-kstep/).
#   KIND=flow K=40 bash scripts/train_qm.sh
# The QM data are stored every 0.25 fs step, so K=40 is a 10 fs jump.
set -euo pipefail
KIND=${KIND:-flow}
K=${K:-40}
python src/04_train.py --dataset "data/qm/dataset_k${K}.npz" \
    --model-config "configs/model_${KIND}.yaml" --train-config configs/train_qm.yaml \
    --splits data/qm_splits/train.json data/qm_splits/val.json \
    --out "outputs/v2/qm_${KIND}_k${K}" "$@"
