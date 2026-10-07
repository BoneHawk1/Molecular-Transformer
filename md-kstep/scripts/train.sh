#!/usr/bin/env bash
# Train the classical-MD k-step models (run from md-kstep/).
#   KIND=flow|mean  K=4|8|12  bash scripts/train.sh
set -euo pipefail
KIND=${KIND:-flow}
K=${K:-4}
python src/04_train.py --dataset "data/md/dataset_k${K}.npz" \
    --model-config "configs/model_${KIND}.yaml" --train-config configs/train.yaml \
    --splits data/splits/train.json data/splits/val.json \
    --out "outputs/v2/${KIND}_k${K}" "$@"
