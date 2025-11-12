#!/bin/bash
# Run hybrid QM integration for validation molecules
# Usage: bash run_hybrid_qm.sh

set -e

CHECKPOINT="outputs/checkpoints_qm/best.pt"
MODEL_CONFIG="configs/model_qm.yaml"
QM_CONFIG="configs/qm.yaml"
QM_TRAJ_DIR="data/qm"
OUT_DIR="outputs/hybrid_qm"
STEPS=100
K_STEPS=4

echo "Running hybrid QM integration"
echo "Checkpoint: $CHECKPOINT"
echo "Steps: $STEPS (k=$K_STEPS)"
echo "Output: $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# Run for each molecule in QM trajectory directory
for mol_dir in "$QM_TRAJ_DIR"/*; do
    if [ -d "$mol_dir" ]; then
        mol_name=$(basename "$mol_dir")
        traj_file="$mol_dir/trajectory.npz"
        
        if [ -f "$traj_file" ]; then
            echo "Processing $mol_name..."
            
            python src/06b_hybrid_integrate_qm.py \
                --checkpoint "$CHECKPOINT" \
                --model-config "$MODEL_CONFIG" \
                --qm-config "$QM_CONFIG" \
                --qm-traj "$traj_file" \
                --out "$OUT_DIR/${mol_name}_hybrid_k${K_STEPS}.npz" \
                --frame 0 \
                --steps $STEPS \
                --k-steps $K_STEPS \
                --device cuda
        else
            echo "Skipping $mol_name (no trajectory found)"
        fi
    fi
done

echo ""
echo "Hybrid QM integration complete!"
echo "Results saved to: $OUT_DIR"

