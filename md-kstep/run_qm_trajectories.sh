#!/bin/bash
# Batch script to generate QM trajectories for all molecules
# Usage: bash run_qm_trajectories.sh

set -e

SMILES_FILE="data/qm_molecules.smi"
CONFIG="configs/qm.yaml"
OUT_DIR="data/qm"

echo "Generating QM trajectories for molecules in $SMILES_FILE"
echo "Output directory: $OUT_DIR"
echo "Config: $CONFIG"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Run all molecules (or specify --molecule <name> to run just one)
python src/01b_run_qm_baselines.py \
    --smiles-file "$SMILES_FILE" \
    --config "$CONFIG" \
    --out "$OUT_DIR" \
    --num-workers 20 \

echo ""
echo "QM trajectory generation complete!"
echo "Results saved to $OUT_DIR"

