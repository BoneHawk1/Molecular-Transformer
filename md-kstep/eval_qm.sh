#!/bin/bash
# Evaluate QM hybrid integrator against baseline
# Usage: bash eval_qm.sh

set -e

BASELINE="data/qm"
HYBRID="outputs/hybrid_qm_scratch"
OUT_DIR="outputs/eval_qm_scratch"

echo "Evaluating QM hybrid integrator"
echo "Baseline: $BASELINE"
echo "Hybrid: $HYBRID"
echo "Output: $OUT_DIR"
echo ""

python src/05b_eval_qm.py \
    --baseline "$BASELINE" \
    --hybrid "$HYBRID" \
    --out-dir "$OUT_DIR"

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUT_DIR"

