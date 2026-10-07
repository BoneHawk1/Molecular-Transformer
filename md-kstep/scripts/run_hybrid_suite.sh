#!/usr/bin/env bash
# Run the hybrid integrator for every molecule in data/raw under one condition.
#
#   scripts/run_hybrid_suite.sh LABEL [args for src/06_hybrid_integrate.py ...]
#
# e.g.  scripts/run_hybrid_suite.sh flow_k4 --checkpoint outputs/v2/flow_k4/best.pt
#       scripts/run_hybrid_suite.sh zero_k4 --predictor zero --k-steps 4
#
# Env: OUT_ROOT (default outputs/v2/hybrid), STEPS (2500 macro-steps), MD_CONFIG, FRAME.
# Existing outputs are skipped, so the script can be re-run after an interruption.
set -euo pipefail
LABEL=$1; shift
OUT="${OUT_ROOT:-outputs/v2/hybrid}/$LABEL"
mkdir -p "$OUT"
for mol_dir in data/raw/*/; do
  mol=$(basename "$mol_dir")
  [ -f "$OUT/$mol.npz" ] && continue
  python src/06_hybrid_integrate.py --md-config "${MD_CONFIG:-configs/md.yaml}" --molecule "$mol_dir" \
    --initial-md "data/md/$mol/trajectory.npz" --frame "${FRAME:-0}" --steps "${STEPS:-2500}" \
    --out "$OUT/$mol.npz" "$@" 2>&1 | grep -v "^\[INFO\] macro-step"
done
