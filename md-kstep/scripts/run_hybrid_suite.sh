#!/usr/bin/env bash
# Run the hybrid integrator for every molecule in data/raw under one condition.
#
#   scripts/run_hybrid_suite.sh LABEL [args for src/06_hybrid_integrate.py ...]
#
# e.g.  scripts/run_hybrid_suite.sh flow_k4 --checkpoint outputs/v2/flow_k4/best.pt
#       scripts/run_hybrid_suite.sh zero_k4 --predictor zero --k-steps 4
#
# Env: OUT_ROOT (default outputs/v2/hybrid), STEPS (2500 macro-steps), MD_CONFIG, FRAME,
#      SPLITS (space-separated split JSONs, e.g. "data/splits_v2/val.json data/splits_v2/test.json";
#      default: every molecule in data/raw).
# Existing outputs are skipped, so the script can be re-run after an interruption; a failing
# molecule is reported and the loop continues.
set -uo pipefail
LABEL=$1; shift
OUT="${OUT_ROOT:-outputs/v2/hybrid}/$LABEL"
mkdir -p "$OUT"
if [ -n "${SPLITS:-}" ]; then
  mapfile -t MOLS < <(python -c "import json,sys; [print(m) for f in sys.argv[1:] for m in json.load(open(f))['molecules']]" $SPLITS)
else
  MOLS=(); for d in data/raw/*/; do MOLS+=("$(basename "$d")"); done
fi
for mol in "${MOLS[@]}"; do
  mol_dir="data/raw/$mol"
  [ -f "$OUT/$mol.npz" ] && continue
  python src/06_hybrid_integrate.py --md-config "${MD_CONFIG:-configs/md.yaml}" --molecule "$mol_dir" \
    --initial-md "data/md/$mol/trajectory.npz" --frame "${FRAME:-0}" --steps "${STEPS:-2500}" \
    --out "$OUT/$mol.npz" "$@" 2>&1 | grep -v "^\[INFO\] macro-step" || true
  [ -f "$OUT/$mol.npz" ] || echo "FAILED: $LABEL $mol"
done
