#!/bin/bash
# Run hybrid QM integration for validation molecules
# Usage: bash run_hybrid_qm.sh

set -euo pipefail

CHECKPOINT=${CHECKPOINT:-"outputs/checkpoints_transformer_qm_scratch/best.pt"}
MODEL_CONFIG=${MODEL_CONFIG:-"configs/model_qm.yaml"}
QM_CONFIG=${QM_CONFIG:-"configs/qm.yaml"}
QM_TRAJ_DIR=${QM_TRAJ_DIR:-"data/qm"}
OUT_DIR=${OUT_DIR:-"outputs/hybrid_qm_scratch"}
STEPS=${STEPS:-100}
K_STEPS=${K_STEPS:-4}
DEVICE=${DEVICE:-"cuda"}

MAX_JOBS=${MAX_JOBS:-2}
THREADS_PER_JOB=${THREADS_PER_JOB:-1}
DRY_RUN=${DRY_RUN:-0}

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer (got '$MAX_JOBS')" >&2
    exit 1
fi

if ! [[ "$THREADS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS_PER_JOB must be a positive integer (got '$THREADS_PER_JOB')" >&2
    exit 1
fi

if ! [[ "$DRY_RUN" =~ ^[01]$ ]]; then
    echo "DRY_RUN must be 0 or 1 (got '$DRY_RUN')" >&2
    exit 1
fi

for required_file in "$MODEL_CONFIG" "$QM_CONFIG"; do
    if [ ! -f "$required_file" ]; then
        echo "Required config not found: $required_file" >&2
        exit 1
    fi
done

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Available checkpoints (best.pt) under outputs/:" >&2
    find outputs -maxdepth 3 -type f -name 'best.pt' -print 2>/dev/null || true
    exit 1
fi

if [ ! -d "$QM_TRAJ_DIR" ]; then
    echo "QM trajectory directory not found: $QM_TRAJ_DIR" >&2
    exit 1
fi

echo "Running hybrid QM integration"
echo "Checkpoint: $CHECKPOINT"
echo "Steps: $STEPS (k=$K_STEPS)"
echo "Output: $OUT_DIR"
echo "Max parallel jobs: $MAX_JOBS"
echo "Device: $DEVICE"
echo "Threads per job (OMP/MKL/BLAS): $THREADS_PER_JOB"
if (( DRY_RUN == 1 )); then
    echo "Dry run enabled: commands will be printed but not executed"
fi
echo ""

mkdir -p "$OUT_DIR"

shopt -s nullglob
mol_dirs=("$QM_TRAJ_DIR"/*)
shopt -u nullglob

if ((${#mol_dirs[@]} == 0)); then
    echo "No molecule directories found in $QM_TRAJ_DIR"
    exit 1
fi

runnable=()
for mol_dir in "${mol_dirs[@]}"; do
    if [ -d "$mol_dir" ]; then
        mol_name=$(basename "$mol_dir")
        traj_file="$mol_dir/trajectory.npz"
        if [ -f "$traj_file" ]; then
            runnable+=("$mol_dir")
        else
            echo "Skipping $mol_name (no trajectory found)"
        fi
    fi
done

if ((${#runnable[@]} == 0)); then
    echo "No trajectories to process."
    exit 0
fi

pids=()
names=()

cleanup() {
    if ((${#pids[@]})); then
        echo "Stopping running jobs..."
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup SIGINT SIGTERM

wait_for_slot() {
    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
}

start_job() {
    local mol_dir="$1"
    local mol_name
    mol_name=$(basename "$mol_dir")
    local traj_file="$mol_dir/trajectory.npz"
    local out_file="$OUT_DIR/${mol_name}_hybrid_k${K_STEPS}.npz"
    local log_file="$OUT_DIR/${mol_name}.log"
    local omp_stack="${OMP_STACKSIZE:-4G}"

    local -a cmd=(
        python src/06b_hybrid_integrate_qm.py
        --checkpoint "$CHECKPOINT"
        --model-config "$MODEL_CONFIG"
        --qm-config "$QM_CONFIG"
        --qm-traj "$traj_file"
        --out "$out_file"
        --frame 0
        --steps "$STEPS"
        --k-steps "$K_STEPS"
        --device "$DEVICE"
        --max-delta-pos 0.02
        --max-delta-vel 2.0
        --delta-scale 0.5
        --pos-threshold 1.0
        --vel-threshold 50.0
	--energy-rescale        
    )

    echo "Launching $mol_name -> $out_file (log: $log_file)"

    if (( DRY_RUN == 1 )); then
        echo "DRY RUN: OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB OPENBLAS_NUM_THREADS=$THREADS_PER_JOB NUMEXPR_NUM_THREADS=$THREADS_PER_JOB OMP_STACKSIZE=$omp_stack ${cmd[*]}"
        return
    fi

    OMP_NUM_THREADS="$THREADS_PER_JOB" \
    MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    OMP_STACKSIZE="$omp_stack" \
    "${cmd[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
    names+=("$mol_name")
}

for mol_dir in "${runnable[@]}"; do
    wait_for_slot
    start_job "$mol_dir"
done

if (( DRY_RUN == 1 )); then
    echo ""
    echo "Dry run complete. No jobs were executed."
    exit 0
fi

failures=()
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        failures+=("${names[$idx]}")
        echo "Job for ${names[$idx]} failed (see $OUT_DIR/${names[$idx]}.log)"
    fi
done

trap - SIGINT SIGTERM

echo ""

if ((${#failures[@]})); then
    echo "Hybrid QM integration finished with failures: ${failures[*]}"
    exit 1
fi

echo "Hybrid QM integration complete!"
echo "Results saved to: $OUT_DIR"
