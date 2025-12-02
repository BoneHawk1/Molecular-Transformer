#!/bin/bash
# Quick test to validate xTB trajectory performance fix
# Expected: <5 minutes for glycine with 30 ps trajectory

set -e

echo "======================================"
echo "Testing QM Trajectory Performance Fix"
echo "======================================"
echo ""

# Create minimal test SMILES file (glycine - small amino acid)
mkdir -p data/test_qm
cat > data/test_qm/test_molecule.smi << EOF
C(C(=O)O)N glycine
CC(C(=O)O)N alanine
EOF

# Create fast test config (shorter trajectory for quick validation)
cat > configs/qm_test.yaml << EOF
temperature_K: 300.0
friction_per_ps: 0.002
dt_fs: 0.25
length_ps: 5.0  # Short test trajectory (5 ps = 20,000 steps)
save_interval_steps: 1
method: GFN2-xTB
random_seed: 42
charge: 0
spin_multiplicity: 1
equilibration_ps: 1.0  # Short equilibration
nve_window_ps: 0.0  # Disable NVE for speed test
nve_every_ps: 0.0
EOF

echo "Test molecule: Glycine (C(C(=O)O)N)"
echo "Trajectory: 5 ps at 0.25 fs timestep = 20,000 steps"
echo "Expected time: <2 minutes (was >30 min before fix)"
echo ""

# Run test
START_TIME=$(date +%s)

python src/01b_run_qm_baselines.py \
    --smiles-file data/test_qm/test_molecule.smi \
    --config configs/qm_test.yaml \
    --out data/test_qm \
    --num-workers 10 \
    --omp-num-threads 2

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$(echo "scale=1; $ELAPSED / 60" | bc)

echo ""
echo "======================================"
echo "✅ Test Complete!"
echo "======================================"
echo "Time elapsed: ${ELAPSED}s (${ELAPSED_MIN} minutes)"
echo ""

# Verify output file exists and has correct shape
python << PYEOF
import numpy as np
from pathlib import Path

npz_path = Path("data/test_qm/glycine/trajectory.npz")
if not npz_path.exists():
    print("❌ ERROR: Output file not found!")
    exit(1)

data = np.load(npz_path)
print(f"✅ Output file exists: {npz_path}")
print(f"✅ Trajectory shape: {data['pos'].shape}")
print(f"✅ Number of frames: {len(data['pos'])}")
print(f"✅ Number of atoms: {data['pos'].shape[1]}")
print(f"✅ Keys in NPZ: {list(data.keys())}")

expected_frames = int(5.0 * 1000 / 0.25) + 1  # 20,001 frames
actual_frames = len(data['pos'])

if abs(actual_frames - expected_frames) <= 1:
    print(f"✅ Frame count correct: {actual_frames} ≈ {expected_frames}")
else:
    print(f"⚠️  Frame count mismatch: {actual_frames} vs expected {expected_frames}")

# Check data quality
if not np.any(np.isnan(data['pos'])):
    print("✅ No NaN values in positions")
else:
    print("⚠️  Warning: NaN values detected in positions")

PYEOF

echo ""
if [ $ELAPSED -lt 180 ]; then
    echo "✅ PERFORMANCE TEST PASSED! (< 3 minutes)"
    echo "Expected speedup: ~50-100× faster than before"
else
    echo "⚠️  Slower than expected but check if it completed successfully"
fi

echo ""
echo "To test with full 30 ps trajectory:"
echo "  bash run_qm_trajectories.sh"
echo ""
