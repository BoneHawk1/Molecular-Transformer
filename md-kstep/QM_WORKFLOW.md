# QM K-Step Integrator: Complete Workflow

This document provides a step-by-step guide to training and using the QM k-step integrator.

## Prerequisites

```bash
# Activate environment
conda activate kstep

# Install QM dependencies
conda install -c conda-forge xtb ase

# Verify installation
python -c "from ase.calculators.xtb import XTB; print('xTB OK')"
```

## Step 1: Generate QM Training Trajectories

```bash
# Run QM trajectories for training molecules
bash run_qm_trajectories.sh

# This will generate trajectories in data/qm/<molecule>/trajectory.npz
# Expected time: ~1-10 hours depending on molecule size and GPU
```

**Output:** `data/qm/*/trajectory.npz` (20-50 ps per molecule)

## Step 2: Create QM k-Step Dataset

```bash
# Convert trajectories to supervised k-step dataset
bash make_qm_dataset.sh

# Creates train/val/test splits and k-step windows
```

**Output:** 
- `data/qm/dataset_k4.npz` (supervised training data)
- `data/qm_splits/{train,val,test}.json` (molecule splits)

## Step 3: Train QM Model with Transfer Learning

```bash
# Fine-tune MM model on QM data
bash train_qm.sh

# Training will use pretrained MM weights from:
# outputs/checkpoints_transformer/best.pt

# Monitor training:
# - Logs: outputs/logs_qm/
# - Checkpoints: outputs/checkpoints_qm/
# - W&B: https://wandb.ai (if enabled)
```

**Expected training time:** 1-3 hours on single GPU

**Key metrics to monitor:**
- Validation loss should be < 2x training loss
- Position loss should decrease steadily
- Structural losses (bonds/angles) should be low

## Step 4: Run Hybrid QM Integration

```bash
# Run hybrid integrator on validation molecules
bash run_hybrid_qm.sh

# This applies the trained model to predict k-step jumps
# followed by xTB corrector steps
```

**Output:** `outputs/hybrid_qm/*_hybrid_k4.npz`

## Step 5: Evaluate Performance

```bash
# Compare hybrid vs baseline QM trajectories
bash eval_qm.sh

# Generates metrics and plots
```

**Output:** `outputs/eval_qm/qm_metrics.json`

**Success criteria:**
- Bond RMSE < 0.02 Å
- Energy drift similar to baseline
- RMSD < 0.5 Å over 100 steps
- Structural coherence maintained

## Expected Results

### MM Model Performance (Reference)
- Bond RMSE: 0.039 Å
- Angle RMSE: 4.23°
- RDF L1: 0.034

### Target QM Model Performance
- Bond RMSE: < 0.02 Å (2x better than MM)
- Angle RMSE: < 2° (2x better than MM)
- Energy drift: < 0.05 kJ/mol/ps
- Force call reduction: 4x (k=4)

## Troubleshooting

### Issue: Training loss not decreasing
**Solution:** 
- Check data quality: Are QM trajectories stable?
- Reduce learning rate: Try `lr: 1.0e-5`
- Increase regularization: `lambda_struct_bond: 0.2`

### Issue: Hybrid integration unstable
**Solution:**
- Reduce delta scale: `--delta-scale 0.5`
- Increase max attempts: `--max-attempts 10`
- Check model displacement bounds: `max_disp_nm: 0.005`

### Issue: Transfer learning not working
**Solution:**
- Verify MM checkpoint exists and loads
- Try training from scratch: Remove `--pretrained` flag
- Check architecture compatibility

### Issue: xTB calculator fails
**Solution:**
- Check xTB installation: `which xtb`
- Verify charge/multiplicity settings
- Try different xTB method: `GFN1-xTB` or `GFN-FF`

## Configuration Tuning

### For more rigid molecules
```yaml
# configs/model_qm.yaml
max_disp_nm: 0.005  # Tighter bound
lambda_struct_bond: 0.2  # Stronger regularization
```

### For larger k-steps (k=8)
```yaml
# configs/train_qm.yaml
k_steps: 8
max_disp_nm: 0.02  # Scale with k
```

### For faster training (development)
```yaml
# configs/train_qm.yaml
batch_size: 64
max_epochs: 20
val_every_steps: 50
```

## Next Steps: QM/MM Integration

Once QM-only validation succeeds:

1. Review `QMMM_INTEGRATION.md` for integration strategies
2. Start with simple QM/MM test system (e.g., solvated small molecule)
3. Use `src/qmmm_example.py` as starting point
4. Validate energy conservation and boundary stability

## File Reference

- **Scripts:** `run_*.sh` - Automated workflows
- **Configs:** `configs/*_qm.yaml` - QM-specific settings
- **Source:** `src/*b_*.py` - QM-specific implementations
- **Docs:** `*.md` - Documentation and guides

## Performance Monitoring

### W&B Integration
```yaml
# configs/train_qm.yaml
wandb:
  enabled: true
  project: md-kstep-qm
```

### Manual Monitoring
```bash
# Watch training logs
tail -f outputs/logs_qm/train_metrics.jsonl

# Plot losses
python -c "
import json
import matplotlib.pyplot as plt
with open('outputs/logs_qm/train_metrics.jsonl') as f:
    metrics = [json.loads(line) for line in f]
steps = [m['step'] for m in metrics]
losses = [m['loss'] for m in metrics]
plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('training_curve.png')
"
```

## FAQ

**Q: How much QM data do I need?**
A: Minimum 10 molecules × 20 ps each. More data = better transfer learning.

**Q: Can I use DFT instead of xTB?**
A: Yes, but it's much slower. Modify calculator in `01b_run_qm_baselines.py`:
```python
from ase.calculators.psi4 import Psi4
calc = Psi4(method='B3LYP', basis='6-31G*')
```

**Q: What if I don't have MM pretrained weights?**
A: Remove `--pretrained` flag and train from scratch. You'll need more QM data (~20-30 molecules).

**Q: How do I visualize trajectories?**
A: Use VMD, PyMOL, or ASE's GUI:
```python
from ase.io import read
from ase.visualize import view
atoms = read('outputs/hybrid_qm/molecule_hybrid_k4.npz')
view(atoms)
```

## Citation

If you use this QM k-step integrator in your research, please cite:
- This repository: [Add citation when published]
- xTB: Grimme et al., J. Chem. Theory Comput. 2017, 13, 1989
- ASE: Larsen et al., J. Phys.: Condens. Matter 2017, 29, 273002

