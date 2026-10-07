# QM K-Step Integrator: Transfer Learning from MM to QM

This extension of the md-kstep project demonstrates transfer learning from molecular mechanics (MM) to semi-empirical quantum mechanics (QM), enabling fast QM molecular dynamics with learned k-step integrators.

## Overview

The QM k-step integrator applies the same principle as the MM version:
1. **ML k-step prediction**: Jump forward k timesteps using a learned model
2. **One-step corrector**: Apply a single xTB force evaluation to project onto the physical manifold

**Key differences from MM:**
- **Smaller timesteps**: 0.25 fs (vs 2 fs for MM)
- **Tighter bounds**: QM geometries are more rigid
- **Force matching**: QM forces are well-defined and important
- **Transfer learning**: Initialize from pretrained MM weights

## Quick Start

```bash
# 1. Install QM dependencies
conda install -c conda-forge xtb ase

# 2. Generate QM training data
bash run_qm_trajectories.sh  # ~1-10 hours

# 3. Create k-step dataset
bash make_qm_dataset.sh

# 4. Train with transfer learning
bash train_qm.sh  # ~1-3 hours

# 5. Run hybrid integration
bash run_hybrid_qm.sh

# 6. Evaluate results
bash eval_qm.sh
```

See `QM_WORKFLOW.md` for detailed instructions.

## File Structure

```
md-kstep/
├── configs/
│   ├── qm.yaml              # QM trajectory config
│   ├── model_qm.yaml        # QM model architecture
│   └── train_qm.yaml        # QM training config
├── src/
│   ├── 01b_run_qm_baselines.py     # Generate xTB trajectories
│   ├── 06b_hybrid_integrate_qm.py  # QM hybrid integrator
│   ├── 05b_eval_qm.py              # QM evaluation metrics
│   └── qmmm_example.py             # QM/MM proof-of-concept
├── data/
│   ├── qm_molecules.smi     # Small molecules for QM (22 molecules)
│   ├── qm/                  # QM trajectories
│   └── qm_splits/           # Train/val/test splits
├── outputs/
│   ├── checkpoints_qm/      # Trained QM models
│   ├── hybrid_qm/           # Hybrid QM trajectories
│   └── eval_qm/             # Evaluation metrics
├── QM_WORKFLOW.md           # Step-by-step guide
├── QMMM_INTEGRATION.md      # QM/MM integration strategy
└── INSTALL_QM.md            # Installation instructions
```

## Key Features

### 1. Transfer Learning
- Load pretrained MM weights (`--pretrained`)
- Optional layer freezing (`--freeze-layers`)
- Lower learning rate for fine-tuning

### 2. QM-Specific Optimizations
- Tighter displacement bounds (0.01 nm vs 0.25 nm)
- Stronger structural regularization
- Force head for auxiliary supervision
- Per-graph data augmentation

### 3. xTB Integration
- ASE-based calculator interface
- GFN2-xTB method (accurate semi-empirical)
- Energy conservation analysis
- Fallback to GFN1-xTB or GFN-FF if needed

### 4. Validation Tools
- Energy drift comparison
- Structural RMSD tracking
- Bond/angle preservation
- Stability analysis

## Performance Expectations

### MM Model (Reference)
- Bond RMSE: 0.039 Å
- Angle RMSE: 4.23°
- Dihedral RMSE: 8.21°
- Force call reduction: 4x (k=4)

### Target QM Model
- Bond RMSE: < 0.02 Å (2x improvement)
- Angle RMSE: < 2° (2x improvement)
- Energy drift: < 0.05 kJ/mol/ps
- Force call reduction: 4-8x (k=4-8)

**Why QM should be more accurate:**
- Stronger electronic forces → better constraints
- Smaller molecules → easier to learn
- More rigid geometries → tighter error bounds
- Transfer learning → head start from MM

## Training Data Requirements

**Minimum (proof-of-concept):**
- 10 molecules × 20 ps = 200 ps total
- ~80,000 frames (at 0.25 fs)
- ~1-5 GPU-hours to generate

**Recommended (production):**
- 20-30 molecules × 50 ps = 1000-1500 ps total
- ~400,000 frames
- ~10-50 GPU-hours to generate

**Molecule selection criteria:**
- 10-30 atoms (small, rigid)
- Diverse chemistry (C, N, O, S, aromatics, etc.)
- Biologically relevant (amino acids, nucleobases, cofactors)

## QM/MM Integration (Future Work)

Once QM-only validation succeeds, integrate into QM/MM pipeline:

1. **ASE-based approach** (simpler):
   - Use `SimpleQMMM` calculator
   - Apply k-step model to QM region only
   - Standard MM for MM region

2. **OpenMM-Torch approach** (more control):
   - Custom OpenMM force for QM region
   - Full integration with OpenMM ecosystem
   - Better performance for large systems

See `QMMM_INTEGRATION.md` for detailed strategy.

## Comparison: MM vs QM Integrator

| Feature | MM | QM |
|---------|----|----|
| Timestep | 2 fs | 0.25 fs |
| Force evaluations | OpenMM | xTB |
| k-step target | 4-12 | 4-8 |
| Speedup | 4-12x | 4-8x |
| Training data | 10-20 ns | 200-1500 ps |
| Data generation | Fast (~hours) | Slow (~days) |
| Accuracy | Bond RMSE ~0.04 Å | Bond RMSE < 0.02 Å |
| Use case | Conformational sampling | Active site dynamics |

## Troubleshooting

### Common Issues

1. **xTB installation fails**
   ```bash
   # Try manual installation
   wget https://github.com/grimme-lab/xtb/releases/latest
   # Or use conda-forge
   conda install -c conda-forge xtb
   ```

2. **Transfer learning not helping**
   - Train from scratch (remove `--pretrained`)
   - Check architecture compatibility
   - Verify MM checkpoint quality

3. **Hybrid integration unstable**
   - Reduce `--delta-scale` to 0.5
   - Increase `--max-attempts` to 10
   - Tighten `max_disp_nm` in config

4. **Low accuracy despite training**
   - Increase training data (add molecules)
   - Strengthen regularization
   - Check for data quality issues

See `QM_WORKFLOW.md` FAQ for more troubleshooting tips.

## Citation

If you use this work, please cite:

```bibtex
@software{md-kstep-qm,
  title = {QM K-Step Integrator: Transfer Learning for Fast Quantum Molecular Dynamics},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-repo]/md-kstep}
}
```

And the key dependencies:
- xTB: Grimme et al., J. Chem. Theory Comput. 2017, 13, 1989
- ASE: Larsen et al., J. Phys.: Condens. Matter 2017, 29, 273002

## Next Steps

1. **Generate QM training data** - Start with 10 small molecules
2. **Transfer learning** - Fine-tune MM model on QM data
3. **Validate hybrid integrator** - Check energy conservation
4. **Increase k-step** - Try k=8 for 8x speedup
5. **QM/MM integration** - Apply to real biomolecular systems

## Support

- **General issues**: See main `README.md`
- **QM-specific**: Check `QM_WORKFLOW.md` and `QMMM_INTEGRATION.md`
- **Installation**: See `INSTALL_QM.md`

## Acknowledgments

This QM extension builds on the MM k-step integrator framework. Thanks to:
- Grimme group for xTB
- ASE developers for calculator interface
- PyTorch and CUDA teams for GPU acceleration

