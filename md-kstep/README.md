# md-kstep

Small equivariant k-step state-jump model for molecular dynamics with a one-step physics corrector.

## Project structure

```
md-kstep/
├─ env/
├─ data/
│  ├─ molecules.smi
│  ├─ raw/
│  ├─ md/
│  └─ splits/
├─ src/
│  ├─ 00_prep_mols.py
│  ├─ 01_run_md_baselines.py
│  ├─ 02_make_dataset.py
│  ├─ 03_model.py
│  ├─ 04_train.py
│  ├─ 05_eval_drift_rdfs.py
│  ├─ 06_hybrid_integrate.py
│  └─ utils.py
├─ configs/
│  ├─ md.yaml
│  ├─ model.yaml
│  └─ train.yaml
└─ README.md
```

## Environment (Conda)

```bash
conda create -n kstep python=3.11 -y
conda activate kstep
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas pyyaml tqdm hydra-core mdtraj MDAnalysis matplotlib seaborn wandb
pip install rdkit-pypi openff-toolkit openmm==8.* openmmforcefields
pip install torchmd-net e3nn torch_geometric
pip install line_profiler rich
```

## Quickstart

1) Prepare molecules
```bash
python src/00_prep_mols.py --smiles data/molecules.smi --out data/raw
```

2) Generate MD baselines (implicit solvent)
```bash
python src/01_run_md_baselines.py --conf configs/md.yaml --raw data/raw --out data/md
```

3) Build dataset windows
```bash
python src/02_make_dataset.py --md data/md --splits data/splits --k 8 --stride 10 --out data/md/dataset_k8.npz
```

4) Train model
```bash
python src/04_train.py --model configs/model.yaml --train configs/train.yaml --dataset data/md/dataset_k8.npz
```

5) Evaluate
```bash
python src/05_eval_drift_rdfs.py --dataset data/md/dataset_k8.npz --md data/md --out outputs/eval_k8
```

6) Hybrid integrator
```bash
python src/06_hybrid_integrate.py --checkpoint outputs/ckpts/best.pt --k 8 --md-conf configs/md.yaml --out outputs/hybrid
```

## Notes
- Positions in nm, velocities in nm/ps.
- Remove COM translation and velocity during dataset creation.
- The hybrid integrator performs one OpenMM force step per macro-step.
