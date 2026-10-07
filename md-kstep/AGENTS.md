# Repository Guidelines

See `CLAUDE.md` for architecture, invariants and cluster notes; `README.md` for the workflow.

## Structure
- Library code: `src/kstep/`. Pipeline CLIs: `src/00_prep_mols.py` … `src/08_time_study.py`.
- Configs: `configs/` (`model_flow.yaml`, `model_mean.yaml`, `train.yaml`, `md.yaml`, `qm*.yaml`);
  2025 configs in `configs/legacy/`.
- Data (`data/`) and outputs (`outputs/`) are git-ignored; new results go under `outputs/v2/`.

## Commands
- Tests: `python -m pytest` (CPU, a few seconds).
- Smoke-train: `python src/04_train.py --train-config configs/train_debug.yaml --model-config configs/model_flow.yaml --dataset data/md/dataset_k4.npz --splits data/splits/train.json data/splits/val.json --out /tmp/dbg --device cpu`.
- Check a model config: `python src/03_model.py --model-config configs/model_flow.yaml`.

## Style
- PEP 8, 4-space indents, type hints, `LOGGER` from `kstep.common`.
- Keep scripts thin; put reusable logic in `src/kstep/` with a test in `tests/`.
- Commit messages: imperative and concise (`Add k=8 dataset`, `Fix OpenMM seed handling`).
