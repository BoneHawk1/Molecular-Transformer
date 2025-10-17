# Repository Guidelines

## Project Structure & Module Organization
- Core scripts live in `src/00_prep_mols.py` through `src/06_hybrid_integrate.py`, mirroring the pipeline from molecule prep to hybrid rollout.
- Training data, splits, and baseline trajectories sit under `data/`; `data/md/dataset_k4.npz` and friends hold supervised windows, while `data/splits/*.json` define molecule-disjoint partitions.
- Configuration defaults are in `configs/` (`md.yaml`, `model.yaml`, `train.yaml`, etc.); use overrides rather than editing checkpoints directly.
- Generated checkpoints and diagnostics belong in `outputs/`; avoid checking large artifacts into git.

## Build, Test, and Development Commands
- Activate the curated environment with `conda activate kstep`; it already bundles OpenMM, RDKit, torch, and helper libraries.
- Prepare molecules from the curated list: `python src/00_prep_mols.py --smiles data/molecules.smi --out-dir data/raw`.
- Produce baseline trajectories: `python src/01_run_md_baselines.py --molecule data/raw/aspirin --config configs/md.yaml --out data/md`.
- Assemble supervised datasets: `python src/02_make_dataset.py --md-root data/md --out-root data/md --splits-dir data/splits --ks 4 8 12`.
- Launch training: `python src/04_train.py --dataset data/md/dataset_k4.npz --model-config configs/model.yaml --train-config configs/train.yaml --splits data/splits/train.json data/splits/val.json`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case for functions, and UpperCamelCase for classes; keep module-level constants uppercase.
- Scripts already use sequential numeric prefixes (e.g., `00_`, `01_`); continue that convention for new stages.
- Annotate functions with type hints when practical and keep logging via the shared `LOGGER` in `utils.py`.
- Use UTF-8 for text files and avoid non-ASCII unless data requires it.

## Testing Guidelines
- Fast sanity checks leverage the lightweight configuration: `python src/04_train.py --train-config configs/train_debug.yaml ...`.
- Validate dataset integrity by reusing `src/02_make_dataset.py` with `--dry-run` style additions if you extend it; ensure new metadata lives in `data/md`.
- After training, run `python src/05_eval_drift_rdfs.py --baseline data/md --hybrid-runs outputs/hybrid --out-dir outputs/eval` to confirm drift metrics remain stable.

## Commit & Pull Request Guidelines
- Write imperative, concise commit messages (`Add k=8 dataset`, `Fix OpenMM seed handling`), following the style in `git log`.
- Every PR should summarize pipeline changes, list affected stages (`prep`, `md`, `train`), and call out new configs or data dependencies.
- Link relevant issues or experiment notes, attach before/after plots when drift or accuracy changes, and mention required re-generation steps for reviewers.
