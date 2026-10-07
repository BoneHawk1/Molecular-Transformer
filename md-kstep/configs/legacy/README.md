# Legacy configs (Dec 2025)

Configs for the original EGNN / Transformer-EGNN models and their trainer. They are
kept so the old checkpoints can be loaded for comparison, e.g.

```bash
python src/06_hybrid_integrate.py --checkpoint outputs/checkpoints_transformer_aug_wide/best.pt \
    --model-config configs/legacy/model_transformer_aug_wide.yaml ...
```

`model_egnn.yaml` matches `outputs/checkpoints_aug_wide/best.pt`;
`model_transformer_aug_wide.yaml` (formerly `configs/model.yaml`) matches
`outputs/checkpoints_transformer_aug_wide/best.pt`. The train configs target the
old trainer, which was replaced by `src/04_train.py` in Oct 2026; several keys
(`attention_heads`, `use_cross_attention`, `attention_type`) never had any effect.
