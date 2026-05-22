# Evaluation Protocol

Use a paired evaluation setup when comparing baseline and augmented AES models.

## Recommended Flow

1. Generate embeddings for the baseline split.
2. Train and evaluate the baseline model.
3. Generate embeddings for the augmented split with the same source examples.
4. Train and evaluate the augmented model with the same seed and training settings.
5. Compare metrics from the same test split.

## Metrics To Report

- Kappa score.
- Pearson correlation.
- MAE or RMSE when available.
- Per-topic or per-prompt behavior if the split supports it.

## Notes

Report the augmentation recipe with the metric table. A performance delta is only meaningful when the source data, split, and training settings are held fixed.
