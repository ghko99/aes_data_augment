# Reproducibility Notes

This repository compares a baseline KoBERT-GRU AES model against a topic-augmented variant.

## Recommended run order

```bash
python3 aes_embedding.py --is_topic=False
python3 aes_train.py --is_topic=False
python3 aes_embedding.py --is_topic=True
python3 aes_train.py --is_topic=True
```

## Inputs

- Keep raw AIHub data outside the repository.
- Confirm that local CSV paths in the scripts point to the same train/validation/test split for both variants.

## Outputs

- Save generated embeddings and result CSV files under ignored output directories.
- Record the random seed, model checkpoint, and dependency versions with each experiment run.
