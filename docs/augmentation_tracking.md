# Augmentation Tracking

Track the exact augmentation recipe used for each AES experiment so baseline and topic-augmented runs stay comparable.

## Recipe Metadata

Record these values with every generated dataset:

- Source dataset revision and split name.
- Topic insertion setting, including sentence interval or per-sentence insertion mode.
- Random seed used during preprocessing.
- Script entry point and command line.
- Generated file path and row count.

## Dataset Checks

Before training, inspect a small sample from each split and confirm that the topic text is inserted in the expected position. Check label ranges and verify that row order still matches the source split when a paired comparison is required.

## Artifact Handling

Keep generated CSV files and embeddings outside normal Git history. Commit only small examples or documentation needed to explain the recipe.
