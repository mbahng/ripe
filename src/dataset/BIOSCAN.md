BIOSCAN image classification
============================

Set `run.dataset.name` to `bioscan1m` or `bioscan5m` in an existing
training configuration. The model's `num_classes` must match the dataset:
the default order task has 16 classes for 1M and 42 for 5M.

```python
from src.dataset import init_dataloader

loaders = init_dataloader({
    "name": "bioscan1m",  # or bioscan5m
    "root": "data",       # parent of bioscan_1m/ and bioscan_5m/
    "rank": "species",
    "image_size": 224,
    "download": True,
    "train": {"batch_size": 32, "num_workers": 2, "shuffle": True},
    "push": {"batch_size": 32, "num_workers": 2, "shuffle": False},
    "val": {"batch_size": 32, "num_workers": 2},
    "test": {"batch_size": 32, "num_workers": 2},
})
batch = next(iter(loaders["train"]))
# batch["img"]: float [B, 3, 224, 224]
# batch["gen"]: list of raw DNA barcode strings (variable length)
# batch["label"]: int64 [B]
```

Local data is reused without network requests or rehashing the image archives.
Missing datasets download automatically from Hugging Face. `download=False`
disables downloading. BIOSCAN-1M reads JPEG bytes lazily from HDF5 with a
separate handle per worker; BIOSCAN-5M reads extracted JPEG files.

For `rank: species`, the default `split_strategy: stratified` pools all rows
with a known species, removes species with fewer than `min_species_samples`
(default 10) across the labeled pool, then splits within each retained species
using `split` (default
`[0.6, 0.2, 0.2]`) and `dataset.seed` (default 0). Every retained species has at least
one training sample. If the threshold is lowered to 1, singletons stay in training; species with enough samples
have at least one example in each enabled partition. Ratios are approximate
because of per-species rounding and rare-species constraints. Splitting precedes
balancing, so duplicated training examples cannot enter validation or test.

For 5M, stratification includes labeled samples from all original partitions,
including pretraining and unseen/key partitions; this is a custom closed-set
split, not the published benchmark. Original directory names remain intact.

Other ranks default to `split_strategy: published`; species can opt into it too.
In published mode, 1M's `split_column` defaults to `large_insect_order`; use
`large_diptera_family` with `rank="family"` for that benchmark. Evaluation-only
classes are excluded with warnings in published mode. Supported targets are
order, family, subfamily, genus, and species.

Each sample is a dictionary with `img` (image tensor), `gen` (raw DNA
barcode string), and `label` (integer class index). DNA comes from `nucraw`
for 1M and `dna_barcode` for 5M; it is not padded, truncated, or encoded.
Missing DNA remains an empty string. The image trainers use `img` and
`label`; genetic modeling requires a separate sequence encoding step.

For 5M in published mode, pretraining and unseen/key partitions are excluded
by default. To add labeled pretraining samples to training, set:

```yaml
split_strategy: published
split_names:
  train: [train, pretrain]
  val: [val]
  test: [test]
```

Images are resized and ImageNet-normalized by default (`normalize: false`
disables normalization). The push view always returns unnormalized images.
Each split supports `subsample` as a fraction (e.g. `0.1`) or an integer
count; subsampling takes the first entries, matching the repo convention.
Without subsampling, inspect `loaders["train"].dataset.classes` and
`.class_to_idx`; with subsampling these are on `.dataset.dataset`.

Tests: `.venv/bin/python -m unittest discover -s tests -v`.

Direct factory calls: `bioscan1m(dataset_cfg)` and `bioscan5m(dataset_cfg)`
from `src.dataset.bioscan` each return a `Dict[str, Dataset]` with
`train`, `push`, `val`, and `test` keys, including automatic downloading.

Set `run.dataset.oversample: X` to balance the training classes while targeting
`round(X * N)` samples, where N is the labeled training size after rank and
split filtering, before `train.subsample`. For example, `1.0` balances at the
original size and `2.0` doubles it. Omit this option or use `null` to disable.
X must be positive, finite, and large enough to retain every training class.

Each class gets the same number of samples (within one for rounding).
Common classes are sampled without replacement; rare classes keep all their
examples and receive random duplicates. The result is shuffled once at dataset
construction using `dataset.seed` (default 0); it is not redrawn each epoch.
Push, validation, and test keep their original samples. Images and genetics
stay paired because sampling selects entire metadata rows. Subsequent
`train.subsample` truncation can reduce the exact class balance.
