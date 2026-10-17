"""Label-stratified partitioning for datasets that ship no usable split.

Several of the torchvision datasets here publish only a train/test partition,
so the validation set has to be carved out of train locally. ``random_split``
does that uniformly at random, which leaves per-class counts to chance: fine
for MNIST at ~6000 images per class, meaningfully lossy for CUB200 or Stanford
Cars at a few dozen. Stratifying instead makes every partition mirror the
label distribution of the pool it came from.

Datasets whose partitions are published upstream (CelebA, Flowers102, the
BIOSCAN benchmark splits) must keep those partitions and do not use this
module; BIOSCAN and Canadian Invertebrates carry their own stratified
splitters because they additionally group by barcode and photograph.
"""

import numpy as np

__all__ = ["stratified_indices"]


def stratified_indices(targets, ratios, seed=0):
  """Positions of ``targets`` split so each partition mirrors the label mix.

  ``ratios`` are relative weights, normalised internally, so callers can pass
  a raw ``[train, val]`` slice of a three-way config without rescaling it.
  Each class is distributed by largest remainder, with ties broken randomly:
  rounding consistently in index order would hand every leftover sample to the
  earliest partition, systematically thinning the last one. Every class keeps
  at least one training sample, and classes with enough samples appear in
  every enabled partition, so a label can never be known only at evaluation
  time. Returns one sorted index array per ratio.
  """
  ratios = np.asarray(ratios, dtype=float)
  if ratios.ndim != 1 or len(ratios) < 2 or not np.isfinite(ratios).all() \
      or (ratios < 0).any() or ratios[0] <= 0:
    raise ValueError("ratios must be two or more finite nonnegative weights "
                     "with a positive training weight")
  ratios = ratios / ratios.sum()
  k = len(ratios)
  enabled = np.count_nonzero(ratios)

  targets = np.asarray(targets)
  if targets.ndim != 1:
    raise ValueError("targets must be one label per sample")
  rng = np.random.default_rng(seed)
  buckets = [[] for _ in range(k)]

  for label in np.unique(targets):
    positions = rng.permutation(np.flatnonzero(targets == label))
    n = len(positions)
    ideal = n * ratios
    counts = np.floor(ideal).astype(int)
    order = rng.permutation(k)
    order = order[np.argsort(-(ideal - counts)[order], kind="stable")]
    counts[order[:n - counts.sum()]] += 1

    minimum = np.zeros(k, dtype=int)
    minimum[0] = 1
    if n >= enabled:
      minimum[ratios > 0] = 1
    for recipient in range(k):
      while counts[recipient] < minimum[recipient]:
        # Take from whichever partition is furthest above its ideal share.
        donors = np.flatnonzero(counts > minimum)
        donor = donors[np.argmax((counts - ideal)[donors])]
        counts[donor] -= 1
        counts[recipient] += 1

    for bucket, chunk in zip(buckets, np.split(positions, np.cumsum(counts)[:-1])):
      bucket.append(chunk)

  # Sorting keeps each subset in dataset order, so an index means the same
  # sample across runs and shuffling stays the DataLoader's job.
  return [np.sort(np.concatenate(bucket)) for bucket in buckets]
