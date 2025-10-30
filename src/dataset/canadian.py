"""Canadian Invertebrates image+DNA factory over a locally assembled BOLD export.

Expects ``<root>/canadian_invertebrates`` containing
``canadian_paired_first_image.tsv`` and ``canadian_images/<processid>.jpg``,
as produced by filter_canadian.py plus the aria2 fetch. Nothing is downloaded.
"""

import hashlib
import math
import os
import re
from numbers import Real
from pathlib import Path

from typing import Dict
from torch.utils.data import Dataset

from .bioscan import _balance_training, build_transform

__all__ = ["canadian_invertebrates"]


_RANKS = {
    "phylum": "phylum_name",
    "class": "class_name",
    "order": "order_name",
    "family": "family_name",
    "genus": "genus_name",
    "species": "species_name",
}
# Literal Excel error string that replaces 420 sequences in the upstream dump.
_CORRUPT_SEQUENCE = "#NAME?"
_HASH_CACHE = "image_hashes.csv"


class _CanadianDataset(Dataset):
    """Image classification view with a shared training-derived label vocabulary.

    Mirrors ``_BIOSCANDataset``: each item is ``{"img", "gen", "label"}`` where
    ``gen`` is the DNA barcode string and ``label`` indexes ``classes``.
    """

    def __init__(self, root, frame, classes, transform):
        self.root = Path(root)
        self.frame = frame.reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.targets = self.frame["label"].map(self.class_to_idx).to_numpy(dtype="int64")
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        from PIL import Image

        row = self.frame.iloc[index]
        path = self.root / "canadian_images" / (row["processid"] + ".jpg")
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return {"img": image, "gen": row["gen"], "label": int(self.targets[index])}


def _image_digest(path):
    with open(path, "rb") as stream:
        return hashlib.blake2b(stream.read(), digest_size=16).hexdigest()


def _image_hashes(destination, processids):
    """Content hashes for duplicate detection, cached beside the dataset.

    193 photographs are reused across 686 specimens, so identical images must be
    kept on one side of a split. Hashing is parallel and cached; the first call
    reads every JPEG once (~1-2 minutes), later calls read the cache.
    """
    import pandas as pd

    cache = destination / _HASH_CACHE
    known = {}
    if cache.is_file():
        frame = pd.read_csv(cache, dtype=str, keep_default_na=False)
        known = dict(zip(frame["processid"], frame["digest"]))
    missing = [p for p in processids if p not in known]
    if missing:
        from multiprocessing import Pool

        paths = [str(destination / "canadian_images" / (p + ".jpg")) for p in missing]
        print(f"Hashing {len(missing)} images for duplicate grouping", flush=True)
        workers = min(8, os.cpu_count() or 1)
        with Pool(workers) as pool:
            for name, digest in zip(missing, pool.imap(_image_digest, paths, chunksize=256)):
                known[name] = digest
        pd.DataFrame({"processid": list(known), "digest": [known[k] for k in known]}).to_csv(
            cache, index=False)
    return known


def _components(frame, key_columns):
    """Union-find component id per row, linking rows that share any key value.

    Sequence identity and image identity are different axes: 131 of 173
    duplicate-photo groups span more than one barcode, so neither key alone
    isolates leakage.
    """
    import numpy as np

    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for column in key_columns:
        first = {}
        for position, value in enumerate(frame[column]):
            if value in first:
                union(first[value], position)
            else:
                first[value] = position
    return np.array([find(i) for i in range(len(frame))], dtype="int64")


def _grouped_stratified_split(frame, ratios, seed, group_column):
    """Stratify by label while keeping each group wholly within one partition.

    Stratification balances classes across partitions; grouping forbids a
    barcode or photograph from straddling them. The two conflict, so ratios are
    met approximately: whole groups are packed into the partition with the
    largest row deficit. Every class keeps at least one training group, and
    classes with enough groups get one in each enabled partition.

    Groups are keyed per class, so the effective unit is (component, label).
    A component spanning two labels is therefore split across partitions - but
    only as (barcode, label) pairs that differ, which is the COI-ambiguous
    species-pair case (85 components, 2.2% of rows at species rank). That is a
    hard example rather than leakage: a memorised barcode gives the wrong answer.
    No (barcode, label) or (image, label) pair ever crosses a partition.
    """
    import numpy as np

    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios < 0).any() or ratios[0] <= 0:
        raise ValueError("split must contain three finite nonnegative weights with a positive training weight")
    ratios = ratios / ratios.sum()
    enabled = np.flatnonzero(ratios > 0)
    rng = np.random.default_rng(seed)
    names = np.array(["train", "val", "test"])
    assignments = np.empty(len(frame), dtype=object)

    for positions in frame.groupby("label", sort=True).indices.values():
        groups = {}
        for position in positions:
            groups.setdefault(frame[group_column].iloc[position], []).append(position)
        blocks = [np.array(v) for v in groups.values()]
        order = rng.permutation(len(blocks))
        blocks = [blocks[i] for i in order]
        target = ratios * len(positions)
        filled = np.zeros(3, dtype=float)
        chosen = [[] for _ in range(3)]
        # Seed one block per enabled partition, largest target first, so every
        # class that can appear in a partition does. Ties are broken randomly:
        # with equal val/test weights a two-group class must not always hand its
        # second group to val, which would leave test systematically thinner.
        seeded = 0
        if len(blocks) >= len(enabled):
            shuffled = enabled[rng.permutation(len(enabled))]
            for partition in shuffled[np.argsort(-target[shuffled], kind="stable")]:
                block = blocks[seeded]
                chosen[partition].append(block)
                filled[partition] += len(block)
                seeded += 1
        # Remaining blocks: largest first into whichever partition is furthest
        # below its target.
        rest = sorted(blocks[seeded:], key=len, reverse=True)
        for block in rest:
            deficit = np.where(ratios > 0, target - filled, -np.inf)
            # argmax would always hand ties to the lowest index, so a two-group
            # class would send its second group to val every time and never test.
            best = np.flatnonzero(deficit == deficit.max())
            recipient = int(best[0] if len(best) == 1 else best[rng.integers(len(best))])
            chosen[recipient].append(block)
            filled[recipient] += len(block)
        if not chosen[0]:
            # Guarantee a training block even when blocks < enabled partitions.
            donor = max((p for p in range(3) if chosen[p]), key=lambda p: len(chosen[p]))
            chosen[0].append(chosen[donor].pop())
        for partition in range(3):
            for block in chosen[partition]:
                assignments[block] = names[partition]

    return frame.assign(_partition=assignments).iloc[rng.permutation(len(frame))].reset_index(drop=True)


def _row_stratified_split(frame, ratios, seed):
    """Per-row stratified split; identical barcodes may straddle partitions."""
    from .bioscan import _stratified_split

    return _stratified_split(frame, ratios, seed)


def canadian_invertebrates(dataset_cfg: dict) -> Dict[str, Dataset]:
    """Return train/push/val/test datasets over the local Canadian export.

    Options: root (data), rank (species), min_samples (10), split
    ([0.6, 0.2, 0.2]), seed, group_by (["sequence", "image"]), allowed_bases
    ("ACGTN"), min_sequence_length, image_size (224), normalize, augmentation,
    oversample.
    Rows whose image is absent, whose sequence is the upstream ``#NAME?``
    artefact, or whose sequence contains a character outside ``allowed_bases``
    are dropped.
    """
    import pandas as pd
    from torchvision import transforms

    destination = Path(dataset_cfg.get("root", "data")) / "canadian_invertebrates"
    metadata = destination / "canadian_paired_first_image.tsv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing {metadata}; this dataset is assembled locally, not downloaded")

    rank = dataset_cfg.get("rank", "species")
    if rank not in _RANKS:
        raise ValueError(f"Unsupported rank: {rank}. Choose from {sorted(_RANKS)}")
    rank_column = _RANKS[rank]

    minimum = dataset_cfg.get("min_samples", 10)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("min_samples must be a positive integer")

    group_by = dataset_cfg.get("group_by", ["sequence", "image"])
    group_by = [] if not group_by else ([group_by] if isinstance(group_by, str) else list(group_by))
    if set(group_by) - {"sequence", "image"}:
        raise ValueError("group_by entries must be 'sequence' and/or 'image'")

    columns = ["processid", "sampleid", "bin_uri", "nucleotides", rank_column]
    frame = pd.read_csv(metadata, sep="\t", usecols=list(dict.fromkeys(columns)),
                        dtype=str, keep_default_na=False, encoding="latin1")

    # Integrity exclusions: 420 corrupt sequences plus any specimen whose image
    # never downloaded (1 upstream 404 as of the current export).
    sequences = frame["nucleotides"].str.strip()
    frame = frame.loc[sequences.str.upper() != _CORRUPT_SEQUENCE].copy()
    frame["gen"] = sequences.loc[frame.index].str.upper()
    # BOLD stores sequences aligned, padding absent coverage with '-'. Restricting
    # the alphabet drops those partial reads outright rather than stripping the
    # padding, which also removes the all-gap records whose stripped form was an
    # empty string. Widen allowed_bases (e.g. "ACGTN-") to keep them.
    allowed = dataset_cfg.get("allowed_bases", "ACGTN")
    if allowed:
        if not isinstance(allowed, str):
            raise ValueError("allowed_bases must be a string of permitted characters, or null")
        allowed = allowed.upper()
        unknown = set(allowed) - set("ACGTURYSWKMBDHVN-")
        if unknown:
            raise ValueError(f"allowed_bases contains non-IUPAC characters: {sorted(unknown)}")
        frame = frame[~frame["gen"].str.contains(f"[^{re.escape(allowed)}]", regex=True)]
        if frame.empty:
            raise ValueError(f"No sequences consist only of {allowed}")
    floor = dataset_cfg.get("min_sequence_length")
    if floor is not None:
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 1:
            raise ValueError("min_sequence_length must be a positive integer")
        frame = frame[frame["gen"].str.len() >= floor]
    available = {os.path.splitext(name)[0]
                 for name in os.listdir(destination / "canadian_images")}
    frame = frame[frame["processid"].isin(available)]

    labels = frame[rank_column].str.strip()
    frame = frame.loc[labels != ""].copy()
    frame["label"] = labels.loc[frame.index]
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= minimum].index)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No {rank} has at least {minimum} usable samples")

    keys = []
    if "sequence" in group_by:
        keys.append("gen")
    if "image" in group_by:
        digests = _image_hashes(destination, list(frame["processid"]))
        frame["_digest"] = frame["processid"].map(digests)
        keys.append("_digest")
    seed = dataset_cfg.get("seed", 0)
    split = dataset_cfg.get("split", [0.6, 0.2, 0.2])
    if keys:
        frame["_group"] = _components(frame, keys)
        frame = _grouped_stratified_split(frame, split, seed, "_group")
    else:
        frame = _row_stratified_split(frame, split, seed)

    classes = sorted(frame.loc[frame["_partition"] == "train", "label"].unique())
    if not classes:
        raise ValueError("No labeled Canadian training samples after filtering")

    size = dataset_cfg.get("image_size", 224)
    transform = build_transform(size, dataset_cfg.get("normalize", True),
                                dataset_cfg.get("augmentation"))
    # Push keeps raw geometry: prototypes are projected onto real patches.
    push_transform = build_transform(size, normalize=False, augmentation=None)

    result = {}
    for partition in ("train", "val", "test"):
        rows = frame[frame["_partition"] == partition]
        rows = rows[rows["label"].isin(classes)]
        result[partition] = _CanadianDataset(destination, rows, classes, transform)
    result["push"] = _CanadianDataset(destination, result["train"].frame, classes,
                                      push_transform)
    if dataset_cfg.get("oversample") is not None:
        rows = _balance_training(result["train"].frame, dataset_cfg["oversample"], seed)
        result["train"] = _CanadianDataset(destination, rows, classes, transform)
    return result
