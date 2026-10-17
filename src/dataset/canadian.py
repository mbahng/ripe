"""Canadian Invertebrates image+DNA factory over a locally assembled BOLD export.

Expects ``<root>/canadian_invertebrates`` containing
``canadian_paired_first_image.tsv`` and ``canadian_images/<processid>.jpg``,
as produced by filter_canadian.py plus the aria2 fetch. Nothing is downloaded.
"""

import os
import re
from pathlib import Path

from typing import Callable, Dict, Tuple
from torch.utils.data import Dataset

from .bioscan import encode_dna
from .split import stratified_indices

__all__ = ["canadian_invertebrates", "augmentations", "balance_samples"]


def balance_samples(frame, seed):
  """Resample the training rows so every label is equally represented.
  The BOLD export is steeply long-tailed even after ``min_samples`` prunes
  the rarest labels: at species rank a few common invertebrates carry orders
  of magnitude more specimens than the median. A classifier trained on those
  raw frequencies can score well while ignoring the tail. This draws a fixed
  quota per label instead. Labels below their quota are oversampled -- every
  original row is kept, then rows are repeated at random to fill -- and
  labels above it are undersampled without replacement. Final counts land
  within one of each other.

  The split keeps its original size: rows are redistributed across labels,
  not added. Only train is resampled; val, test and push keep the real class
  frequencies so metrics reflect the actual distribution.

  Resampling happens after the grouped split, so repeated rows never cross a
  partition boundary: a duplicated barcode or photograph stays wholly inside
  train and cannot leak into val or test.
  """
  import numpy as np

  groups = list(frame.groupby("label", sort=True).indices.values())
  total = len(frame)
  rng = np.random.default_rng(seed)
  quotas = np.full(len(groups), total // len(groups), dtype=np.int64)
  # Spread the remainder at random rather than over the first few labels.
  quotas[rng.permutation(len(groups))[:total % len(groups)]] += 1
  selections = []
  for positions, quota in zip(groups, quotas):
    if quota <= len(positions):
      selections.append(rng.choice(positions, size=quota, replace=False))
    else:
      # Keep each rare-class example, then repeat randomly selected examples.
      selections.append(np.concatenate([
          positions, rng.choice(positions, size=quota - len(positions), replace=True)
      ]))
  indices = np.concatenate(selections)
  rng.shuffle(indices)
  return frame.iloc[indices].reset_index(drop=True)


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


def augmentations(augmentation: dict, shape): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  Like BIOSCAN these are plate-mounted specimen photographs at arbitrary
  orientation, so flips and full rotation are label-preserving, and cropping
  keeps the model off the scale bars at the frame edge. Keep color_jitter
  mild: pigment separates several of the species here.
  ``crop`` re-crops to ``shape``, which is why the output size is needed here.
  """
  from torchvision import transforms

  aug = []
  if augmentation.get("crop"): 
    aug.append(transforms.RandomResizedCrop(shape, scale=(augmentation["crop"], 1.0),
                                            ratio=(0.85, 1.18)))
  if augmentation.get("hflip"): 
    aug.append(transforms.RandomHorizontalFlip(p=augmentation["hflip"]))
  if augmentation.get("vflip"): 
    aug.append(transforms.RandomVerticalFlip(p=augmentation["vflip"]))
  if augmentation.get("rotation"): 
    aug.append(transforms.RandomRotation(augmentation["rotation"]))
  if augmentation.get("perspective"): 
    aug.append(transforms.RandomPerspective(distortion_scale=augmentation["perspective"], p=0.5))
  if augmentation.get("color_jitter"): 
    j = augmentation["color_jitter"]
    aug.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j, 0.5) / 2))
  return aug


def collate(dna_length):
  """Build the collate function for a Canadian dataset of the given barcode length.

    Items are ``{"img", "gen", "label"}`` dicts, which batch into
    ``{"x": (B, 3, H, W), "gen": (B, 4, 1, dna_length), "y": (B,)}``. Barcodes
    are stored as raw strings and one-hot encoded here, so every batch carries
    both modalities; drop the "gen" key for image-only models. Gap characters
    from BOLD's alignment (and any other non-ACGT base) encode as all-zero
    columns, the same "no base called here" convention ``encode_dna`` uses.
    """
  import torch

  def batched(batch):
    return {"x": torch.stack([item["img"] for item in batch]),
            "gen": torch.stack([encode_dna(item["gen"], dna_length) for item in batch]),
            "y": torch.tensor([item["label"] for item in batch], dtype=torch.long)}

  return batched


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


def canadian_invertebrates(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """Return train/push/val/test datasets and their collate over the local export.

    Options: root (data), rank (species), min_samples, split ([0.6, 0.2,
    0.2]), seed, allowed_bases ("ACGTN"), min_sequence_length, shape,
    normalize, augmentation, balance, dna_length (720).
    Rows whose image is absent, whose sequence is the upstream ``#NAME?``
    artefact, or whose sequence contains a character outside ``allowed_bases``
    are dropped.
    """
  import numpy as np
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

  minimum = dataset_cfg["min_samples"]
  if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
    raise ValueError("min_samples must be a positive integer")

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

  seed = dataset_cfg.get("seed", 0)
  split = dataset_cfg.get("split", [0.6, 0.2, 0.2])
  # Row-wise, like the torchvision datasets. Conspecific specimens share an
  # identical COI barcode and some photographs are reused across specimens, so
  # the same barcode or image can land on both sides of the split.
  parts = stratified_indices(frame["label"].to_numpy(), split, seed)
  partition = np.empty(len(frame), dtype=object)
  for name, positions in zip(("train", "val", "test"), parts):
    partition[positions] = name
  frame = frame.assign(_partition=partition).sample(frac=1, random_state=seed) \
               .reset_index(drop=True)

  classes = sorted(frame.loc[frame["_partition"] == "train", "label"].unique())
  if not classes:
    raise ValueError("No labeled Canadian training samples after filtering")

  dna_length = dataset_cfg.get("dna_length", 720)
  if isinstance(dna_length, bool) or not isinstance(dna_length, int) or dna_length < 18:
    raise ValueError("dna_length must be an integer >= 18")

  shape = dataset_cfg["shape"]
  tail = [transforms.ToTensor()]
  if dataset_cfg.get("normalize", True):
    tail.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
  transform = transforms.Compose([transforms.Resize(shape), *tail])
  train_transform = transforms.Compose([transforms.Resize(shape),
                                        *augmentations(dataset_cfg["augmentation"], shape), *tail])
  # Push keeps raw geometry: prototypes are projected onto real patches.
  push_transform = transforms.Compose([transforms.Resize(shape), transforms.ToTensor()])

  result = {}
  for partition in ("train", "val", "test"):
    rows = frame[frame["_partition"] == partition]
    rows = rows[rows["label"].isin(classes)]
    pipeline = train_transform if partition == "train" else transform
    result[partition] = _CanadianDataset(destination, rows, classes, pipeline)
  result["push"] = _CanadianDataset(destination, result["train"].frame, classes,
                                    push_transform)
  if dataset_cfg["balance"]:
    rows = balance_samples(result["train"].frame, seed)
    result["train"] = _CanadianDataset(destination, rows, classes, train_transform)
  return result, collate(dna_length)
