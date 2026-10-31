"""BIOSCAN dataset factories with automatic downloads and local data reuse."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import functools
import io
import os

from typing import Callable, Dict, Tuple
from torch.utils.data import Dataset

from .split import stratified_indices

__all__ = ["bioscan1m", "bioscan5m", "augmentations", "balance_samples", "encode_dna"]


_METADATA_5M = "bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"


def _download(filenames, destination, repo):
  def _curl(url, target):
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
         "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
         "--continue-at", "-", "--output",
         str(target), url],
        check=True,
    )

  destination = Path(destination)
  destination.mkdir(parents=True, exist_ok=True)
  manifest = destination / "huggingface.json"
  if not manifest.exists():
    temporary = manifest.with_suffix(".json.part")
    url = f"https://huggingface.co/api/datasets/{repo}?blobs=true"
    _curl(url, temporary)
    json.loads(temporary.read_text())
    temporary.replace(manifest)
  metadata = json.loads(manifest.read_text())
  files = {
      entry["rfilename"]: {"size": entry["lfs"]["size"],
                            "checksum": "sha256:" + entry["lfs"]["sha256"]}
      for entry in metadata["siblings"] if "lfs" in entry
  }
  for name in filenames:
    entry = files[name]
    target = destination / name
    if not target.exists():
      partial = destination / (name + ".part")
      print(f"Downloading {name} ({entry['size'] / 1e9:.2f} GB)", flush=True)
      url = f"https://huggingface.co/datasets/{repo}/resolve/{metadata['sha']}/{name}"
      _curl(url, partial)
      candidate = partial
    else:
      candidate = target
    algorithm, expected = entry["checksum"].split(":", 1)
    print(f"Verifying {name}", flush=True)
    with candidate.open("rb") as stream:
      actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if candidate.stat().st_size != entry["size"] or actual != expected:
      raise ValueError(f"Integrity check failed: {candidate}. Remove this file and retry.")
    if candidate != target:
      candidate.replace(target)
    marker = destination / (name + ".extracted")
    if name.endswith(".zip"):
      print(f"Extracting {name}", flush=True)
      with zipfile.ZipFile(target) as archive:
        root = destination.resolve()
        for member in archive.infolist():
          if not (root / member.filename).resolve().is_relative_to(root):
            raise ValueError(f"Unsafe archive path: {member.filename}")
        archive.extractall(destination)
      marker.touch()
  print(f"Complete: {destination.resolve()}", flush=True)
  return destination


def augmentations(augmentation: dict, shape): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  Specimen photographs are plate-mounted at arbitrary orientation, so flips
  and full rotation are label-preserving. Crop matters because the frames
  include rulers and scale bars at the edges, and cropping stops the model
  keying on them. Keep color_jitter mild: pigment is taxonomically
  informative.
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


_DNA_CHANNELS = {"A": 0, "C": 1, "G": 2, "T": 3}

import numpy as _np
_DNA_LUT = _np.full(256, 255, dtype=_np.uint8)
for _base, _channel in _DNA_CHANNELS.items():
  _DNA_LUT[ord(_base)] = _channel


def encode_dna(sequence, length):
  """One-hot a barcode into the (4, 1, length) tensor GeneticCNN2D expects.

    Channels are A/C/G/T; anything else (notably the ambiguity code ``N``) and any
    padding past the end of a short read stay all-zero, which is the natural
    "no base called here" encoding. Reads longer than ``length`` are truncated.
    The singleton height axis is required because the model is built from Conv2d
    with (1, k) kernels, so batches arrive as (B, 4, 1, length).
    """
  import numpy as np
  import torch

  # Vectorized: look every byte up in a 256-entry table, then scatter the ones.
  codes = _DNA_LUT[np.frombuffer(sequence[:length].upper().encode("ascii", "replace"), dtype=np.uint8)]
  positions = np.flatnonzero(codes < 4)
  out = np.zeros((4, length), dtype=np.float32)
  out[codes[positions], positions] = 1.0
  return torch.from_numpy(out).reshape(4, 1, length)


def collate(dna_length):
  """Build the collate function for a BIOSCAN dataset of the given barcode length.

    Items arrive as dicts, so batching is a per-key stack rather than a
    positional one. Two item layouts exist (see ``_BIOSCANDataset``):

    - image mode: ``{"img", "gen", "label"}`` becomes ``{"x": (B, 3, H, W),
      "gen": (B, 4, 1, dna_length), "y": (B,)}``. Barcodes are stored as raw
      strings on the dataset and one-hot encoded here, so an image batch also
      carries its paired DNA for multimodal models; drop the key to ignore it.
    - dna mode: ``{"x", "y"}`` is already encoded and only needs stacking.
    """
  import torch

  def batched(batch):
    first = batch[0]
    if "img" not in first:
      return {"x": torch.stack([item["x"] for item in batch]),
              "y": torch.tensor([item["y"] for item in batch], dtype=torch.long)}
    return {"x": torch.stack([item["img"] for item in batch]),
            "gen": torch.stack([encode_dna(item["gen"], dna_length) for item in batch]),
            "y": torch.tensor([item["label"] for item in batch], dtype=torch.long)}

  return batched


_H5_NAME = "cropped_256.hdf5"
_H5_GROUP = "bioscan_dataset"
_INDEX_NAME = "cropped_256.index.npz"


def _build_offset_index(h5_path, index_path):
  """Record where every image's bytes sit inside the HDF5 file.

    Looking a name up in the 1M-member group costs ~33 ms per image, against
    ~0.2 ms for the read itself. The images are contiguous, uncompressed uint8
    datasets, so each one is a plain byte range of the file and can be read
    with ``os.pread`` without HDF5. ``visititems`` walks the group in ~0.3 ms
    per item (about 5 minutes for BIOSCAN-1M); the result is saved next to the
    HDF5 file and reused until the HDF5 file's size changes.
    """
  import fcntl
  import h5py
  import numpy as np

  h5_path, index_path = Path(h5_path), Path(index_path)
  with open(str(index_path) + ".lock", "w") as lock:
    # One process builds; the others wait here, then find the finished index.
    fcntl.flock(lock, fcntl.LOCK_EX)
    if _index_is_current(index_path, h5_path):
      return
    print(f"Building image offset index for {h5_path.name} (one-time, a few minutes)...", flush=True)
    names, offsets, sizes = [], [], []

    def visit(name, obj):
      if not isinstance(obj, h5py.Dataset):
        return None
      offset = obj.id.get_offset()
      if offset is None or obj.chunks is not None or obj.compression is not None:
        raise RuntimeError(f"{name} is not a contiguous uncompressed dataset; os.pread cannot read it")
      names.append(name)
      offsets.append(offset)
      sizes.append(int(np.prod(obj.shape)) * obj.dtype.itemsize)
      if len(names) % 100_000 == 0:
        print(f"  indexed {len(names):,} images", flush=True)
      return None

    with h5py.File(h5_path, "r") as handle:
      handle[_H5_GROUP].visititems(visit)
    temporary = index_path.with_name(index_path.name + f".{os.getpid()}.tmp")
    with open(temporary, "wb") as out:
      np.savez(out, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
               sizes=np.array(sizes, dtype=np.int64), file_size=np.int64(h5_path.stat().st_size))
    os.replace(temporary, index_path)   # atomic: a reader never sees a half-written index
    print(f"Saved offset index for {len(names):,} images to {index_path}", flush=True)


def _index_is_current(index_path, h5_path):
  import numpy as np

  if not Path(index_path).is_file():
    return False
  try:
    with np.load(index_path) as index:
      return int(index["file_size"]) == Path(h5_path).stat().st_size
  except (OSError, ValueError, KeyError):
    return False   # unreadable or from a different layout: rebuild


@functools.lru_cache(maxsize=2)
def _load_offset_index(root):
  """(names, offsets, sizes) for the HDF5 file under ``root``; builds the index if missing or stale."""
  import numpy as np

  h5_path, index_path = Path(root) / _H5_NAME, Path(root) / _INDEX_NAME
  if not _index_is_current(index_path, h5_path):
    _build_offset_index(h5_path, index_path)
  with np.load(index_path) as index:
    return index["names"], index["offsets"], index["sizes"]


class _BIOSCANDataset(Dataset):
  """Image classification view with a shared label vocabulary.

    Image bytes are read straight from the HDF5 file with ``os.pread`` using a
    precomputed offset index (see ``_build_offset_index``); the file descriptor
    is opened lazily per process, so DataLoader workers do not share it. Rows with no label at the requested rank are dropped by the
    factory; ``classes`` and ``class_to_idx`` describe the output labels.
    """

  def __init__(self, root, version, frame, classes, transform, dna_length=None):
    self.root = Path(root)
    # When set, items carry the encoded barcode instead of the image; the
    # JPEG is not read at all, which is the bulk of per-sample cost.
    self.dna_length = dna_length
    self.version = version # 1m or 5m
    self.frame = frame.reset_index(drop=True)
    self.classes = classes
    self.class_to_idx = {name: i for i, name in enumerate(classes)}
    self.targets = self.frame["label"].map(self.class_to_idx).to_numpy(dtype="int64")
    self.transform = transform
    self._fd = None
    self._pid = None
    if version == "1m" and dna_length is None:
      import pandas as pd

      names, offsets, sizes = _load_offset_index(str(self.root))
      positions = pd.Index(names).get_indexer(self.frame["image_file"])
      if (positions < 0).any():
        missing = self.frame["image_file"][positions < 0].head(3).tolist()
        raise KeyError(f"{int((positions < 0).sum())} images are not in the HDF5 file, e.g. {missing}")
      self._offsets = offsets[positions]
      self._sizes = sizes[positions]

  def __len__(self):
    return len(self.frame)

  def __getitem__(self, index):
    from PIL import Image

    row = self.frame.iloc[index]
    if self.dna_length is not None:
      # Keys match the repo's tuple-dataset convention so the existing
      # supervised trainer consumes DNA batches without modification.
      return {"x": encode_dna(row["gen"], self.dna_length),
              "y": int(self.targets[index])}
    if self.version == "1m":
      if self._fd is None or self._pid != os.getpid():
        self.close()
        self._fd = os.open(self.root / _H5_NAME, os.O_RDONLY)
        self._pid = os.getpid()
      data = os.pread(self._fd, int(self._sizes[index]), int(self._offsets[index]))
      if len(data) != self._sizes[index]:
        raise OSError(f"short read for {row['image_file']}: {len(data)} of {self._sizes[index]} bytes")
      source = io.BytesIO(data)
    else:
      source = (self.root / "bioscan5m/images/cropped_256" / row["split"]
                / row["chunk"] / (row["processid"] + ".jpg"))
    with Image.open(source) as image:
      image = image.convert("RGB")
      image = self.transform(image)
    return {"img": image, "gen": row["gen"], "label": int(self.targets[index])}

  def close(self):
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None
      self._pid = None

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_fd"] = None
    state["_pid"] = None
    return state


def balance_samples(frame, seed):
  """Resample the training rows so every label is equally represented.

    BIOSCAN is long-tailed: a few insect orders carry most of the specimens, so
    a classifier can score well on raw frequencies while ignoring the tail.
    This draws a fixed quota per label instead. Labels below their quota are
    oversampled -- every original row is kept, then rows are repeated at random
    to fill -- and labels above it are undersampled without replacement. Final
    counts land within one of each other.

    The split keeps its original size: rows are redistributed across labels,
    not added. Only train is resampled; val, test and push keep the real class
    frequencies so metrics reflect the actual distribution.
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


def _bioscan(cfg, version, destination):
  import numpy as np
  import pandas as pd
  from torchvision import transforms

  rank = cfg.get("rank", cfg.get("target_rank", "order"))
  if rank not in ("order", "family", "subfamily", "genus", "species"):
    raise ValueError(f"Unsupported rank: {rank}")
  if version == "1m":
    metadata = destination / "BIOSCAN_Insect_Dataset_metadata.tsv"
    gen_column = "nucraw"
    columns = list(dict.fromkeys(["image_file", gen_column, rank]))
    separator = "\t"
  else:
    metadata = destination / _METADATA_5M
    gen_column = "dna_barcode"
    # "split" is read as data, not as a partition: it names the image directory.
    columns = list(dict.fromkeys(["processid", "chunk", "split", gen_column, rank]))
    separator = ","
  chunks = []
  # Read only relevant fields and retain only requested splits from the 5M CSV.
  for chunk in pd.read_csv(metadata, sep=separator, usecols=columns, dtype=str,
                           keep_default_na=False, chunksize=100_000):
    # Filter unavailable target ranks before selecting partitions or building labels.
    labels = chunk[rank].str.strip()
    missing = labels.str.lower().isin(
        ["", "not_classified", "unclassified", "unknown", "nan", "none", "null", "na", "n/a"]
    )
    chunk = chunk.loc[~missing].copy()
    chunk[rank] = labels.loc[~missing]
    chunks.append(chunk.rename(columns={rank: "label", gen_column: "gen"}))
  frame = pd.concat(chunks, ignore_index=True)
  # Applied at every rank, not just species: order rank also carries a few
  # labels with one or two rows, and a class that small cannot be split into
  # meaningful train/val/test shares.
  minimum = cfg["min_samples"]
  if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
    raise ValueError("min_samples must be a positive integer")
  counts = frame["label"].value_counts()
  frame = frame[frame["label"].isin(counts[counts >= minimum].index)].reset_index(drop=True)
  if frame.empty:
    raise ValueError(f"No {rank} has at least {minimum} labeled samples")
  seed = cfg.get("seed", 0)
  parts = stratified_indices(frame["label"].to_numpy(),
                             cfg.get("split", [0.6, 0.2, 0.2]), seed)
  partition = np.empty(len(frame), dtype=object)
  for name, positions in zip(("train", "val", "test"), parts):
    partition[positions] = name
  frame = frame.assign(_partition=partition).sample(frac=1, random_state=seed) \
               .reset_index(drop=True)
  # stratified_indices guarantees every label at least one training row, so
  # the class vocabulary is simply every label present.
  classes = sorted(frame["label"].unique())
  if not classes:
    raise ValueError("No labeled BIOSCAN samples for the requested rank")
  # modality: "image" (default) or "dna". In dna mode items are the encoded
  # barcode, so no image archive is touched.
  modality = cfg.get("modality", "image")
  if modality not in ("image", "dna"):
    raise ValueError("modality must be 'image' or 'dna'")
  # The length is needed in both modalities: in dna mode the dataset encodes
  # items itself, in image mode the collate encodes the paired barcode.
  dna_length = cfg.get("dna_length", 720)
  if isinstance(dna_length, bool) or not isinstance(dna_length, int) or dna_length < 18:
    raise ValueError("dna_length must be an integer >= 18")
  item_dna_length = dna_length if modality == "dna" else None
  shape = cfg["shape"]
  transform = transforms.Compose([transforms.Resize(shape), transforms.ToTensor(),
                                  transforms.Normalize(cfg["normalize"]["mean"], cfg["normalize"]["std"])])
  train_transform = transforms.Compose([transforms.Resize(shape),
                                        *augmentations(cfg["augmentation"], shape),
                                        transforms.ToTensor(),
                                        transforms.Normalize(cfg["normalize"]["mean"], cfg["normalize"]["std"])])
  result = {}
  for split in ("train", "val", "test"):
    rows = frame[frame["_partition"] == split]
    pipeline = train_transform if split == "train" else transform
    result[split] = _BIOSCANDataset(destination, version, rows, classes, pipeline, item_dna_length)
  # Push uses the val/test pipeline: same normalization, and crucially no augmentation,
  # since a prototype is projected onto a specific patch of a specific image and the
  # recorded boxes have to refer to it.
  result["push"] = _BIOSCANDataset(destination, version, result["train"].frame, classes,
                                  transform, item_dna_length)
  if cfg["balance"]:
    rows = balance_samples(result["train"].frame, cfg.get("seed", 0))
    result["train"] = _BIOSCANDataset(destination, version, rows, classes, train_transform,
                                      item_dna_length)
  return result, collate(dna_length)


def bioscan1m(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """Return train/push/val/test datasets and their collate, downloading missing 1M data.

    Uses data/bioscan_1m by default. Options: root, download, rank (order),
    min_samples, shape, normalize (required; mean/std per channel, or false),
    augmentation, balance, modality, dna_length, split, seed.
    Splits are always label-stratified from the split ratios and seed; the
    benchmark's own partition columns are not used.
    """
  destination = Path(dataset_cfg.get("root", "data")) / "bioscan_1m"
  files = ["BIOSCAN_Insect_Dataset_metadata.tsv", "cropped_256.hdf5"]
  if dataset_cfg.get("download", True) and not all(
      (destination / name).is_file() for name in files
  ):
    _download(files, destination, repo="bioscan-ml/BIOSCAN-1M")
  return _bioscan(dataset_cfg, "1m", destination)


def bioscan5m(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """Return train/push/val/test datasets and their collate, downloading missing 5M data.

    Uses data/bioscan_5m by default. Options match bioscan1m. Stratification
    pools every labeled sample, including the pretrain and unseen/key
    partitions, while preserving the image paths those names encode.
    """
  destination = Path(dataset_cfg.get("root", "data")) / "bioscan_5m"
  metadata_exists = (destination / _METADATA_5M).is_file()
  images_exist = all(
      (destination / "bioscan5m/images/cropped_256" / split).is_dir()
      for split in ("train", "val", "test", "pretrain")
  )
  if dataset_cfg.get("download", True) and not (metadata_exists and images_exist):
    files = []
    if not metadata_exists:
      files.append("BIOSCAN_5M_Insect_Dataset_metadata_MultiTypes.zip")
    if not images_exist:
      files.append("BIOSCAN_5M_cropped_256.zip")
    _download(files, destination, repo="bioscan-ml/BIOSCAN-5M")
  return _bioscan(dataset_cfg, "5m", destination)
