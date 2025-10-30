"""BIOSCAN dataset factories with automatic downloads and local data reuse."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import io
import os
import warnings
import math
from numbers import Real

from typing import Dict
from torch.utils.data import Dataset

__all__ = ["bioscan1m", "bioscan5m", "build_transform", "encode_dna"]


_METADATA_5M = "bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"


def _curl(url, destination):
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
         "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
         "--continue-at", "-", "--output",
         str(destination), url],
        check=True,
    )


def _download(filenames, destination, repo):
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


_AUGMENTATIONS = ("hflip", "vflip", "rotation", "crop", "color_jitter", "perspective", "erasing")


def build_transform(size, normalize=True, augmentation=None):
    """Compose a specimen-image transform, optionally with train-time augmentation.

    ``augmentation`` is a dict of knobs; omit or pass a falsey value to disable.
    Specimen photographs are plate-mounted at arbitrary orientation, so flips and
    full rotation are label-preserving. ``crop`` matters most for datasets whose
    frames include rulers or scale bars at the edges: random-resized cropping
    stops the model keying on them. Colour jitter is kept mild because pigment is
    taxonomically informative.
    """
    from torchvision import transforms

    size = (size, size) if isinstance(size, int) else tuple(size)
    augmentation = augmentation or {}
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation must be a mapping of option names to values, or null")
    unknown = set(augmentation) - set(_AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unknown augmentation options: {sorted(unknown)}. "
                         f"Choose from {list(_AUGMENTATIONS)}")

    head, tail = [], []
    crop = augmentation.get("crop")
    if crop:
        scale = tuple(crop) if isinstance(crop, (list, tuple)) else (float(crop), 1.0)
        head.append(transforms.RandomResizedCrop(size, scale=scale, ratio=(0.85, 1.18)))
    else:
        head.append(transforms.Resize(size))
    if augmentation.get("hflip"):
        head.append(transforms.RandomHorizontalFlip(p=float(augmentation["hflip"])))
    if augmentation.get("vflip"):
        head.append(transforms.RandomVerticalFlip(p=float(augmentation["vflip"])))
    if augmentation.get("rotation"):
        head.append(transforms.RandomRotation(degrees=float(augmentation["rotation"])))
    if augmentation.get("perspective"):
        head.append(transforms.RandomPerspective(
            distortion_scale=float(augmentation["perspective"]), p=0.5))
    if augmentation.get("color_jitter"):
        j = float(augmentation["color_jitter"])
        head.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j, 0.5) / 2))
    if augmentation.get("erasing"):
        tail.append(transforms.RandomErasing(p=float(augmentation["erasing"]), scale=(0.02, 0.15)))

    steps = head + [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(steps + tail)


_DNA_CHANNELS = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_dna(sequence, length):
    """One-hot a barcode into the (4, 1, length) tensor GeneticCNN2D expects.

    Channels are A/C/G/T; anything else (notably the ambiguity code ``N``) and any
    padding past the end of a short read stay all-zero, which is the natural
    "no base called here" encoding. Reads longer than ``length`` are truncated.
    The singleton height axis is required because the model is built from Conv2d
    with (1, k) kernels, so batches arrive as (B, 4, 1, length).
    """
    import torch

    out = torch.zeros(4, 1, length, dtype=torch.float32)
    for position, base in enumerate(sequence[:length].upper()):
        channel = _DNA_CHANNELS.get(base)
        if channel is not None:
            out[channel, 0, position] = 1.0
    return out


class _BIOSCANDataset(Dataset):
    """Image classification view with a shared training-derived label vocabulary.

    HDF5 handles are opened lazily per process, so DataLoader workers do not
    share handles. Missing labels and evaluation-only classes are excluded by
    the factory; ``classes`` and ``class_to_idx`` describe the output labels.
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
        self._h5 = None
        self._pid = None

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
            import h5py

            if self._h5 is None or self._pid != os.getpid():
                self.close()
                self._h5 = h5py.File(self.root / "cropped_256.hdf5", "r")
                self._pid = os.getpid()
            source = io.BytesIO(self._h5["bioscan_dataset"][row["image_file"]][()].tobytes())
        else:
            source = (self.root / "bioscan5m/images/cropped_256" / row["split"]
                      / row["chunk"] / (row["processid"] + ".jpg"))
        with Image.open(source) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return {"img": image, "gen": row["gen"], "label": int(self.targets[index])}

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_pid"] = None
        return state


def _balance_training(frame, factor, seed):
    """Randomly resample to round(factor * N), with class counts within one."""
    import numpy as np

    if isinstance(factor, bool) or not isinstance(factor, Real) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("oversample must be a finite positive number, or null to disable")
    groups = list(frame.groupby("label", sort=True).indices.values())
    total = round(factor * len(frame))
    if total < len(groups):
        raise ValueError("oversample is too small to retain at least one sample per class")
    rng = np.random.default_rng(seed)
    quotas = np.full(len(groups), total // len(groups), dtype=np.int64)
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


def _stratified_split(frame, ratios, seed):
    """Assign each row once, retaining every class in training."""
    import numpy as np

    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios < 0).any() or ratios[0] <= 0:
        raise ValueError("split must contain three finite nonnegative weights with a positive training weight")
    ratios = ratios / ratios.sum()
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(frame), dtype=object)
    names = np.array(["train", "val", "test"])
    for positions in frame.groupby("label", sort=True).indices.values():
        positions = rng.permutation(positions)
        n = len(positions)
        ideal = n * ratios
        counts = np.floor(ideal).astype(int)
        # Randomize ties rather than consistently favoring validation over test.
        tie_order = rng.permutation(3)
        order = tie_order[np.argsort(-(ideal - counts)[tie_order], kind="stable")]
        counts[order[:n - counts.sum()]] += 1
        minimum = np.zeros(3, dtype=int)
        minimum[0] = 1
        if n >= np.count_nonzero(ratios):
            minimum[ratios > 0] = 1
        for recipient in range(3):
            while counts[recipient] < minimum[recipient]:
                donors = np.flatnonzero(counts > minimum)
                donor = donors[np.argmax((counts - ideal)[donors])]
                counts[donor] -= 1
                counts[recipient] += 1
        assignments[positions] = np.repeat(names, counts)
    return frame.assign(_partition=assignments).iloc[rng.permutation(len(frame))].reset_index(drop=True)


def _bioscan(cfg, version, destination):
    import pandas as pd
    from torchvision import transforms

    rank = cfg.get("rank", cfg.get("target_rank", "order"))
    if rank not in ("order", "family", "subfamily", "genus", "species"):
        raise ValueError(f"Unsupported rank: {rank}")
    strategy = cfg.get("split_strategy", "stratified" if rank == "species" else "published")
    if strategy not in ("stratified", "published"):
        raise ValueError("split_strategy must be stratified or published")
    if strategy == "stratified" and "split_names" in cfg:
        raise ValueError("split_names applies only to split_strategy: published")
    if version == "1m":
        metadata = destination / "BIOSCAN_Insect_Dataset_metadata.tsv"
        split_column = cfg.get("split_column", "large_insect_order")
        gen_column = "nucraw"
        columns = list(dict.fromkeys(["image_file", gen_column, rank, split_column]))
        default_splits = {"train": ["train"], "val": ["validation"], "test": ["test"]}
        separator = "\t"
    else:
        metadata = destination / _METADATA_5M
        split_column = "split"
        gen_column = "dna_barcode"
        columns = ["processid", "chunk", "split", gen_column, rank]
        default_splits = {"train": ["train"], "val": ["val"], "test": ["test"]}
        separator = ","
    split_names = cfg.get("split_names", default_splits)
    if set(split_names) != {"train", "val", "test"}:
        raise ValueError("split_names must define train, val, and test")
    split_names = {k: [v] if isinstance(v, str) else list(v) for k, v in split_names.items()}
    names = [name for values in split_names.values() for name in values]
    if len(names) != len(set(names)):
        raise ValueError("split_names must not overlap")
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
    if rank == "species":
        minimum = cfg.get("min_species_samples", 10)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("min_species_samples must be a positive integer")
        counts = frame["label"].value_counts()
        frame = frame[frame["label"].isin(counts[counts >= minimum].index)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No species have at least {minimum} labeled samples")
    if strategy == "published":
        frame = frame[frame[split_column].isin(names)].reset_index(drop=True)
    if strategy == "stratified":
        frame = _stratified_split(frame, cfg.get("split", [0.6, 0.2, 0.2]), cfg.get("seed", 0))
        # Preserve the original 5M split column: it identifies image directories.
        split_column = "_partition"
        split_names = {name: [name] for name in ("train", "val", "test")}
    classes = sorted(frame.loc[frame[split_column].isin(split_names["train"]), "label"].unique())
    if not classes:
        raise ValueError("No labeled BIOSCAN training samples match the requested splits/rank")
    # modality: "image" (default) or "dna". In dna mode items are the encoded
    # barcode, so no image archive is touched.
    modality = cfg.get("modality", "image")
    if modality not in ("image", "dna"):
        raise ValueError("modality must be 'image' or 'dna'")
    dna_length = cfg.get("dna_length", 720) if modality == "dna" else None
    if dna_length is not None:
        if isinstance(dna_length, bool) or not isinstance(dna_length, int) or dna_length < 18:
            raise ValueError("dna_length must be an integer >= 18")
    size = cfg.get("image_size", 224)
    normalise = cfg.get("normalize", True)
    transform = build_transform(size, normalise, cfg.get("augmentation"))
    # Push keeps raw geometry: prototypes are projected onto real patches.
    push_transform = build_transform(size, normalize=False, augmentation=None)
    result = {}
    for split, values in split_names.items():
        rows = frame[frame[split_column].isin(values)]
        known = rows["label"].isin(classes)
        if not known.all():
            warnings.warn(f"BIOSCAN {split}: excluding {(~known).sum()} samples with labels absent from training")
        rows = rows[known]
        if rows.empty and strategy == "published":
            raise ValueError(f"No labeled BIOSCAN samples in {split}")
        result[split] = _BIOSCANDataset(destination, version, rows, classes, transform, dna_length)
    result["push"] = _BIOSCANDataset(destination, version, result["train"].frame, classes,
                                    push_transform, dna_length)
    if cfg.get("oversample") is not None:
        rows = _balance_training(result["train"].frame, cfg["oversample"], cfg.get("seed", 0))
        result["train"] = _BIOSCANDataset(destination, version, rows, classes, transform, dna_length)
    return result


def bioscan1m(dataset_cfg: dict) -> Dict[str, Dataset]:
    """Return train/push/val/test datasets, downloading missing 1M data.

    Uses data/bioscan_1m by default. Options: root, download, rank
    (order), split_column (large_insect_order), image_size (224), normalize.
    Species defaults to stratified splitting using split ratios and seed.
    Set split_strategy to published to use the original benchmark partitions.
    """
    destination = Path(dataset_cfg.get("root", "data")) / "bioscan_1m"
    files = ["BIOSCAN_Insect_Dataset_metadata.tsv", "cropped_256.hdf5"]
    if dataset_cfg.get("download", True) and not all(
        (destination / name).is_file() for name in files
    ):
        _download(files, destination, repo="bioscan-ml/BIOSCAN-1M")
    return _bioscan(dataset_cfg, "1m", destination)


def bioscan5m(dataset_cfg: dict) -> Dict[str, Dataset]:
    """Return train/push/val/test datasets, downloading missing 5M data.

    Uses data/bioscan_5m by default. Options match bioscan1m except
    split_column. Species stratification pools all labeled samples, including
    pretraining and unseen/key partitions, while preserving image paths.
    Use split_strategy: published to retain the benchmark partitions.
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
