import torch
from typing import Callable, Dict, Tuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Subset

from .split import stratified_indices

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate (image, category_id) pairs into {"x": (B, 3, H, W), "y": (B,)}."""
  images, labels = zip(*batch)
  return {"x": torch.stack(images), "y": torch.tensor(labels, dtype=torch.long)}

def augmentations(augmentation: dict, shape): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  These are field photographs of organisms in situ, so framing and lighting
  vary wildly and crop plus jitter are the useful knobs; hflip is safe, but
  vflip is not offered because the horizon is a real cue.
  ``crop`` re-crops to ``shape``, which is why the output size is needed here.
  """
  aug = []
  if augmentation.get("crop"): 
    aug.append(transforms.RandomResizedCrop(shape, scale=(augmentation["crop"], 1.0)))
  if augmentation.get("hflip"): 
    aug.append(transforms.RandomHorizontalFlip(p=augmentation["hflip"]))
  if augmentation.get("rotation"): 
    aug.append(transforms.RandomRotation(augmentation["rotation"]))
  if augmentation.get("color_jitter"): 
    j = augmentation["color_jitter"]
    aug.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j, 0.5) / 2))
  return aug

def inaturalist(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  # transform and augment
  transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    transforms.ToTensor(),
    transforms.Normalize(dataset_cfg["normalize"]["mean"], dataset_cfg["normalize"]["std"])
  ])
  train_transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    *augmentations(dataset_cfg["augmentation"], dataset_cfg["shape"]),
    transforms.ToTensor(),
    transforms.Normalize(dataset_cfg["normalize"]["mean"], dataset_cfg["normalize"]["std"])
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]

  # Using '2021_train_mini' as a default for training/validation
  # and '2021_valid' for testing. Train and val come out of that one
  # partition, so it is loaded twice: augmented for train, plain for val.
  train_source = datasets.INaturalist(root='./data/iNaturalist', version='2021_train_mini', transform=train_transform, download=True)
  val_source = datasets.INaturalist(root='./data/iNaturalist', version='2021_train_mini', transform=transform, download=True)
  # The mini split has no published val partition, so one is carved out of it,
  # stratified by category. ``index`` pairs each file with its category id, so
  # labels are read from metadata rather than from the images.
  targets = [category_id for category_id, _ in val_source.index]
  train_idx, val_idx = stratified_indices(targets, [train_split, val_split],
                                          dataset_cfg.get("seed", 0))
  train_ds = Subset(train_source, train_idx.tolist())
  val_ds = Subset(val_source, val_idx.tolist())
  test_ds = datasets.INaturalist(root='./data/iNaturalist', version='2021_valid', transform=transform, download=True)

  return {"train" : train_ds, "val": val_ds, "test": test_ds}, collate
