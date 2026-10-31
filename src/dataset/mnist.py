import torch
from typing import Callable, Dict, Tuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Subset

from .split import stratified_indices

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate (image, label) pairs into {"x": (B, 1, H, W), "y": (B,)}."""
  images, labels = zip(*batch)
  return {"x": torch.stack(images), "y": torch.tensor(labels, dtype=torch.long)}

def augmentations(augmentation: dict):
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  Digits are not flip-symmetric (a mirrored 2 is not a digit, and 6/9 collide
  under 180 degrees), so only small label-preserving warps are offered.
  """
  aug = []
  if augmentation.get("rotation"):
    aug.append(transforms.RandomRotation(augmentation["rotation"]))
  if augmentation.get("translate"):
    shift = augmentation["translate"]
    aug.append(transforms.RandomAffine(degrees=0, translate=(shift, shift)))
  if augmentation.get("scale"):
    zoom = augmentation["scale"]
    aug.append(transforms.RandomAffine(degrees=0, scale=(1 - zoom, 1 + zoom)))
  if augmentation.get("shear"):
    aug.append(transforms.RandomAffine(degrees=0, shear=augmentation["shear"]))
  return aug

def mnist(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  # transform and augment
  transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    transforms.ToTensor(),
    transforms.Normalize(dataset_cfg["normalize"]["mean"], dataset_cfg["normalize"]["std"]),
  ])
  train_transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    *augmentations(dataset_cfg["augmentation"]),
    transforms.ToTensor(),
    transforms.Normalize(dataset_cfg["normalize"]["mean"], dataset_cfg["normalize"]["std"]),
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  # Train and val come out of the same published partition, so it is loaded
  # twice: one view augmented for train, one plain for val.
  train_source = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
  val_source = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
  # MNIST publishes only a train/test partition, so val is carved out of
  # train, stratified by digit.
  train_idx, val_idx = stratified_indices(val_source.targets, [train_split, val_split],
                                          dataset_cfg.get("seed", 0))
  train_ds = Subset(train_source, train_idx.tolist())
  val_ds = Subset(val_source, val_idx.tolist())
  test_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

  return {"train" : train_ds, "val": val_ds, "test": test_ds}, collate
