import torch
from typing import Callable, Dict, Tuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate (image, label) pairs into {"x": (B, 3, H, W), "y": (B,)}."""
  images, labels = zip(*batch)
  return {"x": torch.stack(images), "y": torch.tensor(labels, dtype=torch.long)}

def augmentations(augmentation: dict, shape): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  Blooms are photographed at arbitrary roll, so flips and full rotation are
  label-preserving. Keep color_jitter mild: petal colour separates several of
  the 102 species, so heavy jitter erases the signal it is meant to regularise.
  ``crop`` re-crops to ``shape``, which is why the output size is needed here.
  """
  aug = []
  if augmentation.get("crop"): 
    aug.append(transforms.RandomResizedCrop(shape, scale=(augmentation["crop"], 1.0)))
  if augmentation.get("hflip"): 
    aug.append(transforms.RandomHorizontalFlip(p=augmentation["hflip"]))
  if augmentation.get("vflip"): 
    aug.append(transforms.RandomVerticalFlip(p=augmentation["vflip"]))
  if augmentation.get("rotation"): 
    aug.append(transforms.RandomRotation(augmentation["rotation"]))
  if augmentation.get("color_jitter"): 
    j = augmentation["color_jitter"]
    aug.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j, 0.5) / 2))
  return aug

def flowers102(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """
  Needs scipy to load target files
  """
  transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    transforms.ToTensor(),
    transforms.Normalize(dataset_cfg["normalize"]["mean"], dataset_cfg["normalize"]["std"]),
  ])
  train_transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    *augmentations(dataset_cfg["augmentation"], dataset_cfg["shape"]),
    transforms.ToTensor(),
    transforms.Normalize(dataset_cfg["normalize"]["mean"], dataset_cfg["normalize"]["std"]),
  ])

  train_ds = datasets.Flowers102(root='./data', split="train", transform=train_transform, download=True) 
  val_ds = datasets.Flowers102(root='./data', split="val", transform=transform, download=True) 
  test_ds = datasets.Flowers102(root='./data', split="test", transform=transform, download=True) 

  return {"train" : train_ds, "val": val_ds, "test": test_ds}, collate
