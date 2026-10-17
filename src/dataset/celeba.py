import torch
from typing import Callable, Dict, Tuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate (image, attributes) pairs into {"x": (B, 3, H, W), "y": (B, 40)}.

  CelebA's default ``target_type`` is "attr", so each target is already a
  40-dim multi-hot tensor rather than a scalar class index.
  """
  images, targets = zip(*batch)
  return {"x": torch.stack(images), "y": torch.stack(targets)}

def augmentations(augmentation: dict): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  Faces are near mirror-symmetric, so hflip is safe; note that it does flip
  any left/right-specific attribute, and the crops are already face-aligned,
  so translate and rotation are kept small by convention.
  """
  aug = []
  if augmentation.get("hflip"): 
    aug.append(transforms.RandomHorizontalFlip(p=augmentation["hflip"]))
  if augmentation.get("rotation"): 
    aug.append(transforms.RandomRotation(augmentation["rotation"]))
  if augmentation.get("translate"): 
    shift = augmentation["translate"]
    aug.append(transforms.RandomAffine(degrees=0, translate=(shift, shift)))
  if augmentation.get("color_jitter"): 
    j = augmentation["color_jitter"]
    aug.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j, 0.5) / 2))
  return aug

def celebA(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]: 
  transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    transforms.ToTensor(),
  ])
  train_transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    *augmentations(dataset_cfg["augmentation"]),
    transforms.ToTensor(),
  ])

  train_ds = datasets.CelebA(root='./data', split='train', transform=train_transform, download=True)
  val_ds = datasets.CelebA(root='./data', split='valid', transform=transform, download=True)
  test_ds = datasets.CelebA(root='./data', split='test', transform=transform, download=True)

  return {"train" : train_ds, "val": val_ds, "test": test_ds}, collate
