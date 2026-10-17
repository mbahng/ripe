import os
import scipy
import torch
from PIL import Image
from typing import Callable, Dict, Tuple
from torchvision import transforms
from torch.utils.data import Dataset, Subset

from .split import stratified_indices
import kagglehub
import shutil

class StanfordCarsDataset(Dataset):
  def __init__(self, root, split="train", transform=None):
    self.root = root
    self.split = split
    self.transform = transform
    
    # Paths
    devkit = os.path.join(root, "devkit")
    if split == "train":
      self.img_dir = os.path.join(root, "cars_train")
      mat_path = os.path.join(devkit, "cars_train_annos.mat")
    else:
      self.img_dir = os.path.join(root, "cars_test")
      mat_path = os.path.join(devkit, "cars_test_annos.mat")
        
    self.samples = []
    if os.path.exists(mat_path):
      mat = scipy.io.loadmat(mat_path)
      annotations = mat["annotations"][0]
      for ann in annotations:
        fname = str(ann['fname'][0])
        if 'class' in ann.dtype.names:
          label = int(ann['class'][0, 0]) - 1
        else:
          label = -1
        self.samples.append((fname, label))
    # Mirrors torchvision's convention, and lets the split read labels without
    # decoding every JPEG.
    self.targets = [label for _, label in self.samples]

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    fname, label = self.samples[idx]
    path = os.path.join(self.img_dir, fname)
    image = Image.open(path).convert("RGB")
    
    if self.transform:
      image = self.transform(image)
        
    return image, label

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate (image, label) pairs into {"x": (B, 3, 256, 256), "y": (B,)}.

  Test-split annotations carry no class, so those labels arrive as -1.
  """
  images, labels = zip(*batch)
  return {"x": torch.stack(images), "y": torch.tensor(labels, dtype=torch.long)}

def augmentations(augmentation: dict, shape): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  A mirrored car is still the same model, so hflip is safe. Crop matters here
  because the stock frames are loosely cropped around the vehicle, and jitter
  helps because paint colour is not part of the label.
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

def cars(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]: 
  # download dataset 
  if not os.path.exists("data/cars"): 
    data_path = kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset")
    os.mkdir("data/cars")
    shutil.move(f"{data_path}/car_devkit/devkit", "./data/cars/")
    shutil.move(f"{data_path}/cars_train/cars_train", "./data/cars/")
    shutil.move(f"{data_path}/cars_test/cars_test", "./data/cars/")

  # transform and augment
  transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    transforms.ToTensor(), # scales data to [0, 1]
  ])
  train_transform = transforms.Compose([
    transforms.Resize(dataset_cfg["shape"]),
    *augmentations(dataset_cfg["augmentation"], dataset_cfg["shape"]),
    transforms.ToTensor(),
  ])

  # Train and val come out of the same published partition, so it is loaded
  # twice: one view augmented for train, one plain for val.
  train_source = StanfordCarsDataset(root='data/cars', split='train', transform=train_transform)
  val_source = StanfordCarsDataset(root='data/cars', split='train', transform=transform)

  # split: the devkit's test annotations carry no class, so that split is
  # unusable as a labelled set; val is carved out of train, stratified by
  # model. At ~40 images per class a random carve-out skews badly.
  train_split, val_split, _ = dataset_cfg["split"]
  train_idx, val_idx = stratified_indices(val_source.targets, [train_split, val_split],
                                          dataset_cfg.get("seed", 0))
  train_ds = Subset(train_source, train_idx.tolist())
  val_ds = Subset(val_source, val_idx.tolist())
  test_ds = StanfordCarsDataset(root='data/cars', split='test', transform=transform)
 
  return {"train" : train_ds, "val": val_ds, "test": test_ds}, collate

