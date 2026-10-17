import os
import scipy.io
import torch
from PIL import Image
from typing import Callable, Dict, Tuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset

from .split import stratified_indices
import kagglehub
import shutil
from pathlib import Path

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate (image, label) pairs into {"x": (B, 3, 224, 224), "y": (B,)}."""
  images, labels = zip(*batch)
  return {"x": torch.stack(images), "y": torch.tensor(labels, dtype=torch.long)}

def augmentations(augmentation: dict, shape): 
  """
  Online augmentation: these are random transforms, so they are redrawn on
  every __getitem__ and each epoch sees a different view of the same image.
  Birds are photographed upright, so hflip is safe but vflip is not; the
  images are already bounding-box cropped, so warps are kept mild.
  ``crop`` re-crops to ``shape``, which is why the output size is needed here.
  """
  aug = []
  if augmentation.get("crop"): 
    aug.append(transforms.RandomResizedCrop(shape, scale=(augmentation["crop"], 1.0)))
  if augmentation.get("hflip"): 
    aug.append(transforms.RandomHorizontalFlip(p=augmentation["hflip"]))
  if augmentation.get("rotation"): 
    aug.append(transforms.RandomRotation(augmentation["rotation"]))
  if augmentation.get("shear"): 
    aug.append(transforms.RandomAffine(degrees=0, shear=augmentation["shear"]))
  if augmentation.get("perspective"): 
    aug.append(transforms.RandomPerspective(distortion_scale=augmentation["perspective"], p=0.5))
  if augmentation.get("color_jitter"): 
    j = augmentation["color_jitter"]
    aug.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(j, 0.5) / 2))
  return aug

def cub200(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:

  # download dataset
  if not os.path.exists("data/cub200/original"):
    data_path = kagglehub.dataset_download("wenewone/cub2002011")
    src_path = os.path.join(data_path, "CUB_200_2011")
    target_path = "./data/cub200/original"

    os.makedirs("data/cub200", exist_ok=True)
    if os.path.exists(src_path):
      shutil.copytree(src_path, target_path)
    elif os.path.exists(os.path.join(data_path, "images")):
      shutil.copytree(data_path, target_path)
    else:
      raise FileNotFoundError(f"Expected 'CUB_200_2011' folder or 'images' in {data_path}, found: {os.listdir(data_path)}")

  # crop and make train/validation/test split
  if not os.path.exists("data/cub200/cropped"):
    dataset_root = Path("./data/cub200/original")
    output_root = Path("./data/cub200/cropped")

    image_paths = {
      int(img_id): img_path
      for line in (dataset_root / 'images.txt').read_text().splitlines()
      for img_id, img_path in [line.split()]
    }

    bboxes = {}
    for line in (dataset_root / 'bounding_boxes.txt').read_text().splitlines():
      img_id, *coords = line.split()
      bboxes[int(img_id)] = tuple(map(float, coords[:4]))

    split_info, train_ids = {}, []
    for line in (dataset_root / 'train_test_split.txt').read_text().splitlines():
      img_id, is_train = line.split()
      split_info[int(img_id)] = int(is_train)
      if int(is_train) == 1:
        train_ids.append(int(img_id))

    train_split, val_split, _ = dataset_cfg["split"]

    # The upstream train/test partition is published, but the val carve-out is
    # ours, so stratify it by species: with ~30 training images per class a
    # uniformly random carve-out leaves some species thin in val and others
    # absent. Label is the leading directory of the image path.
    species = [image_paths[img_id].split('/')[0] for img_id in train_ids]
    _, val_positions = stratified_indices(species, [train_split, val_split],
                                          dataset_cfg["seed"])
    val_ids = {train_ids[position] for position in val_positions}

    splits = {s: output_root / s for s in ('train', 'val', 'test')}
    for d in splits.values():
      d.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / 'images'
    counts = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}

    for img_id, img_path in sorted(image_paths.items()):
      full_path = images_dir / img_path
      if not full_path.exists():
        counts['errors'] += 1
        continue
      try:
        img = Image.open(full_path).convert('RGB')
        x, y, w, h = bboxes[img_id]
        iw, ih = img.size
        box = (max(0, int(x)), max(0, int(y)), min(iw, int(x + w)), min(ih, int(y + h)))
        cropped = img.crop(box)

        if split_info[img_id] == 1:
          split = 'val' if img_id in val_ids else 'train'
        else:
          split = 'test'

        out_dir = splits[split] / img_path.split('/')[0]
        out_dir.mkdir(exist_ok=True)
        cropped.save(out_dir / Path(img_path).name)
        counts[split] += 1
      except Exception as e:
        print(f"Error processing {full_path}: {e}")
        counts['errors'] += 1

    print(f"Done: train={counts['train']}, val={counts['val']}, test={counts['test']}, errors={counts['errors']}")

  normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225))

  train_transform = transforms.Compose([
      transforms.Resize(dataset_cfg["shape"]),
      *augmentations(dataset_cfg["augmentation"], dataset_cfg["shape"]),
      transforms.ToTensor(),
      normalize,
  ])

  train_ds = datasets.ImageFolder("./data/cub200/cropped/train/", train_transform)

  train_push_ds = datasets.ImageFolder(
      "./data/cub200/cropped/train/",
      transforms.Compose([
          transforms.Resize(dataset_cfg["shape"]),
          transforms.ToTensor(),
      ]))

  val_ds = datasets.ImageFolder(
      "./data/cub200/cropped/val/",
      transforms.Compose([
          transforms.Resize(dataset_cfg["shape"]),
          transforms.ToTensor(),
          normalize,
      ]))

  test_ds = datasets.ImageFolder(
      "./data/cub200/cropped/test/",
      transforms.Compose([
          transforms.Resize(dataset_cfg["shape"]),
          transforms.ToTensor(),
          normalize,
      ]))

  return ({"train" : train_ds, "push": train_push_ds, "val": val_ds, "test": test_ds},
          collate)

