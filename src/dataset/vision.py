import os
import scipy.io
from PIL import Image
from typing import Dict
from torchvision import transforms, datasets
from torch.utils.data import random_split, Dataset

def makedir(path):
  """Create directory if it doesn't exist."""
  if not os.path.exists(path):
      os.makedirs(path)

def mnist(dataset_cfg: dict) -> Dict[str, Dataset]: 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"] 
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True) 

  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def fashion_mnist(dataset_cfg: dict) -> Dict[str, Dataset]:
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"] 
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True) 

  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def cifar10(dataset_cfg: dict) -> Dict[str, Dataset]: 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transform, download=True)
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform, download=True) 

  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def cifar100(dataset_cfg: dict) -> Dict[str, Dataset]: 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.CIFAR100(root='./data/CIFAR100', train=True, transform=transform, download=True)
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.CIFAR100(root='./data/CIFAR100', train=False, transform=transform, download=True) 
 
  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def cars(dataset_cfg: dict) -> Dict[str, Dataset]: 
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

    def __len__(self):
      return len(self.samples)

    def __getitem__(self, idx):
      fname, label = self.samples[idx]
      path = os.path.join(self.img_dir, fname)
      image = Image.open(path).convert("RGB")
      
      if self.transform:
        image = self.transform(image)
          
      return image, label

  # download dataset 
  if not os.path.exists("data/cars"): 
    data_path = kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset")
    os.mkdir("data/cars")
    shutil.move(f"{data_path}/car_devkit/devkit", "./data/cars/")
    shutil.move(f"{data_path}/cars_train/cars_train", "./data/cars/")
    shutil.move(f"{data_path}/cars_test/cars_test", "./data/cars/")

  # transform and augment
  transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # scales data to [0, 1]
  ])

  full_train_ds = StanfordCarsDataset(root='data/cars', split='train', transform=transform)

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  
  train_ds, val_ds = random_split(full_train_ds, [train_split, val_split])
  test_ds = StanfordCarsDataset(root='data/cars', split='test', transform=transform)
 
  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def svhn(dataset_cfg: dict) -> Dict[str, Dataset]: 
  """
  There is also an "extra" split for SVHN
  """
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split, val_split = train_split / total_split, val_split / total_split
  ds = datasets.SVHN(root='./data/svhn', split='train', transform=transform, download=True)
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.SVHN(root='./data/svhn', split='test', transform=transform, download=True) 

  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def celebA(dataset_cfg: dict) -> Dict[str, Dataset]: 
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  train_ds = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
  val_ds = datasets.CelebA(root='./data', split='valid', transform=transform, download=True)
  test_ds = datasets.CelebA(root='./data', split='test', transform=transform, download=True)

  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def flowers102(dataset_cfg: dict) -> Dict[str, Dataset]:
  """
  Needs scipy to load target files
  """
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])


  train_ds = datasets.Flowers102(root='./data', split="train", transform=transform, download=True) 
  val_ds = datasets.Flowers102(root='./data', split="val", transform=transform, download=True) 
  test_ds = datasets.Flowers102(root='./data', split="test", transform=transform, download=True) 

  return {"train" : train_ds, "val": val_ds, "test": test_ds}

def cub200(dataset_cfg: dict):
  import kagglehub
  import shutil
  import random
  from pathlib import Path

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
    random.seed(dataset_cfg["seed"])
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

    random.shuffle(train_ids)
    val_ids = set(train_ids[:int(len(train_ids) * (val_split / (train_split + val_split)))])

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

  from torch.utils.data import ConcatDataset

  train_transform = transforms.Compose([
      transforms.Resize(size=(224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=10),
      transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
      transforms.RandomAffine(degrees=0, shear=10),
      transforms.ToTensor(),
      normalize,
  ])

  train_ds = ConcatDataset([
      datasets.ImageFolder("./data/cub200/cropped/train/", train_transform)
      for _ in range(30)
  ])

  train_push_ds = datasets.ImageFolder(
      "./data/cub200/cropped/train/",
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
      ]))

  val_ds = datasets.ImageFolder(
      "./data/cub200/cropped/val/",
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
          normalize,
      ]))

  test_ds = datasets.ImageFolder(
      "./data/cub200/cropped/test/",
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
          normalize,
      ]))

  return {"train" : train_ds, "push": train_push_ds, "val": val_ds, "test": test_ds}

def inaturalist(dataset_cfg: dict):
  # transform and augment
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_p = train_split / total_split
  val_p = val_split / total_split

  # Using '2021_train_mini' as a default for training/validation
  # and '2021_valid' for testing.
  full_train_ds = datasets.INaturalist(root='./data/iNaturalist', version='2021_train_mini', transform=transform, download=True)
  train_ds, val_ds = random_split(full_train_ds, [train_p, val_p])
  test_ds = datasets.INaturalist(root='./data/iNaturalist', version='2021_valid', transform=transform, download=True)

  return {"train" : train_ds, "push": train_ds, "val": val_ds, "test": test_ds}
