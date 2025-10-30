from .toy import * 
from .vision import *
from .language import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from typing import Tuple, Dict
from torch.utils.data import Subset


def init_dataset(cfg_dataset: dict) -> Dict[str, Dataset]:  
  """
  Returns the train/val/test split of the dataset we want to work with.
  """
  match cfg_dataset["name"]: 
    case "mnist": 
      print("Loading dataset: MNIST")
      datasets = mnist(cfg_dataset) 
    case "fashion_mnist": 
      print("Loading dataset: Fashion MNIST")
      datasets = fashion_mnist(cfg_dataset) 
    case "cifar10":
      print("Loading dataset: CIFAR10")
      datasets = cifar10(cfg_dataset) 
    case "cifar100":
      print("Loading dataset: CIFAR100")
      datasets = cifar100(cfg_dataset) 
    case "svhn": 
      print("Loading dataset: SVHN")
      datasets = svhn(cfg_dataset) 
    case "celeba": 
      print("Loading dataset: CelebA")
      datasets = celebA(cfg_dataset) 
    case "flowers102": 
      print("Loading dataset: Flowers102")
      datasets = flowers102(cfg_dataset)
    case "two_gaussians": 
      print("Loading dataset: Two Gaussians")
      datasets = two_gaussians(cfg_dataset)
    case "four_gaussians": 
      print("Loading dataset: Four Gaussians")
      datasets = four_gaussians(cfg_dataset)
    case "moons": 
      print("Loading dataset: Moons")
      datasets = moons(cfg_dataset)
    case "circles": 
      print("Loading dataset: Concentric Circles")
      datasets = circles(cfg_dataset)
    case "s_curve": 
      print("Loading dataset: S Curve")
      datasets = s_curve(cfg_dataset)
    case "cars": 
      print("Loading dataset: Stanford Cars")
      datasets = cars(cfg_dataset)
    case "shakespeare":
      print("Loading dataset: Shakespeare")
      datasets = shakespeare(cfg_dataset)
    case "cub200": 
      print("Loading dataset: CUB200")
      datasets = cub200(cfg_dataset)
    case "inaturalist": 
      print("Loading dataset: iNaturalist")
      datasets = inaturalist(cfg_dataset)
    case _: 
      raise Exception("Not a valid dataset.")

  train_ds = datasets["train"]
  val_ds = datasets["val"]
  test_ds = datasets["test"]

  # subsample 
  train_samples = cfg_dataset["train"]["subsample"]
  val_samples = cfg_dataset["val"]["subsample"] 
  test_samples = cfg_dataset["test"]["subsample"]
  if isinstance(train_samples, float): 
    train_samples = int(train_samples * len(train_ds))
  if isinstance(val_samples, float): 
    val_samples = int(val_samples * len(val_ds))
  if isinstance(test_samples, float): 
    test_samples = int(test_samples * len(test_ds))
  train_ds = Subset(train_ds, range(train_samples))
  val_ds = Subset(val_ds, range(val_samples))
  test_ds = Subset(test_ds, range(test_samples)) 

  return train_ds, val_ds, test_ds

def collate(batch):
  """
  Standardizes batches from different dataset formats into a unified dictionary.
  Handles:
    - (image, label) tuples
    - (image,) single-item tuples
    - raw image tensors
  """
  images = []
  labels = []
  
  # Check if the first sample is a sequence AND has at least 2 elements
  has_labels = isinstance(batch[0], (tuple, list)) and len(batch[0]) > 1

  for item in batch:
    if has_labels:
      # Assume format is (image, label, ...)
      images.append(item[0])
      labels.append(item[1])
    else:
      # Handle (image,) or raw tensor
      if isinstance(item, (tuple, list)):
        images.append(item[0])
      else:
        images.append(item)

  res = {"x": torch.stack(images)}
  
  if has_labels:
    if isinstance(labels[0], torch.Tensor):
      res["y"] = torch.stack(labels)
    else:
      res["y"] = torch.tensor(labels)
  
  return res

def init_dataloader(cfg_dataset: dict) -> Tuple[DataLoader, DataLoader, DataLoader]: 
  train_ds, val_ds, test_ds = init_dataset(cfg_dataset)

  if cfg_dataset["is_distributed"]: 
    if cfg_dataset["train"]["shuffle"]: 
      UserWarning("You have turned shuffle on when using distributed. Shuffle will be forced off.")
      cfg_dataset["train"]["shuffle"] = False

    train_dl = DataLoader(train_ds, 
                          batch_size=cfg_dataset["train"]["batch_size"], 
                          shuffle=cfg_dataset["train"]["shuffle"], 
                          num_workers=cfg_dataset["train"]["num_workers"],
                          sampler=DistributedSampler(train_ds, seed=cfg_dataset["seed"]),
                          collate_fn=collate)
    val_dl = DataLoader(val_ds, 
                        batch_size=cfg_dataset["val"]["batch_size"], 
                        num_workers=cfg_dataset["val"]["num_workers"],
                        sampler=DistributedSampler(val_ds, seed=cfg_dataset["seed"]),
                        collate_fn=collate)
    test_dl = DataLoader(test_ds, 
                         batch_size=cfg_dataset["test"]["batch_size"], 
                         num_workers=cfg_dataset["test"]["num_workers"],
                         sampler=DistributedSampler(test_ds, seed=cfg_dataset["seed"]),
                         collate_fn=collate)
  else: 
    train_dl = DataLoader(train_ds, 
                          batch_size=cfg_dataset["train"]["batch_size"], 
                          shuffle=cfg_dataset["train"]["shuffle"], 
                          num_workers=cfg_dataset["train"]["num_workers"], 
                          collate_fn=collate)
    val_dl = DataLoader(val_ds, 
                        batch_size=cfg_dataset["val"]["batch_size"], 
                        num_workers=cfg_dataset["val"]["num_workers"], 
                        collate_fn=collate)
    test_dl = DataLoader(test_ds, 
                         batch_size=cfg_dataset["test"]["batch_size"], 
                         num_workers=cfg_dataset["test"]["num_workers"], 
                         collate_fn=collate)

  return train_dl, val_dl, test_dl
