# toy datasets
from .toy import *

# vision
from .mnist import mnist 
from .fashion_mnist import fashion_mnist 
from .cifar10 import cifar10 
from .cifar100 import cifar100 
from .svhn import svhn
from .celeba import celebA
from .flowers102 import flowers102
from .cars import cars
from .inaturalist import inaturalist 
from .cub200 import cub200 
from .bioscan import bioscan1m, bioscan5m
from .canadian import canadian_invertebrates

# language 
from .shakespeare import shakespeare 

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from typing import Callable, Dict, Tuple
from torch.utils.data import Subset


def init_dataset(cfg_dataset: dict) -> Tuple[Dict[str, Dataset], Callable]:  
  """
  Returns the train/val/test split of the dataset we want to work with, plus
  the collate function that dataset's factory supplies.

  Every factory returns ``(splits, collate)``. The collate is dataset-specific
  because item layouts differ (positional tuples, token pairs, image+DNA
  dicts), but all of them emit a batch dict whose "x" is the model input and
  whose "y" is the target, so trainers stay dataset-agnostic. Paired-modality
  datasets add further keys, e.g. BIOSCAN and Canadian carry the one-hot
  barcode under "gen"; unlabeled toy datasets omit "y".
  """
  match cfg_dataset["name"]: 
    case "mnist": 
      print("Loading dataset: MNIST")
      result = mnist(cfg_dataset) 
    case "fashion_mnist": 
      print("Loading dataset: Fashion MNIST")
      result = fashion_mnist(cfg_dataset) 
    case "cifar10":
      print("Loading dataset: CIFAR10")
      result = cifar10(cfg_dataset) 
    case "cifar100":
      print("Loading dataset: CIFAR100")
      result = cifar100(cfg_dataset) 
    case "svhn": 
      print("Loading dataset: SVHN")
      result = svhn(cfg_dataset) 
    case "celeba": 
      print("Loading dataset: CelebA")
      result = celebA(cfg_dataset) 
    case "flowers102": 
      print("Loading dataset: Flowers102")
      result = flowers102(cfg_dataset)
    case "two_gaussians": 
      print("Loading dataset: Two Gaussians")
      result = two_gaussians(cfg_dataset)
    case "four_gaussians": 
      print("Loading dataset: Four Gaussians")
      result = four_gaussians(cfg_dataset)
    case "moons": 
      print("Loading dataset: Moons")
      result = moons(cfg_dataset)
    case "circles": 
      print("Loading dataset: Concentric Circles")
      result = circles(cfg_dataset)
    case "s_curve": 
      print("Loading dataset: S Curve")
      result = s_curve(cfg_dataset)
    case "cars": 
      print("Loading dataset: Stanford Cars")
      result = cars(cfg_dataset)
    case "shakespeare":
      print("Loading dataset: Shakespeare")
      result = shakespeare(cfg_dataset)
    case "cub200": 
      print("Loading dataset: CUB200")
      result = cub200(cfg_dataset)
    case "inaturalist": 
      print("Loading dataset: iNaturalist")
      result = inaturalist(cfg_dataset) 
    case "bioscan1m" | "bioscan_1m": 
      print("Loading dataset: Bioscan 1M")
      result = bioscan1m(cfg_dataset) 
    case "bioscan5m" | "bioscan_5m": 
      print("Loading dataset: Bioscan 5M")
      result = bioscan5m(cfg_dataset) 
    case "caninv" | "canadian_invertebrates": 
      print("Loading dataset: Canadian Invertebrates")
      result = canadian_invertebrates(cfg_dataset) 
    case _: 
      raise Exception("Not a valid dataset.")

  datasets, collate = result

  subsample = cfg_dataset["train"]["subsample"] 
  if not isinstance(subsample, (int, float)) or not 0.0 < subsample <= 1.0: 
    raise ValueError("Not valid subsampling.") 

  splits = {}
  # Iterate the factory's own keys so extra splits (e.g. "push") survive.
  for split, ds in datasets.items(): 
    samples = int(subsample * len(ds)) 
    splits[split] = ds if subsample == 1.0 else Subset(ds, range(samples)) 

  return splits, collate

def init_dataloader(cfg_dataset: dict) -> Dict[str, DataLoader]:
  ds_dict, collate = init_dataset(cfg_dataset) 
  train_ds, val_ds, test_ds = ds_dict["train"], ds_dict["val"], ds_dict["test"]

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

  return {
      "train" : train_dl, 
      "val" : val_dl, 
      "test" : test_dl
      }
