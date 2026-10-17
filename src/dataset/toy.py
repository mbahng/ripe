import torch
import math
from typing import Callable, Dict, Tuple
from torch.utils.data import TensorDataset, Dataset
from sklearn.datasets import make_s_curve, make_moons, make_circles

__all__ = ["two_gaussians", "four_gaussians", "moons", "circles", "s_curve"]

def collate(batch) -> Dict[str, torch.Tensor]:
  """Collate single-tensor samples into {"x": (B, D)}.

  The toy distributions are unlabeled: every factory below wraps points in a
  one-element ``TensorDataset``, so each sample arrives as a ``(point,)``
  tuple and no "y" key is produced.
  """
  return {"x": torch.stack([point for (point,) in batch])}

def _split(ds, dataset_cfg: dict) -> Dict[str, Dataset]:
  train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, dataset_cfg["split"])
  return {"train": train_ds, "val": val_ds, "test": test_ds}

def two_gaussians(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  N = 10000
  variance = 0.8
  gaussian1 = 4 + math.sqrt(variance) * torch.randn(N, 2)
  gaussian2 = -4 + (1/math.sqrt(variance)) * torch.randn(N, 2)
  mask = (torch.rand(N) > 0.5).unsqueeze(1).expand(N, 2)
  Y = mask * gaussian1 + (~mask) * gaussian2
  ds = TensorDataset(Y)
  return _split(ds, dataset_cfg), collate

def four_gaussians(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """
  Toy dataset with 4 Gaussian components arranged in a ring pattern
  """
  N = 10000
  n_per_mode = N // 5

  # Create 4 Gaussians in a ring pattern
  angles = torch.tensor([0, math.pi/2, math.pi, 3*math.pi/2])
  radius = 5.0
  ring_data = []

  for angle in angles:
    center_x = radius * math.cos(angle)
    center_y = radius * math.sin(angle)
    center = torch.tensor([center_x, center_y])
    samples = center + 0.5 * torch.randn(n_per_mode, 2)
    ring_data.append(samples)

  # Combine all samples
  Y = torch.cat(ring_data, dim=0)

  # Shuffle the data
  perm = torch.randperm(Y.size(0))
  Y = Y[perm]

  ds = TensorDataset(Y)
  return _split(ds, dataset_cfg), collate

def moons(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """
  Toy dataset consisting of two cresent moons
  """
  N=10000
  noise=0.05
  Y, _ = make_moons(N, noise=noise)
  Y = torch.tensor(4 * Y, dtype=torch.float)
  ds = TensorDataset(Y)
  return _split(ds, dataset_cfg), collate


def circles(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  """
  Toy dataset consisting of two concentric circles.
  Factor = scale between inner vs outer circle.
  """
  N=10000
  scale=0.5
  noise=0.05
  Y, _ = make_circles(N, factor=scale, noise=noise)
  Y = torch.tensor(4 * Y, dtype=torch.float)
  ds = TensorDataset(Y)
  return _split(ds, dataset_cfg), collate

def s_curve(dataset_cfg: dict) -> Tuple[Dict[str, Dataset], Callable]:
  N = 100
  noise=0.05
  Y, _ = make_s_curve(N, noise=noise)
  Y = torch.tensor(4 * Y, dtype=torch.float)
  ds = TensorDataset(Y)
  return _split(ds, dataset_cfg), collate
