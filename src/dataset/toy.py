import torch
import math
from torch.utils.data import TensorDataset
from sklearn.datasets import make_s_curve, make_moons, make_circles

def two_gaussians(dataset_cfg: dict):
  N = 10000
  variance = 0.8
  gaussian1 = 4 + math.sqrt(variance) * torch.randn(N, 2)
  gaussian2 = -4 + (1/math.sqrt(variance)) * torch.randn(N, 2)
  mask = (torch.rand(N) > 0.5).unsqueeze(1).expand(N, 2)
  Y = mask * gaussian1 + (~mask) * gaussian2
  ds = TensorDataset(Y) 
  train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, dataset_cfg["split"])
  return train_ds, val_ds, test_ds

def four_gaussians(dataset_cfg: dict):
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
  train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, dataset_cfg["split"])
  return train_ds, val_ds, test_ds

def moons(dataset_cfg: dict): 
  """
  Toy dataset consisting of two cresent moons
  """
  N=10000
  noise=0.05
  Y, _ = make_moons(N, noise=noise) 
  Y = torch.tensor(4 * Y, dtype=torch.float)
  ds = TensorDataset(Y)
  train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, dataset_cfg["split"])
  return train_ds, val_ds, test_ds


def circles(dataset_cfg: dict): 
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
  train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, dataset_cfg["split"])
  return train_ds, val_ds, test_ds

def s_curve(dataset_cfg: dict): 
  N = 100
  noise=0.05
  Y, _ = make_s_curve(N, noise=noise) 
  Y = torch.tensor(4 * Y, dtype=torch.float) 
  ds = TensorDataset(Y)
  train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, dataset_cfg["split"])
  return train_ds, val_ds, test_ds

