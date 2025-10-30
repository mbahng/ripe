import torch
import torch.nn as nn
from torch import Tensor
from ..utils import activation

class MLP(nn.Module): 

  def __init__(self, dimensions, normalizations, activations):
    super().__init__()

    self.model = nn.Sequential() 
    for i in range(len(dimensions) - 1): 
      self.model.append(nn.Linear(dimensions[i], dimensions[i+1])) 
      if normalizations[i]: 
        self.model.append(nn.BatchNorm1d(dimensions[i+1]) )
      self.model.append(activation(activations[i]))

    self._init_weights()
    self.device = None

  def forward(self, x: Tensor): 
    x = torch.flatten(x, start_dim=1)
    return self.model(x)

  def _init_weights(self): 
    ...
