import torch.nn as nn
import torch.nn.functional as F

def init_loss(cfg_loss: dict):
  
  match cfg_loss["name"]:
    case 'mse' | 'l2':
      return MSE()
    case 'cross_entropy' | 'ce':
      return CrossEntropy()
    case _:
      raise Exception("Loss not implemented.")


class Loss(nn.Module):
  """Base loss class"""
  def __init__(self):
    super().__init__()
  
  def forward(self, pred, target):
    raise NotImplementedError("Subclass must implement forward method")

class MSE(Loss):
  """Mean Squared Error Loss (L2 Loss)"""
  def __init__(self, reduction='sum'):
    super().__init__()
    self.reduction = reduction
  
  def forward(self, pred, target):
    return F.mse_loss(pred, target, reduction=self.reduction)


class CrossEntropy(Loss):
  """Cross Entropy Loss for classification"""
  def __init__(self, weight=None, reduction='sum', label_smoothing=0.0):
    super().__init__()
    self.weight = weight
    self.reduction = reduction
    self.label_smoothing = label_smoothing
  
  def forward(self, pred, target):
    return F.cross_entropy(pred, target, weight=self.weight, reduction=self.reduction)
