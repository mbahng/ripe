import torch
from copy import deepcopy

class EMA(torch.nn.Module):
    def __init__(self, model, beta=0.995):
        super().__init__()
        self.beta = beta
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        for current_param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            ema_param.data = self.beta * ema_param.data + (1.0 - self.beta) * current_param.data

def activation(name): 
  match name.lower():
    case 'relu':
      return torch.nn.ReLU()
    case 'sigmoid':
      return torch.nn.Sigmoid()
    case 'tanh':
      return torch.nn.Tanh()
    case 'leaky_relu':
      return torch.nn.LeakyReLU()
    case 'elu':
      return torch.nn.ELU()
    case 'selu':
      return torch.nn.SELU()
    case 'gelu':
      return torch.nn.GELU()
    case 'softplus':
      return torch.nn.Softplus()
    case 'softmax':
      return torch.nn.Softmax(dim=1)
    case '': 
      return torch.nn.Identity()
    case _:
      raise ValueError(f"Activation function '{name}' not recognized.")

def one_hot(labels: torch.Tensor, num_classes: int): 
  """int -> vector"""
  return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
