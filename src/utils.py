import torch

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
