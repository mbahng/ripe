import torch
import torch.nn as nn
from torch import Tensor
from ..utils import activation

class CNN(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    
    self.conv_channels = kwargs.get("conv_channels", [])
    self.kernel_sizes = kwargs.get("kernel_sizes", [])
    self.kernel_paddings = kwargs.get("kernel_paddings", [])
    self.poolings = kwargs.get("poolings", [])
    
    self.fc_dimensions = kwargs.get("fc_dimensions", [])
    self.activations = kwargs.get("activations", [])
    self.normalizations = kwargs.get("normalizations", [])
    
    self.layers = nn.ModuleList()
    
    # Build Convolutional Layers
    num_conv_layers = len(self.conv_channels) - 1
    for i in range(num_conv_layers):
      in_c = self.conv_channels[i]
      out_c = self.conv_channels[i+1]
      k_size = self.kernel_sizes[i]
      padding = self.kernel_paddings[i]
      
      self.layers.append(nn.Conv2d(in_c, out_c, kernel_size=k_size, padding=padding))
      
      if i < len(self.normalizations) and self.normalizations[i]:
        self.layers.append(nn.BatchNorm2d(out_c))
          
      if i < len(self.activations):
        self.layers.append(activation(self.activations[i]))
          
      if i < len(self.poolings) and self.poolings[i] > 1:
         self.layers.append(nn.MaxPool2d(kernel_size=self.poolings[i], stride=self.poolings[i]))

    # Build Fully Connected Layers
    self.fc_layers = nn.ModuleList()
    
    # Use fc_dimensions directly (User must ensure the first dim matches the flattened conv output)
    dims = self.fc_dimensions
    
    # The activation/norm indices continue from where conv left off
    start_idx = num_conv_layers 
    
    for i in range(len(dims) - 1):
      in_d = dims[i]
      out_d = dims[i+1]
      
      self.fc_layers.append(nn.Linear(in_d, out_d))
      
      current_idx = start_idx + i
      
      if current_idx < len(self.normalizations) and self.normalizations[current_idx]:
        self.fc_layers.append(nn.BatchNorm1d(out_d))
          
      if current_idx < len(self.activations):
        self.fc_layers.append(activation(self.activations[current_idx]))

  def forward(self, x: Tensor):
    for layer in self.layers:
      x = layer(x)
        
    x = torch.flatten(x, start_dim=1)
    
    for layer in self.fc_layers:
      x = layer(x)
        
    return x


