import torch
import torch.nn as nn
from ..utils import activation

class GAN(nn.Module): 
  """Wrapper class around both discriminator and generator"""

  def __init__(self, **kwargs): 
    super().__init__()
    self.generator = self.Generator(**kwargs["generator"])
    self.discriminator = self.Discriminator(**kwargs["discriminator"])

  class Generator(nn.Module): 
    """
    Generator that takes in a prior and maps it to data space
    """
    def __init__(self, **kwargs): 
      super().__init__()
      dimensions = kwargs["dimensions"] 
      activations = kwargs["activations"]
      normalizations = kwargs["normalizations"] 
      self.latent_dim = dimensions[0]

      self.fc_maps = nn.Sequential() 
      for i in range(len(dimensions) - 1): 
        idim, odim = dimensions[i], dimensions[i+1] 
        self.fc_maps.append(nn.Linear(idim, odim)) 
        if normalizations[i]: 
          self.fc_maps.append(nn.BatchNorm1d(odim))
        self.fc_maps.append(activation(activations[i]))  

    def forward(self, x): 
      """
      x should be the sampled tensor from latent dim
      """
      return self.fc_maps(x) 

    def sample(self, sample_size): 
      return torch.randn(sample_size, self.latent_dim)

    def toggle_grad(self, enable): 
      for param in self.parameters(): 
        param.requires_grad = enable

  class Discriminator(nn.Module): 
    """
    Discriminator MLP that outputs probability that the sample came 
    from the true data generating distribution. 
    """
    
    def __init__(self, **kwargs): 
      super().__init__()
      dimensions = kwargs["dimensions"] 
      activations = kwargs["activations"]
      normalizations = kwargs["normalizations"]

      self.fc_maps = nn.Sequential() 
      for i in range(len(dimensions) - 1): 
        idim, odim = dimensions[i], dimensions[i+1] 
        self.fc_maps.append(nn.Linear(idim, odim)) 
        if normalizations[i]: 
          self.fc_maps.append(nn.BatchNorm1d(odim))
        self.fc_maps.append(activation(activations[i]))  

    def forward(self, x): 
      x = torch.flatten(x, start_dim=1)
      return self.fc_maps(x) 

    def toggle_grad(self, enable): 
      for param in self.parameters(): 
        param.requires_grad = enable


