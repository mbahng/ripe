import torch 
import torch.nn as nn
from .distribution import LogisticDistribution 
from torch import Tensor
from ..utils import activation
import math

class NICE(nn.Module): 
  """
  Simplest finite normalizing flow model by Dinh 2014. 
  """

  def __init__(self, **kwargs):
    super().__init__() 
    self.input_shape = kwargs["input_shape"] 
    self.input_dim = math.prod(self.input_shape)
    self.n_coupling_layers = kwargs["n_coupling_layers"]

    coupling_layers = []
    for i in range(self.n_coupling_layers):
      mask_first = bool(i % 2)
      coupling_layers.append(self.CouplingLayer(self.input_dim, mask_first, **kwargs["coupling_layer"]))
    self.coupling_layers = nn.ModuleList(coupling_layers)

    self.scaling_layer = self.ScalingLayer(self.input_dim)
    self.latent_prior = LogisticDistribution()

  class CouplingLayer(nn.Module): 
    """
    A simple coupling layer.
    """

    def __init__(self, input_dim, mask_first, **kwargs):
      super().__init__()
      self.input_dim = input_dim
      self.register_buffer("mask", self._init_mask(mask_first))
      
      self.mlp = self.MLP(input_dim, **kwargs["mlp"])

    class MLP(nn.Module): 

      def __init__(self, input_dim, **kwargs): 
        super().__init__()  

        dimensions = kwargs["dimensions"] 
        assert input_dim == dimensions[0] == dimensions[-1]
        activations = kwargs["activations"]
        normalizations = kwargs["normalizations"]

        self.layers = nn.Sequential() 
        for i in range(len(dimensions) - 1): 
          self.layers.append(nn.Linear(dimensions[i], dimensions[i+1])) 
          if normalizations[i]: 
            self.layers.append(nn.BatchNorm1d(dimensions[i+1]) )
          self.layers.append(activation(activations[i]))

      def forward(self, x): 
        return self.layers(x)

    def _init_mask(self, mask_first: bool): 
      mask = torch.zeros(self.input_dim) 
      if mask_first: 
        mask[::2] += 1 
      else: 
        mask[1::2] += 1 
      return mask

    def forward(self, x: Tensor, logdet_accum: Tensor): 
      """
      Return output f(x) and log det, which is log(det(I)) = log(1) = 0
      """
      x1, x2 = self.mask * x, (1 - self.mask) * x
      y1 = x1
      y2 = x2 + (self.mlp(x1) * (1. - self.mask))
      return y1 + y2, logdet_accum

    def inverse(self, z: Tensor): 
      """
      The inverse is easy to calculate
      """
      z1, z2 = self.mask * z, (1 - self.mask) * z 
      x1 = z1 
      x2 = z2 - (self.mlp(z1) * (1 - self.mask))
      return x1 + x2

  class ScalingLayer(nn.Module): 
    """
    Use this to make NICE non volume preserving. 
    """

    def __init__(self, input_dim):
      super().__init__()
      # Initialize with small values to prevent numerical instability
      self.log_scale_vector = nn.Parameter(torch.randn(1, input_dim, requires_grad=True))

    def forward(self, x: Tensor, logdet: int):
      log_det_jacobian = torch.sum(self.log_scale_vector)
      return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian 

    def inverse(self, y: Tensor): 
      return torch.exp(- self.log_scale_vector) * y

  def forward(self, x: Tensor): 
    x = torch.flatten(x, start_dim=1)
    logdet_accum = 0.0
    for coupling_layer in self.coupling_layers: 
      x, logdet_accum = coupling_layer(x, logdet_accum) 
    x, logdet_accum = self.scaling_layer(x, logdet_accum) 
    log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=1) + logdet_accum 

    x = x.reshape(-1, *self.input_shape)
    return x, log_likelihood

  def inverse(self, z: Tensor): 
    z = self.scaling_layer.inverse(z)
    for coupling_layer in reversed(self.coupling_layers): 
      z = coupling_layer.inverse(z)  # type: ignore
    z = z.reshape(-1, *self.input_shape)
    return z 

  def sample(self, n_samples: int): 
    z = self.latent_prior.sample([n_samples, self.input_dim])
    return self.inverse(z)



