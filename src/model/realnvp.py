import torch 
import torch.nn as nn
import torch.nn.functional as F
from .distribution import Normal
from torch.nn.utils.parametrizations import weight_norm
from torch import Tensor
from warnings import deprecated 
from typing import Optional
from enum import Enum
import numpy as np
from ..utils import activation
from copy import deepcopy

@deprecated("Used for comparison with NICE. This was used so that we can compare RealNVP with NICE on 1D distributions. However, the squeeze operations can't be implemented here. ")
class RealNVP1d(nn.Module): 
  """
  Simple non-volume-preserving finite normalizing flow model by Dinh 2015.
  One level of complexity above NICE by implementing resdiual connections, 
  affine coupling layer, and convolutions to process 2d images directly. 
  """

  def __init__(self, **kwargs):
    super().__init__()
    self.input_shape = kwargs["input_shape"]
    self.n_scales = kwargs["n_scales"]
    
    coupling_layers = []
    for i in range(self.n_scales): 
      mask_first = bool(i % 2)
      coupling_layers.append(self.AffineCouplingLayer1d(self.input_shape, mask_first, **kwargs["affine_coupling_layer1d"]))
      # coupling_layers.append(self.AffineCouplingLayer1d(self.input_shape, neural_net_layers, latent_channel, mask_first))
    self.coupling_layers = nn.ModuleList(coupling_layers)

    from torch.distributions import Normal
    self.latent_prior = Normal(0, 1)

  class AffineCouplingLayer1d(nn.Module): 
    """
    Affine Coupling Layer that contains neural networks s, t that act as a scaling and translation factor. 
    Designed only for 1d inputs, e.g. flattened images. 
    For 2d inputs, use AffineCouplingLayer2d. 
    """

    mask: Tensor

    def __init__(self, input_shape: int, mask_first: bool, **kwargs):
      super().__init__()

      self.input_shape = input_shape
      # register this as buffer so it's automatically moved to devices
      self.register_buffer("mask", self._init_mask(mask_first))

      self.s = self.MLP(self.input_shape, **kwargs["mlp"])
      self.t = self.MLP(self.input_shape, **kwargs["mlp"])

    class MLP(nn.Module):
      """
      ResNet-style MLP with residual connections and layer normalization. 
      Only supports 1d inputs. 
      """

      def __init__(self, input_shape: int, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.depth = kwargs["depth"] 
        self.latent_channel = kwargs["latent_channel"]
        assert self.depth >= 2

        # Input projection
        self.input_layer = nn.Sequential(
          nn.Linear(self.input_shape, self.latent_channel),
          nn.LayerNorm(self.latent_channel),
          nn.LeakyReLU(0.2)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
          self.ResidualBlock(self.latent_channel) for _ in range(self.depth - 2)
        ])

        # Output projection
        self.output_layer = nn.Sequential(
          nn.LayerNorm(self.latent_channel),
          nn.Linear(self.latent_channel, self.input_shape)
        )

      class ResidualBlock(nn.Module):

        def __init__(self, latent_channel):
          super().__init__()
          self.norm1 = nn.LayerNorm(latent_channel)
          self.linear1 = nn.Linear(latent_channel, latent_channel)
          self.activation = nn.LeakyReLU(0.2)
          self.norm2 = nn.LayerNorm(latent_channel)
          self.linear2 = nn.Linear(latent_channel, latent_channel)

        def forward(self, x):
          residual = x
          out = self.norm1(x)
          out = self.linear1(out)
          out = self.activation(out)
          out = self.norm2(out)
          out = self.linear2(out)
          out = out + residual  # Skip connection
          return self.activation(out)

      def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
          x = block(x)
        return self.output_layer(x)

    def _init_mask(self, mask_first: bool): 
      mask = torch.zeros(self.input_shape) 
      if mask_first: 
        mask[::2] += 1 
      else: 
        mask[1::2] += 1 
      return mask

    def forward(self, x, logdet_accum):
      """
      Return output f(x), and be careful with the masks
      Determinant calclation in section 3.3 of paper
      """
      x1, x2 = self.mask * x, (1 - self.mask) * x
      y1 = x1
      logscale = self.s(x1)
      translation = self.t(x1)
      y2 = (x2 * torch.exp(logscale) + translation) * (1 - self.mask)
      # make sure to sum over only the samples! not the batch dimension! Also mask it
      return y1 + y2, logdet_accum + (logscale * (1 - self.mask)).sum(dim=-1)

    def inverse(self, y): 
      """
      The inverse is easy to calculate
      """
      y1, y2 = self.mask * y, (1 - self.mask) * y 
      x1 = y1 
      x2 = (y2 - self.t(y1)) * torch.exp(- self.s(y1)) * (1 - self.mask)
      return x1 + x2

  def forward(self, x: Tensor): 
    logdet_accum = 0.0
    for coupling_layer in self.coupling_layers: 
      x, logdet_accum = coupling_layer(x, logdet_accum) 
    nonbatch_dims = list(range(1, len(x.size()))) 
    log_likelihood = torch.sum(self.latent_prior.log_prob(x), dim=nonbatch_dims) + logdet_accum
    return x, log_likelihood

  def inverse(self, z: Tensor): 
    for coupling_layer in reversed(self.coupling_layers): 
      z = coupling_layer.inverse(z)  # type: ignore
    return z 

  def sample(self, n_samples: int): 
    z = self.latent_prior.sample([n_samples, *self.input_shape])
    return self.inverse(z)

class Mask(Enum): 
  """Mask has to be in top scope since ResNet also refers to it"""
  Checkerboard0 = 1 
  Checkerboard1 = 2 
  Channel0 = 3 
  Channel1 = 4

class RealNVP(nn.Module):
  """
  Simple non-volume-preserving finite normalizing flow model by Dinh 2015.
  One level of complexity above NICE by implementing resdiual connections,
  affine coupling layer, and convolutions to process 2d images directly.

  The 2nd paragraph in Section 3.6 explains how the layers should be made.
  """

  def __init__(self, **kwargs):
    super().__init__()
    self.input_shape = kwargs["input_shape"] 
    self.n_scales = kwargs["n_scales"]
    self.device = None

    coupling_layers = []
    coupling_layers.append(self.LogitTransform(**kwargs["logit_transform"]))
    squeezed_input_shape = deepcopy(self.input_shape)

    # Dinh 2015 refers to each of these steps as a "scale," hence the name
    for scale in range(self.n_scales - 1):
      # first apply 3 coupling layers with alternating checkboard masks
      coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard0, **kwargs["affine_coupling_layer2d"]))
      coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard1, **kwargs["affine_coupling_layer2d"]))
      coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard0, **kwargs["affine_coupling_layer2d"]))
    
      # apply squeeze operation.
      coupling_layers.append(s := self.Squeeze(squeezed_input_shape))
    
      squeezed_input_shape = s.odim
    
      # Then 3 more layers with alternating channel-wise masking
      coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Channel0, **kwargs["affine_coupling_layer2d"]))
      coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Channel1, **kwargs["affine_coupling_layer2d"]))
      coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Channel0, **kwargs["affine_coupling_layer2d"]))

      # Finally a factor out to split latent space 
      coupling_layers.append(f := self.FactorOut(squeezed_input_shape, scale)) 
      squeezed_input_shape = f.odim
    
    # final scale consists of 4 coupling layers with alternating checkerboard masks.
    coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard0, **kwargs["affine_coupling_layer2d"]))
    coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard1, **kwargs["affine_coupling_layer2d"]))
    coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard0, **kwargs["affine_coupling_layer2d"]))
    coupling_layers.append(self.AffineCouplingLayer2d(squeezed_input_shape, Mask.Checkerboard1, **kwargs["affine_coupling_layer2d"]))
    coupling_layers.append(self.FactorOut(squeezed_input_shape, self.n_scales - 1))


    self.coupling_layers = nn.ModuleList(coupling_layers)

    self.latent_prior = Normal(torch.tensor(0.), torch.tensor(1.))

    odim_x, _, odim_z = self.forward(torch.rand(1, *self.input_shape))
    self.odim_x = odim_x.squeeze(0).size() 
    self.odim_z = odim_z.squeeze(0).size()

  class LogitTransform(nn.Module):
    """
    Logit transform with dequantization as described in RealNVP paper Section 4.1
    Maps [0, 1] to (-inf, inf) for better modeling with Gaussian prior
    """
    def __init__(self, **kwargs):
      super().__init__()
      self.alpha = kwargs["alpha"]
    
    def forward(self, x, logdet_accum, z):
      """x in [0, 1] -> y in (-inf, inf)"""

      # restrict data
      x *= 2.                   # [0, 2]
      x -= 1.                   # [-1, 1]
      x *= (1 - 2 * self.alpha) # [-0.9, 0.9]
      x += 1.                   # [0.1, 1.9]
      x /= 2.                   # [0.05, 0.95]

      
      # Apply logit with numerical stability
      # Map [0, 1] -> [alpha, 1-alpha] to avoid log(0)
      x_scaled = self.alpha + (1 - 2 * self.alpha) * x
      y = torch.log(x_scaled) - torch.log(1 - x_scaled)
      

      # logit data
      logit_x = torch.log(x) - torch.log(1. - x)

      # log-determinant of Jacobian from the transform
      pre_logit_scale = torch.tensor(
          np.log(1 - 2 * self.alpha) - np.log(2 * self.alpha))
      log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
          - F.softplus(-pre_logit_scale)

      logdet = torch.sum(log_diag_J, dim=(1, 2, 3))
      
      return y, logdet_accum + logdet, z
    
    def inverse(self, y, z):
      """y in (-inf, inf) -> x in [0, 1]"""
      # Apply sigmoid to map back to [alpha, 1-alpha]
      x_scaled = torch.sigmoid(y)
      
      # Invert the linear scaling: x = (x_scaled - alpha) / (1 - 2*alpha)
      x = (x_scaled - self.alpha) / (1 - 2 * self.alpha)
      
      # Clamp to ensure bounds (handles numerical errors)
      x = torch.clamp(x, 0, 1)
      return x, z

  class AffineCouplingLayer2d(nn.Module): 
    """
    Affine Coupling Layer that contains neural networks s, t that act as a scaling and translation factor. 
    Designed for 2d inputs, using fully convolutional layers. 
    For 2d inputs, use AffineCouplingLayer2d. 
    """
    mask: Tensor

    def __init__(self, input_shape: list, mask: Mask, **kwargs):
      super().__init__()
      assert len(input_shape) >= 2
      self.input_shape = input_shape
      self.register_buffer("mask", self._init_mask(mask)) # type: ignore 

      self.s = self.ResidualCNN(self.input_shape, **kwargs["resnet"])
      self.t = self.ResidualCNN(self.input_shape, **kwargs["resnet"])

    class ResidualCNN(nn.Module): 
      """
      CNN with residual connections and skip architecture similar to the reference implementation.
      Uses pre-activation residual blocks (BN -> ReLU -> Conv).
      """

      def __init__(self, input_shape, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.n_blocks = kwargs["n_blocks"]
        self.latent_channel = kwargs["latent_channel"]
        self.use_skip = kwargs["use_skip"]
        C, *_ = input_shape 
        
        # Input projection
        self.bn1 = nn.BatchNorm2d(C)
        self.conv_input = weight_norm(nn.Conv2d(C, self.latent_channel, kernel_size=3, stride=1, padding=1)) # type: ignore
        self.bn2 = nn.BatchNorm2d(self.latent_channel)
        self.relu = nn.ReLU()
        
        # Create residual blocks
        self.blocks = nn.ModuleList([self.ResidualBlock(**kwargs["block"]) for _ in range(self.n_blocks)])
        
        # Skip architecture: 1x1 convolutions to accumulate features from all blocks
        if self.use_skip:
          # Initial skip connection
          self.skip_input = weight_norm(nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=1)) # type: ignore
          
          # Skip connection for each residual block
          self.skip_connections = nn.ModuleList([
            weight_norm(nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=1)) for _ in range(self.n_blocks) # type: ignore
          ])
        
        # Output projection
        self.bn_out = nn.BatchNorm2d(self.latent_channel)
        self.relu_out = nn.ReLU()
        self.conv_output = weight_norm(nn.Conv2d(self.latent_channel, C, kernel_size=1)) # type: ignore
        
        # For numerical stability, apply tanh and then scale  
        self.scale = nn.Parameter(torch.ones(1))
      
      class ResidualBlock(nn.Module):
        """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv"""
        # def __init__(self, channels: int):
        def __init__(self, **kwargs): 
          super().__init__()
          # Pre-activation: BN and ReLU come BEFORE convolutions

          normalizations = kwargs["normalizations"] 
          activations = kwargs["activations"] 
          channels = kwargs["channels"] 
          kernel_sizes = kwargs["kernel_sizes"] 
          strides = kwargs["strides"] 
          paddings = kwargs["paddings"] 
          weight_norms = kwargs["weight_norm"]

          self.layers = nn.Sequential()

          for i in range(len(channels) - 1): 
            in_channel, out_channel = channels[i], channels[i+1]
            if normalizations[i]: 
              self.layers.append(nn.BatchNorm2d(out_channel)) 
            self.layers.append(activation(activations[i]))
            conv_layer = nn.Conv2d(in_channel, out_channel, kernel_sizes[i], strides[i], paddings[i]) 
            if weight_norms[i] is True: 
              self.layers.append(weight_norm(conv_layer)) # type: ignore
            else:
              self.layers.append(conv_layer)
        
        def forward(self, x: Tensor) -> Tensor:
          # Pre-activation pattern
          identity = x
          out = self.layers(x)
          # Add residual connection
          return out + identity
      
      def forward(self, x: Tensor): 
        # Input processing
        x = self.bn1(x) 
        x = self.conv_input(x) 
        x = self.bn2(x) 
        x = self.relu(x)
        
        if self.use_skip:
          # Initialize skip with processed input
          skip = self.skip_input(x)
          
          # Process through residual blocks with skip accumulation
          for i, block in enumerate(self.blocks):
            x = block(x)
            # Accumulate skip connections from all blocks
            skip = skip + self.skip_connections[i](x)
          
          # Use accumulated skip as the main path
          x = skip
        else:
          # Simple sequential residual connections (your original approach)
          for block in self.blocks:
            x = block(x)
        
        # Output processing with pre-activation
        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.conv_output(x)
        
        # Numerical stability
        x = torch.tanh(x) * self.scale
        return x

    def _init_mask(self, mask: Mask) -> Tensor: 
      """
      Alternating checkerboard or channel wise masking
      """
      match mask: 
        case Mask.Checkerboard0: 
          C, H, W = self.input_shape
          checker = torch.zeros((1, H, W))
          checker[:, ::2, ::2] = 1  # Even rows, even cols
          checker[:, 1::2, 1::2] = 1  # Odd rows, odd cols
          return checker.repeat(C, 1, 1)
        case Mask.Checkerboard1: 
          C, H, W = self.input_shape
          checker = torch.ones((1, H, W))
          checker[:, ::2, ::2] = 0  # Even rows, even cols
          checker[:, 1::2, 1::2] = 0  # Odd rows, odd cols
          return checker.repeat(C, 1, 1)
        case Mask.Channel0: 
          masker = torch.zeros(*self.input_shape) 
          masker[:self.input_shape[0] // 2] += 1 
          return masker
        case Mask.Channel1: 
          masker = torch.zeros(*self.input_shape) 
          masker[self.input_shape[0] // 2:] += 1 
          return masker

    def forward(self, x, logdet_accum, z):
      """
      Return output f(x), and be careful with the masks
      Determinant calclation in section 3.3 of paper
      """
      x1, x2 = self.mask * x, (1 - self.mask) * x
      y1 = x1
      logscale = self.s(x1)
      translation = self.t(x1)
      y2 = (x2 * torch.exp(logscale) + translation) * (1 - self.mask)
      # make sure to sum over only the samples! not the batch dimension! Also mask it
      nonbatch_dims = list(range(1, len(logscale.size())))
      return y1 + y2, logdet_accum + (logscale * (1 - self.mask)).sum(dim=nonbatch_dims), z

    def inverse(self, y, z): 
      """
      The inverse is easy to calculate
      """
      y1, y2 = self.mask * y, (1 - self.mask) * y 
      x1 = y1 
      # just masking at the end should suffice? 
      x2 = (y2 - self.t(y1)) * torch.exp(- self.s(y1)) * (1 - self.mask)
      return x1 + x2, z 

  class Squeeze(nn.Module): 
    """
    Squeezing operation used in each level to convert shape (C, H, W) to 
    shape (4C, H/2, W/2). See 2016 Dinh Section 3.6 or Figure 3. 
    Note that you can't just simply reshape!
    """

    def __init__(self, input_shape): 
      super().__init__()
      self.input_shape = input_shape
      assert self.input_shape[-1] % 2 == 0 and self.input_shape[-2] % 2 == 0 
      self.odim = input_shape
      self.odim[-1] = self.odim[-1] // 2 
      self.odim[-2] = self.odim[-2] // 2 
      self.odim[-3] = 4 * self.odim[-3] 
      self.odim = self.odim

    def _squeeze(self, x: Tensor): 
      B, C, H, W = x.size()
      x = x.reshape(B, C, H//2, 2, W//2, 2)
      x = x.permute(0, 1, 3, 5, 2, 4)
      x = x.reshape(B, C*4, H//2, W//2)
      return x 

    def _unsqueeze(self, z: Tensor): 
      B, C, H, W = z.size() 
      z = z.reshape(B, C//4, 2, 2, H, W)
      z = z.permute(0, 1, 4, 2, 5, 3)
      z = z.reshape(B, C//4, H*2, W*2)
      return z

    def forward(self, x: Tensor, logdet_accum: Tensor, z: Tensor):  
      x = self._squeeze(x) 
      z = self._squeeze(z) if z is not None else z
      return x, logdet_accum, z

    def inverse(self, x: Tensor, z: Tensor): 
      x = self._unsqueeze(x) 
      z = self._unsqueeze(z) if z is not None else z
      return x, z

  class FactorOut(nn.Module): 
    """Factor out that splits the latent space by forwarding half of it."""

    def __init__(self, input_shape, scale: int):  
      super().__init__()
      self.input_shape = input_shape
      self.scale = scale # keeps track of how much has been factored out in powers of two
      self.odim = input_shape
      self.odim[-1] = self.odim[-1]
      self.odim[-2] = self.odim[-2]
      self.odim[-3] = self.odim[-3] // 2
      self.odim = self.odim

    def forward(self, x: Tensor, logdet_accum: Tensor, z: Optional[Tensor]): 
      _, C, _, _ = x.size() 
      split = C // 2 
      new_z = x[:,:split,:,:] 
      x = x[:,split:,:,:] 
      if z is not None: 
        z = torch.concat([z, new_z], dim=1)
      else: 
        z = new_z 
      return x, logdet_accum, z

    def inverse(self, y: Tensor, z: Tensor):  # 3, 9 -> 6, 6 
      split = y.size(1) # channels 3

      # you want to take back from the right end of z and add it back in
      new_y = z[:,-split:,:,:] 
      z = z[:,:-split,:,:] 

      x = torch.concat([new_y, y], dim=1) 
      return x, z

  def forward(self, x: Tensor):
    logdet_accum = torch.zeros(x.size(0)).to(self.device)
    z = None
    for coupling_layer in self.coupling_layers: # contains both coupling layers and factor out layers
      x, logdet_accum, z = coupling_layer.forward(x, logdet_accum, z)

    # Compute log probability
    nonbatch_dims = list(range(1, len(x.size())))
    K = np.prod(x.shape[1:])  # Total dimensions

    log_density_z = torch.sum(self.latent_prior.log_prob(torch.concat((x, z), dim=1)), dim=nonbatch_dims) # type: ignore
    log_density_x = log_density_z + logdet_accum
    
    # Add discretization correction for 8-bit images
    log_prob_x = log_density_x - K * np.log(256.0)
    
    return x, log_prob_x, z

  def inverse(self, y, z): 
    for coupling_layer in reversed(self.coupling_layers): 
      y, z = coupling_layer.inverse(y, z)  # type: ignore 
    assert z.size(1) == 0 # channels in latent dim should all be gone by now. 
    return y

  def sample(self, n_samples: int): 
    y = self.latent_prior.sample([n_samples, *self.odim_x]).to(self.device)
    z = self.latent_prior.sample([n_samples, *self.odim_z]).to(self.device)
    return self.inverse(y, z)


