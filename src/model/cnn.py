import torch
import torch.nn as nn
from torch import Tensor
from ..utils import activation

class CNN(nn.Module): 
  """
  Shallow convolutional classifier: a stack of conv blocks, then a small MLP head.

  Depth is whatever the config asks for. ``channels``, ``kernel_sizes``,
  ``normalizations``, ``activations`` and ``pools`` are parallel lists with one
  entry per block, in the same style as MLP's ``dimensions``. A block is
  conv -> optional BatchNorm -> activation -> optional max-pool, with padding
  set to keep the spatial size, so pooling alone decides the downsampling.

  ``dimensions`` is the head, ending with the number of classes. Its input
  width is measured with a dry forward pass instead of being read from the
  config, so editing the block list or the input size cannot silently
  disagree with the head.
  """

  def __init__(self, in_channels, in_size, channels, kernel_sizes, normalizations,
               activations, pools, dimensions, dropout=0.0):
    super().__init__()

    for name, values in (("kernel_sizes", kernel_sizes), ("normalizations", normalizations),
                         ("activations", activations), ("pools", pools)):
      if len(values) != len(channels):
        raise ValueError(f"{name} has {len(values)} entries but channels has {len(channels)}")
    if not dimensions:
      raise ValueError("dimensions must end with the number of classes")

    self.features = nn.Sequential()
    previous = in_channels
    for out_channels, kernel, normalize, act, pool in zip(
        channels, kernel_sizes, normalizations, activations, pools):
      # BatchNorm subtracts its own mean, so the conv bias would be redundant.
      self.features.append(nn.Conv2d(previous, out_channels, kernel,
                                     padding=kernel // 2, bias=not normalize))
      if normalize:
        self.features.append(nn.BatchNorm2d(out_channels))
      self.features.append(activation(act))
      if pool and pool > 1:
        self.features.append(nn.MaxPool2d(pool))
      previous = out_channels

    self.classifier = nn.Sequential()
    previous = self._feature_width(in_channels, in_size)
    for width in dimensions[:-1]:
      self.classifier.append(nn.Linear(previous, width))
      self.classifier.append(nn.ReLU())
      if dropout:
        self.classifier.append(nn.Dropout(dropout))
      previous = width
    self.classifier.append(nn.Linear(previous, dimensions[-1]))

    self._init_weights()
    self.device = None

  def _feature_width(self, in_channels, in_size): 
    """Flattened conv-stack output for one (in_channels, in_size, in_size) input."""
    size = (in_size, in_size) if isinstance(in_size, int) else tuple(in_size)
    training = self.features.training
    # eval() so the probe does not fold itself into BatchNorm's running stats.
    self.features.eval()
    with torch.no_grad():
      width = self.features(torch.zeros(1, in_channels, *size)).numel()
    self.features.train(training)
    return width

  def forward(self, x: Tensor): 
    x = self.features(x)
    x = torch.flatten(x, start_dim=1)
    return self.classifier(x)

  def _init_weights(self): 
    """Kaiming for the ReLU-family stack; BatchNorm starts out as the identity."""
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)
