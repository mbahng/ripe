import torch
import torch.nn as nn
import torch.nn.functional as F 

class GeneticCNN2D(nn.Module):
  """
  Takes a (4, length) tensor and returns a (class_count,) tensor.
  Layers were chosen arbitrarily, and should be optimized. I have no idea what I'm doing.
  """

  def __init__(self, length=720, class_count=40, include_connected_layer=False):
    super().__init__()
    
    self.length = length
    self.class_count = class_count
    self.include_connected_layer = include_connected_layer
    
    self.conv1 = nn.Conv2d(4, 16, kernel_size=(1,3), padding=(0,1))
    self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,3), padding=(0,1))
    self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
    self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
    self.pool3 = nn.MaxPool2d(kernel_size=(1,3), stride=3)

    if include_connected_layer:
      self.fc1 = nn.Linear(64 * (length // 18), class_count)

  def forward(self, x):
    x = self.pool3(F.relu(self.conv1(x)))
    x = self.pool3(F.relu(self.conv2(x)))
    x = self.pool(self.conv3(x))

    if hasattr(self, 'fc1'):
      x = torch.flatten(x, 1)
      x = self.fc1(x)
    return x


