import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import gdown

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    "resnet50[pretraining=inaturalist]": "1yY9LcLN7bVy7CmH77Mc4TfmW7qT00oS9",
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_dir = './src/model/backbone'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(nn.Module):
  # class attribute
  expansion = 4
  num_layers = 3

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = conv1x1(inplanes, planes)
    self.bn1 = nn.BatchNorm2d(planes)
    # only conv with possibly not 1 stride
    self.conv2 = conv3x3(planes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = conv1x1(planes, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)

    # if stride is not 1 then self.downsample cannot be None
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

  def block_conv_info(self):
    block_kernel_sizes = [1, 3, 1]
    block_strides = [1, self.stride, 1]
    block_paddings = [0, 1, 0]

    return block_kernel_sizes, block_strides, block_paddings


class ResNet(nn.Module):
  '''
  the convolutional layers of ResNet
  the average pooling and final fully convolutional layer is removed
  '''

  def __init__(self, block, layers, num_classes: int = 0, pretrained="", zero_init_residual=False):
    super(ResNet, self).__init__()

    self.inplanes = 64

    # the first convolutional layer before the structured sequence of blocks
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # comes from the first conv and the following max pool
    self.kernel_sizes = [7, 3]
    self.strides = [2, 2]
    self.paddings = [3, 1]

    # the following layers, each layer is a sequence of blocks
    self.block = block
    self.layers = layers
    self.layer1 = self._make_layer(block=self.block, planes=64, num_blocks=self.layers[0])
    self.layer2 = self._make_layer(block=self.block, planes=128, num_blocks=self.layers[1], stride=2)
    self.layer3 = self._make_layer(block=self.block, planes=256, num_blocks=self.layers[2], stride=2)
    self.layer4 = self._make_layer(block=self.block, planes=512, num_blocks=self.layers[3], stride=2)

    self._backbone = nn.Sequential(
      self.conv1, self.bn1, self.relu, self.maxpool,
      self.layer1, self.layer2, self.layer3, self.layer4,
    )

    # initialize the parameters
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

    # layer4 outputs 512 channels for BasicBlock, 2048 for Bottleneck
    backbone_out = 512 * self.block.expansion
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(backbone_out, num_classes) if num_classes > 0 else None

    if pretrained:
        url_or_id = model_urls[pretrained]
        if url_or_id.startswith('https://'):
            my_dict = model_zoo.load_url(url_or_id, model_dir=model_dir)
        else:
            # Google Drive file ID — download with gdown
            dest = os.path.join(model_dir, f"{pretrained.replace('[', '_').replace(']', '').replace('=', '_')}.pth")
            gdown.download(id=url_or_id, output=dest, quiet=False)
            my_dict = torch.load(dest, map_location='cpu', weights_only=False)
        # BBN checkpoints wrap weights under a 'state_dict' key
        if 'state_dict' in my_dict:
            my_dict = my_dict['state_dict']
        # strip DataParallel 'module.' prefix if present
        my_dict = {k.removeprefix('module.'): v for k, v in my_dict.items()}
        my_dict.pop('fc.weight', None)
        my_dict.pop('fc.bias', None)
        self.load_state_dict(my_dict, strict=False)

  def _make_layer(self, block, planes, num_blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    # only the first block has downsample that is possibly not None
    layers.append(block(self.inplanes, planes, stride, downsample))

    self.inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
      layers.append(block(self.inplanes, planes))

    # keep track of every block's conv size, stride size, and padding size
    for each_block in layers:
      block_kernel_sizes, block_strides, block_paddings = each_block.block_conv_info()
      self.kernel_sizes.extend(block_kernel_sizes)
      self.strides.extend(block_strides)
      self.paddings.extend(block_paddings)

    return nn.Sequential(*layers)

  def backbone(self) -> nn.Module:
    """Return the feature extractor module; outputs (B, 512, 7, 7) for (B, 3, 224, 224) input."""
    return self._backbone

  def forward(self, x):
    x = self._backbone(x)          # (B, 512, 7, 7)
    if self.fc is not None:
      x = self.avgpool(x)          # (B, 512, 1, 1)
      x = x.flatten(1)             # (B, 512)
      x = self.fc(x)               # (B, num_classes)
    return x

  def conv_info(self):
    return self.kernel_sizes, self.strides, self.paddings

  def num_layers(self):
    '''
    the number of conv layers in the network, not counting the number
    of bypass layers
    '''

    return (self.block.num_layers * self.layers[0]
          + self.block.num_layers * self.layers[1]
          + self.block.num_layers * self.layers[2]
          + self.block.num_layers * self.layers[3]
          + 1)


  def __repr__(self):
    template = 'resnet{}_features'
    return template.format(self.num_layers() + 1)

