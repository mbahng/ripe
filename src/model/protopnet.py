from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from .backbone import resnet34_features
from ..receptive_field import compute_proto_layer_rf_info_v2

class ProtoPNet(nn.Module): 

  def __init__(self, img_size, prototype_shape = (2000, 128, 1, 1), num_classes = 200): 
    super().__init__()
    self.img_size = img_size
    self.backbone = resnet34_features(pretrained=True)

    first_add_on_layer_in_channels = [i for i in self.backbone.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

    self.prototype_shape = prototype_shape 
    assert self.prototype_shape[0] % num_classes == 0 
    self.num_classes = num_classes
    self.num_prototypes = prototype_shape[0]

    self.add_on_layers = nn.Sequential(
      nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
      nn.Sigmoid()
      )
    self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

    # the default weights for convolution after add-on layers, should not be backpropped
    # this may be more efficient with torch.repeat?  
    self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

    self.prototype_activation_function = 'linear' 
    self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False) # do not use bias  

    # need this for calculating loss after forward call
    self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
    num_prototypes_per_class = self.num_prototypes // self.num_classes
    for j in range(self.num_prototypes):
      self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
    
    # you need this init method. Learning is trash if we don't set this. 
    self._initialize_weights()

    # this is only useful for visualization of prototype heatmaps
    layer_filter_sizes, layer_strides, layer_paddings = self.backbone.conv_info()
    self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=self.prototype_shape[2])
    # evaluates to [7, 32, 899, 0.5]

  def set_last_layer_incorrect_connection(self, incorrect_strength):
    '''
    the incorrect strength will be actual strength if -0.5 then input -0.5
    '''
    positive_one_weights_locations = torch.t(self.prototype_class_identity)
    negative_one_weights_locations = 1 - positive_one_weights_locations

    correct_class_connection = 1
    incorrect_class_connection = incorrect_strength
    self.last_layer.weight.data.copy_(
      correct_class_connection * positive_one_weights_locations
      + incorrect_class_connection * negative_one_weights_locations)

  def _initialize_weights(self):
    for m in self.add_on_layers.modules():
      if isinstance(m, nn.Conv2d):
        # every init technique has an underscore _ in the name
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


  def distance_2_similarity(self, distances):
    return -distances

  def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]: 
    """
    Original implementation had (x - p)^2 = x^2 + p^2 - 2 x p weirdly. 
    Chaofan told me that it was because PyTorch didn't have fast support for these operations, 
    so they were forced to use convolutions. Let's not take this path and reimplement this. 
    We have input patch of shape (B, D, 7, 7) and prototypes of shape (2000, D, 1, 1) 
    We want to compute the distances (B, 2000, 7, 7), where (i1, i2, i3, i4) represents the distance of the 'i2'th prototype 
    with the (i3, i4)th patch in sample 'i1'. We then max pool to get (B, 2000). 
   
    The differences between this and original function seem like numerical precision errors. 
    """
    x = self.backbone(x) 
    x = self.add_on_layers(x) # (B, 128, 7, 7)  

    B, D, H, W = x.size()
    p = self.prototype_vectors  # (2000, 128, 1, 1)
    p = p.expand(-1, -1, H, W)  # (2000, 128, 7, 7)

    x = x.unsqueeze(1).expand(-1, p.size(0), -1, -1, -1) # (B, 2000, 128, 7, 7) 
    
    # compute squared differences of all elements 
    distances = (x - p) ** 2  # (B, 2000, 128, 7, 7)

    # now compute distances between vectors by summing them up 
    distances = distances.sum(dim=2) # (B, 2000, 7, 7)

    # Now min over the dimensions 
    min_distances = distances.amin(dim=(-2, -1))  # (B, 2000)
    prototype_activations = self.distance_2_similarity(min_distances) # B, 2000
    logits = self.last_layer(prototype_activations) # B, 200

    return logits, min_distances

  def _forward_original(self, x) -> Tuple[torch.Tensor, torch.Tensor]: 
    x = self.backbone(x) 
    x = self.add_on_layers(x) # (B, 128, 7, 7)
    x2 = x ** 2 
    x2_patch_sum = F.conv2d(input=x2, weight=self.ones) # (B, 128, 7, 7) * (2000, 128, 1, 1) -> (B, 2000, 7, 7) 

    p2 = self.prototype_vectors ** 2 # 2000, 128, 1, 1
    p2 = torch.sum(p2, dim=(1, 2, 3)) # 2000
    p2_reshape = p2.view(-1, 1, 1) # 2000, 1, 1

    xp = F.conv2d(input=x, weight=self.prototype_vectors) # B, 2000, 7, 7 
    intermediate_result = -2 * xp + p2_reshape  # broadcast
    distances = F.relu(x2_patch_sum + intermediate_result) # B, 2000, 7, 7 
    # B, i, j, k represents the distance between ith prototype  (of shape 128, 1, 1) and the (j, k)th patch

    # global min pooling
    min_distances = -F.max_pool2d(-distances,       # B, 2000, 1, 1
                                  kernel_size=(distances.size()[2],
                                               distances.size()[3]))
    min_distances = min_distances.view(-1, self.num_prototypes) # B, 2000, min_distances[i] = minimum distance from ith prototype to each of the 7x7 patches
    prototype_activations = self.distance_2_similarity(min_distances) # B, 2000
    logits = self.last_layer(prototype_activations) # B, 200

    return logits, min_distances

  def push_forward(self, x): 
    """
    Original implementation had (x - p)^2 = x^2 + p^2 - 2 x p weirdly. 
    Chaofan told me that it was because PyTorch didn't have fast support for these operations, 
    so they were forced to use convolutions. Let's not take this path and reimplement this. 
    We have input patch of shape (B, D, 7, 7) and prototypes of shape (2000, D, 1, 1) 
    We want to compute the distances (B, 2000, 7, 7), where (i1, i2, i3, i4) represents the distance of the 'i2'th prototype 
    with the (i3, i4)th patch in sample 'i1'. We then max pool to get (B, 2000). 
   
    The differences between this and original function seem like numerical precision errors. 
    """
    x = self.backbone(x) 
    x = self.add_on_layers(x) # (B, 128, 7, 7)  

    B, D, H, W = x.size()
    p = self.prototype_vectors  # (2000, 128, 1, 1)
    p = p.expand(-1, -1, H, W)  # (2000, 128, 7, 7)

    x_ = x.unsqueeze(1).expand(-1, p.size(0), -1, -1, -1) # (B, 2000, 128, 7, 7) 
    
    # compute squared differences of all elements 
    distances = (x_ - p) ** 2  # (B, 2000, 128, 7, 7)

    # now compute distances between vectors by summing them up 
    distances = distances.sum(dim=2) # (B, 2000, 7, 7)

    return x, distances

  def warm_only(self): 
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.add_on_layers.parameters():
      p.requires_grad = True
    self.prototype_vectors.requires_grad = True
    for p in self.last_layer.parameters():
      p.requires_grad = True
    print("warm")
  
  def joint(self): 
    for p in self.backbone.parameters():
      p.requires_grad = True 
    for p in self.add_on_layers.parameters():
      p.requires_grad = True
    self.prototype_vectors.requires_grad = True
    for p in self.last_layer.parameters():
      p.requires_grad = True
    print("joint")

  def last_only(self): 
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.add_on_layers.parameters():
      p.requires_grad = False 
    self.prototype_vectors.requires_grad = False 
    for p in self.last_layer.parameters():
      p.requires_grad = True
    print("last")
