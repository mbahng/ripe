from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from .resnet import ResNet, BasicBlock, Bottleneck
from ..receptive_field import compute_proto_layer_rf_info_v2

class ProtoPNet(nn.Module): 

  def __init__(self, img_size, backbone: str, prototype_shape: tuple, num_classes: int,
               backbone_checkpoint: str = ""): 
    super().__init__()
    self.img_size = img_size

    # num_classes=0 drops ResNet's fc/avgpool, so forward() returns the
    # (B, C, H, W) feature map ProtoPNet needs, and conv_info() supplies the
    # receptive-field geometry. Replaces the deleted backbone.py helpers.
    match backbone: 
      case "resnet18": 
        self.backbone = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                               num_classes=0, pretrained="resnet18")
      case "resnet34": 
        self.backbone = ResNet(block=BasicBlock, layers=[3, 4, 6, 3],
                               num_classes=0, pretrained="resnet34")
      case "resnet50": 
        self.backbone = ResNet(block=Bottleneck, layers=[3, 4, 6, 3],
                               num_classes=0, pretrained="resnet50")
      case "resnet101": 
        self.backbone = ResNet(block=Bottleneck, layers=[3, 4, 23, 3],
                               num_classes=0, pretrained="resnet101")
      case "resnet152": 
        self.backbone = ResNet(block=Bottleneck, layers=[3, 8, 36, 3],
                               num_classes=0, pretrained="resnet152")
      case _: 
        raise ValueError(
          f"Backbone {backbone!r} not valid. Choose resnet18, resnet34, resnet50, resnet101 or resnet152.")

    if backbone_checkpoint:
      # Weights from training this backbone as a plain classifier on the same data.
      # Their keys sit at the top level, so they are re-prefixed onto self.backbone,
      # and the classifier head is dropped because num_classes=0 leaves no fc to
      # receive it. Everything else keeps its fresh initialisation. strict=True then
      # still has to see the exact key set, and the trunk is checked separately --
      # a partial load would otherwise succeed in silence and train from scratch.
      weights = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)
      weights = {f"backbone.{k}": v for k, v in weights.items() if not k.startswith("fc.")}
      expected = {k for k in self.state_dict() if k.startswith("backbone.")}
      if set(weights) != expected:
        missing, extra = sorted(expected - set(weights)), sorted(set(weights) - expected)
        raise ValueError(
            f"{backbone_checkpoint} does not match the {backbone} trunk: "
            f"{len(missing)} missing (e.g. {missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")
      merged = self.state_dict()
      merged.update(weights)
      self.load_state_dict(merged, strict=True)
      print(f"Loaded {len(weights)} backbone tensors from {backbone_checkpoint}")

    first_add_on_layer_in_channels = [i for i in self.backbone.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

    self.prototype_shape = prototype_shape 
    if self.prototype_shape[0] % num_classes != 0: 
        raise ValueError("The number of prototypes must be a multiple of the number of classes.")
    self.num_classes = num_classes
    self.num_prototypes = prototype_shape[0]
    if self.prototype_shape[2] != 1 or self.prototype_shape[3] != 1:
      raise ValueError(
        f"Only 1x1 prototypes are supported, got {self.prototype_shape[2]}x{self.prototype_shape[3]}. "
        "The cosine is taken over the channel dimension of a single latent patch.")

    self.add_on_layers = nn.Sequential(
      nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
      nn.Sigmoid()
      )
    self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

    self.prototype_activation_function = 'cosine'
    # Upper bound on the dissimilarity this model can report: cosine distance
    # 1 - cos(x, p) lives in [0, 2]. The cluster and separation costs need it to
    # invert a masked minimum, so it is published here rather than re-derived.
    self.max_distance = 2.0
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
    """Cosine distance back to the cosine similarity it was built from."""
    return 1.0 - distances

  def distances(self, features, prototypes=None):
    """Cosine distance from every prototype to every latent patch.

    features is the (B, D, H, W) add-on output; the result is (B, P, H, W), where
    [i1, i2, i3, i4] is 1 - cos(patch (i3, i4) of sample i1, prototype i2), in [0, 2].
    prototypes defaults to all of them; pass a (K, D, 1, 1) subset to score only those,
    which is how the visualisation keeps its distance tensor small.

    Once both sides are L2 normalized along the channel dimension a 1x1 convolution
    *is* the cosine, so there is no (B, P, D, H, W) broadcast to materialize and no
    x^2 + p^2 - 2xp rewrite to be careful about -- the dot product is the whole of it.
    Normalizing the prototypes rather than constraining them keeps push exact: a
    prototype set to a real patch has cosine 1, whatever that patch's magnitude is.
    """
    if prototypes is None:
      prototypes = self.prototype_vectors
    x = F.normalize(features, p=2, dim=1, eps=1e-8)      # (B, D, H, W)
    w = F.normalize(prototypes, p=2, dim=1, eps=1e-8)    # (P, D, 1, 1)
    return 1.0 - F.conv2d(input=x, weight=w)             # (B, P, H, W)

  def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
    x = self.backbone(x)
    x = self.add_on_layers(x)                        # (B, 128, 7, 7)

    distances = self.distances(x)                    # (B, 2000, 7, 7)
    # Closest patch per prototype, i.e. the most similar one: min over cosine
    # distance is max over cosine similarity.
    min_distances = distances.amin(dim=(-2, -1))     # (B, 2000)
    prototype_activations = self.distance_2_similarity(min_distances) # (B, 2000), in [-1, 1]
    logits = self.last_layer(prototype_activations)  # (B, 200)

    return logits, min_distances

  def push_forward(self, x):
    """The latent patches and their cosine distances to every prototype, for push."""
    x = self.backbone(x)
    x = self.add_on_layers(x)       # (B, 128, 7, 7)
    return x, self.distances(x)     # (B, 128, 7, 7), (B, 2000, 7, 7)

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

