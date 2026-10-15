from typing import Tuple, Dict
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from .resnet import ResNet


class ProtoTree(nn.Module):
  """
  ProtoTree: interpretable image classification via a soft binary decision tree.
  Each Branch node routes samples via prototype similarity; each Leaf holds a class distribution.
  Leaf distributions are updated via a derivative-free method each batch.
  """

  class Node(nn.Module):
    """Abstract base class for tree nodes."""

    def __init__(self, index: int):
      super().__init__()
      self._index = index

    def forward(self, *args, **kwargs):
      raise NotImplementedError

    @property
    def index(self) -> int:
      return self._index

    @property
    def size(self) -> int:
      raise NotImplementedError

    @property
    def nodes(self) -> set:
      return self.branches.union(self.leaves)

    @property
    def leaves(self) -> set:
      raise NotImplementedError

    @property
    def branches(self) -> set:
      raise NotImplementedError

    @property
    def nodes_by_index(self) -> dict:
      raise NotImplementedError

    @property
    def num_branches(self) -> int:
      return len(self.branches)

    @property
    def num_leaves(self) -> int:
      return len(self.leaves)

    @property
    def depth(self) -> int:
      raise NotImplementedError

  class Branch(Node):
    """Internal branch node — routes via prototype similarity score."""

    def __init__(self, index: int, left: 'ProtoTree.Node', right: 'ProtoTree.Node', log_probabilities: bool = False):
      super().__init__(index)
      self.l = left
      self.r = right
      self.log_probabilities = log_probabilities

    def g(self, out_map, conv_net_output):
      """Returns the probability of routing to the right subtree."""
      out = conv_net_output[out_map[self]]
      return out.squeeze(dim=1)

    def forward(self, xs, out_map, conv_net_output, attr):
      batch_size = xs.size(0)
      pa = attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
      ps = self.g(out_map, conv_net_output)
      attr[self, 'ps'] = ps
      attr[self.l, 'pa'] = (1 - ps) * pa
      attr[self.r, 'pa'] = ps * pa
      l_dists, _ = self.l.forward(xs, out_map, conv_net_output, attr)
      r_dists, _ = self.r.forward(xs, out_map, conv_net_output, attr)
      ps = ps.view(batch_size, 1)
      return (1 - ps) * l_dists + ps * r_dists, attr

    @property
    def size(self) -> int:
      return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
      return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
      return {self}.union(self.l.branches).union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
      return {self.index: self, **self.l.nodes_by_index, **self.r.nodes_by_index}

    @property
    def num_branches(self) -> int:
      return 1 + self.l.num_branches + self.r.num_branches

    @property
    def num_leaves(self) -> int:
      return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) -> int:
      return self.l.depth + 1

  class Leaf(Node):
    """Leaf node — holds a learnable class distribution updated derivative-free."""

    def __init__(self, index: int, num_classes: int):
      super().__init__(index)
      self._dist_params = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

    def distribution(self):
      return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)

    def forward(self, xs: Tensor, out_map, conv_net_output, attr: dict) -> Tuple[Tensor, Dict]:
      batch_size = xs.size(0)
      attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
      dist = self.distribution()       # (num_classes,)
      dist = dist.view(1, -1)          # (1, num_classes)
      dists = dist.expand(batch_size, -1)  # (B, num_classes)
      attr[self, 'ds'] = dists
      return dists, attr

    @property
    def requires_grad(self) -> bool:
      return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
      self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
      return 1

    @property
    def leaves(self) -> set:
      return {self}

    @property
    def branches(self) -> set:
      return set()

    @property
    def nodes_by_index(self) -> dict:
      return {self.index: self}

    @property
    def num_branches(self) -> int:
      return 0

    @property
    def num_leaves(self) -> int:
      return 1

    @property
    def depth(self) -> int:
      return 0

  def __init__(self, depth: int, hdim: int = 128, num_classes: int = 200, log_probabilities: bool = False):
    super().__init__()
    self.depth = depth
    self.num_prototypes = 2 ** depth - 1
    self.prototype_shape = (self.num_prototypes, hdim, 1, 1)
    self.num_classes = num_classes
    self.log_probabilities = log_probabilities

    self.backbone = resnet34_features(pretrained=True)

    first_add_on_in_ch = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)][-1].out_channels
    self.add_on_layers = nn.Sequential(
      nn.Conv2d(first_add_on_in_ch, hdim, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(hdim, hdim, kernel_size=1),
      nn.Sigmoid()
    )

    self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

    self._root = self._init_tree(self.num_classes)
    self._parents = dict()
    self._set_parents()

    self._out_map = {n: i for i, n in zip(range(self.num_prototypes), self.branches)}

    # Register all nodes so leaf _dist_params appear in model.parameters()
    self._all_nodes = nn.ModuleList(list(self._root.nodes))

    self._initialize_weights()

  def _initialize_weights(self):
    for m in self.add_on_layers.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
    nn.init.normal_(self.prototype_vectors, mean=0.5, std=0.1)

  def _init_tree(self, num_classes: int) -> 'ProtoTree.Node':
    def _recursive(i: int, d: int):
      if d == self.depth:
        return self.Leaf(i, num_classes)
      left = _recursive(i + 1, d + 1)
      right = _recursive(i + left.size + 1, d + 1)
      return self.Branch(i, left, right, self.log_probabilities)
    return _recursive(0, 0)

  def _set_parents(self) -> None:
    self._parents.clear()
    self._parents[self._root] = None

    def _recurse(node):
      if isinstance(node, self.Branch):
        self._parents[node.l] = node
        self._parents[node.r] = node
        _recurse(node.l)
        _recurse(node.r)
    _recurse(self._root)

  @property
  def nodes(self) -> set:
    return self._root.nodes

  @property
  def leaves(self) -> set:
    return self._root.leaves

  @property
  def branches(self) -> set:
    return self._root.branches

  def prototype_layer(self, x: Tensor) -> Tensor:
    """Compute L2 distance between all input patches and all prototype vectors."""
    ones = torch.ones_like(self.prototype_vectors, device=x.device)
    x_squared_l2 = F.conv2d(x ** 2, weight=ones)
    p_squared_l2 = torch.sum(self.prototype_vectors ** 2, dim=(1, 2, 3)).view(-1, 1, 1)
    xp = F.conv2d(x, weight=self.prototype_vectors)
    distance = x_squared_l2 + p_squared_l2 - 2 * xp
    distance = torch.sqrt(torch.abs(distance) + 1e-14)
    if torch.isnan(distance).any():
      raise Exception('NaN in prototype distances; try log_probabilities=True')
    return distance

  def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
    x = self.backbone(x)
    x = self.add_on_layers(x)
    B, _, H, W = x.size()

    distances = self.prototype_layer(x)
    min_distances = -F.max_pool2d(-distances, kernel_size=(H, W))
    min_distances = min_distances.view(B, -1)
    similarities = torch.exp(-min_distances)

    conv_net_output = similarities.chunk(similarities.size(1), dim=1)
    out_map = dict(self._out_map)
    attr = dict()

    out, attr = self._root.forward(similarities, out_map, conv_net_output, attr)

    info = dict()
    info['pa_tensor'] = {n.index: attr[n, 'pa'].unsqueeze(1) for n in self.nodes}
    info['ps'] = {n.index: attr[n, 'ps'].unsqueeze(1) for n in self.branches}
    info['min_distances'] = min_distances

    return out, info

  def push_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns (latent_features, distances) for prototype push step."""
    x = self.backbone(x)
    x = self.add_on_layers(x)
    _, _, H, W = x.size()
    p = self.prototype_vectors.expand(-1, -1, H, W)
    x_ = x.unsqueeze(1).expand(-1, p.size(0), -1, -1, -1)
    distances = ((x_ - p) ** 2).sum(dim=2)
    return x, distances

  def batch_derivative_free_update(self, info: dict, targets: Tensor, num_batches: int, old_leaf_dist: dict) -> None:
    """
    Update leaf class distributions without gradient descent.
    For each leaf, blend its distribution toward the arrival-weighted class counts from this batch.
    old_leaf_dist maps leaf node -> _dist_params tensor snapshot from before the forward pass.
    """
    B = targets.size(0)
    one_hot = torch.zeros(B, self.num_classes, device=targets.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    for leaf in self.leaves:
      pa = info['pa_tensor'][leaf.index].squeeze(1)  # (B,)
      leaf_class_counts = (pa.unsqueeze(1) * one_hot).sum(dim=0)
      total = leaf_class_counts.sum()
      if total > 0:
        new_dist = leaf_class_counts / total
      else:
        new_dist = torch.ones(self.num_classes, device=targets.device) / self.num_classes
      leaf._dist_params.data = old_leaf_dist[leaf] + (1.0 / num_batches) * (new_dist - old_leaf_dist[leaf])

  def warm_only(self):
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.add_on_layers.parameters():
      p.requires_grad = True
    self.prototype_vectors.requires_grad = True
    for leaf in self.leaves:
      leaf._dist_params.requires_grad = True
    print("warm")

  def joint(self):
    for p in self.backbone.parameters():
      p.requires_grad = True
    for p in self.add_on_layers.parameters():
      p.requires_grad = True
    self.prototype_vectors.requires_grad = True
    for leaf in self.leaves:
      leaf._dist_params.requires_grad = True
    print("joint")

  def last_only(self):
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.add_on_layers.parameters():
      p.requires_grad = False
    self.prototype_vectors.requires_grad = False
    for leaf in self.leaves:
      leaf._dist_params.requires_grad = True
    print("last_only")
