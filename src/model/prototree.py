import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .resnet import ResNet, BasicBlock, Bottleneck


class ProtoTree(nn.Module):
  """ProtoTree: interpretable classification by a soft binary decision tree.

  Every branch owns one prototype and routes a sample right with probability equal to
  that prototype's similarity to the sample; every leaf holds a class distribution that
  is fitted by a derivative-free update rather than by gradient descent. The prediction
  is the path-probability-weighted mixture of the leaf distributions.

  This follows protopnext-image-genetic (protopnet/models/prototree.py plus
  ProtoTreePredictionHead in protopnet/prediction_heads.py), which in turn follows
  Nauta et al.'s reference implementation. Three things are worth stating up front
  because they differ from this repo's ProtoPNet:

  - similarity is the cosine activation relu(cos(x, p)), never an L2 distance, so the
    routing probability of a branch is directly the cosine similarity of its prototype;
  - the add-on stack has no sigmoid by default -- it would compress every latent patch
    into the positive orthant and flatten the cosine, which is also why the reference
    refuses to build a tree with one;
  - forward returns *log* probabilities, so the classification loss is a plain NLL.
  """

  # Constants of the cosine activation, named so the arithmetic below reads as
  # the reference does. The normalization constant cancels out of the result.
  ACTIVATION_EPSILON = 1e-4
  ACTIVATION_NORMALIZATION = 64

  class Node(nn.Module):
    """Shared interface for the two node types."""

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
    """Internal node; routes right with the similarity of its own prototype."""

    def __init__(self, index: int, left: 'ProtoTree.Node', right: 'ProtoTree.Node',
                 log_probabilities: bool):
      super().__init__(index)
      # Assigning the children as attributes is what registers the whole tree as
      # submodules of the model, so leaf distributions land in state_dict under a
      # stable path. A set or a ModuleList built from one would not be ordered.
      self.left = left
      self.right = right
      self._log_probabilities = log_probabilities

    def g(self, out_map, conv_net_output):
      """This branch's routing probability: the similarity of its own prototype."""
      return conv_net_output[out_map[self]].squeeze(dim=1)

    def forward(self, xs: Tensor, out_map: dict, conv_net_output, attr: dict,
                hard_routing: bool = False) -> Tuple[Tensor, dict]:
      batch_size = xs.size(0)
      # The probability of arriving here. A parent sets it on its children before
      # recursing, so a node that does not find one is the root and arrives with
      # certainty.
      #
      # That certainty is written as 1 in both modes, which in log space should be 0.
      # Both Nauta's implementation and protopnext do the same, so the root's path
      # probability -- and through it every descendant's, since pa accumulates
      # additively -- carries a constant offset of 1 under log_probabilities, which
      # the derivative-free leaf update then inherits. Reproduced deliberately: the
      # logits are unaffected, and changing it here would silently diverge from both
      # references. Leaves, in contrast, do default to 0 in log space, in both.
      pa = attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))

      ps = self.g(out_map, conv_net_output)   # (B,)
      attr[self, 'ps'] = ps

      if hard_routing:
        # Evaluation-only mode: commit to one child instead of mixing. The decision
        # is the same quantity, thresholded, so ps is untouched and the result is
        # exactly the distribution of the single leaf the sample walks to.
        decision = (ps > 0.5).float()
        attr[self.left, 'pa'] = (1 - decision) * pa
        attr[self.right, 'pa'] = decision * pa
        l_dists, _ = self.left.forward(xs, out_map, conv_net_output, attr, hard_routing)
        r_dists, _ = self.right.forward(xs, out_map, conv_net_output, attr, hard_routing)
        decision = decision.view(batch_size, 1)
        return (1 - decision) * l_dists + decision * r_dists, attr

      if not self._log_probabilities:
        attr[self.left, 'pa'] = (1 - ps) * pa
        attr[self.right, 'pa'] = ps * pa
        l_dists, _ = self.left.forward(xs, out_map, conv_net_output, attr, hard_routing)
        r_dists, _ = self.right.forward(xs, out_map, conv_net_output, attr, hard_routing)
        ps = ps.view(batch_size, 1)
        return (1 - ps) * l_dists + ps * r_dists, attr

      # Log-probability arithmetic. log(1 - exp(-x)) is split at log 2 because
      # expm1 and log1p each lose precision on the other side of it.
      # Ported from tensorflow_probability's log1mexp.
      x = torch.abs(ps) + 1e-7
      oneminusp = torch.where(x < math.log(2),
                              torch.log(-torch.expm1(-x)),
                              torch.log1p(-torch.exp(-x)))
      attr[self.left, 'pa'] = oneminusp + pa
      attr[self.right, 'pa'] = ps + pa
      l_dists, _ = self.left.forward(xs, out_map, conv_net_output, attr, hard_routing)
      r_dists, _ = self.right.forward(xs, out_map, conv_net_output, attr, hard_routing)
      ps = ps.view(batch_size, 1)
      oneminusp = oneminusp.view(batch_size, 1)
      return torch.logsumexp(torch.stack((oneminusp + l_dists, ps + r_dists)), dim=0), attr

    @property
    def size(self) -> int:
      return 1 + self.left.size + self.right.size

    @property
    def leaves(self) -> set:
      return self.left.leaves.union(self.right.leaves)

    @property
    def branches(self) -> set:
      return {self}.union(self.left.branches).union(self.right.branches)

    @property
    def nodes_by_index(self) -> dict:
      return {self.index: self, **self.left.nodes_by_index, **self.right.nodes_by_index}

    @property
    def num_branches(self) -> int:
      return 1 + self.left.num_branches + self.right.num_branches

    @property
    def num_leaves(self) -> int:
      return self.left.num_leaves + self.right.num_leaves

    @property
    def depth(self) -> int:
      return self.left.depth + 1

  class Leaf(Node):
    """Terminal node; holds the class distribution fitted without gradients."""

    def __init__(self, index: int, num_classes: int, log_probabilities: bool,
                 disable_derivative_free_leaf_optim: bool):
      super().__init__(index)
      self._log_probabilities = log_probabilities
      if disable_derivative_free_leaf_optim:
        # Trained by the optimizer like any other parameter.
        self._dist_params = nn.Parameter(torch.randn(num_classes), requires_grad=True)
      else:
        # Fitted by batch_derivative_free_update, which writes the tensor in place.
        self._dist_params = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

    def distribution(self) -> Tensor:
      if self._log_probabilities:
        return F.log_softmax(self._dist_params + 1e-10, dim=0)
      # Shift by the max before the softmax: the update rule drives dist_params to
      # unbounded positive values, which overflow exp() without it.
      return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)

    def forward(self, xs: Tensor, out_map: dict, conv_net_output, attr: dict,
                hard_routing: bool = False) -> Tuple[Tensor, dict]:
      batch_size = xs.size(0)
      attr.setdefault(
          (self, 'pa'),
          torch.zeros(batch_size, device=xs.device) if self._log_probabilities
          else torch.ones(batch_size, device=xs.device))
      dists = self.distribution().repeat(batch_size, 1)   # (B, num_classes)
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

  def __init__(self, img_size, backbone: str, depth: int, num_classes: int,
               proto_channels: int = 0, num_addon_layers: int = 1,
               include_sigmoid: bool = False, log_probabilities: bool = False,
               k_for_topk: int = 1, pruning_threshold: float = 0.0,
               disable_derivative_free_leaf_optim: bool = False,
               backbone_checkpoint: str = ""):
    super().__init__()
    self.img_size = img_size
    self.tree_depth = depth
    self.num_classes = num_classes
    self.log_probabilities = log_probabilities
    self.k_for_topk = k_for_topk
    self.pruning_threshold = pruning_threshold
    # True means the leaves are ordinary parameters trained by an optimizer; False
    # (the reference default) means the derivative-free update owns them.
    self.gradient_leaves = disable_derivative_free_leaf_optim

    # num_classes=0 drops ResNet's fc/avgpool, so the backbone returns the
    # (B, C, H, W) feature map the prototype layer needs.
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
      case "resnet50[pretraining=inaturalist]":
        self.backbone = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=0,
                               pretrained="resnet50[pretraining=inaturalist]")
      case "resnet101":
        self.backbone = ResNet(block=Bottleneck, layers=[3, 4, 23, 3],
                               num_classes=0, pretrained="resnet101")
      case "resnet152":
        self.backbone = ResNet(block=Bottleneck, layers=[3, 8, 36, 3],
                               num_classes=0, pretrained="resnet152")
      case _:
        raise ValueError(
          f"Backbone {backbone!r} not valid. Choose resnet18, resnet34, resnet50, "
          "resnet50[pretraining=inaturalist], resnet101 or resnet152.")

    if backbone_checkpoint:
      # Weights from training this backbone as a plain classifier on the same data.
      # Their keys sit at the top level, so they are re-prefixed onto self.backbone,
      # and the classifier head is dropped because num_classes=0 leaves no fc to
      # receive it. strict=True then still has to see the exact key set -- a partial
      # load would otherwise succeed in silence and train from scratch.
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

    backbone_channels = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)][-1].out_channels
    # proto_channels: 0 means "keep whatever the trunk emits", matching the reference,
    # where the add-on is a width-preserving 1x1 conv unless asked otherwise.
    self.proto_channels = int(proto_channels) if proto_channels else backbone_channels

    # One 1x1 conv by default, and no sigmoid. A sigmoid would push every latent
    # patch into the positive orthant, where all cosines crowd towards 1 and the
    # routing probabilities lose their spread; the reference refuses to build with one.
    mid_layers = []
    for _ in range(num_addon_layers - 1):
      mid_layers += [nn.ReLU(), nn.Conv2d(self.proto_channels, self.proto_channels, kernel_size=1)]
    self.add_on_layers = nn.Sequential(
      nn.Conv2d(backbone_channels, self.proto_channels, kernel_size=1),
      *mid_layers,
      nn.Sigmoid() if include_sigmoid else nn.Identity(),
    )

    # A complete binary tree of this depth has one prototype per internal node.
    self.num_prototypes = 2 ** depth - 1
    self.prototype_shape = (self.num_prototypes, self.proto_channels, 1, 1)
    self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

    self._root = self._init_tree(num_classes, depth, log_probabilities,
                                 disable_derivative_free_leaf_optim)
    self._parents: dict = {}
    self._set_parents()

    # Branch -> prototype index. Sorted by node index so the mapping is identical on
    # every construction; iterating the branch *set* would shuffle it run to run and
    # make two runs of the same seed disagree.
    self._out_map = {n: i for i, n in enumerate(sorted(self.branches, key=lambda n: n.index))}

    self._initialize_weights()

  def _initialize_weights(self):
    """Reference init: xavier-for-sigmoid on the add-on convs, N(0.5, 0.1) prototypes."""
    for m in self.add_on_layers.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
    with torch.no_grad():
      nn.init.normal_(self.prototype_vectors, mean=0.5, std=0.1)

  def _init_tree(self, num_classes: int, depth: int, log_probabilities: bool,
                 disable_derivative_free_leaf_optim: bool) -> 'ProtoTree.Node':
    def _recursive(i: int, d: int) -> 'ProtoTree.Node':
      if d == depth:
        return ProtoTree.Leaf(i, num_classes, log_probabilities,
                              disable_derivative_free_leaf_optim)
      left = _recursive(i + 1, d + 1)
      return ProtoTree.Branch(i, left, _recursive(i + left.size + 1, d + 1),
                              log_probabilities)
    return _recursive(0, 0)

  def _set_parents(self) -> None:
    self._parents.clear()
    self._parents[self._root] = None

    def _recurse(node):
      if isinstance(node, ProtoTree.Branch):
        self._parents[node.right] = node
        self._parents[node.left] = node
        _recurse(node.right)
        _recurse(node.left)
    _recurse(self._root)

  @property
  def root(self) -> 'ProtoTree.Node':
    return self._root

  @property
  def nodes(self) -> set:
    return self._root.nodes

  @property
  def leaves(self) -> set:
    return self._root.leaves

  @property
  def branches(self) -> set:
    return self._root.branches

  @property
  def num_branches(self) -> int:
    return self._root.num_branches

  @property
  def num_leaves(self) -> int:
    return self._root.num_leaves

  @property
  def nodes_by_index(self) -> dict:
    return self._root.nodes_by_index

  def path_to(self, node: 'ProtoTree.Node') -> List['ProtoTree.Node']:
    assert node in self.leaves or node in self.branches
    path = [node]
    while isinstance(self._parents[node], ProtoTree.Node):
      node = self._parents[node]
      path = [node] + path
    return path

  # -- similarity -------------------------------------------------------------

  def prototype_activations(self, features: Tensor,
                            prototypes: Optional[Tensor] = None) -> Tensor:
    """relu(cos) between every latent patch and every prototype, in [0, 1].

    features is the (B, D, H, W) add-on output; the result is (B, P, H, W). prototypes
    defaults to all of them; pass a (K, D, 1, 1) subset to score only those.

    This reproduces CosPrototypeActivation from the reference arithmetic step for step,
    including the asymmetry in where the epsilon sits (inside the square root for the
    patches, added to the norm for the prototypes) and the 1 + 1e-5 denominator that
    keeps the result strictly below 1. Patch and prototype are both scaled by the same
    normalization constant, which then divides back out, so the value really is the
    cosine -- the constant only keeps the convolution off denormal magnitudes.
    """
    if prototypes is None:
      prototypes = self.prototype_vectors
    eps = self.ACTIVATION_EPSILON
    scale = self.ACTIVATION_NORMALIZATION
    # 1 for the 1x1 prototypes used throughout; kept so the expression matches.
    normalizing_factor = (prototypes.shape[-2] * prototypes.shape[-1]) ** 0.5

    x_length = torch.sqrt(torch.sum(torch.square(features), dim=-3) + eps)
    x_length = x_length.view(x_length.size(0), 1, x_length.size(1), x_length.size(2))
    x_normalized = scale * features / x_length / normalizing_factor

    p_length = torch.sqrt(torch.sum(torch.square(prototypes), dim=-3) + eps)
    p_length = p_length.view(p_length.size(0), 1, p_length.size(1), p_length.size(2))
    p_normalized = prototypes / (p_length + eps) / normalizing_factor

    activations = F.conv2d(x_normalized, p_normalized) / (scale * (1 + 1e-5))
    return torch.relu(activations)


  # -- forward ----------------------------------------------------------------

  def tree_forward(self, activations: Tensor, hard_routing: bool = False) -> Tuple[Tensor, Dict]:
    """Route a batch of prototype activations through the tree.

    Returns (logits, info). logits are log class probabilities either way: the tree
    mixes probabilities when log_probabilities is off, and the log is taken at the end
    so the caller always has something an NLL loss can consume.
    """
    batch_size, num_prototypes = activations.shape[0], activations.shape[1]
    flat = activations.view(batch_size, num_prototypes, -1)

    # One similarity per prototype: the mean of its k strongest patches. k=1 is the max.
    k = min(self.k_for_topk, flat.shape[-1])
    topk_activations, _ = torch.topk(flat, k, dim=-1)
    similarity_score_to_each_prototype = torch.mean(topk_activations, dim=-1)  # (B, P)

    similarities = similarity_score_to_each_prototype
    if self.log_probabilities:
      similarities = torch.log(torch.clamp(similarities, min=1e-8))

    # One (B, 1) column per branch; out_map says which column each branch reads.
    conv_net_output = similarities.chunk(similarities.size(1), dim=1)

    attr: dict = {}
    logits, attr = self._root.forward(flat, dict(self._out_map), conv_net_output,
                                      attr, hard_routing)

    if not self.log_probabilities:
      logits = torch.log(torch.clamp(logits, min=1e-8))

    info = {
      "logits": logits,
      "pa_tensor": {n.index: attr[n, 'pa'].unsqueeze(1) for n in self.nodes},
      "ps": {n.index: attr[n, 'ps'].unsqueeze(1) for n in self.branches},
      "similarity_score_to_each_prototype": similarity_score_to_each_prototype,
    }
    return logits, info

  def forward(self, x: Tensor, hard_routing: bool = False) -> Tuple[Tensor, Dict]:
    x = self.backbone(x)
    x = self.add_on_layers(x)                       # (B, D, H, W)
    activations = self.prototype_activations(x)     # (B, P, H, W)
    return self.tree_forward(activations, hard_routing=hard_routing)

  def push_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    """The latent patches and their activations against every prototype, for push."""
    x = self.backbone(x)
    x = self.add_on_layers(x)            # (B, D, H, W)
    return x, self.prototype_activations(x)

  def hard_routing_predictions(self, info: Dict) -> Tensor:
    """Argmax class of the single leaf each sample reaches when routed hard.

    A branch's routing probability does not depend on how its siblings were routed, so
    the hard path can be read off the ps values a soft forward already produced -- same
    answer as forward(hard_routing=True), without a second pass over the network.
    """
    ps = info["ps"]
    batch_size = next(iter(ps.values())).shape[0]
    device = next(iter(ps.values())).device
    nodes = self.nodes_by_index

    # Walk every sample down the tree together, one level per iteration.
    current = torch.zeros(batch_size, dtype=torch.long, device=device)   # root index 0
    for _ in range(self.tree_depth):
      nxt = current.clone()
      for node_index in current.unique().tolist():
        node = nodes[node_index]
        if not isinstance(node, ProtoTree.Branch):
          continue
        here = current == node_index
        go_right = ps[node_index].squeeze(1) > 0.5
        nxt[here & go_right] = node.right.index
        nxt[here & ~go_right] = node.left.index
      if torch.equal(nxt, current):
        break      # every sample has reached a leaf
      current = nxt

    predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
    for leaf_index in current.unique().tolist():
      predictions[current == leaf_index] = int(nodes[leaf_index].distribution().argmax())
    return predictions

  # -- leaf distributions -----------------------------------------------------

  @torch.no_grad()
  def batch_derivative_free_update(self, info: Dict, target: Tensor, num_batches: int,
                                   old_leaf_dist: Dict) -> None:
    """Fit the leaf distributions to this batch without gradients.

    Each leaf accumulates the class mass that actually arrived at it, weighted by how
    much of each sample's path probability reached it and discounted by how much the
    model already predicted. old_leaf_dist is the snapshot taken once at the start of
    the epoch, and one num_batches-th of it is retired per batch, so over a full epoch
    the previous distribution is replaced rather than added to.
    """
    logits = info["logits"]
    target_dist = F.one_hot(target, num_classes=self.num_classes)

    if self.log_probabilities:
      target_dist = torch.log(torch.clamp(target_dist.float(), min=1e-8))
      predicted_dist = logits
    else:
      # forward always hands back log probabilities; the update rule divides by the
      # predicted probability, so undo the log.
      predicted_dist = torch.clamp(torch.exp(logits), min=1e-8, max=1.0)

    for leaf in self.leaves:
      if self.log_probabilities:
        update = torch.exp(torch.logsumexp(
            info["pa_tensor"][leaf.index] + leaf.distribution() + target_dist
            - predicted_dist, dim=0))
      else:
        update = torch.sum(
            (info["pa_tensor"][leaf.index] * leaf.distribution() * target_dist)
            / predicted_dist, dim=0)
      leaf._dist_params -= old_leaf_dist[leaf] / num_batches
      # Floating point can take the retired mass slightly below zero; clamp before
      # adding, or the softmax starts seeing negative counts.
      F.relu_(leaf._dist_params)
      leaf._dist_params += update

  def leaf_distribution_snapshot(self) -> Dict['ProtoTree.Leaf', Tensor]:
    """The per-leaf state batch_derivative_free_update retires over the coming epoch."""
    with torch.no_grad():
      return {leaf: leaf._dist_params.detach().clone() for leaf in self.leaves}

  # -- pruning ----------------------------------------------------------------

  def check_prune_threshold(self) -> list:
    """Indices of nodes whose every descendant leaf is too unconfident to keep."""
    def has_max_prob_lower_threshold(node) -> bool:
      leaves = [node] if isinstance(node, ProtoTree.Leaf) else node.leaves
      for leaf in leaves:
        dist = leaf.distribution()
        prob = torch.max(torch.exp(dist)) if leaf._log_probabilities else torch.max(dist)
        if prob.item() > self.pruning_threshold:
          return False
      return True

    return [node.index for node in self.nodes if has_max_prob_lower_threshold(node)]

  def prune_tree(self) -> list:
    """Cut the unconfident subtrees out and return the prototype indices freed.

    Removing a node means its parent is left with one child, so the parent collapses
    and the surviving sibling is spliced into the grandparent. The prototypes that
    belonged to the removed branches are then dropped and the rest re-indexed, keeping
    out_map contiguous and aligned with the prototype tensor.
    """
    prune_node_indices = self.check_prune_threshold()

    # A node already inside a doomed subtree does not need pruning of its own.
    to_prune = list(prune_node_indices)
    for node_idx in prune_node_indices:
      node = self.nodes_by_index[node_idx]
      if isinstance(node, ProtoTree.Branch) and node_idx > 0:
        for child in node.nodes:
          if child.index in to_prune and child.index != node_idx:
            to_prune.remove(child.index)

    for node_idx in to_prune:
      node = self.nodes_by_index[node_idx]
      parent = self._parents[node]
      # The root has no grandparent to splice into, so a cut that high is skipped.
      if parent is None or parent.index == 0:
        continue
      grandparent = self._parents[parent]
      sibling = parent.right if node is parent.left else parent.left
      self._parents[sibling] = grandparent
      if parent is grandparent.left:
        grandparent.left = sibling
      else:
        grandparent.right = sibling

    remaining = self.branches
    pruned_prototype_indices = []
    for branch, prototype_idx in dict(self._out_map).items():
      if branch not in remaining:
        del self._out_map[branch]
        pruned_prototype_indices.append(prototype_idx)

    for new_idx, (branch, _) in enumerate(sorted(self._out_map.items(), key=lambda x: x[1])):
      self._out_map[branch] = new_idx

    return sorted(pruned_prototype_indices)

  def prune_prototypes(self) -> list:
    """Prune the tree and drop the prototype vectors its removed branches owned."""
    pruned = self.prune_tree()
    if pruned:
      keep = [i for i in range(self.prototype_vectors.shape[0]) if i not in set(pruned)]
      # Written through .data rather than rebound, matching the reference. The
      # Parameter object survives, but its shape changes, so any optimizer already
      # holding moment estimates for it has to be rebuilt -- the trainer does that.
      self.prototype_vectors.data = self.prototype_vectors.data[keep].clone()
      self.num_prototypes = len(keep)
      self.prototype_shape = tuple(self.prototype_vectors.shape)
    return pruned

  # -- training stages --------------------------------------------------------

  def warm_only(self, backbone_overrides=()):
    """Add-on layers and prototypes only; the trunk stays frozen.

    backbone_overrides names trunk submodules that train anyway -- the reference thaws
    the last residual block during warm-up so the features the prototypes are being
    fitted to are not completely fixed.
    """
    for name, p in self.backbone.named_parameters():
      p.requires_grad = any(override in name for override in backbone_overrides)
    for p in self.add_on_layers.parameters():
      p.requires_grad = True
    self.prototype_vectors.requires_grad = True
    self._freeze_leaves()
    print("warm")

  def joint(self):
    for p in self.backbone.parameters():
      p.requires_grad = True
    for p in self.add_on_layers.parameters():
      p.requires_grad = True
    self.prototype_vectors.requires_grad = True
    self._freeze_leaves()
    print("joint")

  def _freeze_leaves(self):
    """Take gradients off the leaf distributions for the warm and joint stages.

    The reference sets train_prototype_prediction_head=False in both, and its
    set_training_layers writes requires_grad on every parameter of the named
    component, so the leaves are frozen there whatever they were constructed with.
    Without this they would stay trainable under disable_derivative_free_leaf_optim,
    and the derivative-free update -- which writes _dist_params in place -- would be
    fighting autograd over the same tensor.
    """
    for leaf in self.leaves:
      leaf._dist_params.requires_grad = False

  def last_only(self):
    """Freeze the trunk, add-on and prototypes; leave only the leaves free to move.

    There is no counterpart to this stage in the reference, which runs warm then
    joint and stops. It exists so ProtoTree fits the schedule the other models in
    this repo use. Under the default derivative-free leaves nothing here is
    gradient-trained and only batch_derivative_free_update moves anything; with
    disable_derivative_free_leaf_optim the leaves are handed back to the optimizer,
    which is the only reading of a "last layer only" stage that does any work.
    """
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.add_on_layers.parameters():
      p.requires_grad = False
    self.prototype_vectors.requires_grad = False
    for leaf in self.leaves:
      leaf._dist_params.requires_grad = self.gradient_leaves
    print("last")
