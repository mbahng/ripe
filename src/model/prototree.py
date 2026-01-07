from typing import Tuple, Dict
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from .backbone import resnet34_features

    
class ProtoTree(nn.Module): 
  """
  The node attributes contain: 
    - pa : probability of landing in this node, basically accumlated ps (on all nodes)
    - ps : probability of going to the right child (only on branches)
    - ds : distributions on the leaves (only on leaves)
  """

  def __init__(self, depth: int, hdim = 128, num_classes = 200):
    super().__init__() 
    self.depth = depth
    self.num_prototypes = 2 ** depth - 1
    self.prototype_shape = (self.num_prototypes, hdim, 1, 1)
    self.num_classes = num_classes 
    self.backbone = resnet34_features(pretrained=True) 
    self.addon_layers = nn.Identity()
    self._root = self._init_tree()
    self._parents = dict()
    self._set_parents()

    self._out_map = {n: i for i, n in zip(range(2 ** depth - 1), self.branches)}

    self.prototype_vectors = ...

  class Node(nn.Module): 
    """
    Abstract class for nodes (internal and leaves in the tree)
    """

    def __init__(self, index: int): 
      super().__init__() 

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
    """
    Branch node. 
    """
    def __init__(self, index: int, left: Node, right: Node): 
      super().__init__(index)
      self.left = left 
      self.right = right 

      # Flag that indicates whether probabilities or log probabilities are computed
      self.log_probabilities = log_probabilities

    def g(self, out_map, conv_net_output): 
      """
      returns the probabilities of taking the right subtree
      """ 
      out = conv_net_output[out_map[self]] 
      return out.squeeze(dim=1)

    def forward(self, xs, out_map, conv_net_output, attr): 
      batch_size = xs.size(0)
      pa = attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device)) # its only really torch.ones if it is root
      ps = self.g(out_map, conv_net_output)
      attr[self, 'ps'] = ps 
      attr[self.l, 'pa'] = (1 - ps) * pa 
      attr[self.r, 'pa'] = ps * pa
      l_dists, _ = self.l.forward(xs, out_map, conv_net_output, attr) # (bs, k)
      r_dists, _ = self.r.forward(xs, out_map, conv_net_output, attr) # (bs, k)
      ps = ps.view(batch_size, 1) # reshape to broadcast 
      return (1 - ps) * l_dists + ps * r_dists, attr

    @property
    def size(self) -> int:
      return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
      return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
      return {self} \
          .union(self.l.branches) \
          .union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
      return {self.index: self,
              **self.l.nodes_by_index,
              **self.r.nodes_by_index}

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
    """
    Leaf node.
    """
    def __init__(self, index: int, num_classes: int):
      super().__init__(index) 
      self._dist_params = nn.Parameter(torch.zeros(num_classes), requires_grad=False) 

    def distribution(self): 
      return F.softmax(self.dist_params - torch.max(self._dist_params), dim=0) 

    def forward(self, xs: Tensor, attr: dict) -> Tuple[Tensor, Dict]:  
      batch_size = xs.size(0)
      attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
      dist = self.distribution()  # (k,)
      dist = dist.view(1, -1)     # (1, k)
      dists = torch.cat((dist,) * batch_size, dim=0) # (batch_size, k) 
      attr[self, 'ds'] = dists 
      return dists, _attr
    
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

  def _init_tree(self, num_classes: int) -> Node: 
    """
    Initialize a tree
    """
    def _init_tree_recursive(i: int, d: int):  
      if d == self.depth: 
        return self.Leaf(i, num_classes)
      else: 
        left = _init_tree_recursive(i + 1, d + 1) 
        right = _init_tree_recursive(i + left.size + 1, d + 1)
        return self.Branch(i, left, right)

    return _init_tree_recursive(0, 0)

  def _set_parents(self) -> None:
    self._parents.clear()
    self._parents[self._root] = None

    def _set_parents_recursively(node):
      if isinstance(node, self.Branch):
        self._parents[node.r] = node
        self._parents[node.l] = node
        _set_parents_recursively(node.r)
        _set_parents_recursively(node.l)
        return
      elif isinstance(node, self.Leaf):
        return  # Nothing to do here!
      raise Exception('Unrecognized node type!')

    # Set all parents by traversing the tree starting from the root
    _set_parents_recursively(self._root)

  def prototype_layer(self, x): 
    """
    Perform convolution over the input using the squared L2 distance for all prototypes in the layer
    :param xs: A batch of input images obtained as output from some convolutional neural network F. Following the
               notation from the paper, let the shape of xs be (batch_size, D, W, H), where
                 - D is the number of output channels of the conv net F
                 - W is the width of the convolutional output of F
                 - H is the height of the convolutional output of F
    :return: a tensor of shape (batch_size, num_prototypes, W, H) obtained from computing the squared L2 distances
             for patches of the input using all prototypes
    """
    # Adapted from ProtoPNet
    # Computing ||xs - ps ||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
    # where ps is some prototype image

    # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
    # with weights set to 1 so each patch just has its values summed)
    ones = torch.ones_like(self.prototype_vectors,
                           device=xs.device)  # Shape: (num_prototypes, num_features, w_1, h_1)
    xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)  # Shape: (bs, num_prototypes, w_in, h_in)

    # Now compute ||ps||^2
    # We can just use a sum here since ||ps||^2 is the same for each patch in the input image when computing the
    # squared L2 distance
    ps_squared_l2 = torch.sum(self.prototype_vectors ** 2,
                              dim=(1, 2, 3))  # Shape: (num_prototypes,)
    # Reshape the tensor so the dimensions match when computing ||xs||^2 + ||ps||^2
    ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)

    # Compute xs * ps (for all patches in the input image)
    xs_conv = F.conv2d(xs, weight=self.prototype_vectors)  # Shape: (bs, num_prototypes, w_in, h_in)

    # Use the values to compute the squared L2 distance
    distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
    distance = torch.sqrt(torch.abs(distance)+1e-14) #L2 distance (not squared). Small epsilon added for numerical stability
    
    if torch.isnan(distance).any():
        raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')
    return distance  # Shape: (bs, num_prototypes, w_in, h_in)

  def forward(self, x): 
    x = self.backbone(x) 
    x = self.addon_layers(x) 
    B, D, H, W = x.size()

    distances = self.prototype_layer(x) 
    # Perform global min pooling to see the minimal distance for each prototype to any patch of the input image
    min_distances = -F.max_pool2d(-distances, kernel_size=(H, W))
    min_distances = min_distances.view(B, -1)
    similarities = torch.exp(-min_distances)

    # Add the conv net output to the kwargs dict to be passed to the decision nodes in the tree
    # Split (or chunk) the conv net output tensor of shape (batch_size, num_decision_nodes) into individual tensors
    # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
    conv_net_output = similarities.chunk(similarities.size(1), dim=1)
    # Add the mapping of decision nodes to conv net outputs to the kwargs dict to be passed to the decision nodes in
    # the tree

    out_map = dict(self._out_map)  # Use a copy of self._out_map, as the original should not be modified
    attr = dict()

    out, attr = self._root.forward(x, out_map, conv_net_output, attr)

    info = dict()
    # Store the probability of arriving at all nodes in the decision tree
    info['pa_tensor'] = {n.index: attr[n, 'pa'].unsqueeze(1) for n in self.nodes}
    # Store the output probabilities of all decision nodes in the tree
    info['ps'] = {n.index: attr[n, 'ps'].unsqueeze(1) for n in self.branches}

    return out, info

