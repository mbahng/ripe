import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
from enum import Enum, auto
import numpy as np
from ..receptive_field import compute_rf_prototype
import cv2
import os 
import matplotlib.pyplot as plt
import wandb
from pprint import pprint
from ..optimizer import init_optimizer
from ..dataset import init_dataloader
from ..model import ProtoPNet
from ..optimizer import init_optimizer
from ..loss import init_loss
from ..dataset import init_dataloader
from ..model import init_model 

# One image per prototype is untenable once prototype counts scale with the label
# space: 5730 prototypes x ~6 pushes is ~34k uploads per run, and unreadable in the
# UI. A fixed random subset is visualised instead, one side-by-side pair each.
MAX_PROTOTYPE_PAIRS = 64

class ProtoPNetTrainer(Trainer):

  model: ProtoPNet

  class Stage(Enum):
    EVAL = auto()  # one pass over the train set with no optimizer step and no epoch increment
    WARM = auto()     # only the add-on layers and prototypes are trained
    JOINT = auto()    # the backbone is trained along with the rest
    LAST = auto()     # only the last layer is trained

  def __init__(self, device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs):
    super().__init__(device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs)

    self.device = device

    dls = init_dataloader(dataset_cfg)
    self.train_dl, self.push_dl, self.val_dl, self.test_dl = dls["train"], dls["push"], dls["val"], dls["test"]

    self.model = init_model(model_cfg).to(self.device)
    self.model.device = self.device

    # last_layer.weight is (classes, prototypes); 1 where a prototype belongs to another class
    self.l1_mask = (1 - torch.t(self.model.prototype_class_identity)).to(self.device)

    # Every split including push is normalized by its transform, so the model needs no
    # fixing up here. These are kept only to undo it for the prototype visualisations,
    # where a standardized image renders blown out. Shaped to broadcast over (C, H, W).
    self.input_norm = tuple(
        torch.tensor(dataset_cfg["normalize"][key], dtype=torch.float32,
                     device=self.device).view(-1, 1, 1)
        for key in ("mean", "std"))

    # The same random subset is visualised at every push, so a prototype can be watched
    # evolving across pushes while still being spread over the label space -- prototypes
    # are laid out by class, so a leading slice would only ever show the first few.
    generator = torch.Generator().manual_seed(int(dataset_cfg.get("seed", 0)))
    n_prototypes = self.model.prototype_shape[0]
    self.visualized = torch.randperm(n_prototypes, generator=generator)[:MAX_PROTOTYPE_PAIRS]
    self.visualized = sorted(self.visualized.tolist())

    self.loss = init_loss(loss_cfg)
    self.warm_optimizer, self.joint_optimizer, self.last_optimizer = init_optimizer(optimizer_cfg, self.model)

    self.epoch = epoch
    self.warm_epochs = kwargs["warm_epochs"] 
    self.joint_epochs = kwargs["joint_epochs"] 
    self.last_only_epochs = kwargs["last_only_epochs"] 
    self.push_every = kwargs["push_every"]

    # the metrics to keep track of
    self._metrics = {
      "total_loss" : 0.0, 
      "cross_entropy" : 0.0, 
      "cluster" : 0.0, 
      "separation" : 0.0,
      "l1" : 0.0,
      "n_correct" : 0, 
      "n_samples" : 0, 
      "accuracy" : 0.0,
      "balanced_accuracy" : 0.0,
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
    }

    self.ce_coef = float(loss_cfg["ce"])
    self.cluster_coef = float(loss_cfg["cluster"])
    self.separation_coef = float(loss_cfg["separation"])
    self.l1_coef = float(loss_cfg["l1"])


  def _push_epoch(self, logger):
    """Push prototypes onto real patches, then record it like any other epoch.

    A push changes the model, so it is followed by an evaluation pass over the
    train set and a validation pass, and counts as an epoch: post-push metrics
    for both splits are printed, saved and sent to wandb at a new epoch number.
    The train pass goes through train_dl, so it sees the train augmentations.
    """
    print(f"Epoch: {self.epoch} (push)")
    self.push(visualize=True)
    self.train(stage=self.Stage.EVAL)
    self.epoch += 1
    self.val()
    pprint(self.metrics)
    logger.save(self)

  def run(self, logger): 
    for _ in range(self.warm_epochs): 
      print(f"Epoch: {self.epoch}")
      self.train(stage=self.Stage.WARM)
      self.val()
      pprint(self.metrics)
      logger.save(self)  

    pushed_last = False
    for joint_epoch in range(1, self.joint_epochs + 1): 
      if joint_epoch % self.push_every == 0:
        # this joint epoch is a push instead of a training epoch
        self._push_epoch(logger)
        pushed_last = joint_epoch == self.joint_epochs
        continue

      print(f"Epoch: {self.epoch}")
      self.train(stage=self.Stage.JOINT)
      self.val()
      pprint(self.metrics)
      logger.save(self)
      pushed_last = False

    # push once more after joint training, unless the last joint epoch already was a push
    if not pushed_last:
      self._push_epoch(logger)

    for _ in range(self.last_only_epochs): 
      print(f"Epoch: {self.epoch}")
      self.train(stage=self.Stage.LAST) 
      self.val()
      pprint(self.metrics)
      logger.save(self)

  def train(self, stage=None):
    # None means evaluation only: there is no optimizer to step without a stage,
    # and main.py's final call relies on that. Evaluation also runs the model in
    # eval mode without gradients, so it neither updates BatchNorm statistics
    # nor keeps autograd graphs alive.
    if stage is None: stage = self.Stage.EVAL
    with torch.set_grad_enabled(stage != self.Stage.EVAL):
      return self._train(stage)

  def _train(self, stage):
    update = stage != self.Stage.EVAL
    if stage == self.Stage.EVAL: 
      print("evaluation")
    elif stage == self.Stage.WARM: 
      self.model.warm_only() 
      optimizer = self.warm_optimizer
    elif stage == self.Stage.JOINT: 
      self.model.joint() 
      optimizer = self.joint_optimizer
    elif stage == self.Stage.LAST: 
      self.model.last_only()
      optimizer = self.last_optimizer
    else: 
      raise ValueError(f"Not a valid stage: {stage}")

    train_metrics = deepcopy(self._metrics) 
    # Per-class recall tallies, for the accuracy a long-tailed split hides: plain
    # accuracy is dominated by the common labels, balanced accuracy weights each
    # label equally no matter how few samples it has.
    n_classes = self.model.prototype_class_identity.shape[1]
    class_correct = torch.zeros(n_classes, device=self.device)
    class_total = torch.zeros(n_classes, device=self.device)
    start = time.time()
    self.model.train(update)
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      label_cpu = (batch["label"] if "label" in batch else batch["y"])
      x, label = (batch["img"] if "img" in batch else batch["x"]).to(self.device), (batch["label"] if "label" in batch else batch["y"]).to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = self.model.prototype_shape[1] # squared distances of [0, 1] vectors sum to at most the prototype dimension
      _, predicted = torch.max(logits.data, 1)

      # calculate cluster cost by looking at minimum distances between prototype and nearest patch. 
      # 1. Filter prototype_class_identity: (2000, 200) -> (2000, B). It consists of rows that represent each prototype. 
      # 2. If we transpose it, then (B, 2000). Given the ith sample from batch, matrix[i] is a vector that 
      # shows all prototypes associated with it of form: [0 ... 0 1 ... 1 0 ... 0]
      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)
      # prototypes_of_correct_class[i][j] = 1 if jth prototype corresponds to class in ith sample of batch

      # we have the min_distances of shape (B, 2000). We don't care about the distances from ith sample to a prototype that isn't associated with it, 
      # so we just mask it by doing A = min_distances * prototypes_of_correct_class, of shape (B, 2000) 
      # A[i][j] is the distance from the jth prototype to the class that the prototype corresponds to. 
      # invert them and then compute the max (so min distance) across the prototype dimension. This ignores the majority of distances which are 0. 
      # and finds the prototype that has the min distance. 
      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

      # we can't just take the min of the masked min_distances directly since it will always be 0 
      # we want to minimize only over the elements that are not masked
      # this is why we take the max of inverted distances, mask them out, and then invert again. 
      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = (self.model.last_layer.weight * self.l1_mask).norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      if update: 
        self.model.zero_grad()
        total_loss.backward()
        optimizer.step()

      train_metrics["cross_entropy"] += ce_loss.item() * label.size(0)
      train_metrics["cluster"] += cluster_cost.item() * label.size(0)
      train_metrics["separation"] += separation_cost.item() * label.size(0)
      train_metrics["l1"] += l1_cost.item() * label.size(0)
      train_metrics["total_loss"] += total_loss.item() * label.size(0)

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      train_metrics["n_correct"] += correct
      train_metrics["n_samples"] += label.size(0)
      class_correct.index_add_(0, label, (pred_labels == label).float())
      class_total.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

      del label
      del predicted
      del min_distances

    train_metrics["time"] = time.time() - start
    train_metrics["total_loss"] /= train_metrics["n_samples"]
    train_metrics["accuracy"] = train_metrics["n_correct"] / train_metrics["n_samples"]
    seen = class_total > 0
    train_metrics["balanced_accuracy"] = (class_correct[seen] / class_total[seen]).mean().item()
    train_metrics["cross_entropy"] /= train_metrics["n_samples"]
    train_metrics["cluster"] /= train_metrics["n_samples"]
    train_metrics["separation"] /= train_metrics["n_samples"]
    train_metrics["l1"] /= train_metrics["n_samples"]

    self.metrics["train"] = train_metrics
    if update: self.epoch += 1 
    return train_metrics

  @torch.no_grad
  def val(self):
    val_metrics = deepcopy(self._metrics)
    # Per-class recall tallies, for the accuracy a long-tailed split hides: plain
    # accuracy is dominated by the common labels, balanced accuracy weights each
    # label equally no matter how few samples it has.
    n_classes = self.model.prototype_class_identity.shape[1]
    class_correct = torch.zeros(n_classes, device=self.device)
    class_total = torch.zeros(n_classes, device=self.device)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.val_dl, leave=False, file=tqdm_file):
      label_cpu = (batch["label"] if "label" in batch else batch["y"])
      x, label = (batch["img"] if "img" in batch else batch["x"]).to(self.device), (batch["label"] if "label" in batch else batch["y"]).to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = self.model.prototype_shape[1] # squared distances of [0, 1] vectors sum to at most the prototype dimension
      _, predicted = torch.max(logits.data, 1)

      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)

      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = (self.model.last_layer.weight * self.l1_mask).norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      val_metrics["cross_entropy"] += ce_loss.item() * label.size(0)
      val_metrics["cluster"] += cluster_cost.item() * label.size(0)
      val_metrics["separation"] += separation_cost.item() * label.size(0)
      val_metrics["l1"] += l1_cost.item() * label.size(0)
      val_metrics["total_loss"] += total_loss.item() * label.size(0)

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      val_metrics["n_correct"] += correct
      val_metrics["n_samples"] += label.size(0)
      class_correct.index_add_(0, label, (pred_labels == label).float())
      class_total.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

      del label
      del predicted
      del min_distances

    val_metrics["time"] = time.time() - start
    val_metrics["total_loss"] /= val_metrics["n_samples"]
    val_metrics["accuracy"] = val_metrics["n_correct"] / val_metrics["n_samples"]
    seen = class_total > 0
    val_metrics["balanced_accuracy"] = (class_correct[seen] / class_total[seen]).mean().item()
    val_metrics["cross_entropy"] /= val_metrics["n_samples"]
    val_metrics["cluster"] /= val_metrics["n_samples"]
    val_metrics["separation"] /= val_metrics["n_samples"]
    val_metrics["l1"] /= val_metrics["n_samples"]

    self.metrics["val"] = val_metrics
    return val_metrics

  @torch.no_grad
  def test(self): 
    test_metrics = deepcopy(self._metrics)
    # Per-class recall tallies, for the accuracy a long-tailed split hides: plain
    # accuracy is dominated by the common labels, balanced accuracy weights each
    # label equally no matter how few samples it has.
    n_classes = self.model.prototype_class_identity.shape[1]
    class_correct = torch.zeros(n_classes, device=self.device)
    class_total = torch.zeros(n_classes, device=self.device)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.test_dl, leave=False, file=tqdm_file):
      label_cpu = (batch["label"] if "label" in batch else batch["y"])
      x, label = (batch["img"] if "img" in batch else batch["x"]).to(self.device), (batch["label"] if "label" in batch else batch["y"]).to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = self.model.prototype_shape[1] # squared distances of [0, 1] vectors sum to at most the prototype dimension
      _, predicted = torch.max(logits.data, 1)

      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)
      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = (self.model.last_layer.weight * self.l1_mask).norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      test_metrics["cross_entropy"] += ce_loss.item() * label.size(0)
      test_metrics["cluster"] += cluster_cost.item() * label.size(0)
      test_metrics["separation"] += separation_cost.item() * label.size(0)
      test_metrics["l1"] += l1_cost.item() * label.size(0)
      test_metrics["total_loss"] += total_loss.item() * label.size(0)

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      test_metrics["n_correct"] += correct
      test_metrics["n_samples"] += label.size(0)
      class_correct.index_add_(0, label, (pred_labels == label).float())
      class_total.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

      del label
      del predicted
      del min_distances
    test_metrics["time"] = time.time() - start
    test_metrics["total_loss"] /= test_metrics["n_samples"]
    test_metrics["accuracy"] = test_metrics["n_correct"] / test_metrics["n_samples"]
    seen = class_total > 0
    test_metrics["balanced_accuracy"] = (class_correct[seen] / class_total[seen]).mean().item()
    test_metrics["cross_entropy"] /= test_metrics["n_samples"]
    test_metrics["cluster"] /= test_metrics["n_samples"]
    test_metrics["separation"] /= test_metrics["n_samples"]
    test_metrics["l1"] /= test_metrics["n_samples"]

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def push(self, visualize=True): 
    def find_high_activation_crop(activation_map, percentile=95):
      threshold = np.percentile(activation_map, percentile)
      mask = np.ones(activation_map.shape)
      mask[activation_map < threshold] = 0
      lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
      for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
          lower_y = i
          break
      for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
          upper_y = i
          break
      for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
          lower_x = j
          break
      for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
          upper_x = j
          break
      return lower_y, upper_y+1, lower_x, upper_x+1

    self.model.eval()
    prototype_shape = self.model.prototype_shape
    n_prototypes, _, proto_h, proto_w = prototype_shape

    device = self.model.prototype_vectors.device
    assert proto_h == 1 and proto_w == 1, "push assumes 1x1 prototypes"
    # global_min_proto_dist[i]: closest distance from prototype i to any patch of any image of its class so far
    global_min_proto_dist = torch.full((n_prototypes,), float("inf"), device=device)
    # the patch (feature vector) that achieved it
    global_min_fmap_patches = torch.zeros(n_prototypes, prototype_shape[1], device=device)

    target_classes = self.model.prototype_class_identity.argmax(dim=1).to(device)  # (n_prototypes,)
    prototype_ids = torch.arange(n_prototypes, device=device)
    visualized = torch.tensor(self.visualized, device=device) if visualize else None

    # Where each visualised prototype ends up: the push image it is projected onto, as
    # display-ready pixels plus the distance map that located the patch. Recorded rather
    # than drawn, because a prototype improves many times over a sweep and only the last
    # one survives -- rendering inside the loop throws most of that work away.
    sources = {}

    for push_iter, batch in enumerate(tqdm(self.push_dl, desc="push")):
      search_batch = (batch["img"] if "img" in batch else batch["x"])
      search_y = (batch["label"] if "label" in batch else batch["y"])
      search_y_list = search_y.tolist()

      start_index_of_search_batch = push_iter * self.push_dl.batch_size

      search_batch = search_batch.cuda()
      search_y = search_y.to(device)
      # latent features (B, D, H, W) and the distance from every prototype to every patch (B, P, H, W)
      features, dist = self.model.push_forward(search_batch)

      # closest patch of each image to each prototype, and which patch that is
      per_image_min, per_image_patch = dist.flatten(2).min(dim=2)         # (B, P), (B, P)
      # a prototype may only be pushed onto images of its own class
      same_class = search_y[:, None] == target_classes[None, :]            # (B, P)
      per_image_min = per_image_min.masked_fill(~same_class, float("inf"))
      # best image in the batch for each prototype
      batch_min, batch_img = per_image_min.min(dim=0)                      # (P,), (P,)

      # strict: a prototype with no image of its class in this batch has batch_min = inf and never improves
      improved = batch_min < global_min_proto_dist                         # (P,)
      if not improved.any():
        continue
      patch_index = per_image_patch[batch_img, prototype_ids]              # (P,) flat index into H*W
      patches = features.flatten(2)[batch_img, :, patch_index]             # (P, D)
      global_min_fmap_patches = torch.where(improved[:, None], patches, global_min_fmap_patches)
      global_min_proto_dist = torch.where(improved, batch_min, global_min_proto_dist)

      if visualize:
        # One sync per batch rather than one per prototype: pull the whole hit list at once.
        hits = visualized[improved[visualized]]
        if hits.numel():
          hit_ids = hits.tolist()
          for j, i in zip(hit_ids, batch_img[hits].tolist()):
            sources[j] = {"image": self._displayable(search_batch[i]),
                          "distance": dist[i, j].cpu().numpy(),
                          "index": start_index_of_search_batch + i,
                          "label": search_y_list[i]}

    # a prototype with no image of its class anywhere in the push set keeps its current value
    updated = torch.isfinite(global_min_proto_dist).view(-1, 1, 1, 1)
    new_vectors = global_min_fmap_patches.reshape(tuple(prototype_shape))
    self.model.prototype_vectors.data.copy_(torch.where(updated, new_vectors, self.model.prototype_vectors.data))

    # Only meaningful against the pushed vectors, so this has to follow the copy above.
    if visualize and sources:
      self._log_prototype_pairs(sources, find_high_activation_crop)

  def _displayable(self, image):
    """A normalized (C, H, W) tensor as a uint8 (H, W, 3) array fit for drawing on."""
    mean, std = self.input_norm
    image = (image * std + mean).clamp(0, 1).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 1:
      image = np.repeat(image, 3, axis=2)
    return np.uint8(image * 255)

  @torch.no_grad
  def _best_val_matches(self, prototype_ids):
    """For each prototype, the validation image whose closest patch is nearest to it.

    Scoring only the visualised prototypes keeps the distance tensor at (B, 64, H, W)
    instead of (B, P, H, W) -- megabytes rather than a gigabyte. No class restriction:
    which class a prototype fires hardest on is the thing worth seeing.
    """
    device = self.model.prototype_vectors.device
    selected = torch.tensor(prototype_ids, device=device)
    vectors = self.model.prototype_vectors[selected]                       # (K, D, 1, 1)
    best = {}
    best_dist = torch.full((len(prototype_ids),), float("inf"), device=device)

    self.model.eval()
    tqdm_file = getattr(sys.stdout, "terminal", sys.stdout)
    for batch in tqdm(self.val_dl, desc="prototype match", leave=False, file=tqdm_file):
      x = (batch["img"] if "img" in batch else batch["x"]).to(device)
      features = self.model.add_on_layers(self.model.backbone(x))          # (B, D, H, W)
      dist = ((features.unsqueeze(1) - vectors.unsqueeze(0)) ** 2).sum(2)  # (B, K, H, W)

      per_image_min, _ = dist.flatten(2).min(dim=2)                        # (B, K)
      image_min, image_of = per_image_min.min(dim=0)                       # (K,), (K,)
      better = image_min < best_dist
      if better.any():
        for k, i in zip(better.nonzero().flatten().tolist(),
                        image_of[better].tolist()):
          best[prototype_ids[k]] = {"image": self._displayable(x[i]),
                                    "distance": dist[i, k].cpu().numpy()}
        best_dist = torch.where(better, image_min, best_dist)
    return best

  def _activation_overlay(self, image, distance_map):
    """Blend a prototype's upsampled activation over a uint8 image as a JET heat map."""
    size = image.shape[0]
    activation = self.model.prototype_shape[1] - distance_map
    upsampled = cv2.resize(activation, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    upsampled = upsampled - np.amin(upsampled)
    peak = np.amax(upsampled)
    if peak > 0:
      upsampled = upsampled / peak
    heatmap = cv2.applyColorMap(np.uint8(255 * upsampled), cv2.COLORMAP_JET)
    heatmap = np.ascontiguousarray(heatmap[..., ::-1])
    # Both operands are already full range, so the blend is too; rescaling it here would
    # brighten each prototype by its own arbitrary factor and make them incomparable.
    return np.uint8(np.clip(0.5 * image.astype(np.float32) + 0.5 * heatmap, 0, 255))

  def _log_prototype_pairs(self, sources, find_high_activation_crop):
    """Send one image per prototype to wandb: its best val match beside its own source.

    Left is the validation image this prototype fires hardest on, under its activation
    map. Right is the push image the prototype was projected onto, boxed at the region
    that defines it. Together they read as "this (val) looks like that (prototype)".
    """
    if wandb.run is None:   # wandb.run is None when log.wandb.enabled is false
      return
    matches = self._best_val_matches(sorted(sources))
    pairs = {}
    for j, source in sorted(sources.items()):
      match = matches.get(j)
      if match is None:
        continue
      size = source["image"].shape[0]
      activation = self.model.prototype_shape[1] - source["distance"]
      upsampled = cv2.resize(activation, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
      lo_y, hi_y, lo_x, hi_x = find_high_activation_crop(upsampled)

      boxed = source["image"].copy()
      cv2.rectangle(boxed, (lo_x, lo_y), (hi_x - 1, hi_y - 1), (255, 255, 0), 2)
      side_by_side = np.concatenate([self._activation_overlay(match["image"], match["distance"]),
                                     boxed], axis=1)
      pairs[f"prototype_{j:04d}"] = wandb.Image(
          side_by_side,
          caption=f"Prototype {j:04d} | left: best val match | "
                  f"right: push image {source['index']} (class {source['label']})")
    if pairs:
      # Pinned to the current epoch: a bare wandb.log() advances the global step, and
      # Logger.save then writes its metrics at a step that has already passed, which
      # wandb drops. push() runs before the epoch counter moves, so this merges into
      # the step the surrounding metrics already occupy.
      wandb.log(pairs, step=self.epoch)
