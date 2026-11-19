import copy
import sys
import time
from copy import deepcopy
from enum import Enum, auto
from pprint import pprint

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from .base import Convergence, Trainer
from ..dataset import init_dataloader
from ..model import ProtoTree, init_model
from ..optimizer import init_optimizer

# One image per prototype is untenable once the tree gets deep: depth 9 is 511
# prototypes, depth 11 is 2047, and every push would upload all of them. A fixed
# random subset is visualised instead, one side-by-side pair each.
MAX_PROTOTYPE_PAIRS = 64


class ProtoTreeTrainer(Trainer):
  """Trainer for ProtoTree, following protopnext-image-genetic's schedule.

  Two things set this apart from the ProtoPNet trainer next door:

  - the leaf distributions are not trained by the optimizer. Once per epoch their
    state is snapshotted, and after every batch a derivative-free update retires one
    batch's worth of that snapshot and adds the class mass that actually arrived at
    each leaf. Over a full epoch the old distribution is replaced, not accumulated on.
  - projection does not touch the trained model. The reference projects a *copy* onto
    real patches and reports its accuracy as a post-push metric, leaving the live
    prototypes free to keep moving; set push.in_place to get the destructive push
    ProtoPNet performs instead.
  """

  model: ProtoTree

  class Stage(Enum):
    EVAL = auto()     # one pass over the train set with no optimizer step and no epoch increment
    WARM = auto()     # add-on layers and prototypes only; the trunk is frozen but for the override
    JOINT = auto()    # the backbone trains along with the rest
    LAST = auto()     # everything frozen; only the derivative-free leaf update moves

  def __init__(self, device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs):
    super().__init__(device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs)

    self.device = device

    dls = init_dataloader(dataset_cfg)
    self.train_dl, self.push_dl, self.val_dl, self.test_dl = dls["train"], dls["push"], dls["val"], dls["test"]

    self.model = init_model(model_cfg).to(self.device)
    self.model.device = self.device

    # Every split including push is normalized by its transform, so the model needs no
    # fixing up here. These are kept only to undo it for the prototype visualisations,
    # where a standardized image renders blown out. Shaped to broadcast over (C, H, W).
    self.input_norm = tuple(
        torch.tensor(dataset_cfg["normalize"][key], dtype=torch.float32,
                     device=self.device).view(-1, 1, 1)
        for key in ("mean", "std"))

    # The same random subset is visualised at every push, so a prototype can be watched
    # evolving across pushes while still being spread over the tree -- prototypes are
    # laid out by node index, so a leading slice would only ever show the top branches.
    generator = torch.Generator().manual_seed(int(dataset_cfg.get("seed", 0)))
    n_prototypes = self.model.num_prototypes
    self.visualized = sorted(
        torch.randperm(n_prototypes, generator=generator)[:MAX_PROTOTYPE_PAIRS].tolist())

    self.warm_optimizer, self.joint_optimizer, self.last_optimizer = init_optimizer(optimizer_cfg, self.model)
    self.optimizer_cfg = optimizer_cfg

    # The trunk layers the warm stage trains anyway. Same names the warm optimizer
    # gave their own group, so the two stay consistent.
    override = optimizer_cfg["warm"].get("backbone_override") or {}
    self.backbone_overrides = tuple(override.get("layers", []))

    self.epoch = epoch
    self.warm_epochs = kwargs["warm_epochs"]
    self.joint_epochs = kwargs["joint_epochs"]
    self.last_only_epochs = kwargs.get("last_only_epochs", 0)
    push_cfg = kwargs.get("push") or {}
    self.push_every = int(push_cfg.get("every", 1))
    self.push_in_place = bool(push_cfg.get("in_place", False))
    self.push_visualize = bool(push_cfg.get("visualize", True))
    self.prune_before_push = bool(push_cfg.get("prune", False))

    # the metrics to keep track of
    self._metrics = {
      "total_loss": 0.0,
      "nll": 0.0,
      "cluster": 0.0,
      "variability": 0.0,
      "orthogonality": 0.0,
      "n_correct": 0,
      "n_correct_hard": 0,
      "n_samples": 0,
      "accuracy": 0.0,
      "balanced_accuracy": 0.0,
      "hard_accuracy": 0.0,
      "hard_balanced_accuracy": 0.0,
      "time": 0.0,
    }

    self.metrics = {
      "train": deepcopy(self._metrics),
      "val": deepcopy(self._metrics),
    }

    # Optional early stopping; absent or patience 0 means it never fires.
    self.convergence = Convergence(kwargs.get("convergence"))

    self.nll_coef = float(loss_cfg.get("nll", 1.0))
    self.cluster_coef = float(loss_cfg.get("cluster", 0.0))
    self.variability_coef = float(loss_cfg.get("variability", 0.0))
    self.orthogonality_coef = float(loss_cfg.get("orthogonality", 0.0))

  # -- losses -----------------------------------------------------------------

  def _losses(self, model, logits, info, label):
    """The four loss terms and their weighted total, for one batch.

    nll is the classification term; logits are already log probabilities. cluster
    rewards every sample having *some* prototype it matches strongly, and variability
    rewards prototypes that discriminate within the batch rather than firing on
    everything -- both carry negative coefficients, so the optimizer maximises them.
    orthogonality pushes the prototypes apart and is a property of the model alone,
    recomputed per batch because it joins the same backward pass.
    """
    similarity = info["similarity_score_to_each_prototype"]       # (B, P)

    nll = F.nll_loss(logits, label)
    # Not class specific: a tree prototype belongs to a branch, not to a class.
    cluster = torch.mean(torch.max(similarity, dim=1)[0])
    variability = torch.var(similarity, dim=0).mean()

    protos = model.prototype_vectors.reshape(model.prototype_vectors.shape[0], -1)
    protos = F.normalize(protos, p=2, dim=-1)
    orthogonalities = protos @ protos.t()
    orthogonalities = orthogonalities - torch.eye(orthogonalities.shape[0],
                                                  device=orthogonalities.device)
    n = orthogonalities.shape[0]
    orthogonality = torch.norm(orthogonalities, p=2) / (n * (n - 1))

    total = (self.nll_coef * nll
             + self.cluster_coef * cluster
             + self.variability_coef * variability
             + self.orthogonality_coef * orthogonality)
    return total, {"nll": nll, "cluster": cluster, "variability": variability,
                   "orthogonality": orthogonality}

  @staticmethod
  def _batch(batch, device):
    x = (batch["img"] if "img" in batch else batch["x"]).to(device)
    label = (batch["label"] if "label" in batch else batch["y"]).to(device)
    return x, label

  def _tallies(self):
    """Per-class recall counters, for the accuracy a long-tailed split hides.

    Plain accuracy is dominated by the common labels on a long-tailed split; balanced
    accuracy weights every label equally. Kept for soft and hard routing alike.
    """
    zeros = lambda: torch.zeros(self.model.num_classes, device=self.device)
    return {"correct": zeros(), "total": zeros(), "hard_correct": zeros()}

  def _accumulate(self, model, metrics, tallies, terms, total, logits, info, label):
    """Fold one batch into the running epoch metrics."""
    n = label.size(0)
    for name, value in terms.items():
      metrics[name] += value.item() * n
    metrics["total_loss"] += total.item() * n

    pred_labels = logits.argmax(dim=1)
    metrics["n_correct"] += (pred_labels == label).sum().item()
    metrics["n_samples"] += n
    tallies["correct"].index_add_(0, label, (pred_labels == label).float())
    tallies["total"].index_add_(0, label, torch.ones_like(label, dtype=torch.float))

    # Hard routing sends each sample down one branch instead of mixing, which is how
    # the tree is actually read as an explanation. A branch's routing probability does
    # not depend on its siblings, so this costs no extra forward pass.
    hard_pred = model.hard_routing_predictions(info)
    metrics["n_correct_hard"] += (hard_pred == label).sum().item()
    tallies["hard_correct"].index_add_(0, label, (hard_pred == label).float())

  def _finalize(self, metrics, tallies, start):
    metrics["time"] = time.time() - start
    n = metrics["n_samples"]
    for name in ("total_loss", "nll", "cluster", "variability", "orthogonality"):
      metrics[name] /= n
    seen = tallies["total"] > 0
    metrics["accuracy"] = metrics["n_correct"] / n
    metrics["balanced_accuracy"] = (tallies["correct"][seen] / tallies["total"][seen]).mean().item()
    metrics["hard_accuracy"] = metrics["n_correct_hard"] / n
    metrics["hard_balanced_accuracy"] = (tallies["hard_correct"][seen] / tallies["total"][seen]).mean().item()
    return metrics

  # -- schedule ---------------------------------------------------------------

  def run(self, logger):
    stopped = False
    for _ in range(self.warm_epochs):
      print(f"Epoch: {self.epoch}")
      self.train(stage=self.Stage.WARM)
      self.val()
      self._maybe_push_eval()
      pprint(self.metrics)
      logger.save(self)
      if self.convergence.update(self.metrics, self.epoch):
        stopped = True
        break

    for _ in range(0 if stopped else self.joint_epochs):
      print(f"Epoch: {self.epoch}")
      self.train(stage=self.Stage.JOINT)
      self.val()
      self._maybe_push_eval()
      pprint(self.metrics)
      logger.save(self)
      if self.convergence.update(self.metrics, self.epoch):
        stopped = True
        break

    for _ in range(0 if stopped else self.last_only_epochs):
      print(f"Epoch: {self.epoch}")
      self.train(stage=self.Stage.LAST)
      self.val()
      pprint(self.metrics)
      logger.save(self)
      if self.convergence.update(self.metrics, self.epoch):
        break

  def _maybe_push_eval(self):
    """Post-push metrics, on the configured cadence. push.every 0 turns them off."""
    if self.push_every and self.epoch % self.push_every == 0:
      self.push_eval()

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
    optimizer = None
    if stage == self.Stage.EVAL:
      print("evaluation")
    elif stage == self.Stage.WARM:
      self.model.warm_only(self.backbone_overrides)
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
    tallies = self._tallies()
    start = time.time()
    self.model.train(update)

    # Snapshotted once for the whole epoch, not per batch: each batch retires one
    # num_batches-th of it, so by the end of the epoch the previous distribution has
    # been fully replaced by what this epoch's batches delivered.
    old_leaf_dist = self.model.leaf_distribution_snapshot() if update else None
    num_batches = len(self.train_dl)

    # The last-only stage freezes everything the optimizer holds unless the leaves
    # were built as gradient parameters, which leaves the loss with no grad_fn and
    # makes backward() raise. Nothing is wrong in that case -- the derivative-free
    # update is the only thing meant to move -- so the gradient step is skipped
    # rather than attempted.
    steps_gradients = update and any(
        p.requires_grad for group in optimizer.param_groups for p in group["params"])
    if update and not steps_gradients:
      print("no trainable parameters this stage; leaf distributions only")

    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x, label = self._batch(batch, self.device)
      logits, info = self.model(x)
      total_loss, terms = self._losses(self.model, logits, info, label)

      if update:
        assert optimizer is not None and old_leaf_dist is not None
        if steps_gradients:
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
        # Runs after the gradient step, on this batch's forward results, exactly as
        # the reference does -- the leaves are fitted to the predictions that were
        # actually made, not to the ones the updated weights would make.
        self.model.batch_derivative_free_update(info, label, num_batches, old_leaf_dist)

      with torch.no_grad():
        self._accumulate(self.model, train_metrics, tallies, terms, total_loss,
                         logits, info, label)
      del info, logits

    self._finalize(train_metrics, tallies, start)
    self.metrics["train"] = train_metrics
    if update: self.epoch += 1
    return train_metrics

  @torch.no_grad()
  def _evaluate(self, dataloader, desc, model=None):
    model = model if model is not None else self.model
    metrics = deepcopy(self._metrics)
    tallies = self._tallies()
    start = time.time()
    model.eval()
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(dataloader, desc=desc, leave=False, file=tqdm_file):
      x, label = self._batch(batch, self.device)
      logits, info = model(x)
      total_loss, terms = self._losses(model, logits, info, label)
      self._accumulate(model, metrics, tallies, terms, total_loss, logits, info, label)
      del info, logits
    return self._finalize(metrics, tallies, start)

  @torch.no_grad()
  def val(self):
    self.metrics["val"] = self._evaluate(self.val_dl, "val")
    return self.metrics["val"]

  @torch.no_grad()
  def test(self):
    self.metrics["test"] = self._evaluate(self.test_dl, "test")
    return self.metrics["test"]

  # -- projection -------------------------------------------------------------

  @torch.no_grad()
  def push(self, visualize=True, model=None):
    """Project every prototype onto the latent patch that activates it most.

    Unlike ProtoPNet there is no class restriction: a tree prototype belongs to a
    branch rather than to a class, so it takes the strongest patch anywhere in the
    push set. Returns the prototype sources, keyed by index, for visualisation.
    """
    model = model if model is not None else self.model
    model.eval()
    device = model.prototype_vectors.device
    n_prototypes = model.prototype_vectors.shape[0]

    global_max_proto_act = torch.full((n_prototypes,), -float("inf"), device=device)
    global_max_fmap_patches = torch.zeros(n_prototypes, model.prototype_vectors.shape[1],
                                          device=device)
    prototype_ids = torch.arange(n_prototypes, device=device)
    visualized = torch.tensor(self.visualized, device=device) if visualize else None

    # Where each visualised prototype ends up: the push image it is projected onto, as
    # display-ready pixels plus the activation map that located the patch. Recorded
    # rather than drawn, because a prototype improves many times over a push and only
    # the last one survives -- rendering inside the loop throws most of that work away.
    sources = {}

    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for push_iter, batch in enumerate(tqdm(self.push_dl, desc="push", leave=False, file=tqdm_file)):
      search_batch, search_y = self._batch(batch, device)
      start_index_of_search_batch = push_iter * (self.push_dl.batch_size or 0)

      features, activations = model.push_forward(search_batch)      # (B, D, H, W), (B, P, H, W)

      # strongest patch of each image for each prototype, then the best image
      per_image_max, per_image_patch = activations.flatten(2).max(dim=2)   # (B, P), (B, P)
      batch_max, batch_img = per_image_max.max(dim=0)                      # (P,), (P,)

      improved = batch_max > global_max_proto_act
      if not improved.any():
        continue
      patch_index = per_image_patch[batch_img, prototype_ids]              # (P,) flat index into H*W
      patches = features.flatten(2)[batch_img, :, patch_index]             # (P, D)
      global_max_fmap_patches = torch.where(improved[:, None], patches, global_max_fmap_patches)
      global_max_proto_act = torch.where(improved, batch_max, global_max_proto_act)

      if visualized is not None:
        # One sync per batch rather than one per prototype: pull the whole hit list at once.
        hits = visualized[improved[visualized]]
        if hits.numel():
          for j, i in zip(hits.tolist(), batch_img[hits].tolist()):
            sources[j] = {"image": self._displayable(search_batch[i]),
                          "activation": activations[i, j].cpu().numpy(),
                          "index": start_index_of_search_batch + i,
                          "label": int(search_y[i])}

    model.prototype_vectors.data.copy_(
        global_max_fmap_patches.reshape(model.prototype_vectors.shape))
    return sources

  @torch.no_grad()
  def push_eval(self):
    """Project a copy of the model and record how it scores once projected.

    The reference never projects the model it is training: prototypes stay free
    vectors, and projection is an evaluation-time question about how much accuracy
    survives being pinned to real patches. With push.in_place the projected
    prototypes are copied back and the live model becomes the projected one.
    """
    model = self.model if self.push_in_place else copy.deepcopy(self.model).to(self.device)
    if self.prune_before_push:
      pruned = model.prune_prototypes()
      if pruned:
        print(f"pruned {len(pruned)} prototypes")
        if model is self.model:
          # The optimizers hold moment estimates shaped to the old prototype tensor.
          self.warm_optimizer, self.joint_optimizer, self.last_optimizer = \
              init_optimizer(self.optimizer_cfg, self.model)

    sources = self.push(visualize=self.push_visualize, model=model)
    self.metrics["postpush_val"] = self._evaluate(self.val_dl, "postpush val", model=model)
    if self.push_visualize and sources:
      self._log_prototype_pairs(sources, model)
    if model is not self.model:
      del model
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return self.metrics["postpush_val"]

  # -- visualisation ----------------------------------------------------------

  def _displayable(self, image):
    """A normalized (C, H, W) tensor as a uint8 (H, W, 3) array fit for drawing on."""
    mean, std = self.input_norm
    image = (image * std + mean).clamp(0, 1).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 1:
      image = np.repeat(image, 3, axis=2)
    return np.uint8(image * 255)

  @staticmethod
  def _high_activation_crop(activation_map, percentile=95):
    """The bounding box of the top few percent of an upsampled activation map."""
    mask = activation_map >= np.percentile(activation_map, percentile)
    rows, cols = np.where(mask)
    if rows.size == 0:
      return 0, activation_map.shape[0], 0, activation_map.shape[1]
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

  def _activation_overlay(self, image, activation_map):
    """Blend a prototype's upsampled activation over a uint8 image as a JET heat map."""
    size = image.shape[0]
    upsampled = cv2.resize(activation_map, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    upsampled = upsampled - np.amin(upsampled)
    peak = np.amax(upsampled)
    if peak > 0:
      upsampled = upsampled / peak
    heatmap = cv2.applyColorMap(np.uint8(255 * upsampled), cv2.COLORMAP_JET)
    heatmap = np.ascontiguousarray(heatmap[..., ::-1])
    # Both operands are already full range, so the blend is too; rescaling it here would
    # brighten each prototype by its own arbitrary factor and make them incomparable.
    return np.uint8(np.clip(0.5 * image.astype(np.float32) + 0.5 * heatmap, 0, 255))

  @torch.no_grad()
  def _best_val_matches(self, prototype_ids, model):
    """For each prototype, the validation image whose best patch activates it most.

    Scoring only the visualised prototypes keeps the activation tensor at (B, 64, H, W)
    instead of (B, P, H, W) -- megabytes rather than gigabytes.
    """
    device = model.prototype_vectors.device
    vectors = model.prototype_vectors[torch.tensor(prototype_ids, device=device)]
    best = {}
    best_act = torch.full((len(prototype_ids),), -float("inf"), device=device)

    model.eval()
    tqdm_file = getattr(sys.stdout, "terminal", sys.stdout)
    for batch in tqdm(self.val_dl, desc="prototype match", leave=False, file=tqdm_file):
      x, _ = self._batch(batch, device)
      features = model.add_on_layers(model.backbone(x))
      acts = model.prototype_activations(features, vectors)              # (B, K, H, W)

      per_image_max, _ = acts.flatten(2).max(dim=2)                      # (B, K)
      image_max, image_of = per_image_max.max(dim=0)                     # (K,), (K,)
      better = image_max > best_act
      if better.any():
        for k, i in zip(better.nonzero().flatten().tolist(), image_of[better].tolist()):
          best[prototype_ids[k]] = {"image": self._displayable(x[i]),
                                    "activation": acts[i, k].cpu().numpy()}
        best_act = torch.where(better, image_max, best_act)
    return best

  def _log_prototype_pairs(self, sources, model):
    """Send one image per prototype to wandb: its best val match beside its own source.

    Left is the validation image this prototype fires hardest on, under its activation
    map. Right is the push image the prototype was projected onto, boxed at the region
    that defines it. Together they read as "this (val) looks like that (prototype)".
    """
    if wandb.run is None:   # wandb.run is None when log.wandb.enabled is false
      return
    matches = self._best_val_matches(sorted(sources), model)
    pairs = {}
    for j, source in sorted(sources.items()):
      match = matches.get(j)
      if match is None:
        continue
      size = source["image"].shape[0]
      upsampled = cv2.resize(source["activation"], dsize=(size, size),
                             interpolation=cv2.INTER_CUBIC)
      lo_y, hi_y, lo_x, hi_x = self._high_activation_crop(upsampled)

      boxed = source["image"].copy()
      cv2.rectangle(boxed, (int(lo_x), int(lo_y)), (int(hi_x) - 1, int(hi_y) - 1), (255, 255, 0), 2)
      side_by_side = np.concatenate(
          [self._activation_overlay(match["image"], match["activation"]), boxed], axis=1)
      pairs[f"prototype_{j:04d}"] = wandb.Image(
          side_by_side,
          caption=f"Prototype {j:04d} | left: best val match | "
                  f"right: push image {source['index']} (class {source['label']})")
    if pairs:
      # Pinned to the current epoch: a bare wandb.log() advances the global step, and
      # Logger.save then writes its metrics at a step that has already passed, which
      # wandb drops.
      wandb.log(pairs, step=self.epoch)
