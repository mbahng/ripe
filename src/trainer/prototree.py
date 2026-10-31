import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
from enum import Enum, auto
from pprint import pprint
from ..model import ProtoTree

class ProtoTreeTrainer(Trainer):

  model: ProtoTree

  class Stage(Enum):
    EVAL = auto()  # one pass over the train set with no optimizer step and no epoch increment
    WARM = auto()     # only the add-on layers and prototypes are trained
    JOINT = auto()    # the backbone is trained along with the rest
    LAST = auto()     # only the last layer is trained

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._metrics = {
      "total_loss" : 0.0,
      "cross_entropy" : 0.0,
      "n_correct" : 0,
      "n_samples" : 0,
      "accuracy" : 0.0,
      "balanced_accuracy" : 0.0,
      "time" : 0.0
    }
    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }

    self.epoch = kwargs["epoch"]
    self.warm_epochs = kwargs["warm_epochs"] 
    self.joint_epochs = kwargs["joint_epochs"] 
    self.last_only_epochs = kwargs["last_only_epochs"] 
    self.push_every = kwargs["push_every"]

    self.warm_optimizer, self.joint_optimizer, self.last_optimizer = init_optimizer(kwargs["optimizer"], self.model)

  def run(self, logger): 
    for _ in range(self.epoch + 1, self.epoch + self.warm_epochs + 1):
      self.train(stage=self.Stage.WARM)
      self.val()
      pprint(self.metrics)
      logger.save(self)  

    for joint_epoch in range(self.epoch + self.warm_epochs + 1, self.epoch + self.warm_epochs + self.joint_epochs + 1): 
      if joint_epoch % self.push_every == 0:
        self.push(visualize=True) 
        continue

      self.train(stage=self.Stage.JOINT)
      self.val()
      pprint(self.metrics)
      logger.save(self)

    # push last time 
    self.push(visualize=True) 

    for _ in range(self.last_only_epochs): 
      self.train(stage=self.Stage.LAST) 
      self.val()
      pprint(self.metrics)
      logger.save(self)

  @torch.enable_grad
  def train(self, stage=None):
    # None means evaluation only: there is no optimizer to step without a stage,
    # and main.py's final call relies on that.
    if stage is None: stage = self.Stage.EVAL
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
    # Per-class recall tallies: plain accuracy is dominated by the common labels on
    # a long-tailed split, balanced accuracy weights every label equally.
    class_correct = torch.zeros(self.model.num_classes, device=self.device)
    class_total = torch.zeros(self.model.num_classes, device=self.device)
    start = time.time()
    self.model.train()
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x, label = (batch["img"] if "img" in batch else batch["x"]).to(self.device), (batch["label"] if "label" in batch else batch["y"]).to(self.device)

      # Snapshot leaf distributions before forward pass for derivative-free update
      old_leaf_dist = {leaf: leaf._dist_params.data.clone() for leaf in self.model.leaves}

      out, info = self.model(x)  # out: (B, num_classes) class distribution

      # NLL loss on the soft-routed class distribution
      ce_loss = F.nll_loss(torch.log(out + 1e-10), label)
      total_loss = ce_loss

      if update:
        self.model.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Derivative-free leaf distribution update (no gradients)
        with torch.no_grad():
          self.model.batch_derivative_free_update(info, label, len(self.train_dl), old_leaf_dist)

      train_metrics["cross_entropy"] += ce_loss.item()
      train_metrics["total_loss"] += total_loss.item()

      pred_labels = out.argmax(dim=1)
      correct = (pred_labels == label).sum().item()
      train_metrics["n_correct"] += correct
      train_metrics["n_samples"] += label.size(0)
      class_correct.index_add_(0, label, (pred_labels == label).float())
      class_total.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

    train_metrics["time"] = time.time() - start
    n = train_metrics["n_samples"]
    train_metrics["total_loss"] /= n
    train_metrics["cross_entropy"] /= n
    train_metrics["accuracy"] = train_metrics["n_correct"] / n
    seen = class_total > 0
    train_metrics["balanced_accuracy"] = (class_correct[seen] / class_total[seen]).mean().item()

    self.metrics["train"] = train_metrics
    if update:
      self.epoch += 1
    return train_metrics

  @torch.no_grad()
  def val(self):
    val_metrics = deepcopy(self._metrics)
    # Per-class recall tallies: plain accuracy is dominated by the common labels on
    # a long-tailed split, balanced accuracy weights every label equally.
    class_correct = torch.zeros(self.model.num_classes, device=self.device)
    class_total = torch.zeros(self.model.num_classes, device=self.device)
    start = time.time()
    self.model.eval()
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.val_dl, leave=False, file=tqdm_file):
      x, label = (batch["img"] if "img" in batch else batch["x"]).to(self.device), (batch["label"] if "label" in batch else batch["y"]).to(self.device)

      out, info = self.model(x)
      ce_loss = F.nll_loss(torch.log(out + 1e-10), label)

      val_metrics["cross_entropy"] += ce_loss.item()
      val_metrics["total_loss"] += ce_loss.item()

      pred_labels = out.argmax(dim=1)
      correct = (pred_labels == label).sum().item()
      val_metrics["n_correct"] += correct
      val_metrics["n_samples"] += label.size(0)
      class_correct.index_add_(0, label, (pred_labels == label).float())
      class_total.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

    val_metrics["time"] = time.time() - start
    n = val_metrics["n_samples"]
    val_metrics["total_loss"] /= n
    val_metrics["cross_entropy"] /= n
    val_metrics["accuracy"] = val_metrics["n_correct"] / n
    seen = class_total > 0
    val_metrics["balanced_accuracy"] = (class_correct[seen] / class_total[seen]).mean().item()

    self.metrics["val"] = val_metrics
    return val_metrics

  @torch.no_grad()
  def test(self):
    test_metrics = deepcopy(self._metrics)
    # Per-class recall tallies: plain accuracy is dominated by the common labels on
    # a long-tailed split, balanced accuracy weights every label equally.
    class_correct = torch.zeros(self.model.num_classes, device=self.device)
    class_total = torch.zeros(self.model.num_classes, device=self.device)
    start = time.time()
    self.model.eval()
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.test_dl, leave=False, file=tqdm_file):
      x, label = (batch["img"] if "img" in batch else batch["x"]).to(self.device), (batch["label"] if "label" in batch else batch["y"]).to(self.device)

      out, info = self.model(x)
      ce_loss = F.nll_loss(torch.log(out + 1e-10), label)

      test_metrics["cross_entropy"] += ce_loss.item()
      test_metrics["total_loss"] += ce_loss.item()

      pred_labels = out.argmax(dim=1)
      correct = (pred_labels == label).sum().item()
      test_metrics["n_correct"] += correct
      test_metrics["n_samples"] += label.size(0)
      class_correct.index_add_(0, label, (pred_labels == label).float())
      class_total.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

    test_metrics["time"] = time.time() - start
    n = test_metrics["n_samples"]
    test_metrics["total_loss"] /= n
    test_metrics["cross_entropy"] /= n
    test_metrics["accuracy"] = test_metrics["n_correct"] / n
    seen = class_total > 0
    test_metrics["balanced_accuracy"] = (class_correct[seen] / class_total[seen]).mean().item()

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad()
  def push(self, visualize=True):
    ...

