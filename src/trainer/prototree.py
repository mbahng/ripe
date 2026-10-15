import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
from pprint import pprint
from ..model import ProtoTree

class ProtoTreeTrainer(Trainer):

  model: ProtoTree

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._metrics = {
      "total_loss" : 0.0,
      "cross_entropy" : 0.0,
      "n_correct" : 0,
      "n_samples" : 0,
      "accuracy" : 0.0,
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
      self.train(stage="warm")
      self.val()
      pprint(self.metrics)
      logger.save(self)  

    for joint_epoch in range(self.epoch + self.warm_epochs + 1, self.epoch + self.warm_epochs + self.joint_epochs + 1): 
      if joint_epoch % self.push_every == 0:
        self.push(visualize=True) 
        continue

      self.train(stage="joint")
      self.val()
      pprint(self.metrics)
      logger.save(self)

    # push last time 
    self.push(visualize=True) 

    for _ in range(self.last_only_epochs): 
      self.train(stage="last") 
      self.val()
      pprint(self.metrics)
      logger.save(self)

  @torch.enable_grad
  def train(self, initial=False, stage="warm"):
    if stage == "warm":
      self.model.warm_only()
      optimizer = self.warm_optimizer
    elif stage == "joint":
      self.model.joint()
      optimizer = self.joint_optimizer
    elif stage == "last":
      self.model.last_only()
      optimizer = self.last_optimizer
    else:
      raise ValueError("Not a valid stage.")

    train_metrics = deepcopy(self._metrics)
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

      if not initial:
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

    train_metrics["time"] = time.time() - start
    n = train_metrics["n_samples"]
    train_metrics["total_loss"] /= n
    train_metrics["cross_entropy"] /= n
    train_metrics["accuracy"] = train_metrics["n_correct"] / n

    self.metrics["train"] = train_metrics
    if not initial:
      self.epoch += 1
    return train_metrics

  @torch.no_grad()
  def val(self):
    val_metrics = deepcopy(self._metrics)
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

    val_metrics["time"] = time.time() - start
    n = val_metrics["n_samples"]
    val_metrics["total_loss"] /= n
    val_metrics["cross_entropy"] /= n
    val_metrics["accuracy"] = val_metrics["n_correct"] / n

    self.metrics["val"] = val_metrics
    return val_metrics

  @torch.no_grad()
  def test(self):
    test_metrics = deepcopy(self._metrics)
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

    test_metrics["time"] = time.time() - start
    n = test_metrics["n_samples"]
    test_metrics["total_loss"] /= n
    test_metrics["cross_entropy"] /= n
    test_metrics["accuracy"] = test_metrics["n_correct"] / n

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad()
  def push(self, visualize=True):
    ...
