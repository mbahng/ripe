import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer

class ProtoTreeTrainer(Trainer):

  model: torch.nn.Module

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
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
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }
    self.ce_coef = float(kwargs["loss"]["ce"])
    self.cluster_coef = float(kwargs["loss"]["cluster"])
    self.separation_coef = float(kwargs["loss"]["separation"])
    self.l1_coef = float(kwargs["loss"]["l1"])

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
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      label_cpu = batch["y"]
      x, label = batch["x"].to(self.device), batch["y"].to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
      _, predicted = torch.max(logits.data, 1)
      train_metrics["n_correct"] += (predicted == label).sum().item()

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

      l1_cost = self.model.last_layer.weight.norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      if not initial: 
        self.model.zero_grad()
        total_loss.backward()
        optimizer.step()

      train_metrics["cross_entropy"] += ce_loss.item() 
      train_metrics["cluster"] += cluster_cost.item() 
      train_metrics["separation"] += separation_cost.item() 
      train_metrics["l1"] += l1_cost.item() 
      train_metrics["total_loss"] += total_loss.item() 

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      train_metrics["n_correct"] += correct
      train_metrics["n_samples"] += label.size(0)

      del label
      del predicted
      del min_distances

    train_metrics["time"] = time.time() - start
    train_metrics["total_loss"] /= train_metrics["n_samples"]
    train_metrics["accuracy"] = train_metrics["n_correct"] / train_metrics["n_samples"]
    train_metrics["cross_entropy"] /= train_metrics["n_samples"]
    train_metrics["cluster"] /= train_metrics["n_samples"]
    train_metrics["separation"] /= train_metrics["n_samples"]
    train_metrics["l1"] /= train_metrics["n_samples"]

    self.metrics["train"] = train_metrics
    if not initial: self.epoch += 1 
    return train_metrics

  @torch.no_grad
  def val(self):
    val_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.val_dl, leave=False, file=tqdm_file):
      label_cpu = batch["y"]
      x, label = batch["x"].to(self.device), batch["y"].to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
      _, predicted = torch.max(logits.data, 1)
      val_metrics["n_correct"] += (predicted == label).sum().item()

      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)

      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = self.model.last_layer.weight.norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      val_metrics["cross_entropy"] += ce_loss.item() 
      val_metrics["cluster"] += cluster_cost.item() 
      val_metrics["separation"] += separation_cost.item() 
      val_metrics["l1"] += l1_cost.item() 
      val_metrics["total_loss"] += total_loss.item() 

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      val_metrics["n_correct"] += correct
      val_metrics["n_samples"] += label.size(0)

      del label
      del predicted
      del min_distances

    val_metrics["time"] = time.time() - start
    val_metrics["total_loss"] /= val_metrics["n_samples"]
    val_metrics["accuracy"] = val_metrics["n_correct"] / val_metrics["n_samples"]
    val_metrics["cross_entropy"] /= val_metrics["n_samples"]
    val_metrics["cluster"] /= val_metrics["n_samples"]
    val_metrics["separation"] /= val_metrics["n_samples"]
    val_metrics["l1"] /= val_metrics["n_samples"]

    self.metrics["val"] = val_metrics
    return val_metrics

  @torch.no_grad
  def test(self): 
    test_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.test_dl, leave=False, file=tqdm_file):
      label_cpu = batch["y"]
      x, label = batch["x"].to(self.device), batch["y"].to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
      _, predicted = torch.max(logits.data, 1)
      test_metrics["n_correct"] += (predicted == label).sum().item()

      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)
      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = self.model.last_layer.weight.norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      test_metrics["cross_entropy"] += ce_loss.item() 
      test_metrics["cluster"] += cluster_cost.item() 
      test_metrics["separation"] += separation_cost.item() 
      test_metrics["l1"] += l1_cost.item() 
      test_metrics["total_loss"] += total_loss.item() 

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      test_metrics["n_correct"] += correct
      test_metrics["n_samples"] += label.size(0)

      del label
      del predicted
      del min_distances
    test_metrics["time"] = time.time() - start
    test_metrics["total_loss"] /= test_metrics["n_samples"]
    test_metrics["accuracy"] = test_metrics["n_correct"] / test_metrics["n_samples"]
    test_metrics["cross_entropy"] /= test_metrics["n_samples"]
    test_metrics["cluster"] /= test_metrics["n_samples"]
    test_metrics["separation"] /= test_metrics["n_samples"]
    test_metrics["l1"] /= test_metrics["n_samples"]

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def push(self, visualize=True): 
    ...


