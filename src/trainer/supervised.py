import torch
from enum import Enum, auto
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from ..optimizer import init_optimizer
from ..loss import init_loss
from ..dataset import init_dataloader
from ..model import init_model 

class SupervisedTrainer(Trainer):
  """Trainer implementation for supervised models (MLP, CNN, etc.)"""

  model: torch.nn.Module

  class Stage(Enum):
    EVAL = auto()  # one pass over the train set with no optimizer step and no epoch increment
    TRAIN = auto()    # a normal epoch: forward, backward, step

  def __init__(self, device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs):
    super().__init__(device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs)

    self.device = device

    dls = init_dataloader(dataset_cfg)
    self.train_dl, self.val_dl, self.test_dl = dls["train"], dls["val"], dls["test"]

    self.model = init_model(model_cfg).to(self.device)
    self.model.device = self.device

    self.loss = init_loss(loss_cfg)
    self.optimizer = init_optimizer(optimizer_cfg, self.model)

    self.epoch = epoch 
    self.total_epochs = kwargs["total_epochs"]

    # the metrics to keep track of
    self._metrics = {
      "total_loss" : 0.0,
      "n_correct" : 0,
      "n_samples" : 0,
      "accuracy" : 0.0,
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
    }

    # define custom training hyperparameters here 
    # self.hp  = kwargs.get("hp", 1)

  def run(self, logger): 
    for epoch in range(self.epoch  + 1, self.total_epochs):
      print(f"Epoch: {epoch}")
      self.train(stage=self.Stage.TRAIN)
      self.val()

      pprint(self.metrics)
      logger.save(self)

  def train(self, stage=None):
    # Required rather than defaulted: TRAIN and EVAL differ by whether the run
    # advances, and a caller that omits the stage is far more likely to have
    # forgotten it than to want whichever one happened to be the default.
    if stage is None:
      raise ValueError(
          f"train() needs an explicit stage, one of {[s.name for s in self.Stage]}")
    # EVAL also runs without gradients and without putting the model in train mode,
    # so it neither keeps autograd graphs alive nor updates BatchNorm statistics.
    with torch.set_grad_enabled(stage != self.Stage.EVAL):
      return self._train(stage)

  def _train(self, stage):
    update = stage != self.Stage.EVAL
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train(update)
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)

      if update: 
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
      train_metrics["total_loss"] += loss.item() 
      train_metrics["average_loss"] = train_metrics["total_loss"] / len(self.train_dl.dataset) # type: ignore

      # Calculate accuracy
      pred_labels = y_pred.argmax(dim=1)
      correct = (pred_labels == y).sum().item() 
      train_metrics["n_correct"] += correct
      train_metrics["n_samples"] += y.size(0)

    train_metrics["time"] = time.time() - start
    train_metrics["accuracy"] = train_metrics["n_correct"] / train_metrics["n_samples"] if train_metrics["n_samples"] else 0.0

    self.metrics["train"] = train_metrics
    if update: self.epoch += 1 
    return train_metrics

  @torch.no_grad
  def val(self):
    val_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.val_dl, leave=False, file=tqdm_file):
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)
      val_metrics["total_loss"] += loss.item()
      val_metrics["average_loss"] = val_metrics["total_loss"] / len(self.val_dl.dataset) # type: ignore

      # Calculate accuracy
      pred_labels = y_pred.argmax(dim=1)
      correct = (pred_labels == y).sum().item() 
      val_metrics["n_correct"] += correct
      val_metrics["n_samples"] += y.size(0)
    val_metrics["time"] = time.time() - start
    val_metrics["accuracy"] = val_metrics["n_correct"] / val_metrics["n_samples"] if val_metrics["n_samples"] else 0.0

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
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)
      test_metrics["total_loss"] += loss.item()
      test_metrics["average_loss"] = test_metrics["total_loss"] / len(self.test_dl.dataset) # type: ignore

      # Calculate accuracy
      pred_labels = y_pred.argmax(dim=1)
      correct = (pred_labels == y).sum().item() 
      test_metrics["n_correct"] += correct
      test_metrics["n_samples"] += y.size(0)
    test_metrics["time"] = time.time() - start
    test_metrics["accuracy"] = test_metrics["n_correct"] / test_metrics["n_samples"] if test_metrics["n_samples"] else 0.0

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def predict_batch(self):
    """
    Plot the actual samples, predictions, and true labels. 
    Just for sanity checks so that we can see the predictions on a small batch. 
    """
    self.model.eval()
    batch = next(iter(self.val_dl))
    x, y = batch["x"].to(self.device), batch["y"].to(self.device)
    y_pred = self.model(x)
    pred_labels = y_pred.argmax(dim=1)

    # Plot
    n_samples = 5
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    
    for i in range(n_samples):
      # Image
      img = x[i].cpu().numpy()
      # Val images arrive standardized whenever dataset.normalize is on (ImageNet
      # stats put them in roughly [-2.1, 2.6]), and imshow clips float RGB to
      # [0, 1], which renders them blown out. Rescale for display only; images
      # already in [0, 1] are left exactly as they are.
      if img.min() < 0.0 or img.max() > 1.0:
        img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
      # Handle (C, H, W) -> (H, W, C) or (H, W) for plotting
      if img.ndim == 3:
        if img.shape[0] == 1:
          img = img.squeeze(0)
          cmap = 'gray'
        else:
          img = np.transpose(img, (1, 2, 0))
          cmap = None
      else:
        cmap = 'gray'
        
      axes[i, 0].imshow(img, cmap=cmap)
      axes[i, 0].axis('off')
      axes[i, 0].set_title("Input")

      # Predicted
      axes[i, 1].text(0.5, 0.5, str(pred_labels[i].item()), fontsize=20, ha='center', va='center')
      axes[i, 1].axis('off')
      axes[i, 1].set_title("Prediction")

      # True
      axes[i, 2].text(0.5, 0.5, str(y[i].item()), fontsize=20, ha='center', va='center')
      axes[i, 2].axis('off')
      axes[i, 2].set_title("True Label")

    plt.tight_layout()
    return fig

