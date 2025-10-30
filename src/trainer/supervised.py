import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class SupervisedTrainer(Trainer):
  """Trainer implementation for supervised models (MLP, CNN, etc.)"""

  model: torch.nn.Module

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # the metrics to keep track of
    self._metrics = {
      "total_loss" : 0.0, 
      "n_correct" : 0, 
      "n_samples" : 0, 
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }

    # define custom training hyperparameters here 
    # self.hp  = kwargs.get("hp", 1)

  @torch.enable_grad
  def train(self, initial=False):
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)

      if not initial: 
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

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def predict_batch(self):
    """
    Plot the actual samples, predictions, and true labels.   
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

