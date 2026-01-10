import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
import torch.nn.functional as F
import matplotlib.pyplot as plt

def show_tensor_image(image):
    if len(image.shape) == 4:
        image = image[0]
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0)
    if image.shape[-1] == 1:
        plt.imshow(image.cpu().squeeze(), cmap='gray')
    else:
        plt.imshow(image.cpu())

class DiffusionTrainer(Trainer):
  """Trainer implementation for supervised models (MLP, CNN, etc.)"""

  model: torch.nn.Module

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # the metrics to keep track of
    self._metrics = {
      "total_loss" : 0.0, 
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }

    self.ema_model = deepcopy(self.model)
    for param in self.ema_model.parameters():
        param.requires_grad = False

    # define custom training hyperparameters here 
    # self.hp  = kwargs.get("hp", 1)

  def step_ema(self):
      for current_params, ema_params in zip(self.model.parameters(), self.ema_model.parameters()):
          ema_params.data.mul_(0.995).add_(current_params.data, alpha=1 - 0.995)

  @torch.enable_grad
  def train(self, initial=False):
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x = batch["x"].to(self.device)
      t = torch.randint(0, self.model.T, (x.size(0), ), device=self.device).long()

      x_noisy, noise = self.model.forward_diffusion_sample(x, t) 
      noise_pred = self.model.noise_predictor(x_noisy, t)
      loss = F.mse_loss(noise, noise_pred)

      if not initial: 
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_ema()
      train_metrics["total_loss"] += loss.item() 
      train_metrics["average_loss"] = train_metrics["total_loss"] / len(self.train_dl.dataset) # type: ignore

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
      x = batch["x"].to(self.device)
      t = torch.randint(0, self.model.T, (x.size(0), ), device=self.device).long()

      x_noisy, noise = self.model.forward_diffusion_sample(x, t) 
      noise_pred = self.model.noise_predictor(x_noisy, t)
      loss = F.mse_loss(noise, noise_pred)

      val_metrics["total_loss"] += loss.item()
      val_metrics["average_loss"] = val_metrics["total_loss"] / len(self.val_dl.dataset) # type: ignore

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
      x = batch["x"].to(self.device)
      t = torch.randint(0, self.model.T, (x.size(0), ), device=self.device).long()

      x_noisy, noise = self.model.forward_diffusion_sample(x, t) 
      noise_pred = self.model.noise_predictor(x_noisy, t)
      loss = F.mse_loss(noise, noise_pred)

      test_metrics["total_loss"] += loss.item()
      test_metrics["average_loss"] = test_metrics["total_loss"] / len(self.test_dl.dataset) # type: ignore

    test_metrics["time"] = time.time() - start

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad()
  def sample_across_time(self): 
    in_channels = self.model.noise_predictor.channels
    # infer image size from dataset
    x = next(iter(self.train_dl))["x"]
    img_size = x.shape[-1]
    
    num_samples = 8
    num_snapshots = 10
    img = torch.randn((num_samples, in_channels, img_size, img_size), device=self.device)
    fig = plt.figure(figsize=(15,12))
    
    stepsize = int(self.model.T/num_snapshots)

    for i in range(self.model.T - 1, -1, -1):
      t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
      img = self.ema_model.sample(img, t)
      
      if i % stepsize == 0:
        col = (num_snapshots - 1) - int(i / stepsize)
        col = max(0, min(col, num_snapshots - 1))
        
        img_cpu = img.detach().cpu()
        for row in range(num_samples):
          idx = row * num_snapshots + col + 1
          ax = plt.subplot(num_samples, num_snapshots, idx)
          show_tensor_image(img_cpu[row])
          if col == 0:
            ax.set_ylabel(f"Sample {row}")
          if row == 0:
            ax.set_title(f"t = {i}")
          ax.set_xticks([])
          ax.set_yticks([])
    
    plt.tight_layout()
    return fig

