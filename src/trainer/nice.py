import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
import matplotlib.pyplot as plt
from ..model.nice import NICE

class NICETrainer(Trainer):

  model: NICE

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
    self.dequantize = kwargs["dequantize"]

  @torch.enable_grad
  def train(self, initial=False):
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x = batch["x"].to(self.device)

      # dequantize data
      if self.dequantize: 
        x += torch.rand_like(x) / 256
        x = torch.clamp(x, 0, 1)

      _, log_likelihood = self.model(x)
      loss = -torch.mean(log_likelihood)

      if not initial: 
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
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

      _, log_likelihood = self.model(x)
      loss = -torch.mean(log_likelihood)
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

      _, log_likelihood = self.model(x)
      loss = -torch.mean(log_likelihood)
      test_metrics["total_loss"] += loss.item()
      test_metrics["average_loss"] = test_metrics["total_loss"] / len(self.test_dl.dataset) # type: ignore

    test_metrics["time"] = time.time() - start

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def sample(self): 
    # make sure to permute so that channel dimension is last
    z_sample = self.model.latent_prior.sample([25, self.model.input_dim]).to(self.device)

    img = self.model.inverse(z_sample).permute(0, 2, 3, 1).detach().cpu().numpy()
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
      ax.imshow(img[i], cmap='gray'); ax.axis('off')

    plt.tight_layout()
    return fig

  @torch.no_grad
  def distribution(self):
    if not (self.model.input_shape == 2 or self.model.input_shape == [2]):
      raise AttributeError("The dimension must be 2. ")

    ds = self.val_dl.dataset
    if hasattr(ds, "tensors"):
      data = ds.tensors[0]
    else:
      data = torch.stack([ds[i][0] for i in range(len(ds))])

    self.model.eval()
    # Create a grid of points
    x_range = torch.linspace(-10, 10, 200).to(self.device)
    y_range = torch.linspace(-10, 10, 200).to(self.device)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Compute log-likelihood for each point
    _, log_likelihood = self.model(grid_points)
    log_likelihood = log_likelihood.detach().cpu().reshape(200, 200)

    # Convert to probability density
    prob_density = torch.exp(log_likelihood)

    # Plot the heatmap and scatter 
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot data scatter
    axes[0].scatter(data[:,0].detach().cpu(), data[:,1].detach().cpu(), c="r", s=3, alpha=0.5)
    axes[0].set_title('Data Distribution')
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    axes[0].set_aspect('equal')

    # Plot learned distribution heatmap
    im = axes[1].imshow(prob_density.T, extent=[-10, 10, -10, 10], origin='lower', cmap='viridis', aspect='auto') # type: ignore
    axes[1].set_title('Learned Probability Distribution (Flow Model)')
    axes[1].set_aspect('equal')
    fig.colorbar(im, ax=axes[1], label='Probability Density')
    
    return fig

