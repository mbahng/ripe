import torch
from tqdm import tqdm
import sys, time
import numpy as np
from copy import deepcopy
from .base import Trainer
from ..model.gan import GAN
import matplotlib.pyplot as plt

class GANTrainer(Trainer):
  """Trainer implementation for supervised models (MLP, CNN, etc.)"""

  model: GAN

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # the metrics to keep track of
    self._metrics = { 
      "discriminator_loss" : 0.0,  
      "generator_loss" : 0.0, 
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

    self.n_critic = kwargs.get("n_critic", 1)

  @torch.enable_grad 
  def train(self, initial=False):
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for i, batch in enumerate(tqdm(self.train_dl, leave=False, file=tqdm_file)): 
      x_true = batch["x"]
      # Set up gradients based on the current step (Discriminator vs Generator)
      if i % (self.n_critic + 1) != 0: 
        # train discriminator
        self.model.discriminator.toggle_grad(True)
        self.model.generator.toggle_grad(False)
        # sample minibatch from the true data generating distribution
        x_true = x_true.to(self.device) 
        # sample minibatch from the generator
        minibatch_size = x_true.size(0)
        z_gen = self.model.generator.sample(minibatch_size) 
        x_gen = self.model.generator(z_gen).to(self.device) 

        should_be_true = self.model.discriminator(x_true) 
        should_be_false = self.model.discriminator(x_gen)

        loss = - torch.mean(torch.log(should_be_true) + torch.log(1 - should_be_false))  
        train_metrics["discriminator_loss"] += loss.item()

        if not initial: 
          self.model.discriminator.zero_grad()
          loss.backward()
          self.optimizer.step()
      else: 
        # train generator 
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(True)
        # sample minibatch from the true data generating distribution
        x_true = x_true.to(self.device) 
        # sample minibatch from the generator
        minibatch_size = x_true.size(0)
        z_gen = self.model.generator.sample(minibatch_size) 
        x_gen = self.model.generator(z_gen).to(self.device) 

        should_be_false = self.model.discriminator(x_gen)
        loss = torch.mean(torch.log(1 - should_be_false))
        train_metrics["generator_loss"] += loss.item()
        if not initial: 
          self.model.generator.zero_grad()
          loss.backward()
          self.optimizer.step()

    train_metrics["time"] = time.time() - start
    train_metrics["total_loss"] = train_metrics["generator_loss"] + train_metrics["discriminator_loss"]
    self.metrics["train"] = train_metrics
    if not initial: self.epoch += 1 
    return train_metrics

  def val(self): 
    val_metrics = deepcopy(self._metrics)
    # Get the original terminal stdout if logger has redirected it
    start = time.time()
    self.model.eval()
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)

    for i, batch in enumerate(tqdm(self.train_dl, leave=False, file=tqdm_file)): 
      x_true = batch["x"]
      # sample minibatch from the true data generating distribution
      x_true = x_true.to(self.device) 
      # sample minibatch from the generator
      minibatch_size = x_true.size(0)
      z_gen = self.model.generator.sample(minibatch_size) 
      x_gen = self.model.generator(z_gen).to(self.device) 

      if i % 2 != 0: 
        # train discriminator
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_true = self.model.discriminator(x_true) 
        should_be_false = self.model.discriminator(x_gen)

        loss = - torch.mean(torch.log(should_be_true) + torch.log(1 - should_be_false))  
        val_metrics["discriminator_loss"] += loss.item()

      else: 
        # train generator 
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_false = self.model.discriminator(x_gen)
        loss = torch.mean(torch.log(1 - should_be_false))
        val_metrics["generator_loss"] += loss.item()
    val_metrics["time"] = time.time() - start

    val_metrics["total_loss"] = val_metrics["generator_loss"] + val_metrics["discriminator_loss"]
    self.metrics["val"] = val_metrics
    return val_metrics

  def test(self): 
    test_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)

    for i, batch in enumerate(tqdm(self.train_dl, leave=False, file=tqdm_file)): 
      x_true = batch["x"]
      # sample minibatch from the true data generating distribution
      x_true = x_true.to(self.device) 
      # sample minibatch from the generator
      minibatch_size = x_true.size(0)
      z_gen = self.model.generator.sample(minibatch_size) 
      x_gen = self.model.generator(z_gen).to(self.device) 

      if i % 2 != 0: 
        # train discriminator
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_true = self.model.discriminator(x_true) 
        should_be_false = self.model.discriminator(x_gen)

        loss = - torch.mean(torch.log(should_be_true) + torch.log(1 - should_be_false))  
        test_metrics["discriminator_loss"] += loss.item()

      else: 
        # train generator 
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_false = self.model.discriminator(x_gen)
        loss = torch.mean(torch.log(1 - should_be_false))
        test_metrics["generator_loss"] += loss.item()
    test_metrics["time"] = time.time() - start

    test_metrics["total_loss"] = test_metrics["generator_loss"] + test_metrics["discriminator_loss"]
    self.metrics["test"] = test_metrics
    return test_metrics

  def sample(self): 
    shape = next(iter(self.train_dl))["x"][0].shape
    z = self.model.generator.sample(12) 
    x = self.model.generator.forward(z).reshape(-1, *shape) 
    x = x.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
      ax.imshow(x[i, 0, :, :], cmap='gray')
      ax.axis('off')
    plt.tight_layout()
    return fig

  def visualize_distribution(self):
    # 1. Create grid for decision boundary
    range_limit = 8 
    x = np.linspace(-range_limit, range_limit, 100)
    y = np.linspace(-range_limit, range_limit, 100)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

    # Move to device
    device = next(self.model.parameters()).device
    grid = grid.to(device)

    # Get discriminator output
    with torch.no_grad():
      D_out = self.model.discriminator(grid).cpu().numpy().reshape(100, 100)

    # 2. Sample from generator for density
    z = self.model.generator.sample(10000).to(device)
    with torch.no_grad():
      G_out = self.model.generator(z).cpu().numpy()

    # Get true data samples
    true_samples = []
    current_count = 0
    for batch in self.train_dl:
      x = batch["x"]
      true_samples.append(x.numpy())
      current_count += x.size(0)
      if current_count >= 10000:
        break
    true_samples = np.concatenate(true_samples, axis=0)[:10000]

    # 3. Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Plot 1: Samples ---
    ax = axes[0]
    
    # Generated samples scatter plot
    ax.scatter(true_samples[:, 0], true_samples[:, 1], c='red', alpha=0.1, s=5, label='True Data')
    ax.scatter(G_out[:, 0], G_out[:, 1], c='blue', alpha=0.1, s=5, label='Generated Data')
    
    ax.set_xlim([-range_limit, range_limit])
    ax.set_ylim([-range_limit, range_limit])
    ax.set_title(f"True vs Generated Samples (Epoch {self.epoch})")
    ax.legend()

    # --- Plot 2: Discriminator Output ---
    ax = axes[1]

    # Discriminator heatmap
    cont = ax.contourf(X, Y, D_out, levels=20, cmap='RdBu_r', alpha=0.8, vmin=0, vmax=1)
    fig.colorbar(cont, ax=ax, label='D(x) Probability')

    ax.set_xlim([-range_limit, range_limit])
    ax.set_ylim([-range_limit, range_limit])
    ax.set_title(f"Discriminator Probability (Epoch {self.epoch})")
    # ax.legend() # Removed as colorbar serves as legend

    return fig
