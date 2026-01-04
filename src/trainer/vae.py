import torch
from tqdm import tqdm
import sys, time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt 
from torchvision.utils import make_grid
from .base import Trainer

from ..model.vae import VAE

class VAETrainer(Trainer):
  """Trainer implementation for VAE models"""
  
  model: VAE

  def __init__(self, device, **cfg): 
    super().__init__(device, **cfg)
    # the metrics to keep track of
    self._metrics = {
      "total_loss" : 0.0, 
      "average_loss" : 0.0, 
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }

  @torch.enable_grad
  def train(self, initial=False):
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x = batch["x"].to(self.device)
      x_recon, mu, logvar = self.model(x)
      loss = self.loss(x_recon, x, mu, logvar)

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
      x_recon, mu, logvar = self.model(x)
      loss = self.loss(x_recon, x, mu, logvar)
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
      x_recon, mu, logvar = self.model(x)
      loss = self.loss(x_recon, x, mu, logvar)
      test_metrics["total_loss"] += loss.item()
      test_metrics["average_loss"] = test_metrics["total_loss"] / len(self.test_dl.dataset) # type: ignore

    test_metrics["time"] = time.time() - start

    self.metrics["test"] = test_metrics
    return test_metrics
    
  def sample_digits(self):
    """
    Plots a 5x10 grid of generated digits from a VAE.
    Returns: matplotlib figure
    """
    fig, axes = plt.subplots(5, 10, figsize=(15, 8)) 

    latent_samples = torch.randn(50, self.model.latent_dim).to(self.device)

    with torch.no_grad(): 
      x_samples = self.model.decoder(latent_samples)
      x_samples = x_samples.reshape(-1, 28, 28).cpu().numpy()

    for i in range(50): 
      row, col = i // 10, i % 10 
      # Plot the image
      axes[row, col].imshow(x_samples[i], cmap='gray')
      axes[row, col].axis('off')

      # Add column title for the first row
      if row == 0:
        axes[row, col].set_title(f'{col}', fontsize=12)

    plt.tight_layout()
    return fig

  def reconstruct(self): 
    """
    Compares original images to reconstructed for VAE models.
    """
    batch = next(iter(self.val_dl))
    x = batch["x"]
    x = x.squeeze(1)[:25]
    xhat, _, _ = self.model(x.view(-1, 28 * 28).to(self.device))
    xhat = xhat.reshape(-1, 28, 28).detach().cpu().numpy()
    x = x.detach().cpu().reshape(25, 28, 28).numpy()

    fig = plt.figure(figsize=(15, 8))
    subfigs = fig.subfigures(nrows=1, ncols=2)

    axes_left = subfigs[0].subplots(nrows=5, ncols=5)
    for i in range(25):
      axes_left[i // 5, i % 5].imshow(x[i], cmap="gray")
      axes_left[i // 5, i % 5].axis('off')
    subfigs[0].suptitle('Original')

    axes_right = subfigs[1].subplots(nrows=5, ncols=5)
    for i in range(25):
      axes_right[i // 5, i % 5].imshow(xhat[i], cmap="gray")
      axes_right[i // 5, i % 5].axis('off')
    subfigs[1].suptitle('Reconstructed')

    plt.tight_layout()
    return fig

  def interpolate(self):
    """
    Visualize interpolations between images in latent space.
    You'll see that this interpolates much better than regular autoencoders.
    """

    batch = next(iter(self.val_dl))
    X_test, Y_test = batch["x"], batch["y"]
    X_batch, Y_batch = X_test[:128], Y_test[:128]
    data_size = X_test.size()
    data = X_test.view(X_test.size(0),-1).to(self.device)
    Z, _ = self.model.encoder(data)

    def get_centroid(x):
      """
      Computes the centroid of images in the latent space.
      Args: x: torcch.Tensor of shape: batch x 1 x 28 x 28
      Returns: z_centroid: Centroid in latent space.
      """
      data = x.view(x.size(0),-1).to(self.device)
      Z, _ = self.model.encoder(data)
      Z_centroid = Z.mean(axis=0)
      return Z_centroid

    def get_a2b(a_label: int, b_label: int):
      """Computes the vector in latent space from centroid of a to centroid of b.

      Args:
          a_label: Class `a`
          b_label: Class `b`

      Returns:
          z_a2b: Vector from centroid of `a` to centroid of `b`.
      """

      x_a = X_test[Y_test == a_label]
      x_b = X_test[Y_test == b_label]

      z_a = get_centroid(x_a)
      z_b = get_centroid(x_b)
      z_a2b = z_b - z_a
      return z_a2b

    def interpolate_(a_label = 0):
      """Interpolate in latent space from one class to another class."""

      all_classes = np.arange(0, 10)
      all_classes = np.delete(all_classes, a_label)
      z_a2b_all = []
      for b_label in all_classes:
        z_a2b_all.append(get_a2b(a_label, b_label))

      x_a = X_test[Y_test == a_label]
      data = x_a.view(x_a.size(0),-1).to(self.device)
      z_a, _ = self.model.encode(data)
      z_in = z_a[0]

      x_interpolated = []
      for z_a2b in z_a2b_all:
          for alpha in np.arange(0, 2, 0.2):
              z = z_in + alpha*z_a2b
              x_vae = self.model.decode(z).detach()
              x_interpolated.append(x_vae)

      nrow = len(x_interpolated)
      x_all = torch.stack(x_interpolated)
      img = make_grid(x_all.reshape((nrow, 1, 28, 28)), padding=0, nrow=nrow//9)
      npimg = img.cpu().numpy()

      return npimg
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for a_label, ax in enumerate(axes.flat):
      img = interpolate_(a_label=a_label)
      ax.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')

    return fig

  def visualize_latent_space(self):
    """
    Visualizes the 2D latent space for a VAE model.
    Each digit is plotted with a different color.
    Returns: matplotlib figure
    """
    self.model.eval()

    # Collect all latent representations and labels
    z_list = []
    labels_list = []

    with torch.no_grad():
      for batch in self.val_dl:
        x, y = batch["x"], batch["y"]
        x = x.view(x.size(0), -1).to(self.device)
        mu, _ = self.model.encode(x)
        z_list.append(mu.cpu().numpy())
        labels_list.append(y.numpy())

    # Concatenate all batches
    z = np.concatenate(z_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10), label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    
    return fig


