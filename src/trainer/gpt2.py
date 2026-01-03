import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer 
from ..model import GPT2

class GPT2Trainer(Trainer):
  """Trainer implementation for supervised models (MLP, CNN, etc.)"""

  model: GPT2

  def __init__(self, **cfg):
    super().__init__(**cfg)
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

    assert cfg["model"]["block_size"] == cfg["dataset"]["block_size"]
    assert cfg["model"]["vocab_size"] == cfg["dataset"]["vocab_size"]

    # define custom training hyperparameters here 
    self.itos = self.train_dl.dataset.dataset.itos # type: ignore

  @torch.enable_grad
  def train(self, initial=False):
    train_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      logits = self.model(x)
      loss = self.loss(logits.view(-1, self.model.vocab_size), y.view(-1))

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
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      logits = self.model(x)
      loss = self.loss(logits.view(-1, self.model.vocab_size), y.view(-1))
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
      x, y = batch["x"].to(self.device), batch["y"].to(self.device)
      logits = self.model(x)
      loss = self.loss(logits.view(-1, self.model.vocab_size), y.view(-1))
      test_metrics["total_loss"] += loss.item()
      test_metrics["average_loss"] = test_metrics["total_loss"] / len(self.test_dl.dataset) # type: ignore

    test_metrics["time"] = time.time() - start
    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def generate_text(self):  
    self.model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=self.device)  # Start with empty context
    generated = self.model.generate(context, max_tokens=500)
    generated_text = ''.join([self.itos[i] for i in generated[0].tolist()])
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50)
    print(generated_text)



