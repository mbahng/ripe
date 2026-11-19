import torch
from enum import Enum
from typing import Dict
from copy import deepcopy


class Convergence:
  """Early stopping on a plateau, for sweeps that should not pay for a flat tail.

  Configured under ``run.convergence``; absent, or with patience 0, it never fires::

      convergence:
        metric: val.balanced_accuracy   # "<split>.<name>" from trainer.metrics
        min_delta: 0.005                # smaller than this is not an improvement
        patience: 3                     # consecutive non-improving epochs to stop
        goal: max                       # max | min; inferred from the name if absent

  An epoch improves only if the metric beats the *best seen so far* by more than
  min_delta; patience counts how many consecutive epochs fail to. Because the bar
  is the running best rather than the previous epoch, a slow climb still clears it
  eventually and resets the count: min_delta is the improvement required per
  patience window, not per epoch. A run gaining 0.002 an epoch with min_delta
  0.005 and patience 3 therefore keeps going, because every third epoch is worth
  0.006. Raise min_delta, not patience, to be stricter about slow climbs.

  Prototype models perturb themselves on purpose: a ProtoPNet push projects the
  prototypes onto real patches and costs accuracy for an epoch or two afterwards.
  Those epochs are counted like any other, so set patience above the recovery
  window -- with pushes every 10 epochs, patience 3 is comfortable and patience 1
  would stop on the first push.
  """

  # Loss-like names get minimized, everything else maximized. Only consulted when
  # the config does not say, so a surprising name can always be settled explicitly.
  _MINIMIZE = ("loss", "cost", "entropy", "error", "nll", "cluster", "separation",
               "l1", "variability", "orthogonality")

  def __init__(self, cfg: dict | None, default_metric: str = "val.balanced_accuracy"):
    cfg = cfg or {}
    self.patience = int(cfg.get("patience", 0) or 0)
    self.metric = str(cfg.get("metric", default_metric))
    self.min_delta = float(cfg.get("min_delta", 0.0))
    goal = cfg.get("goal")
    if goal is None:
      leaf = self.metric.rsplit(".", 1)[-1].lower()
      goal = "min" if any(tok in leaf for tok in self._MINIMIZE) else "max"
    if goal not in ("max", "min"):
      raise ValueError(f"run.convergence.goal must be 'max' or 'min', got {goal!r}")
    self.goal = goal
    self.best = None
    self.best_epoch = None
    self.strikes = 0
    self.reason = ""

  @property
  def enabled(self) -> bool:
    return self.patience > 0

  def _improved(self, value: float) -> bool:
    if self.best is None:
      return True
    return (value > self.best + self.min_delta if self.goal == "max"
            else value < self.best - self.min_delta)

  def update(self, metrics: Dict, epoch: int) -> bool:
    """Record this epoch's metric; True means the run should stop now."""
    if not self.enabled:
      return False
    split, _, name = self.metric.partition(".")
    if split not in metrics or name not in metrics[split]:
      available = sorted(f"{s}.{n}" for s, m in metrics.items() for n in m)
      raise KeyError(f"run.convergence.metric {self.metric!r} is not one of {available}")
    value = float(metrics[split][name])

    if self._improved(value):
      self.best, self.best_epoch, self.strikes = value, epoch, 0
      return False

    self.strikes += 1
    print(f"[convergence] {self.metric}={value:.4f} did not beat {self.best:.4f} "
          f"(epoch {self.best_epoch}) by {self.min_delta}: {self.strikes}/{self.patience}")
    if self.strikes < self.patience:
      return False
    self.reason = (f"{self.metric} stalled at {self.best:.4f} (epoch {self.best_epoch}) "
                   f"for {self.patience} epochs")
    print(f"[convergence] {self.reason}; stopping early at epoch {epoch}", flush=True)
    return True


class Trainer: 
  """
  Wrapper class that unifies the dataloader, model, loss, and optimizer. 
  Supports the train, val, and test functions. 
  
  This class acts as a Factory to instantiate the correct Trainer subclass
  based on the model type.

  Each subclass defines its own nested ``Stage`` enum, which must include an
  ``EVAL`` member (an evaluation-only pass); ``train`` takes one of its members.
  """
  Stage: type[Enum]

  def __new__(cls, **kwargs):
    # Import inside method to avoid circular import
    from .supervised import SupervisedTrainer 
    from .protopnet import ProtoPNetTrainer 
    from .prototree import ProtoTreeTrainer 

    model_name = kwargs["model_cfg"]["name"]

    if model_name in ("mlp", "cnn") or model_name.startswith("resnet"):
      return super().__new__(SupervisedTrainer) # type: ignore
    elif model_name == "protopnet": 
      return super().__new__(ProtoPNetTrainer) # type: ignore
    elif model_name == "prototree": 
      return super().__new__(ProtoTreeTrainer) # type: ignore

    else: raise NotImplementedError

  def __init__(self, device, dataset_cfg, model_cfg, loss_cfg, optimizer_cfg, epoch, **kwargs):

    self.device = None 
    self.train_dl, self.val_dl, self.test_dl = None, None, None
    self.model = None  
    self.loss = None 
    self.optimizer = None 
    self.epoch = None 

    # the metrics to keep track of
    self._metrics = {} 
    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
    }

  def profile(self):
    """
    Time the dataloaders and sample system utilization. Call right after construction;
    afterwards call self.profiler.stop() and self.profiler.report().
    """
    from ..profiler import Profiler
    self.profiler = Profiler(device=self.device)
    self.train_dl = self.profiler.wrap(self.train_dl, "train")
    self.val_dl = self.profiler.wrap(self.val_dl, "val")
    self.test_dl = self.profiler.wrap(self.test_dl, "test")
    self.profiler.start()

  def run(self, logger) -> None:
    ...

  @torch.enable_grad
  def train(self, stage=None) -> Dict: 
    ...

  @torch.no_grad
  def val(self) -> Dict:
    ...

  @torch.no_grad
  def test(self) -> Dict: 
    ...
