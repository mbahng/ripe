import torch
from enum import Enum
from typing import Dict
from copy import deepcopy

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
    
    model_name = kwargs["model_cfg"]["name"]

    if model_name in ("mlp", "cnn") or model_name.startswith("resnet"):
      return super().__new__(SupervisedTrainer) # type: ignore
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
