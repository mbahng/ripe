import torch
from typing import Dict
import src
from copy import deepcopy

class Trainer: 
  """
  Wrapper class that unifies the dataloader, model, loss, and optimizer. 
  Supports the train, val, and test functions. 
  
  This class acts as a Factory to instantiate the correct Trainer subclass
  based on the model type.
  """
  def __new__(cls, **kwargs):
    # Import inside method to avoid circular import
    from .supervised import SupervisedTrainer 
    from .protopnet import ProtoPNetTrainer 
    from .prototree import ProtoTreeTrainer 
    
    model_name = kwargs["model"]["name"]

    if model_name == "convolutional" or "resnet" in model_name: return super().__new__(SupervisedTrainer) # type: ignore
    elif model_name == "protopnet": return super().__new__(ProtoPNetTrainer) # type: ignore
    elif model_name == "prototree": return super().__new__(ProtoTreeTrainer) # type: ignore
    else: raise NotImplementedError

  def __init__(self, device, model, **kwargs):
    self.device = device

    self.model = src.model.init_model(model).to(self.device)
    self.model.device = self.device

    self.epoch = 0

    # the metrics to keep track of
    self._metrics = {} 
    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }

  def run(self, logger) -> None: 
    raise NotImplementedError

  @torch.enable_grad
  def train(self, initial=False) -> Dict: 
    raise NotImplementedError

  @torch.no_grad
  def val(self) -> Dict:
    raise NotImplementedError

  @torch.no_grad
  def test(self) -> Dict: 
    raise NotImplementedError
