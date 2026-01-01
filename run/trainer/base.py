import torch
from torch.nn import Module 
from torch.utils.data import DataLoader 
from torch.optim import Optimizer
from typing import Dict
from torch.nn.parallel import DistributedDataParallel as DDP
from ..model import *

class Trainer: 
  """
  Wrapper class that unifies the dataloader, model, loss, and optimizer. 
  Supports the train, val, and test functions. 
  
  This class acts as a Factory to instantiate the correct Trainer subclass
  based on the model type.
  """
  
  def __new__(cls, *args, **kwargs):
    # Only act as factory if instantiating Trainer directly
    if cls is Trainer:
      # Import inside method to avoid circular import
      from .supervised import SupervisedTrainer 
      from .gan import GANTrainer
      
      model = kwargs.get("model")
      if model is None: 
        model = args[3] 

      if isinstance(model, DDP): 
        model_to_check = model.module 
      else: 
        model_to_check = model

      if isinstance(model_to_check, MLP): return super().__new__(SupervisedTrainer) # type: ignore
      elif isinstance(model_to_check, CNN): return super().__new__(SupervisedTrainer) # type: ignore
      elif isinstance(model_to_check, GAN): return super().__new__(GANTrainer) # type: ignore
      else: raise NotImplementedError

    return super().__new__(cls)

  def __init__(self, 
               train_dl: DataLoader, 
               val_dl: DataLoader, 
               test_dl: DataLoader, 
               model: Module, 
               loss: Module, 
               optimizer: Optimizer, 
               epoch: int = 0,
               device = None,
               **kwargs): 
    self.train_dl = train_dl
    self.val_dl = val_dl
    self.test_dl = test_dl

    self.model = model

    self.loss = loss
    self.optimizer = optimizer
    self.device = device
    self.epoch = epoch
    self.kwargs = kwargs

    self.model.to(self.device)

    # the metrics to keep track of
    self._metrics = {} 
    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }

  @torch.enable_grad
  def train(self, initial=False) -> Dict:
    raise NotImplementedError

  @torch.no_grad
  def val(self) -> Dict:
    raise NotImplementedError

  @torch.no_grad
  def test(self) -> Dict: 
    raise NotImplementedError
