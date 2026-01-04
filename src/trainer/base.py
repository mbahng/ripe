import torch
from typing import Dict
import src
from copy import deepcopy
from ..model import *

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
    from .nice import NICETrainer
    from .realnvp import RealNVPTrainer
    from .glow import GlowTrainer
    
    model_name = kwargs["model"]["name"]

    if model_name == "mlp": return super().__new__(SupervisedTrainer) # type: ignore
    elif model_name == "nice": return super().__new__(NICETrainer) # type: ignore
    elif model_name == "realnvp1d": return super().__new__(NICETrainer) # type: ignore
    elif model_name == "realnvp": return super().__new__(RealNVPTrainer) # type: ignore
    elif model_name == "glow": return super().__new__(GlowTrainer) # type: ignore
    else: raise NotImplementedError

  def __init__(self, device, dataset, epoch, total_epochs, model, loss, optimizer, **kwargs):
    self.device = device

    self.train_dl, self.val_dl, self.test_dl = src.dataset.init_dataloader(dataset)

    self.model = src.model.init_model(model).to(self.device)
    self.model.device = self.device

    self.loss = src.loss.init_loss(loss)
    self.optimizer = src.optimizer.init_optimizer(optimizer, self.model)

    self.epoch = epoch 
    self.total_epochs = total_epochs

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
