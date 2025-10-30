import torch.optim as optim 
import torch.nn as nn
import torch
from .scheduler import *

def init_optimizer(cfg_optim: dict, model: nn.Module): 
  match cfg_optim["name"]: 
    case "sgd": 
      optimizer = optim.SGD(
          params=model.parameters(), 
          lr=float(cfg_optim["lr"]), 
          momentum=float(cfg_optim["momentum"]), 
          weight_decay=float(cfg_optim["weight_decay"]))
    case "adam": 
      optimizer = optim.Adam(
          params=model.parameters(), 
          lr=float(cfg_optim["lr"]), 
          betas=cfg_optim["betas"],
          weight_decay=cfg_optim["weight_decay"]) 
    case _: 
      raise Exception("Optimizer not defined")

  if ckpt_path := cfg_optim["checkpoint"]: 
    optimizer.load_state_dict(torch.load(ckpt_path))
  return optimizer

