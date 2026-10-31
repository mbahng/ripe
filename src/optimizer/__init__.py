import torch.optim as optim 
import torch.nn as nn
import torch
from .scheduler import *

def init_optimizer(cfg_optim: dict, model: nn.Module): 
  if "all" in cfg_optim:
    spec = cfg_optim["all"]
    match spec["name"]: 
      case "sgd": 
        optimizer = optim.SGD(
            params=model.parameters(), 
            lr=float(spec["lr"]), 
            momentum=float(spec["momentum"]), 
            weight_decay=float(spec["weight_decay"]))
      case "adam": 
        optimizer = optim.Adam(
            params=model.parameters(), 
            lr=float(spec["lr"]), 
            betas=spec["betas"],
            weight_decay=float(spec["weight_decay"])) 
      case _: 
        raise Exception("Optimizer not defined")

    if ckpt_path := spec["checkpoint"]: 
      optimizer.load_state_dict(torch.load(ckpt_path))
    return optimizer
