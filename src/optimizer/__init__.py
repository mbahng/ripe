import torch.optim as optim 
import torch.nn as nn
import torch
from .scheduler import *

def init_optimizer(cfg_optim: dict, model: nn.Module): 
  match cfg_optim["warm"]["name"]: 
    case "adam":
      warm_optimizer = torch.optim.Adam([
        {
          'params': model.add_on_layers.parameters(), 
          'lr': float(cfg_optim["warm"]["addon"]["lr"]), 
          'weight_decay': float(cfg_optim["warm"]["addon"]["weight_decay"]), 
        },
        {
          'params': model.prototype_vectors, 
          'lr': float(cfg_optim["warm"]["prototype"]["lr"])
        },
      ])
    case _: 
      raise Exception("Optimizer not defined")

  match cfg_optim["joint"]["name"]: 
    case "adam":
      joint_optimizer = torch.optim.Adam([
          {
            'params': model.backbone.parameters(), 
            'lr': float(cfg_optim["joint"]["backbone"]["lr"]), 
            'weight_decay': float(cfg_optim["joint"]["backbone"]["weight_decay"]), 
          },
          {
            'params': model.add_on_layers.parameters(), 
            'lr': float(cfg_optim["joint"]["addon"]["lr"]), 
            'weight_decay': float(cfg_optim["joint"]["addon"]["weight_decay"]), 
          },
          {
            'params': model.prototype_vectors, 
            'lr': float(cfg_optim["joint"]["prototype"]["lr"]), 
          },
      ])
    case _: 
      raise Exception("Optimizer not defined")


  match cfg_optim["joint"]["name"]: 
    case "adam":
      last_layer_optimizer = torch.optim.Adam([
          {
            'params': model.last_layer.parameters(), 
            'lr': float(cfg_optim["last_layer"]["lr"])
          }
      ])
    case _: 
      raise Exception("Optimizer not defined")

  if ckpt_path := cfg_optim["warm"]["checkpoint"]: 
    last_layer_optimizer.load_state_dict(torch.load(ckpt_path))

  if ckpt_path := cfg_optim["joint"]["checkpoint"]: 
    last_layer_optimizer.load_state_dict(torch.load(ckpt_path))

  if ckpt_path := cfg_optim["last_layer"]["checkpoint"]: 
    last_layer_optimizer.load_state_dict(torch.load(ckpt_path))

  return warm_optimizer, joint_optimizer, last_layer_optimizer

