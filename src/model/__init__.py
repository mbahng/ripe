import torch
from .mlp import * 
from .protopnet import *
from torch.nn import Module
from copy import deepcopy

def init_model(cfg_model: dict) -> Module: 
  """
  Returns the model from the config. 
  """
  args = deepcopy(cfg_model)
  name = args.pop("name")
  ckpt_path = args.pop("checkpoint", None) 
  match name: 
    case "mlp": 
      print("Loading Model: MLP")
      model = MLP(**args)
    case "protopnet": 
      print("Loading Model: ProtoPNet")
      model = ProtoPNet(**args)
    case _: 
      raise Exception("Model not defined")

  if ckpt_path: 
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

  return model

