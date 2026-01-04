import torch
from .mlp import * 
from .nice import NICE
from .realnvp import RealNVP, RealNVP1d
from .glow import Glow
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
    case "nice": 
      print("Loading Model: NICE")
      model = NICE(**args)
    case "realnvp1d": 
      print("Loading Model: RealNVP1d")
      model = RealNVP1d(**args)
    case "realnvp": 
      print("Loading Model: RealNVP")
      model = RealNVP(**args)
    case "glow": 
      print("Loading Model: GLOW")
      model = Glow(**args)
    case _: 
      raise Exception("Model not defined")

  if ckpt_path: 
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

  return model


