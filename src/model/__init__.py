import torch
from .mlp import * 
from .cnn import * 
from .resnet import *
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
    case "cnn": 
      print("Loading Model: CNN")
      model = CNN(**args)
    case "resnet18": 
      print("Loading Model: Resnet 18") 
      model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], pretrained="resnet18", **args)
    case "resnet34": 
      print("Loading Model: Resnet 34") 
      model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], pretrained="resnet34", **args)
    case "resnet50[pretraining=inaturalist]": 
      print("Loading Model: Resnet 50 [pretraining=inaturalist]") 
      model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], pretrained="resnet50[pretraining=inaturalist]", **args)
    case "resnet50": 
      print("Loading Model: Resnet 50") 
      model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], pretrained="resnet50", **args)
    case "resnet101": 
      print("Loading Model: Resnet 101") 
      model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], pretrained="resnet101", **args)
    case "resnet152": 
      print("Loading Model: Resnet 152") 
      model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], pretrained="resnet152", **args)
    case _: 
      raise Exception("Model not defined")

  if ckpt_path: 
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

  return model

