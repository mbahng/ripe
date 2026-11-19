import torch.optim as optim 
import torch.nn as nn
import torch
from .scheduler import *

def _backbone_param_split(model, spec: dict):
  """Split the trunk into the layers an override names and everything else.

  ProtoTree thaws its last residual block during warm-up while the rest of the trunk
  stays frozen, so those parameters need their own group with their own learning rate.
  Returns (standard, override); override is empty when no spec is given.
  """
  names = list(spec.get("layers", [])) if spec else []
  if not names:
    return list(model.backbone.parameters()), []
  standard, override = [], []
  for name, param in model.backbone.named_parameters():
    (override if any(n in name for n in names) else standard).append(param)
  return standard, override


def _adamw_groups(model, spec: dict, with_backbone: bool):
  """Add-on, prototype and (optionally) backbone groups for one AdamW phase."""
  decay = lambda block: float(block.get("weight_decay", 0.0))
  groups = [
    {'params': list(model.add_on_layers.parameters()),
     'lr': float(spec["addon"]["lr"]), 'weight_decay': decay(spec["addon"])},
    {'params': [model.prototype_vectors],
     'lr': float(spec["prototype"]["lr"]), 'weight_decay': decay(spec["prototype"])},
  ]
  standard, override = _backbone_param_split(model, spec.get("backbone_override"))
  if with_backbone:
    groups.append({'params': standard, 'lr': float(spec["backbone"]["lr"]),
                   'weight_decay': decay(spec["backbone"])})
  if override:
    block = spec["backbone_override"]
    groups.append({'params': override, 'lr': float(block["lr"]),
                   'weight_decay': decay(block)})
  return groups


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
    case "adamw":
      warm_optimizer = torch.optim.AdamW(
        _adamw_groups(model, cfg_optim["warm"], with_backbone=False),
        eps=float(cfg_optim["warm"].get("eps", 1e-8)),
        weight_decay=float(cfg_optim["warm"].get("weight_decay", 0.0)))
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
    case "adamw":
      joint_optimizer = torch.optim.AdamW(
        _adamw_groups(model, cfg_optim["joint"], with_backbone=True),
        eps=float(cfg_optim["joint"].get("eps", 1e-8)),
        weight_decay=float(cfg_optim["joint"].get("weight_decay", 0.0)))
    case _: 
      raise Exception("Optimizer not defined")

  # ProtoTree has no last layer: its leaf distributions stand in for one. They are
  # fitted by the derivative-free update rather than by this optimizer, which only
  # exists so the last-only stage and the checkpoint layout match the other models.
  if hasattr(model, 'last_layer'):
    last_layer_params = list(model.last_layer.parameters())
  else:
    last_layer_params = [leaf._dist_params for leaf in sorted(model.leaves, key=lambda n: n.index)]
  last_layer_group = [{'params': last_layer_params, 'lr': float(cfg_optim["last_layer"]["lr"])}]

  match cfg_optim["last_layer"].get("name", cfg_optim["joint"]["name"]):
    case "adam":
      last_layer_optimizer = torch.optim.Adam(last_layer_group)
    case "adamw":
      last_layer_optimizer = torch.optim.AdamW(
        last_layer_group,
        eps=float(cfg_optim["last_layer"].get("eps", 1e-8)),
        weight_decay=float(cfg_optim["last_layer"].get("weight_decay", 0.0)))
    case _:
      raise Exception("Optimizer not defined")

  if ckpt_path := cfg_optim["warm"]["checkpoint"]:
    warm_optimizer.load_state_dict(torch.load(ckpt_path))

  if ckpt_path := cfg_optim["joint"]["checkpoint"]:
    joint_optimizer.load_state_dict(torch.load(ckpt_path))

  if ckpt_path := cfg_optim["last_layer"]["checkpoint"]: 
    last_layer_optimizer.load_state_dict(torch.load(ckpt_path))

  return warm_optimizer, joint_optimizer, last_layer_optimizer
