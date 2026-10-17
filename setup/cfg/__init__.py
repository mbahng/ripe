from datetime import datetime
import yaml
import os
import random
import torch
import numpy as np


def parse_overrides(tokens): 
  """
  Parse a list of CLI tokens of the form --run.optimizer.all.lr=0.0003 into
  {"run.optimizer.all.lr": 0.0003}. Values are run through the yaml parser so
  that numbers, bools, and lists arrive with the right type instead of as
  strings. This is the format a wandb sweep agent emits for ${args}.
  """
  overrides = {} 
  for token in tokens: 
    if not token.startswith("--") or "=" not in token: 
      raise Exception(f"Overrides must look like --path.to.key=value, got '{token}'")
    key, _, raw = token[2:].partition("=")
    overrides[key] = "" if raw == "" else yaml.safe_load(raw)
  return overrides


class Config: 

  def __init__(self, path: str, resume=False, overrides=None): 
    """
    There are two ways you can start a run. 
      1. You can make a fresh run A, which should have checkpoint=False and the 
      run name should not override any existing runs. 
      2. You can make a new run A that initializes from a previously saved run B, 
      but you save in new directory A. 
      3. You continue an old run A and save in A. 
    """
    with open(path, "r") as f:
      self._cfg = yaml.safe_load(f)

    # applied before anything is derived from the config, since name, savedir,
    # the dataset seed, and the distributed flags are all computed below
    self._apply_overrides(overrides)

    self._set_distributed_status() 
    # need to set seed for datasets since DistributedSampler also takes in seed
    self._cfg["run"]["dataset"]["seed"] = self._cfg["seed"]

    if not resume: 
      # make a new run directory, which may or may not start from a checkpoint
      if self["name"] in os.listdir("saved"): 
        raise Exception("You are overriding a previous run with the same run name. Choose a different name.")
      if not self._cfg["name"]: 
        self._create_name() 
      self["log"]["savedir"] = os.path.join("saved", self["name"])

    else: 
      # continue from an old run directory with exact same config 
      if self["name"] not in os.listdir("saved"): 
        raise Exception(f"Cannot find {self['name']} in saved/. There is no existing run to continue off of.")
      if path != f"saved/{self['name']}/cfg.yml": 
        raise Exception(f"The config file path should be the previous saved run directory. Use 'saved/{self['name']}/cfg.yml'")
      self._prepare_cfg_from_checkpoint()
      print(f"Resuming previous run ({path}) at Epoch {self['run']['epoch']}")

    self.post_init_override()
    self.set_seed()

  def __getitem__(self, key): 
    return self._cfg[key]

  def _apply_overrides(self, overrides): 
    """
    Override config entries by dotted path, e.g. "run.optimizer.all.lr". Only
    existing keys may be overridden, so a typo in a sweep parameter name fails
    loudly instead of silently training with the default value.
    """
    for dotted, value in (overrides or {}).items(): 
      keys = dotted.split(".")
      current = self._cfg
      for i, key in enumerate(keys[:-1]): 
        if not isinstance(current, dict) or key not in current: 
          raise Exception(f"Cannot override '{dotted}': no '{'.'.join(keys[:i+1])}' in config.")
        current = current[key]
      if not isinstance(current, dict) or keys[-1] not in current: 
        raise Exception(f"Cannot override '{dotted}': no '{dotted}' in config.")
      current[keys[-1]] = value
      print(f"Override: {dotted} = {value}")

  def _create_name(self): 
    self._cfg["name"] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f') 
    if self._cfg["name"] in os.listdir("saved"): 
      raise Exception("Name of run already exists, choose a different one.")

  def _set_distributed_status(self): 
    if self._cfg["n_gpus"] > 1: 
      self._cfg["run"]["dataset"]["is_distributed"] = True
      self._cfg["run"]["is_distributed"] = True
    else: 
      self._cfg["run"]["dataset"]["is_distributed"] = False 
      self._cfg["run"]["is_distributed"] = False 

  def _prepare_cfg_from_checkpoint(self): 
    """
    Does the following. 
    1. Looks through the previous run and finds the last epoch which has the model/optimizer saved. 
    2. Updates the current epoch in cfg
    3. Updates the model/optimizer path in cfg 
    """
    # update current epoch, model/optimizer checkpoints  
    # look at the dirs that have actual model and optimizer weights saved. 
    dirs = [] 
    for f in os.listdir(f"saved/{self['name']}"): 
      if os.path.isdir(epoch_dir := f"saved/{self['name']}/{f}") and f != "wandb":  
        if "model.pt" in os.listdir(epoch_dir) and "optimizer.pt" in os.listdir(epoch_dir): 
          dirs.append(f) 
    final_ep_dir = sorted(dirs)[-1]
    self["run"]["model"]["checkpoint"] = f"saved/{self['name']}/{final_ep_dir}/model.pt"
    self["run"]["optimizer"]["checkpoint"] = f"saved/{self['name']}/{final_ep_dir}/optimizer.pt"
    self["run"]["epoch"] = int(final_ep_dir)

  def post_init_override(self): 
    """
    Add stuff here if you want to manually override anything. 
    Usually, you would use this if you want to do a grid search with a batch of runs.  
    """
    pass

  def set_seed(self):
    seed = self._cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

