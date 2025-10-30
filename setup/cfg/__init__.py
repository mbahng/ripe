from datetime import datetime 
import yaml
import os
import torch 
import numpy as np

class Config: 

  def __init__(self, path: str, resume=False): 
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
    torch.manual_seed(self._cfg["seed"])
    torch.cuda.manual_seed(self._cfg["seed"])
    np.random.seed(self._cfg["seed"]) 

