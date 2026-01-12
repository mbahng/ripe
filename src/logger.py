import os
import sys
import yaml
import torch
from .trainer import Trainer
import wandb
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Logger: 
  """
  Logs and saves the following. 
    - configuration files 
    - Metrics as json 
    - Model and optimize state_dicts 
    - figures and visuals per epoch
    - logs to wandb 
  Should add custom functions for saving other data specific to your project. 
  """

  def __init__(self, cfg, device):
    cfg_log = cfg["log"]
    self.savedir = cfg_log["savedir"]

    self.track = cfg_log["track"]
    self.checkpoint = cfg_log["checkpoint"] 
    self.diagnose = cfg_log["diagnose"]

    self.wandb_enabled = cfg_log["wandb"]["enabled"]
    self.is_distributed = True if cfg["n_gpus"] > 1 else False
    self.device = device

    os.makedirs(self.savedir, exist_ok=True)
    self._save_cfg(cfg)
    self._setup_stdout_logging()

    # Only initialize wandb on device 0 to avoid creating multiple runs
    if self.wandb_enabled and (not self.is_distributed or self.device == 0):
      self.wandb_logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=cfg_log["wandb"]["entity"],
        # Set the wandb project where this run will be logged.
        project=cfg_log["wandb"]["project"],
        # Track hyperparameters and run metadata.
        config=cfg,
        name=cfg["name"],
        # Store wandb data in the same directory as other run data
        dir=self.savedir
      )
    else:
      self.wandb_logger = None

  def _save_cfg(self, cfg):
    cfg_path = os.path.join(self.savedir, "cfg.yml")
    with open(cfg_path, "w") as f:
      yaml.dump(cfg._cfg, f, default_flow_style=False)

  def _setup_stdout_logging(self):
    """Redirect stdout and stderr to both console and stdout.txt"""
    stdout_path = os.path.join(self.savedir, "stdout.txt")
    self.stdout_file = open(stdout_path, 'a+')
    self.terminal = sys.stdout
    self.terminal_err = sys.stderr
    self._at_line_start = True  # Track if we're at the beginning of a line
    sys.stdout = self
    sys.stderr = self

  def write(self, message):
    """Required method for stdout redirection - called by print()"""
    if not message:
      return
    if not self._is_main_process(): 
      return

    device = f"GPU {self.device}" if self.device is not None else "CPU"

    # Process the message character by character to add prefix at line starts
    output = []
    for char in message:
      if self._at_line_start and char not in ('\n', '\r'):
        output.append(f"[{device}] ")
        self._at_line_start = False
      output.append(char)
      if char == '\n':
        self._at_line_start = True

    prefixed_message = ''.join(output)
    self.terminal.write(prefixed_message)
    self.stdout_file.write(prefixed_message)
    self.stdout_file.flush()

  def flush(self):
    """Required method for stdout redirection"""
    self.terminal.flush()
    self.stdout_file.flush()

  def isatty(self):
    """Required method for stdout/stderr redirection - checks if underlying stream is a TTY"""
    return self.terminal.isatty()

  def _is_main_process(self): 
    if self.is_distributed and self.device != 0: 
      return False 
    return True

  def save(self, trainer: Trainer): 
    if not self._is_main_process(): 
      return

    # tracking (metrics), always should be done
    epoch_save_dir = os.path.join(self.savedir, str(trainer.epoch).zfill(4)) 
    os.makedirs(epoch_save_dir, exist_ok=True)
    with open(os.path.join(epoch_save_dir, "metrics.json"), 'w') as f:
      json.dump(trainer.metrics, f, indent=2)
    if self.wandb_enabled and self.wandb_logger is not None:
      self.wandb_logger.log(trainer.metrics, step=trainer.epoch)
      
    model = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model 

    # checkpointing (model, optimizer), i.e. save locally. 
    if trainer.epoch % self.checkpoint["every"] == 0: 
      if self.checkpoint.get("model"): 
        torch.save(model.state_dict(), os.path.join(epoch_save_dir, "model.pt")) 
      if self.checkpoint.get("optimizer"): 
        torch.save(trainer.warm_optimizer.state_dict(), os.path.join(epoch_save_dir, "warm_optimizer.pt")) 
        torch.save(trainer.joint_optimizer.state_dict(), os.path.join(epoch_save_dir, "joint_optimizer.pt")) 
        torch.save(trainer.last_optimizer.state_dict(), os.path.join(epoch_save_dir, "last_layer_optimizer.pt")) 

    # diagnosing (weights, gradients, visualizations), i.e. save on wandb 
    if trainer.epoch % self.diagnose["every"] == 0: 
      save_local = self.diagnose.get("save_local", False) 

      if self.diagnose.get("weights") and self.wandb_logger:
        weights_log = {}
        for name, param in model.named_parameters():
          weights_log[f"weights/{name}"] = wandb.Histogram(param.detach().cpu().numpy()) # type: ignore
        self.wandb_logger.log(weights_log, step=trainer.epoch)

      if self.diagnose.get("gradients") and self.wandb_logger: 
        grads_log = {}
        for name, param in model.named_parameters():
          if param.grad is not None:
            grads_log[f"gradients/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy()) # type: ignore
        self.wandb_logger.log(grads_log, step=trainer.epoch)

      if self.diagnose.get("sample"): 
        if hasattr(trainer, "predict_batch"):
          fig = trainer.predict_batch() # type: ignore
          if self.wandb_logger:
            self.wandb_logger.log({f"sample_digits": wandb.Image(fig)}, step=trainer.epoch) 
          
          if save_local:
            fig.savefig(os.path.join(epoch_save_dir, "sample_digits.png"))
          
          plt.close(fig)
