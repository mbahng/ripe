from setup.cfg import Config
from src.trainer import Trainer
import src.model
from src.loss import *
from src.logger import Logger
import argparse
from pprint import pprint
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_reduce
import os

def ddp_setup(rank, world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "1235"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

def aggregate_metrics(metrics_dict, rank):
  """Aggregate metrics across all GPUs by summing total_loss, n_correct, n_samples"""

  aggregated = {}
  for split_name, split_metrics in metrics_dict.items():
    # Convert metrics to tensors for all_reduce
    total_loss = torch.tensor(split_metrics['total_loss'], dtype=torch.float32, device=f'cuda:{rank}')
    n_correct = torch.tensor(split_metrics['n_correct'], dtype=torch.long, device=f'cuda:{rank}')
    n_samples = torch.tensor(split_metrics['n_samples'], dtype=torch.long, device=f'cuda:{rank}')

    # Sum across all GPUs
    all_reduce(total_loss)
    all_reduce(n_correct)
    all_reduce(n_samples)

    # Convert back and compute derived metrics
    aggregated[split_name] = {
      'total_loss': total_loss.item(),
      'n_correct': int(n_correct.item()),
      'n_samples': int(n_samples.item()),
      'accuracy': n_correct.item() / n_samples.item() if n_samples.item() > 0 else 0.0,
      'mean_loss': total_loss.item() / n_samples.item() if n_samples.item() > 0 else 0.0
    }

  return aggregated

def main_distributed(rank: int, cfg: Config): 
  """
  Main script for distributed training. Rank (device) is variable and taken care of by 
  torch.multiprocessing
  """

  ddp_setup(rank=rank, world_size=cfg["n_gpus"])

  train_dl, val_dl, test_dl = src.dataset.init_dataloader(cfg["dataset"])
  model = src.model.init_model(cfg["run"]["model"]).to(rank)
  model = DDP(model, device_ids=[rank])
  loss = src.loss.init_loss(cfg["run"]["loss"])
  optimizer = src.optimizer.init_optimizer(cfg["run"]["optimizer"], model)

  # first initialize trainer for each device
  trainer = Trainer(train_dl, val_dl, test_dl, model, loss, optimizer, device=rank) 

  # then run initial epoch to validate that everything runs.
  train_metrics = trainer.train(initial=True)
  val_metrics = trainer.val()
  metrics = {"train": train_metrics, "val": val_metrics}

  # then initialize logger, create directory, save config, and save epoch 0 metrics
  logger = Logger(cfg, rank)

  # Aggregate metrics across all GPUs
  agg_metrics = aggregate_metrics(metrics, rank)

  logger.save_metrics(agg_metrics, epoch=cfg["run"]["epoch"])
  print(f"Epoch: {cfg['run']['epoch']}")
  pprint(agg_metrics)
  logger.save_state(trainer, epoch=cfg["run"]["epoch"])

  try:
    for epoch in range(cfg["run"]["epoch"] + 1, cfg["run"]["total_epochs"]):
      train_metrics = trainer.train()
      val_metrics = trainer.val()
      metrics = {"train": train_metrics, "val": val_metrics}

      # Aggregate metrics across all GPUs
      agg_metrics = aggregate_metrics(metrics, rank)

      print(f"Epoch: {epoch}")
      pprint(agg_metrics)

      logger.save_metrics(agg_metrics, epoch)
      logger.save_state(trainer, epoch)

    train_metrics = trainer.train()
    val_metrics = trainer.val()
    test_metrics = trainer.test()
    metrics = {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    # Aggregate metrics across all GPUs
    agg_metrics = aggregate_metrics(metrics, rank)

    print(f"Epoch: {cfg['run']['total_epochs']}")
    pprint(agg_metrics)

    logger.save_metrics(agg_metrics, cfg['run']['total_epochs'])
    logger.save_state(trainer, cfg['run']['total_epochs'])

  except KeyboardInterrupt:
    print("Run Failed. Keyboard Interrupt.")
    print(f"Logs in: {cfg['log']['savedir']}")

  except Exception as e: 
    print(e) 
    print(f"Run Failed. Logs in: {cfg['log']['savedir']}")

  finally:
    destroy_process_group()

if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', type=str, help='Path to config file.')
  parser.add_argument('--resume', action='store_true', help='Continue from prev run.')
  args = parser.parse_args()

  cfg = Config(args.cfg, resume=args.resume)
  import time
  now = time.time()

  if cfg["n_gpus"] > 1:
    mp.spawn(main_distributed, args=(cfg,), nprocs=cfg["n_gpus"])
  else:
    main(cfg)

  print(time.time() - now)

