from setup.cfg import Config
from src.trainer import Trainer
from src.loss import *
from src.logger import Logger
import argparse
from pprint import pprint

def main(cfg: Config): 
  """
  Main script for nondistributed training
  """
  device = 0 if cfg["n_gpus"] == 1 else None
  trainer = Trainer(**cfg["run"], device=device)

  # then run initial epoch to validate that everything runs.
  trainer.train(initial=True)
  trainer.val() 

  # then initialize logger, create directory, save config, and save epoch 0 metrics
  logger = Logger(cfg, device=device)
  pprint(trainer.metrics)
  logger.save(trainer)

  for epoch in range(cfg["run"]["epoch"] + 1, cfg["run"]["total_epochs"]):
    print(f"Epoch: {epoch}")
    trainer.train()
    trainer.val()

    pprint(trainer.metrics)
    logger.save(trainer)

  # final epoch with test dataset evaluation
  trainer.train()
  trainer.val()
  trainer.test()

  print(f"Epoch: {cfg['run']['total_epochs']}")
  pprint(trainer.metrics)
  logger.save(trainer)


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', type=str, help='Path to config file.')
  parser.add_argument('--resume', action='store_true', help='Continue from prev run.')
  args = parser.parse_args()

  cfg = Config(args.cfg, resume=args.resume)

  main(cfg)
