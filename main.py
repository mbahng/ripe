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
  logger = Logger(cfg, device=device)

  run = dict(cfg["run"])
  trainer = Trainer(device=device,
                    dataset_cfg=run.pop("dataset"),
                    model_cfg=run.pop("model"),
                    loss_cfg=run.pop("loss"),
                    optimizer_cfg=run.pop("optimizer"),
                    **run)

  # then run initial epoch to validate that everything runs.
  trainer.train(stage=trainer.Stage.EVAL) # type: ignore
  trainer.val() 

  # then initialize logger, create directory, save config, and save epoch 0 metrics
  pprint(trainer.metrics)
  logger.save(trainer)

  trainer.run(logger)

  # final epoch with test dataset evaluation
  trainer.train()
  trainer.val()
  trainer.test()
  pprint(trainer.metrics)
  logger.save(trainer)


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', type=str, help='Path to config file.')
  parser.add_argument('--resume', action='store_true', help='Continue from prev run.')
  args = parser.parse_args()

  cfg = Config(args.cfg, resume=args.resume)

  main(cfg)
