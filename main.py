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

  for _ in range(cfg["run"]["warm_epochs"]):
    trainer.train(stage="warm")
    trainer.val()
    pprint(trainer.metrics)
    logger.save(trainer)  

  for joint_epoch in range(cfg["run"]["joint_epochs"]): 
    if joint_epoch % cfg["run"]["push_every"]: 
      trainer.push(visualize=False) 
      continue

    trainer.train(stage="joint")
    trainer.val()
    pprint(trainer.metrics)
    logger.save(trainer)

  # push last time 
  trainer.push(visualize=True) 

  for _ in range(5): 
    trainer.train(stage="last") 
    trainer.val()
    pprint(trainer.metrics)
    logger.save(trainer)  

  # final epoch with test dataset evaluation
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
