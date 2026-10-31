from setup.cfg import Config
from src.trainer import Trainer
from src.loss import *
from src.logger import Logger
import argparse
from pprint import pprint

def main(cfg: Config, profile: bool = False):
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
  if profile: trainer.profile()

  # then run initial epoch to validate that everything runs.
  trainer.train(stage=trainer.Stage.EVAL) # type: ignore
  trainer.val() 

  # then initialize logger, create directory, save config, and save epoch 0 metrics
  pprint(trainer.metrics)
  logger.save(trainer)

  trainer.run(logger)

  # final evaluation pass over every split; no optimizer step, no epoch increment
  trainer.train(stage=trainer.Stage.EVAL) # type: ignore
  trainer.val()
  trainer.test()
  pprint(trainer.metrics)
  logger.save(trainer)

  if profile:
    trainer.profiler.stop()
    trainer.profiler.report()


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', type=str, help='Path to config file.')
  parser.add_argument('--resume', action='store_true', help='Continue from prev run.')
  parser.add_argument('--profile', action='store_true', help='Print a CPU/GPU bottleneck report at the end of the run.')
  args = parser.parse_args()

  cfg = Config(args.cfg, resume=args.resume)

  main(cfg, profile=args.profile)
