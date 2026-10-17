"""
Runs a W&B sweep by generating a trial config file per trial and handing it to main.py.

main.py takes only --cfg, so sampled hyperparameters are baked into a generated
config rather than passed as arguments. Three modes:

  python sweep.py --cfg setup/sweep/<name>.yml              create the sweep, print its id
  python sweep.py --cfg setup/sweep/<name>.yml --count 20    create it and run 20 trials
  python sweep.py --cfg ... --id <id> --count 20             attach another agent to a sweep

Either budget flag also ends the sweep: --walltime caps the whole sweep and is
checked between trials, --trialtime caps each trial and is enforced by stopping
main.py. Both take fractional hours.

The agent launches each trial as a fresh subprocess, which is what keeps runs
isolated: Logger._setup_stdout_logging replaces sys.stdout without restoring
it, so several trials in one interpreter would nest the redirects and leak file
handles.

The agent exports WANDB_SWEEP_ID and WANDB_RUN_ID into that subprocess and
main.py inherits them, so the wandb.init() already inside Logger attaches to the
right sweep run by itself. Nothing here calls wandb.init.

Everything a sweep produces is grouped under one directory:

  saved/sweep-<sweep id>/
    sweep.yml              copy of the spec this sweep was created from
    base.yml               copy of the base config it starts every trial from
    <wandb run id>/        one per trial, laid out like any other run
      cfg.yml                this trial's resolved config
      stdout.txt
      0000/ 0001/ ...        metrics.json and checkpoints per epoch
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Seconds a timed-out trial is given to unwind and let wandb flush before it is
# killed outright.
_GRACE_SECONDS = 60
# How the agent hands --trialtime down to the trials it spawns. The sweep's
# command is fixed when the sweep is created, so it cannot travel as an argument.
_TRIAL_SECONDS_VAR = "RIPE_TRIAL_SECONDS"

def set_nested_value(cfg, path, value):
  """Set nested value using dot notation (e.g., 'run.optimizer.lr')."""
  keys = path.split('.')
  current = cfg
  for key in keys[:-1]:
    if key not in current:
      current[key] = {}
    current = current[key]
  current[keys[-1]] = value

def build_sweep_config(spec, sweep_cfg_path):
  """Translate our sweep spec into the payload wandb.sweep expects.

  ``program`` and ``command`` are injected rather than hand-written: a trial has
  to re-invoke this script with --trial plus the sampled values, and
  ${args_no_hyphens} is what appends them as bare key=value pairs.
  """
  config = {
      "name": spec["name"],
      "method": spec["method"],
      "metric": spec["metric"],
      "parameters": spec["parameters"],
      "program": "sweep.py",
      "command": ["${env}", sys.executable, "sweep.py",
                  "--cfg", str(sweep_cfg_path), "--trial", "${args_no_hyphens}"],
  }
  for optional in ("early_terminate", "run_cap", "description"):
    if optional in spec:
      config[optional] = spec[optional]
  return config


def sweep_dir(spec):
  """``saved/sweep-<id>`` — where this sweep's runs and its spec copy live.

  The agent exports WANDB_SWEEP_ID, so a trial can find its own sweep without
  being told. A manual --trial invocation has no sweep, so it falls back to the
  spec name and still groups rather than scattering into saved/.
  """
  return Path("saved") / f"sweep-{os.environ.get('WANDB_SWEEP_ID') or spec['name']}"


def write_trial_config(spec, overrides):
  """Write the base config with this trial's values applied; return its path."""
  with open(spec["base_config"]) as f:
    cfg = yaml.safe_load(f)

  for item in overrides:
    path, separator, raw = item.lstrip("-").partition("=")
    if not separator:
      raise ValueError(f"Expected key=value from the sweep agent, got {item!r}")
    # The agent hands everything over as text; let YAML recover the real type.
    set_nested_value(cfg, path, yaml.safe_load(raw))

  # The trial has to report to wandb or the sweep has no metric to optimise, and
  # it has to land in the same project the sweep itself lives in.
  cfg["log"]["wandb"]["enabled"] = True
  cfg["log"]["wandb"]["entity"] = spec.get("entity", "")
  cfg["log"]["wandb"]["project"] = spec.get("project", "")

  # Config derives savedir as saved/<name>, so a name with a separator in it is
  # what nests this run under its sweep: saved/sweep-<id>/<run id>. The run id is
  # the agent's, so the directory and the wandb run line up.
  run_id = os.environ.get("WANDB_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  directory = sweep_dir(spec)
  cfg["name"] = f"{directory.name}/{run_id}"

  # Write the trial config into the run's own directory rather than off in
  # setup/cfg: Logger._save_cfg later rewrites this same path with the resolved
  # config (seed and savedir filled in), so the run keeps exactly one config.
  out_dir = directory / run_id
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / "cfg.yml"
  with open(out_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
  return out_path


def run_trial(trial_cfg, budget):
  """Run main.py on ``trial_cfg``, stopping it after ``budget`` seconds.

  Returns main.py's exit code, or 0 if the trial was stopped on its budget: a
  nonzero exit reads as a failed run to the agent, and agents give up after
  repeated early failures, whereas a trial that used its whole budget did what
  it was asked to.

  SIGINT rather than SIGTERM, because it raises KeyboardInterrupt and lets the
  interpreter unwind through atexit. Nothing in this repo calls wandb.finish(),
  so wandb's own atexit hook is the only thing that finalises the run; killed
  outright it would be left dangling. SIGKILL is only the backstop.
  """
  proc = subprocess.Popen([sys.executable, "main.py", "--cfg", str(trial_cfg)])
  if budget is None:
    return proc.wait()

  try:
    return proc.wait(timeout=budget)
  except subprocess.TimeoutExpired:
    print(f"\n[sweep] trial hit its {budget / 3600:.3g}h budget; stopping it", flush=True)
    proc.send_signal(signal.SIGINT)
  try:
    proc.wait(timeout=_GRACE_SECONDS)
  except subprocess.TimeoutExpired:
    print(f"[sweep] trial ignored SIGINT for {_GRACE_SECONDS}s; killing it", flush=True)
    proc.kill()
    proc.wait()
  return 0


def main():
  parser = argparse.ArgumentParser(description="W&B sweep driver")
  parser.add_argument("--cfg", type=str, required=True, help="Path to sweep YAML")
  parser.add_argument("--trial", action="store_true",
                      help="Run one trial; this is the form the wandb agent invokes")
  parser.add_argument("--count", type=int, default=None,
                      help="Run an agent for this many trials")
  parser.add_argument("--id", type=str, default=None,
                      help="Attach to an existing sweep id instead of creating one")
  parser.add_argument("--finish", action="store_true",
                      help="Mark the sweep finished once this agent's budget is "
                           "spent, instead of leaving it open for more agents")
  parser.add_argument("--walltime", type=float, default=None,
                      help="Stop the sweep after this many hours. Checked between "
                           "trials, so the last trial may overrun by its own length")
  parser.add_argument("--trialtime", type=float, default=None,
                      help="Stop each trial after this many hours. Metrics for "
                           "every completed epoch survive; only the partial epoch "
                           "is lost. Checkpoints follow log.checkpoint.every "
                           "(5 by default), so lower it to keep more state")
  args, overrides = parser.parse_known_args()

  with open(args.cfg) as f:
    spec = yaml.safe_load(f)

  if args.trial:
    # An explicit --trialtime wins, so a manual trial can be time-boxed; the env
    # var is how an agent passes its own --trialtime down.
    budget = args.trialtime * 3600 if args.trialtime else None
    if budget is None and os.environ.get(_TRIAL_SECONDS_VAR):
      budget = float(os.environ[_TRIAL_SECONDS_VAR])
    trial_cfg = write_trial_config(spec, overrides)
    print(f"Trial config: {trial_cfg}", flush=True)
    sys.exit(run_trial(trial_cfg, budget))

  # A floor set before the first wandb call, so nothing lands at the repo root
  # even in the window before the sweep id exists and the precise directory
  # below can be named. wandb appends "wandb/" to this itself.
  os.environ.setdefault("WANDB_DIR", "saved")

  import wandb

  sweep_id = args.id
  if sweep_id is None:
    sweep_id = wandb.sweep(build_sweep_config(spec, args.cfg),
                           entity=spec.get("entity") or None,
                           project=spec.get("project") or None)
    print(f"\nCreated sweep: {sweep_id}")

  # Keep a copy of the spec and the base config beside the runs, so the sweep
  # directory is self-describing even if either file is edited afterwards.
  # WANDB_SWEEP_ID is what the trials use to find this same directory.
  os.environ["WANDB_SWEEP_ID"] = sweep_id
  directory = sweep_dir(spec)
  directory.mkdir(parents=True, exist_ok=True)
  # wandb.sweep/agent take no dir argument; the agent builds its per-trial
  # handoff path as os.path.join(WANDB_DIR, "wandb", f"sweep-{id}", ...)
  # (pyagent.py:328), so this is what keeps it out of ./wandb at the repo root.
  # Trials are unaffected: Logger passes wandb.init an explicit dir, which wins.
  os.environ["WANDB_DIR"] = str(directory)
  shutil.copy(args.cfg, directory / "sweep.yml")
  shutil.copy(spec["base_config"], directory / "base.yml")
  print(f"Sweep directory: {directory}")

  if args.count is None and args.walltime is None:
    print(f"Run agents with:\n"
          f"  python sweep.py --cfg {args.cfg} --id {sweep_id} --count <n>")
    return

  if args.trialtime:
    # wandb.agent spawns the trial command inheriting os.environ, so this is how
    # the per-trial budget reaches a trial that cannot take it as an argument.
    os.environ[_TRIAL_SECONDS_VAR] = str(args.trialtime * 3600)

  # One trial per wandb.agent call so the deadline is re-checked between trials.
  # monotonic, not time(), so a clock adjustment cannot skew the budget.
  deadline = time.monotonic() + args.walltime * 3600 if args.walltime else None
  completed = 0
  while True:
    if args.count is not None and completed >= args.count:
      print(f"[sweep] ran {completed} trials; stopping", flush=True)
      break
    if deadline is not None and time.monotonic() >= deadline:
      print(f"[sweep] {args.walltime:.3g}h budget spent after {completed} trials; "
            f"stopping", flush=True)
      break
    if deadline is not None:
      print(f"[sweep] trial {completed + 1}, "
            f"{(deadline - time.monotonic()) / 3600:.2f}h left", flush=True)
    wandb.agent(sweep_id, count=1,
                entity=spec.get("entity") or None,
                project=spec.get("project") or None)
    completed += 1

  # A sweep's server-side state is not tied to its agents: --count and the
  # budgets bound *this* agent, so the sweep stays RUNNING, waiting for more
  # agents, until it is stopped explicitly or run_cap is reached.
  qualified = "/".join(p for p in (spec.get("entity"), spec.get("project"), sweep_id) if p)
  if args.finish:
    print(f"[sweep] marking sweep finished", flush=True)
    subprocess.run([sys.executable, "-m", "wandb", "sweep", "--stop", qualified], check=False)
  else:
    print(f"\nSweep is still accepting agents. To close it:\n"
          f"  wandb sweep --stop {qualified}        # let active runs finish\n"
          f"  wandb sweep --cancel {qualified}      # kill active runs too\n"
          f"Or set run_cap in the spec to have it close itself.")


if __name__ == "__main__":
  main()
