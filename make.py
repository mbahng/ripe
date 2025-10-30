"""
Generates trial config files and list of commands to run a study (a set of trials)
"""
import argparse
import copy
import itertools
from pathlib import Path
import yaml


def set_nested_value(cfg, path, value):
  """Set nested value using dot notation (e.g., 'run.optimizer.lr')."""
  keys = path.split('.')
  current = cfg
  for key in keys[:-1]:
    if key not in current:
      current[key] = {}
    current = current[key]
  current[keys[-1]] = value


def generate_grid_combinations(search_space):
  """Generate all combinations for grid search."""
  param_names = list(search_space.keys())
  param_values = [search_space[name] for name in param_names]

  combinations = []
  for values in itertools.product(*param_values):
    combo = dict(zip(param_names, values))
    combinations.append(combo)

  return combinations


def make_run_name(base_name, combination):
  """Create run name from hyperparameters."""
  parts = [base_name]
  for param_path, value in combination.items():
    param_short = param_path.split('.')[-1]  # Use last part of path
    parts.append(f"{param_short}={value}")
  return "_".join(parts)


def main():
  parser = argparse.ArgumentParser(description='Grid Search Config Generator')
  parser.add_argument('--cfg', type=str, required=True,
                      help='Path to study configuration YAML')
  args = parser.parse_args()

  # Load study config
  with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)

  trial_cfg_path = cfg['trial_config']
  search_space = cfg['search_space']
  study_name = cfg['name']

  # Load trial config
  with open(trial_cfg_path, 'r') as f:
    trial_cfg = yaml.safe_load(f)

  # Output directory
  output_dir = Path('setup/cfg') / study_name

  # Check if study directory already exists
  if output_dir.exists():
    raise Exception(f"Study directory already exists: {output_dir}")

  # Generate combinations
  combinations = generate_grid_combinations(search_space)
  print(f"Generating {len(combinations)} configurations...\n")

  output_dir.mkdir(parents=True, exist_ok=True)

  commands = []

  # Generate configs
  for combo in combinations:
    cfg = copy.deepcopy(trial_cfg)

    # Apply hyperparameter overrides
    for param_path, value in combo.items():
      set_nested_value(cfg, param_path, value)

    # Set run name
    run_name = make_run_name(study_name, combo)
    cfg['name'] = run_name

    # Save config
    cfg_filename = f"{run_name}.yml"
    cfg_path = output_dir / cfg_filename
    with open(cfg_path, 'w') as f:
      yaml.dump(cfg, f, default_flow_style=False)

    # Build command
    commands.append(f"python main.py --cfg={cfg_path}")

  print("Generated configs:")
  for cmd in commands:
    print(cmd)


if __name__ == '__main__':
  main()
