import torch

# Loss-like names are minimized, everything else maximized. Only consulted when the
# config does not say, so a surprising metric name can always be settled explicitly.
_MINIMIZE = ("loss", "cost", "entropy", "error", "nll", "cluster", "separation",
             "l1", "variability", "orthogonality")


def init_scheduler(phase_cfg: dict, optimizer):
  """Build the learning rate scheduler one optimizer phase asks for.

  Configured under that phase's ``scheduler`` block; absent, or name ``none``,
  returns (None, None) and the phase trains at a fixed rate::

      optimizer:
        joint:
          scheduler:
            name: plateau             # plateau | step | none
            metric: val.balanced_accuracy
            factor: 0.5               # multiply the lr by this on each decay
            patience: 10              # non-improving epochs before decaying
            threshold: 0.01           # smaller than this is not an improvement
            threshold_mode: abs       # abs | rel
            cooldown: 10              # epochs to wait after a decay before counting again
            min_lr: 1.0e-8

  Returns (scheduler, metric). A plateau scheduler needs the metric's value at
  every step; a step scheduler ignores it, so metric comes back None and the
  caller steps without one.

  Why plateau rather than step by default: step decay is not invariant to the
  length of the run. The schedule this repo used to have -- halve every 10 joint
  epochs -- is 12 halvings across 120 joint epochs but 25 across 250, which drives
  the rate into float noise two thirds of the way through. Plateau adapts to the
  curve instead, so the same block stays correct when the epoch budget changes.

  The thresholds matter more than they look. ProtoPNet's validation accuracy on
  BIOSCAN-1M oscillates with a standard deviation near 0.01 and a 6 point
  peak-to-peak band once it is near convergence, and torch defaults to
  threshold 1e-4 in 'rel' mode -- which that noise clears constantly, so the
  scheduler would effectively never fire. An absolute threshold set above the
  noise, with a cooldown so one decay is given time to take effect, is what makes
  it track the real trend.
  """
  cfg = (phase_cfg or {}).get("scheduler") or {}
  name = str(cfg.get("name", "none")).lower()
  if name in ("", "none"):
    return None, None

  if name == "plateau":
    metric = str(cfg.get("metric", "val.balanced_accuracy"))
    goal = cfg.get("goal")
    if goal is None:
      leaf = metric.rsplit(".", 1)[-1].lower()
      goal = "min" if any(token in leaf for token in _MINIMIZE) else "max"
    if goal not in ("max", "min"):
      raise ValueError(f"scheduler.goal must be 'max' or 'min', got {goal!r}")
    threshold_mode = str(cfg.get("threshold_mode", "abs"))
    if threshold_mode not in ("abs", "rel"):
      raise ValueError(f"scheduler.threshold_mode must be 'abs' or 'rel', got {threshold_mode!r}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=goal,
        factor=float(cfg.get("factor", 0.5)),
        patience=int(cfg.get("patience", 10)),
        threshold=float(cfg.get("threshold", 0.01)),
        threshold_mode="abs" if threshold_mode == "abs" else "rel",
        cooldown=int(cfg.get("cooldown", 0)),
        min_lr=float(cfg.get("min_lr", 0.0)))
    return scheduler, metric

  if name == "step":
    if "step_size" not in cfg:
      raise ValueError("scheduler.name 'step' requires scheduler.step_size")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["step_size"]),
        gamma=float(cfg.get("gamma", 0.5)))
    return scheduler, None

  raise ValueError(f"Unknown scheduler.name {name!r}; expected plateau, step or none")
