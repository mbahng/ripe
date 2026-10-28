import os
import time
import threading
import numpy as np
import psutil

try:
  import pynvml # type: ignore
except ImportError:
  pynvml = None

# thresholds for the suggestions in Profiler.report
WAIT_HI = 0.30   # fraction of step time spent waiting on the dataloader
GPU_LO = 50.0    # mean gpu utilization (%) below which the gpu counts as starved
GPU_HI = 80.0
CPU_HI = 85.0
RAM_HI = 90.0


class _TimedLoader:
  """
  Transparent proxy around a DataLoader that times every fetch.
  Time blocked in next() is the data wait. The gap between receiving one batch and asking
  for the next is the compute time of the training step.
  """

  def __init__(self, dl, name, prof):
    self.dl, self.name, self.prof = dl, name, prof

  def __iter__(self):
    it = iter(self.dl)
    first, last_end = True, None
    while True:
      t0 = time.perf_counter()
      if last_end is not None: self.prof._add(self.name, "compute", t0 - last_end)
      try: batch = next(it)
      except StopIteration: return
      t1 = time.perf_counter()
      # the first fetch of an iteration pays the worker startup cost, so track it separately
      self.prof._add(self.name, "startup" if first else "wait", t1 - t0)
      first, last_end = False, t1
      yield batch

  def __len__(self):
    return len(self.dl)

  def __getattr__(self, attr):
    # .dataset, .batch_size, .num_workers, ...
    return getattr(self.dl, attr)


class Profiler:
  """
  Samples system utilization in a background thread and collects data wait / compute times
  from the wrapped dataloaders. Call start() then report() once training is finished.
  """

  def __init__(self, device=None, interval=0.5):
    self.device, self.interval = device, interval
    self.times = {}   # split -> {"wait": [], "compute": [], "startup": []}
    self.cpu, self.cpu_core_max, self.ram = [], [], []
    self.gpu_util, self.gpu_mem = [], []
    self.loaders = {}
    self._stop = threading.Event()
    self._thread = None
    self._lock = threading.Lock()
    self._nvml = self._init_nvml()
    self.start_time = self.end_time = None

  def _init_nvml(self):
    if pynvml is None or self.device is None: return None
    try:
      pynvml.nvmlInit()
      # torch device indices are relative to CUDA_VISIBLE_DEVICES, nvml's are physical
      visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
      idx = int(visible[self.device]) if self.device < len(visible) and visible[self.device].strip().isdigit() else self.device
      return pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception:
      return None

  def wrap(self, dl, name):
    self.loaders[name] = dl
    return _TimedLoader(dl, name, self)

  def _add(self, name, kind, t):
    with self._lock:
      self.times.setdefault(name, {"wait": [], "compute": [], "startup": []})[kind].append(t)

  def _sample(self):
    psutil.cpu_percent(percpu=True)  # prime, the first call always returns 0
    while not self._stop.wait(self.interval):
      per_core = psutil.cpu_percent(percpu=True)
      self.cpu.append(float(np.mean(per_core)))
      self.cpu_core_max.append(float(np.max(per_core)))
      self.ram.append(psutil.virtual_memory().percent)
      if self._nvml is not None and pynvml is not None:
        try:
          util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml)
          mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml)
          self.gpu_util.append(float(util.gpu))
          self.gpu_mem.append(100.0 * mem.used / mem.total)
        except Exception:
          pass

  def start(self):
    self.start_time = time.time()
    self._thread = threading.Thread(target=self._sample, daemon=True)
    self._thread.start()

  def stop(self):
    self._stop.set()
    if self._thread is not None: self._thread.join()
    self.end_time = time.time()

  @staticmethod
  def _stats(xs):
    return (float(np.mean(xs)), float(np.percentile(xs, 95)), float(np.max(xs))) if len(xs) else None

  def report(self):
    """Print a summary table followed by suggestions."""
    out = []
    elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
    out.append("=" * 70)
    out.append(f"PROFILE ({elapsed:.0f}s, {len(self.cpu)} system samples @ {self.interval}s, {len(os.sched_getaffinity(0))} cpu cores)")
    out.append("=" * 70)
    out.append(f"{'split':<7}{'steps':>8}{'wait/step':>12}{'compute/step':>14}{'wait frac':>11}{'startup':>10}")
    total_wait = total_compute = 0.0
    for name, t in self.times.items():
      wait, compute, startup = sum(t["wait"]), sum(t["compute"]), sum(t["startup"])
      steps = len(t["wait"])
      total_wait += wait
      total_compute += compute
      frac = wait / (wait + compute) if wait + compute else 0.0
      out.append(f"{name:<7}{steps:>8}{wait / max(steps, 1):>11.4f}s{compute / max(len(t['compute']), 1):>13.4f}s{frac:>11.2f}{startup:>9.1f}s")

    cpu, core, ram = self._stats(self.cpu), self._stats(self.cpu_core_max), self._stats(self.ram)
    gpu, gmem = self._stats(self.gpu_util), self._stats(self.gpu_mem)
    out.append("")
    out.append(f"{'':<14}{'mean':>8}{'p95':>8}{'max':>8}")
    for label, s in (("cpu %", cpu), ("busiest core %", core), ("ram %", ram), ("gpu util %", gpu), ("gpu mem %", gmem)):
      if s: out.append(f"{label:<14}{s[0]:>8.1f}{s[1]:>8.1f}{s[2]:>8.1f}")
      else: out.append(f"{label:<14}{'n/a':>8}")

    wait_frac = total_wait / (total_wait + total_compute) if total_wait + total_compute else 0.0
    out.append("")
    out.append("SUGGESTIONS")
    tips = self._suggestions(wait_frac, cpu, core, ram, gpu, gmem)
    out.extend(f"  - {tip}" for tip in tips)
    print("\n".join(out))

  def _suggestions(self, wait_frac, cpu, core, ram, gpu, gmem):
    tips = []
    dl = self.loaders.get("train") or next(iter(self.loaders.values()), None)
    workers = getattr(dl, "num_workers", None)
    cores = len(os.sched_getaffinity(0))
    gpu_low = gpu is not None and gpu[0] < GPU_LO

    if wait_frac > WAIT_HI:
      tips.append(f"Data-loader bound: {wait_frac:.0%} of step time is spent waiting for batches"
                  + (f" (GPU util mean {gpu[0]:.0f}%)." if gpu else "."))
      tips.append(f"Raise num_workers (currently {workers}, machine has {cores} cores), "
                  "and make sure workers are not oversubscribed by torch/cv2/numpy threads (set OMP_NUM_THREADS=1).")
      if dl is not None:
        if not getattr(dl, "pin_memory", False):
          tips.append("pin_memory is off on the dataloader (init_dataloader currently does not pass it from the config).")
        if workers and not getattr(dl, "persistent_workers", False):
          tips.append("persistent_workers is off, so workers are respawned every epoch. Turn it on and consider prefetch_factor > 2.")
      tips.append("Make the per-sample transforms cheaper (do resizing/decoding once offline, cache the dataset in RAM, or move augmentations to the GPU).")
    elif cpu and (cpu[0] > CPU_HI or core[2] > 98) and gpu_low:
      tips.append(f"CPU is busy (mean {cpu[0]:.0f}%) and the GPU is starved (mean {gpu[0]:.0f}%), but data wait is only {wait_frac:.0%}. "
                  "The bottleneck is CPU work inside the training step (per-step .item() syncs, metrics, python overhead) "
                  "or the main process itself being saturated. Check for host-side ops and reduce per-step logging.")
    elif gpu and gpu[0] > GPU_HI:
      tips.append(f"GPU bound (util mean {gpu[0]:.0f}%), the input pipeline is keeping up. Consider mixed precision (torch.autocast) or channels_last.")

    if gpu is not None and gpu_low and gmem is not None and gmem[2] < 50:
      tips.append(f"GPU memory peaked at {gmem[2]:.0f}% and utilization is low, so a larger batch_size will likely improve throughput.")
    if ram and ram[2] > RAM_HI:
      tips.append(f"RAM peaked at {ram[2]:.0f}%. Reduce num_workers/prefetch_factor or avoid holding the dataset in every worker.")
    if gpu is None:
      tips.append("No GPU stats collected (CPU run, device unset, or nvidia-ml-py is not installed), so GPU-based advice is limited.")
    if not tips:
      tips.append("No obvious bottleneck detected.")
    startup = sum(sum(t["startup"]) for t in self.times.values())
    if startup > 0.05 * max(total_time(self.times), 1e-9):
      tips.append(f"{startup:.0f}s went to dataloader worker startup (first fetch of each epoch). persistent_workers=True removes it.")
    return tips


def total_time(times):
  return sum(sum(t["wait"]) + sum(t["compute"]) + sum(t["startup"]) for t in times.values())
