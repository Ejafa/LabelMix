#!/usr/bin/env python3
"""Persistent job daemon for GPU training orchestration.

Replaces Ray Tune with a simple, transparent YAML-based job queue.
Runs in a tmux session, watches an inbox directory for new jobs,
assigns GPUs, monitors processes, and auto-retries on failure.

Usage::

    # Start daemon in tmux (schedule + node-isolated context)
    tmux new -s daemon
    python jobdaemon.py -s imagenet_sweep --node-index 0 start --gpus 0,1,2,3,4,5,6,7

    # Exclusive placement (never stack jobs on the same GPU group)
    python jobdaemon.py -s imagenet_sweep --node-index 0 start \
        --gpus 0,1,2,3,4,5,6,7 --no-overcommit

    # Submit jobs (must use the same -s + --node-index context as start)
    python jobdaemon.py -s imagenet_sweep --node-index 0 submit imagenet_sweep_node_0_jobs.yaml

    # Check status
    python jobdaemon.py -s imagenet_sweep --node-index 0 status
    python jobdaemon.py -s imagenet_sweep --node-index 0 status --watch

    # Manage
    python jobdaemon.py -s imagenet_sweep --node-index 0 cancel <job_name>
    python jobdaemon.py -s imagenet_sweep --node-index 0 pause
    python jobdaemon.py -s imagenet_sweep --node-index 0 resume
    python jobdaemon.py -s imagenet_sweep --node-index 0 retry --all-failed
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_POLL_INTERVAL = 5
DEFAULT_PORT_RANGE = (29500, 29599)
DEFAULT_MAX_RETRIES = 3
DEFAULT_LOG_DIR = "./logs/jobs"
DEFAULT_STATE_DIR = "./state"
DEFAULT_INBOX_DIR = "./inbox"
SCHEDULES_ROOT = "./schedules"

DEFAULT_SATURATION_COOLDOWN = 30  # seconds to wait after OOM before trying next launch
DEFAULT_OOM_CRASH_THRESHOLD = 120  # if a job dies within this many seconds, treat as OOM
DEFAULT_BURST_DELAY = 15 # seconds between consecutive burst launches
DEFAULT_FIFO_LAUNCH_COOLDOWN = 240  # seconds between FIFO VRAM polls / launch attempts after burst
DEFAULT_GPU_CHECK_DELAY = 30 # seconds grace period before checking GPU presence
DEFAULT_BURST_GPU_WAIT = 300 # seconds to wait for a burst-launched job to appear on GPU before launching next
DEFAULT_STABILIZATION_WAIT = 300 # base seconds (5 min) for saturation checkpoint; actual wait = checkpoint_number × this value
DEFAULT_MEMORY_SAFETY_MARGIN = 0.0  # 15% safety margin — predict OOM if free memory < avg_per_job × (1 + margin)
DEFAULT_OOM_WAIT_TIMEOUT = 600  # 10 minutes — max time to wait for memory headroom before queueing
DEFAULT_SENTINEL_TIMEOUT = 600  # 10 minutes — max time to wait for .training_started sentinel before fallback

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """Represents a single training job in the queue."""
    name: str
    cmd: str
    cmd_template: Optional[str] = None  # original unresolved cmd with {gpus}/{port} placeholders
    status: str = "pending"  # pending | running | completed | failed | cancelled
    pid: Optional[int] = None
    pgid: Optional[int] = None
    gpu_ids: Optional[List[int]] = None
    master_port: Optional[int] = None
    retries: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    gpus_needed: int = 1
    exit_code: Optional[int] = None
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    log_file: Optional[str] = None
    error_snippet: Optional[str] = None
    working_dir: Optional[str] = None
    memory_at_stable: Optional[float] = None  # MiB per GPU when job stabilized on GPU
    oom_requeue_count: int = 0  # number of times this job was requeued due to OOM
    launched_at_burst_index: Optional[int] = None  # burst_jobs_launched when this job was launched
    model_key: Optional[str] = None  # model key propagated by scheduler for diagnostics
    estimated_memory_mib_per_gpu: Optional[float] = None  # scheduler-profiled expected memory per GPU

    # Runtime-only (not persisted)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)
    _log_fh: Optional[Any] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        """Serialize for YAML (exclude runtime fields)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        # Filter out unknown keys
        valid = {f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class DaemonConfig:
    """Daemon-level configuration persisted in state.yaml."""
    started_at: Optional[str] = None
    pid: Optional[int] = None
    gpus: List[int] = field(default_factory=list)
    max_concurrent: int = 1
    fixed_gpus_needed: Optional[int] = None  # enforce one GPUs-per-job shape per daemon
    master_port_range: List[int] = field(default_factory=lambda: list(DEFAULT_PORT_RANGE))
    paused: bool = False
    overcommit: bool = True  # allow multiple jobs to share a GPU group
    saturated: bool = False  # True when machine is at capacity (OOM detected)
    saturation_time: Optional[str] = None  # when saturation was detected
    burst_phase: bool = True  # True = burst launching, False = FIFO timed VRAM polling
    burst_jobs_launched: int = 0  # how many jobs launched in current burst
    saturation_checkpoints_passed: int = 0  # how many saturation checkpoints completed (for progressive wait)
    overcommit_rr_index: int = 0  # round-robin index for GPU pinning in overcommit mode
    fill_gpu_index: int = 0  # index into gpus list: the GPU group we are currently probing/filling
    round_robin: bool = True  # round-robin GPU placement: advance fill_gpu_index after every launch in burst phase (spreads jobs evenly across GPUs instead of stacking on one GPU until OOM)
    prev_running: int = 0  # legacy FIFO state retained for backward-compatible persistence
    peak_memory_per_job: Optional[float] = None  # MiB: best estimate of memory per job from stabilized jobs
    last_checkpoint_burst_index: int = 0  # burst_jobs_launched at last successful saturation checkpoint
    last_launched_job_name: Optional[str] = None  # name of the last job launched (waiting for its sentinel)
    last_fifo_launch_time: Optional[str] = None  # timestamp of last successful FIFO launch
    last_fifo_check_time: Optional[str] = None  # timestamp of last FIFO VRAM poll / launch attempt


@dataclass
class State:
    """Full daemon state — persisted atomically to state.yaml."""
    state_dir: str = DEFAULT_STATE_DIR
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    jobs: List[Job] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "daemon": self.daemon.__dict__,
            "jobs": [j.to_dict() for j in self.jobs],
        }

    @classmethod
    def from_dict(cls, data: dict, state_dir: str = DEFAULT_STATE_DIR) -> "State":
        st = cls(state_dir=state_dir)
        if "daemon" in data and data["daemon"]:
            st.daemon = DaemonConfig(**data["daemon"])
        if "jobs" in data and data["jobs"]:
            st.jobs = [Job.from_dict(j) for j in data["jobs"]]
        return st


# ---------------------------------------------------------------------------
# State persistence (atomic writes)
# ---------------------------------------------------------------------------

def _state_path(state_dir: str) -> str:
    return os.path.join(state_dir, "state.yaml")


def save_state(state: State) -> None:
    """Atomically save state to YAML (write-tmp-then-rename)."""
    os.makedirs(state.state_dir, exist_ok=True)
    path = _state_path(state.state_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(state.to_dict(), f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def load_state(state_dir: str) -> Optional[State]:
    """Load state from YAML. Returns None if file doesn't exist."""
    path = _state_path(state_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not data:
        return None
    return State.from_dict(data, state_dir=state_dir)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def is_process_alive(pid: Optional[int]) -> bool:
    """Check if a process is still running using psutil.

    Also verifies the process is actually a training process (not a recycled PID)
    by checking if the cmdline contains 'torchrun' or 'train.py'.
    """
    if pid is None:
        return False
    if HAS_PSUTIL:
        try:
            proc = psutil.Process(pid)
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return False
            cmdline = " ".join(proc.cmdline())
            if "torchrun" not in cmdline and "train.py" not in cmdline:
                return False
            return True
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except psutil.AccessDenied:
            return True  # exists but we can't inspect it
    else:
        # Fallback: os.kill(pid, 0) + /proc/cmdline
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace")
                if "torchrun" not in cmdline and "train.py" not in cmdline:
                    return False
        except (FileNotFoundError, PermissionError):
            return False
        return True


def is_port_free(port: int) -> bool:
    """Check if a TCP port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def allocate_port(state: State) -> int:
    """Find an unused master port for torchrun."""
    used = {j.master_port for j in state.jobs if j.status == "running" and j.master_port}
    lo, hi = state.daemon.master_port_range
    for port in range(lo, hi + 1):
        if port not in used and is_port_free(port):
            return port
    raise RuntimeError(f"No free ports in range {lo}-{hi}")


def _is_object_detection_cmd(cmd: str) -> bool:
    """Heuristic: does this command launch a detectron2 / ViTDet (object
    detection) training run?

    Object-detection jobs are launched via ``train_net.py --config-file …``
    with a config living under ``detectron2_vitdet/…`` and use Hydra-style
    ``key=value`` overrides that may legitimately contain ``=`` inside the
    value (e.g. checkpoint paths like ``…/baseline_seed=42.pth``).

    Classification jobs (ImageNet-1k, CIFAR-100, …) are launched via
    ``torchrun … train.py --config …`` in timm style and must NOT have
    their ``--experiment foo__seed=44`` tokens rewritten — otherwise the
    experiment name ends up wrapped in literal double quotes.
    """
    lowered = cmd.lower()
    if "train_net.py" in lowered and "--config-file" in lowered:
        return True
    if "detectron2_vitdet" in lowered:
        return True
    return False


def resolve_placeholders(cmd: str, gpu_ids: List[int], port: int, job: Job) -> str:
    """Replace {gpus}, {port}, {gpu_ids}, {job_name}, {log_file} placeholders.

    For object-detection jobs (see :func:`_is_object_detection_cmd`) we
    additionally wrap Hydra-style ``key=value`` overrides whose VALUE
    itself contains an un-quoted ``=`` in double quotes.  Hydra's override
    parser rejects such tokens with
        OverrideParseException: mismatched input '=' expecting <EOF>
    which happens for checkpoint paths like
    ``…/baseline_seed=42__runid-xyz.pth``.

    For classification jobs (ImageNet-1k, CIFAR-100, …) this defensive
    quoting is DISABLED, because their CLIs do not use Hydra overrides
    and the extra quotes would corrupt values such as
    ``--experiment vit-betwixt__in1k__img256__seed=44``.
    """
    resolved = (
        cmd.replace("{gpus}", str(len(gpu_ids)))
        .replace("{port}", str(port))
        .replace("{gpu_ids}", ",".join(str(g) for g in gpu_ids))
        .replace("{job_name}", job.name)
        .replace("{log_file}", job.log_file or "")
    )

    # Only object-detection (detectron2 / ViTDet) commands need the
    # Hydra-override quoting fix-up.  For everything else, return the
    # resolved command verbatim so we don't accidentally mangle tokens
    # like ``--experiment foo__seed=44``.
    if not _is_object_detection_cmd(resolved):
        return resolved

    try:
        tokens = shlex.split(resolved)
    except ValueError:
        return resolved  # let downstream surface the syntax error

    def _fix(tok: str) -> str:
        if "=" not in tok or tok.startswith("--"):
            return tok
        key, _, value = tok.partition("=")
        if "=" not in value:
            return tok
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return tok
        escaped = value.replace('"', r"\"")
        return f'{key}="{escaped}"'

    fixed = [_fix(t) for t in tokens]
    return " ".join(shlex.quote(t) for t in fixed)


def _compute_cpu_affinity(gpu_ids: List[int], total_gpus: int = 8) -> Optional[str]:
    """Compute a taskset CPU core range for the given GPU IDs.

    Splits the available CPU cores evenly across GPUs (NUMA-aware heuristic).
    For a 192-thread / 8-GPU system, each GPU gets 24 cores:
      GPU 0 → 0-23, GPU 1 → 24-47, ..., GPU 7 → 168-191.

    Returns a comma-separated core list like "0-23,48-71" for taskset -c,
    or None if taskset is unavailable.
    """
    total_cores = os.cpu_count() or 192
    cores_per_gpu = total_cores // total_gpus
    if cores_per_gpu < 1:
        return None

    # Check that taskset exists
    if shutil.which("taskset") is None:
        return None

    ranges = []
    for gid in sorted(gpu_ids):
        start = (gid % total_gpus) * cores_per_gpu
        end = start + cores_per_gpu - 1
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def launch_job(job: Job, gpu_ids: List[int], port: int, log_dir: str, working_dir: str) -> None:
    """Launch a training job as a subprocess with its own process group."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["HF_DATASETS_OFFLINE"] = "1"

    # ---- Weights & Biases: always-on by default ------------------------
    # train_net.py already instantiates a WandbWriter and calls
    # init_wandb_from_cfg; here we just make sure the subprocess env does
    # not accidentally disable it (e.g. inherited WANDB_MODE=disabled from
    # an offline login shell). Operators can still opt out explicitly by
    # exporting WANDB_MODE=disabled|offline before launching the daemon.
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("WANDB_PROJECT", "labelmix-vitdet")
    # Make crashes flush data (best-effort; _wandb_writer has an atexit hook too).
    env.setdefault("WANDB_START_METHOD", "thread")
    # Silence the interactive login prompt — in a subprocess it would just hang.
    env.setdefault("WANDB_SILENT", "true")

    log_file = os.path.join(log_dir, f"{job.name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    job.log_file = log_file

    # Preserve the original template so retries can re-resolve placeholders
    if job.cmd_template is None:
        job.cmd_template = job.cmd
    resolved_cmd = resolve_placeholders(job.cmd_template, gpu_ids, port, job)
    job.cmd = resolved_cmd  # Store last-resolved cmd for display/debugging

    fh = open(log_file, "a")
    fh.write(f"\n{'=' * 60}\n")
    fh.write(f"=== Attempt {job.retries + 1} at {datetime.now().isoformat()} ===\n")
    fh.write(f"=== CMD: {resolved_cmd}\n")
    # CPU affinity: pin this job's processes to cores aligned with its GPUs
    cpu_affinity = _compute_cpu_affinity(gpu_ids)
    if cpu_affinity:
        resolved_cmd = f"taskset -c {cpu_affinity} {resolved_cmd}"

    fh.write(f"=== GPUs: {gpu_ids} (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})\n")
    fh.write(f"=== Port: {port}\n")
    if cpu_affinity:
        fh.write(f"=== CPU affinity: {cpu_affinity}\n")
    fh.write(f"{'=' * 60}\n\n")
    fh.flush()

    proc = subprocess.Popen(
        shlex.split(resolved_cmd),
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=working_dir,
        start_new_session=True,  # new process group
    )

    job._proc = proc
    job._log_fh = fh
    job.pid = proc.pid
    job.pgid = os.getpgid(proc.pid)
    job.gpu_ids = list(gpu_ids)
    job.master_port = port
    job.status = "running"
    job.started_at = datetime.now().isoformat()

    _log(f"Launched {job.name} (PID={proc.pid}, GPUs={gpu_ids}, port={port})")


def tail_log(log_file: Optional[str], lines: int = 5) -> Optional[str]:
    """Return the last N lines of a log file."""
    if not log_file or not os.path.exists(log_file):
        return None
    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:]).strip()
    except Exception:
        return None


def is_nccl_timeout(job: Job) -> bool:
    """Check if a job failed due to NCCL communication timeout (NOT OOM).

    NCCL timeouts happen when distributed workers can't establish
    communication within the timeout window — typically caused by
    GPU contention during initialization, NOT by memory exhaustion.
    These should be retried (counted as retry) rather than requeued as OOM.
    """
    snippet = tail_log(job.log_file, lines=50)
    if not snippet:
        return False
    snippet_lower = snippet.lower()
    nccl_timeout_keywords = [
        "wait timeout",
        "nccluniqueid",
        "timed out after",
        "nccl communicator",
        "key-value store",
        "distbackenderror",
        "socket.cpp",
    ]
    matches = sum(1 for kw in nccl_timeout_keywords if kw.lower() in snippet_lower)
    if matches >= 2:
        _log(f"🔍 {job.name}: NCCL timeout detected in log (matched {matches} keywords) "
             f"— this is NOT OOM, treating as a normal failure")
        return True
    return False


def check_dmesg_oom(pid: int, pgid: Optional[int] = None,
                    lookback_seconds: float = 300) -> bool:
    """Check kernel ring buffer (dmesg) for OOM kills of a specific PID or
    any process in its process group.

    This is the ground-truth method for detecting system-level OOM kills.
    The Linux kernel logs every OOM kill to the ring buffer, so this is
    far more reliable than log-file keyword matching.

    Args:
        pid: The main process ID to check for.
        pgid: Optional process group ID — if provided, also matches any
              PID in the group that was OOM-killed.
        lookback_seconds: Only consider dmesg entries from the last N seconds.

    Returns:
        True if dmesg contains an OOM kill entry for the given PID/PGID.
    """
    try:
        result = subprocess.run(
            ["dmesg", "-T"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False

        # Collect PIDs to match: the main pid + process-group children
        pids_to_check = {str(pid)}
        if pgid is not None:
            # Collect all PIDs that belonged to this process group.
            # After the process is dead, /proc won't help, but dmesg
            # logs the killed PID directly so we match broadly.
            pids_to_check.add(str(pgid))

        now = datetime.now()
        for line in result.stdout.splitlines():
            line_lower = line.lower()
            if "oom" not in line_lower and "out of memory" not in line_lower:
                continue

            # Parse timestamp: dmesg -T produces lines like:
            # [Fri Mar 14 01:23:45 2026] Out of memory: Killed process 12345
            ts_match = None
            if line.startswith("["):
                bracket_end = line.find("]")
                if bracket_end > 0:
                    ts_str = line[1:bracket_end].strip()
                    for fmt in ("%a %b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S%z"):
                        try:
                            ts_match = datetime.strptime(ts_str, fmt)
                            break
                        except ValueError:
                            continue

            # Skip entries older than lookback window
            if ts_match:
                age = (now - ts_match).total_seconds()
                if age > lookback_seconds:
                    continue

            # Check if any of our PIDs appear in this OOM line
            for p in pids_to_check:
                if p in line:
                    _log(f"🔍 dmesg OOM hit for PID {p}: {line.strip()[:200]}")
                    return True

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        _log(f"⚠️ dmesg OOM check failed: {exc}")
    return False


def is_oom_crash(job: Job, oom_threshold: float = DEFAULT_OOM_CRASH_THRESHOLD) -> bool:
    """Determine if a job crash was likely caused by OOM (resource exhaustion).

    Detection methods (in priority order — NO log parsing):
    1. dmesg kernel ring buffer — ground truth for system OOM kills
    2. Exit code -9/137 (SIGKILL) + quick death — strong signal
    3. Quick death + any non-zero exit — heuristic fallback

    NOTE: NCCL timeouts are explicitly excluded — they are communication
    failures, not memory failures.
    """
    # First, check if this is an NCCL timeout (NOT OOM)
    if is_nccl_timeout(job):
        return False

    # ── Method 0: Sticky OOM flag ───────────────────────────────────
    # If this job was previously requeued for OOM, it's VERY likely to
    # OOM again when retried. Treat any non-zero exit as OOM to prevent
    # the retry from masking the real issue (the job runs longer on
    # retry because other jobs freed some memory, so it no longer
    # triggers the quick_death heuristic, creating a false negative).
    if job.oom_requeue_count > 0 and job.exit_code not in (0, None):
        _log(f"🔍 {job.name}: previously OOM-requeued {job.oom_requeue_count}× "
             f"+ non-zero exit ({job.exit_code}) → treating as OOM")
        return True

    # ── Method 1: dmesg (ground truth) ──────────────────────────────
    if job.pid and check_dmesg_oom(job.pid, job.pgid):
        _log(f"🔍 {job.name}: OOM confirmed via dmesg (kernel ring buffer)")
        return True

    # ── Method 2 & 3: exit code + timing heuristics ─────────────────
    quick_death = False
    if job.started_at:
        try:
            start = datetime.fromisoformat(job.started_at)
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed < oom_threshold:
                quick_death = True
        except Exception:
            pass

    # Exit code -9 (SIGKILL) is the Linux OOM killer's signature
    killed_by_signal = job.exit_code in (-9, 137)  # 137 = 128 + 9

    if killed_by_signal and quick_death:
        _log(f"🔍 {job.name}: killed by SIGKILL + quick death → likely OOM")
        return True
    if quick_death and job.exit_code not in (0, None):
        _log(f"🔍 {job.name}: quick death ({job.exit_code}) → treating as resource exhaustion")
        return True

    return False


def _requeue_job(job: Job, reason: str) -> None:
    """Reset a job back to pending after a failed launch attempt."""
    _log(f"↩ Requeueing {job.name}: {reason}")
    job.error_snippet = reason
    # Track OOM requeues for sticky OOM detection
    if "OOM" in reason.upper() or "oom" in reason:
        job.oom_requeue_count += 1
        _log(f"   📌 {job.name} OOM requeue count: {job.oom_requeue_count}")
    job.status = "pending"
    job.pid = None
    job.pgid = None
    job.master_port = None
    # Clear memory snapshot — will be re-measured on next launch
    job.memory_at_stable = None
    job.launched_at_burst_index = None
    # Don't release gpu_ids here — caller handles free_gpus tracking
    gpu_ids = job.gpu_ids
    job.gpu_ids = None
    if job._log_fh:
        try:
            job._log_fh.write(f"\n=== LAUNCH FAILED: {reason} ===\n")
            job._log_fh.close()
        except Exception:
            pass
        job._log_fh = None
    job._proc = None
    return gpu_ids  # return freed GPUs so caller can reuse them


def cascade_requeue_since_checkpoint(state: "State", trigger_job_name: str,
                                      oom_threshold: float) -> int:
    """Kill and requeue all running jobs launched AFTER the last saturation checkpoint.

    When OOM is detected during burst, the jobs launched between the last
    checkpoint and now were launched under a false assumption (the memory
    prediction was wrong because earlier jobs were still loading). Those
    jobs are at high risk of cascade-OOMing. Rather than waiting for them
    to die one-by-one, proactively kill + requeue them.

    Args:
        state: daemon state
        trigger_job_name: name of the job that triggered the cascade
        oom_threshold: passed through for is_oom_crash

    Returns:
        Number of jobs killed and requeued.
    """
    last_cp_index = state.daemon.last_checkpoint_burst_index
    requeued = 0

    candidates = [
        j for j in state.jobs
        if j.status == "running"
        and j.launched_at_burst_index is not None
        and j.launched_at_burst_index > last_cp_index
        and j.name != trigger_job_name  # trigger job already handled by caller
    ]

    if not candidates:
        return 0

    _log(f"🔄 CASCADE REQUEUE: {trigger_job_name} caused OOM. "
         f"Killing {len(candidates)} job(s) launched after checkpoint "
         f"(burst index > {last_cp_index}):")

    for job in candidates:
        _log(f"   🔄 Killing {job.name} (burst_index={job.launched_at_burst_index})")

        # Kill the process tree
        if job._proc is not None:
            try:
                if job.pgid:
                    os.killpg(job.pgid, signal.SIGTERM)
                else:
                    job._proc.terminate()
                # Give it a moment then force kill
                try:
                    job._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if job.pgid:
                        os.killpg(job.pgid, signal.SIGKILL)
                    else:
                        job._proc.kill()
            except (OSError, ProcessLookupError):
                pass
        elif job.pid:
            try:
                if job.pgid:
                    os.killpg(job.pgid, signal.SIGTERM)
                    time.sleep(2)
                    os.killpg(job.pgid, signal.SIGKILL)
                else:
                    os.kill(job.pid, signal.SIGTERM)
                    time.sleep(2)
                    os.kill(job.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        # Close log handle
        if job._log_fh:
            try:
                job._log_fh.write(f"\n=== CASCADE REQUEUE: killed due to {trigger_job_name} OOM ===\n")
                job._log_fh.close()
            except Exception:
                pass
            job._log_fh = None
        job._proc = None

        # Requeue without counting as retry (it's not the job's fault)
        job.status = "pending"
        job.pid = None
        job.pgid = None
        job.gpu_ids = None
        job.master_port = None
        job.exit_code = None
        job.launched_at_burst_index = None
        job.memory_at_stable = None
        requeued += 1

    _log(f"🔄 CASCADE REQUEUE: {requeued} job(s) killed and requeued "
         f"(not counted as retries)")
    return requeued


def get_gpu_pids() -> Optional[set]:
    """Query NVML for all PIDs currently using GPU compute.

    Returns:
        set of int PIDs  – PIDs with active GPU processes (may be empty if
                           NVML works but nothing is on GPU yet).
        None             – NVML not available / failed.

    Uses pynvml for direct driver-level queries (much faster and more reliable
    than shelling out to nvidia-smi). Falls back to nvidia-smi if pynvml is
    not installed.
    """
    if HAS_PYNVML:
        pids = set()
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in procs:
                        pids.add(p.pid)
                except pynvml.NVMLError:
                    continue
            return pids
        except pynvml.NVMLError as e:
            _log(f"⚠️ pynvml query failed: {e}")
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
    else:
        # Fallback: nvidia-smi subprocess
        pids = set()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if line and line.isdigit():
                        pids.add(int(line))
            else:
                _log(f"⚠️ nvidia-smi query failed: {result.stderr.strip()}")
        except FileNotFoundError:
            _log("⚠️ nvidia-smi not found")
            return None
        except subprocess.TimeoutExpired:
            _log("⚠️ nvidia-smi query timed out")
            return None
        except Exception as e:
            _log(f"⚠️ nvidia-smi query error: {e}")
            return None
        return pids


def get_gpu_pids_per_device() -> Optional[Dict[int, set]]:
    """Query NVML for PIDs per GPU device.

    Returns:
        dict of {gpu_index: set of PIDs} – per-GPU process mapping.
        None – NVML not available / failed.

    This is more informative than get_gpu_pids() because it tells us
    WHICH GPU each process is on, enabling precise per-GPU health checks.
    """
    if not HAS_PYNVML:
        return None
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        result = {}
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            pids = set()
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    pids.add(p.pid)
            except pynvml.NVMLError:
                pass
            result[i] = pids
        return result
    except pynvml.NVMLError as e:
        _log(f"⚠️ pynvml per-device query failed: {e}")
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def check_pid_has_gpu_fd(pid: int) -> bool:
    """Check if a PID has /dev/nvidia* device files open.

    Uses psutil if available (cleaner API), falls back to /proc/<pid>/fd
    scanning. This is more reliable than nvidia-smi on some systems
    (e.g. H20 + driver 535.x).

    Returns True if the PID has at least one /dev/nvidia<N> fd open.
    """
    if HAS_PSUTIL:
        try:
            proc = psutil.Process(pid)
            for of in proc.open_files():
                path = of.path
                if path.startswith("/dev/nvidia") and path[11:].isdigit():
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return False
    else:
        # Fallback: /proc/<pid>/fd scanning
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                    if target.startswith("/dev/nvidia") and target[11:].isdigit():
                        return True
                except (OSError, ValueError):
                    continue
        except (OSError, PermissionError):
            pass
        return False


def get_gpu_memory_usage() -> Optional[list]:
    """Query per-GPU memory usage via NVML.

    Returns a list of (gpu_index, used_mib) tuples, or None on failure.
    Uses pynvml for direct driver queries; falls back to nvidia-smi.
    """
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            usage = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mib = mem_info.used // (1024 * 1024)
                usage.append((i, used_mib))
            return usage
        except pynvml.NVMLError as e:
            _log(f"⚠️ pynvml memory query failed: {e}")
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
    else:
        # Fallback: nvidia-smi subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            usage = []
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        usage.append((int(parts[0].strip()), int(parts[1].strip())))
                    except ValueError:
                        continue
            return usage
        except Exception:
            return None


def get_gpu_memory_total_and_used() -> Optional[List[Tuple[int, int, int]]]:
    """Query per-GPU total and used memory via NVML.

    Returns a list of (gpu_index, total_mib, used_mib) tuples, or None on failure.
    This extends get_gpu_memory_usage() by also returning total memory so we
    can compute free headroom for OOM prediction.
    """
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            result = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mib = mem_info.total // (1024 * 1024)
                used_mib = mem_info.used // (1024 * 1024)
                result.append((i, total_mib, used_mib))
            return result
        except pynvml.NVMLError as e:
            _log(f"⚠️ pynvml memory total/used query failed: {e}")
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
    else:
        # Fallback: nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            usage = []
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(",")
                if len(parts) == 3:
                    try:
                        usage.append((
                            int(parts[0].strip()),
                            int(parts[1].strip()),
                            int(parts[2].strip()),
                        ))
                    except ValueError:
                        continue
            return usage
        except Exception:
            return None


def get_per_job_avg_memory(state: "State") -> Optional[float]:
    """Calculate the average GPU memory (MiB) used per running job.

    Priority order (to avoid the "loading undercount" problem):
      1. Use memory_at_stable from jobs that have already stabilized on GPU.
         This is the most accurate: it's the memory recorded AFTER a job
         finished loading its model and appeared on the GPU.
      2. Fall back to daemon.peak_memory_per_job (persisted best estimate
         from previous saturation checkpoints).
      3. Last resort: live query of current VRAM usage (may undercount if
         jobs are still loading).

    Returns the average in MiB, or None if no data available.
    """
    # Method 1: Average from stabilized jobs
    stabilized = [j for j in state.jobs
                  if j.status == "running" and j.memory_at_stable is not None]
    if stabilized:
        avg = sum(j.memory_at_stable for j in stabilized) / len(stabilized)
        return avg

    # Method 2: Persisted peak memory from previous checkpoint
    if state.daemon.peak_memory_per_job is not None and state.daemon.peak_memory_per_job > 100:
        return state.daemon.peak_memory_per_job

    # Method 3: Live query (last resort — may undercount during loading)
    running_jobs = [j for j in state.jobs if j.status == "running" and j.gpu_ids]
    if not running_jobs:
        return None

    mem_info = get_gpu_memory_total_and_used()
    if not mem_info:
        return None

    # Build a lookup: gpu_index -> used_mib
    gpu_used = {gpu_idx: used for gpu_idx, _total, used in mem_info}

    # Sum memory used on GPUs that have at least one running job
    # Count how many job-GPU assignments there are (a multi-GPU job counts
    # once per GPU it uses)
    total_mem = 0
    job_gpu_count = 0
    gpus_counted = set()
    for job in running_jobs:
        for gid in job.gpu_ids:
            if gid in gpu_used and gid not in gpus_counted:
                total_mem += gpu_used[gid]
                gpus_counted.add(gid)
        job_gpu_count += len(job.gpu_ids)

    if job_gpu_count == 0:
        return None

    # Average memory per job-GPU slot
    avg = total_mem / job_gpu_count

    # Warn that this is a live estimate (may be low)
    _log(f"⚠️ OOM prediction using LIVE memory estimate ({avg:.0f} MiB/job) — "
         f"jobs may still be loading; estimate may be low")
    return avg


def _snapshot_peak_memory(state: "State") -> None:
    """Record per-job memory for all running jobs and update the daemon's
    peak_memory_per_job estimate.

    Called after a successful saturation checkpoint — this is the MOST
    reliable moment to measure, because all jobs have been running stably
    for the full checkpoint duration (models loaded, first steps done).
    """
    mem_info = get_gpu_memory_total_and_used()
    if not mem_info:
        return

    gpu_used = {idx: used for idx, _total, used in mem_info}
    running_jobs = [j for j in state.jobs if j.status == "running" and j.gpu_ids]
    if not running_jobs:
        return

    # Compute per-GPU-slot memory by dividing each GPU's used memory
    # by the number of jobs assigned to that GPU
    gpu_job_counts: dict = {}
    for job in running_jobs:
        for gid in job.gpu_ids:
            gpu_job_counts[gid] = gpu_job_counts.get(gid, 0) + 1

    # Estimate per-job memory on each GPU
    per_job_estimates = []
    for job in running_jobs:
        job_mem = 0.0
        for gid in job.gpu_ids:
            if gid in gpu_used and gid in gpu_job_counts:
                job_mem += gpu_used[gid] / gpu_job_counts[gid]
        if job_mem > 0:
            job.memory_at_stable = job_mem
            per_job_estimates.append(job_mem)

    if per_job_estimates:
        avg = sum(per_job_estimates) / len(per_job_estimates)
        state.daemon.peak_memory_per_job = avg
        _log(f"📊 Peak memory snapshot: {avg:.0f} MiB/job avg "
             f"(from {len(per_job_estimates)} jobs, "
             f"range {min(per_job_estimates):.0f}-{max(per_job_estimates):.0f} MiB)")


def estimate_oom_risk(
    state: "State",
    gpus_needed: int = 1,
    safety_margin: float = DEFAULT_MEMORY_SAFETY_MARGIN,
) -> Tuple[bool, str]:
    """Predict whether launching a new job is likely to cause OOM.

    Strategy:
      1. Get total and used memory for each GPU.
      2. Compute average memory per running job from historical observation.
      3. For the GPUs that would be assigned to the new job (round-robin),
         check if free memory >= avg_per_job × (1 + safety_margin).
      4. If not enough headroom on ANY assigned GPU → predict OOM.

    Returns:
        (at_risk: bool, reason: str)
        at_risk=True means "don't launch, likely OOM"
    """
    running_jobs = [j for j in state.jobs if j.status == "running" and j.gpu_ids]
    if not running_jobs:
        return False, "No running jobs yet — no memory baseline, safe to launch"

    mem_info = get_gpu_memory_total_and_used()
    if not mem_info:
        return False, "Cannot query GPU memory — skipping OOM prediction"

    avg_per_job = get_per_job_avg_memory(state)
    if avg_per_job is None or avg_per_job < 100:
        # Less than 100 MiB average is not meaningful (jobs still initializing)
        return False, f"Avg memory per job too low ({avg_per_job:.0f} MiB) — jobs may still be loading"

    # IMPORTANT: avg_per_job is the TOTAL memory a job uses across ALL its
    # GPUs (e.g. a 2-GPU job using 5GB on each GPU → avg_per_job=10GB).
    # For the per-GPU comparison, we need the per-GPU-slot expected usage.
    per_gpu_slot_mem = avg_per_job / max(gpus_needed, 1)

    # Compute how much memory the new job is expected to need PER GPU
    expected_mem = per_gpu_slot_mem * (1.0 + safety_margin)

    # Check which GPUs would be assigned (round-robin)
    all_gpus = list(state.daemon.gpus)
    num_gpus = len(all_gpus)
    rr_idx = state.daemon.overcommit_rr_index
    target_gpus = [
        all_gpus[(rr_idx + g) % num_gpus]
        for g in range(gpus_needed)
    ]

    # Build lookup: gpu_index -> (total, used)
    gpu_mem = {idx: (total, used) for idx, total, used in mem_info}

    worst_gpu = None
    worst_free = float("inf")
    for gid in target_gpus:
        if gid not in gpu_mem:
            continue
        total, used = gpu_mem[gid]
        free = total - used
        if free < worst_free:
            worst_free = free
            worst_gpu = gid

    if worst_gpu is None:
        return False, "Could not find target GPU memory info — skipping prediction"

    total_for_worst, used_for_worst = gpu_mem[worst_gpu]

    if worst_free < expected_mem:
        return True, (
            f"GPU {worst_gpu}: {worst_free:.0f} MiB free "
            f"< {expected_mem:.0f} MiB needed "
            f"(per_gpu_slot={per_gpu_slot_mem:.0f} MiB + {safety_margin*100:.0f}% margin, "
            f"avg_total/job={avg_per_job:.0f} MiB across {gpus_needed} GPU(s)). "
            f"[{used_for_worst:.0f}/{total_for_worst:.0f} MiB used]"
        )
    else:
        return False, (
            f"GPU {worst_gpu}: {worst_free:.0f} MiB free "
            f">= {expected_mem:.0f} MiB needed "
            f"(per_gpu_slot={per_gpu_slot_mem:.0f} MiB) — safe to launch"
        )


def memory_wait_for_headroom(
    state: "State",
    gpus_needed: int,
    safety_margin: float,
    timeout: float,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    """Wait up to `timeout` seconds for GPU memory headroom to become available.

    Polls every 15 seconds. During the wait, also performs health checks
    on all running jobs (kills zombies, detects OOM crashes).

    Returns:
        (headroom_ok: bool, reason: str)
        headroom_ok=True  → memory is now available, safe to launch
        headroom_ok=False → timed out or a running job died (caller should queue)
    """
    start = time.time()
    poll_interval = 15  # check every 15 seconds
    check_count = 0

    _log(f"⏳ OOM PREDICTION: Waiting for memory headroom "
         f"(timeout={timeout:.0f}s, checking every {poll_interval}s)...")

    while time.time() - start < timeout:
        check_count += 1
        elapsed = time.time() - start

        # Re-check memory
        at_risk, reason = estimate_oom_risk(state, gpus_needed, safety_margin)
        if not at_risk:
            _log(f"✅ Memory headroom available after {elapsed:.0f}s "
                 f"(check #{check_count}): {reason}")
            return True, reason

        # Log progress every 4th check (~60s)
        if check_count % 4 == 0:
            _log(f"⏳ Still waiting for memory ({elapsed:.0f}s/{timeout:.0f}s): {reason}")

        # Health check: verify running jobs are still alive
        for rj in state.jobs:
            if rj.status != "running":
                continue
            rj_dead = False
            rj_exit = None
            if rj._proc is not None:
                ret = rj._proc.poll()
                if ret is not None:
                    rj_dead = True
                    rj_exit = ret
            else:
                if not is_process_alive(rj.pid):
                    rj_dead = True

            if rj_dead:
                _log(f"🛑 MEMORY WAIT: {rj.name} died during headroom wait "
                     f"(exit={rj_exit}, elapsed={elapsed:.0f}s)")
                rj.exit_code = rj_exit
                rj.error_snippet = tail_log(rj.log_file, lines=5)
                if rj._log_fh:
                    try:
                        rj._log_fh.close()
                    except Exception:
                        pass
                    rj._log_fh = None
                oom_threshold = getattr(args, 'oom_threshold', DEFAULT_OOM_CRASH_THRESHOLD)
                if is_oom_crash(rj, oom_threshold=oom_threshold):
                    _requeue_job(rj, "OOM during memory headroom wait")
                elif rj.retries < rj.max_retries:
                    rj.retries += 1
                    rj.status = "pending"
                    rj.pid = None
                    rj.pgid = None
                    rj.gpu_ids = None
                    rj.master_port = None
                else:
                    rj.status = "failed"
                    rj.completed_at = datetime.now().isoformat()

                # A job died — memory may have freed up, re-check immediately
                at_risk2, reason2 = estimate_oom_risk(state, gpus_needed, safety_margin)
                if not at_risk2:
                    _log(f"✅ Memory freed after {rj.name} died: {reason2}")
                    return True, reason2
                # If still at risk, return False — caller decides whether to
                # switch to FIFO or keep waiting
                return False, f"Job {rj.name} died but still at risk: {reason2}"

        time.sleep(poll_interval)

    # Timed out
    elapsed = time.time() - start
    _, final_reason = estimate_oom_risk(state, gpus_needed, safety_margin)
    _log(f"⏰ OOM PREDICTION: Timed out after {elapsed:.0f}s waiting for "
         f"memory headroom. {final_reason}")
    return False, f"Timed out after {elapsed:.0f}s: {final_reason}"


def get_child_pids(parent_pid: int) -> set:
    """Get all descendant PIDs of a parent process (recursive).

    Uses psutil for reliable process tree traversal; falls back to pgrep.
    torchrun spawns child processes that actually allocate GPU memory,
    so we need to check children too.
    """
    if HAS_PSUTIL:
        children = set()
        try:
            proc = psutil.Process(parent_pid)
            for child in proc.children(recursive=True):
                children.add(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return children
    else:
        # Fallback: pgrep subprocess
        children = set()
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    child_pid = int(line)
                    children.add(child_pid)
                    children.update(get_child_pids(child_pid))
        except Exception:
            pass
        return children


def get_process_gpu_info(pid: int) -> Optional[Dict[str, Any]]:
    """Get detailed GPU info for a specific process using pynvml+psutil.

    Returns a dict with:
        - gpu_indices: set of GPU indices the process is using
        - gpu_memory_mib: dict of {gpu_index: memory_used_mib}
        - alive: whether the process is still running
        - children: set of child PIDs
    Or None if the process doesn't exist.
    """
    info: Dict[str, Any] = {
        "gpu_indices": set(),
        "gpu_memory_mib": {},
        "alive": False,
        "children": set(),
    }

    # Check process alive + get children
    if HAS_PSUTIL:
        try:
            proc = psutil.Process(pid)
            info["alive"] = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            for child in proc.children(recursive=True):
                info["children"].add(child.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None
        except psutil.AccessDenied:
            info["alive"] = True
    else:
        info["alive"] = is_process_alive(pid)
        info["children"] = get_child_pids(pid)

    if not info["alive"]:
        return None

    # Get per-GPU process info via NVML
    all_pids = {pid} | info["children"]
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in procs:
                        if p.pid in all_pids:
                            info["gpu_indices"].add(i)
                            current = info["gpu_memory_mib"].get(i, 0)
                            if p.usedGpuMemory is not None:
                                info["gpu_memory_mib"][i] = current + p.usedGpuMemory // (1024 * 1024)
                except pynvml.NVMLError:
                    continue
        except pynvml.NVMLError:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    return info


def check_gpu_presence(job: Job) -> bool:
    """Check if a job's PID (or any of its child PIDs) have GPU presence.

    Uses three strategies (in order):
      1. pynvml: Direct NVML query for compute processes per GPU (fastest,
         most reliable — no subprocess overhead).
      2. /proc/<pid>/fd or psutil open_files: Check for /dev/nvidia* device
         files (works even when NVML process table has issues on H20/driver 535.x).
      3. GPU memory heuristic: Check if assigned GPUs have significant memory
         usage with alive child processes.

    torchrun itself may not show in NVML, but its child workers will.
    Returns True if the job has GPU presence, False otherwise.
    """
    if job.pid is None:
        return False

    # Collect the job's PID tree (main + all children/grandchildren)
    all_pids = {job.pid}
    all_pids.update(get_child_pids(job.pid))

    # Strategy 1: NVML direct query (preferred — fastest, no subprocess)
    gpu_pids = get_gpu_pids()
    if gpu_pids is not None:
        overlap = all_pids & gpu_pids
        if overlap:
            return True
    elif gpu_pids is None and not HAS_PYNVML:
        # NVML/nvidia-smi completely unavailable — can't check
        _log(f"⚠️ GPU query unavailable, assuming {job.name} is OK")
        return True

    # Strategy 2: Check /dev/nvidia* file descriptors (psutil or /proc)
    for pid in all_pids:
        if check_pid_has_gpu_fd(pid):
            return True

    # Strategy 3: Check if assigned GPUs have significant memory usage
    if job.gpu_ids:
        mem_usage = get_gpu_memory_usage()
        if mem_usage:
            for gpu_idx, used_mib in mem_usage:
                if gpu_idx in job.gpu_ids and used_mib > 500:  # >500 MiB = not idle
                    alive_children = [p for p in all_pids if is_process_alive(p)]
                    if len(alive_children) > 1:  # main + at least 1 worker
                        return True

    return False


# ---------------------------------------------------------------------------
# Sentinel-based training-started detection
# ---------------------------------------------------------------------------

def _parse_cli_flag(cmd: str, flag: str) -> Optional[str]:
    """Extract the value of *flag* (e.g. ``--output``) from a shell command string.

    Handles both ``--flag value`` and ``--flag=value`` forms.  Returns
    *None* when the flag is absent.
    """
    tokens = shlex.split(cmd)
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(f"{flag}="):
            return tok.split("=", 1)[1]
    return None


def get_job_sentinel_path(job: Job) -> Optional[str]:
    """Derive the ``.training_started`` sentinel path from a job's command.

    The sentinel lives at ``<output>/<experiment>/.training_started``.
    Both ``--output`` and ``--experiment`` must be present in the command.
    """
    cmd = job.cmd_template or job.cmd
    output_root = _parse_cli_flag(cmd, "--output")
    experiment = _parse_cli_flag(cmd, "--experiment")
    if output_root and experiment:
        return os.path.join(output_root, experiment, ".training_started")
    return None


def _is_descendant_of(pid: int, ancestor_pid: int) -> bool:
    """Return *True* if *pid* is a descendant of *ancestor_pid* (or equal).

    Walks up the process tree via ``/proc/<pid>/stat`` (Linux-specific).
    Returns *False* on any error (process already exited, non-Linux, etc.).
    """
    if pid == ancestor_pid:
        return True
    visited = {pid}
    current = pid
    try:
        while True:
            stat_path = f"/proc/{current}/stat"
            with open(stat_path, "r") as f:
                # Format: pid (comm) state ppid ...
                data = f.read()
            # The comm field can contain spaces/parens, so find the last ')'
            close_paren = data.rfind(")")
            fields_after = data[close_paren + 2:].split()
            ppid = int(fields_after[1])  # ppid is field index 3 (0-based after comm)
            if ppid == ancestor_pid:
                return True
            if ppid in visited or ppid <= 1:
                return False
            visited.add(ppid)
            current = ppid
    except (OSError, ValueError, IndexError):
        return False


def check_training_started(job: Job) -> bool:
    """Return *True* if the job has written its ``.training_started`` sentinel.

    The sentinel file written by ``train.py`` contains a ``pgid=<N>`` line.
    We verify that the PGID recorded in the sentinel belongs to the same
    process tree as this job, so that stale sentinels from a previous run
    (or manually created files) are not mistaken for a valid start signal.

    When ``torchrun`` is used with ``start_new_session=True``, child worker
    processes may get their own process group (via ``multiprocessing``), so
    the sentinel PGID won't match the ``torchrun`` launcher's PGID exactly.
    We accept the sentinel if:
      1. The sentinel PGID matches ``job.pgid`` exactly, OR
      2. The sentinel PID/PGID is a descendant of ``job.pid``.

    Falls back to :func:`check_gpu_presence` when the sentinel path cannot
    be determined.
    """
    sentinel = get_job_sentinel_path(job)
    if sentinel is None:
        # Cannot determine sentinel path — fall back to GPU presence check
        return check_gpu_presence(job)
    if not os.path.isfile(sentinel):
        return False

    # ------------------------------------------------------------------
    # Verify ownership: the sentinel's PGID/PID must belong to this job's
    # process tree (exact PGID match OR descendant of job.pid).
    # ------------------------------------------------------------------
    if job.pgid is None:
        # No pgid recorded (shouldn't happen) — accept presence only
        return True
    try:
        with open(sentinel, "r") as f:
            content = f.read()
        sentinel_pgid = None
        sentinel_pid = None
        for line in content.splitlines():
            if line.startswith("pgid="):
                sentinel_pgid = int(line.split("=", 1)[1])
            elif line.startswith("pid="):
                sentinel_pid = int(line.split("=", 1)[1])

        if sentinel_pgid is None:
            # No pgid line found — sentinel is invalid / not from our train.py
            _log(
                f"⚠ Sentinel {sentinel} has no pgid line "
                f"— ignoring invalid sentinel"
            )
            return False

        # Accept: exact PGID match
        if sentinel_pgid == job.pgid:
            return True

        # Accept: sentinel process is a descendant of the job's launcher
        # (handles torchrun workers that get their own process group)
        check_pid = sentinel_pid if sentinel_pid is not None else sentinel_pgid
        if job.pid is not None and _is_descendant_of(check_pid, job.pid):
            _log(
                f"✓ Sentinel {sentinel} pgid={sentinel_pgid} differs from "
                f"job pgid={job.pgid}, but pid={check_pid} is a descendant "
                f"of job pid={job.pid} — accepting"
            )
            return True

        _log(
            f"⚠ Sentinel {sentinel} has pgid={sentinel_pgid} "
            f"but job {job.name} expects pgid={job.pgid} "
            f"and pid={check_pid} is not a descendant of job pid={job.pid} "
            f"— ignoring stale/foreign sentinel"
        )
        return False
    except (OSError, ValueError) as exc:
        _log(f"⚠ Failed to read sentinel {sentinel}: {exc}")
        return False


def get_gpu_free_memory(gpu_index: int) -> Optional[int]:
    """Return free memory (MiB) on a single GPU, or *None* on failure."""
    mem_info = get_gpu_memory_total_and_used()
    if not mem_info:
        return None
    for idx, total, used in mem_info:
        if idx == gpu_index:
            return total - used
    return None


def _resolve_profiled_memory_per_gpu_for_job(
    job: Optional[Job],
    state: "State",
    gpus_needed: int,
) -> Tuple[Optional[float], str]:
    """Resolve expected per-GPU memory for admission checks.

    Priority:
      1) Scheduler profiling attached to the pending job
         (job.estimated_memory_mib_per_gpu).
      2) Running jobs with the same model_key (stable memory snapshots).
      3) Runtime global average fallback (converted to per-GPU slot).

    Returns:
      (expected_per_gpu_mib, source)
    """
    if job is not None and job.estimated_memory_mib_per_gpu is not None:
        try:
            profiled = float(job.estimated_memory_mib_per_gpu)
        except (TypeError, ValueError):
            profiled = None
        if profiled is not None and profiled > 100:
            return profiled, "scheduler-profiled"

    if job is not None and job.model_key:
        same_model = [
            j for j in state.jobs
            if j.status == "running"
            and j.model_key == job.model_key
            and j.memory_at_stable is not None
            and j.gpu_ids
        ]
        if same_model:
            per_gpu_values = []
            for running_job in same_model:
                slots = max(len(running_job.gpu_ids), 1)
                per_gpu_values.append(running_job.memory_at_stable / slots)
            if per_gpu_values:
                return (
                    sum(per_gpu_values) / len(per_gpu_values),
                    f"observed model_key={job.model_key}",
                )

    avg_per_job = get_per_job_avg_memory(state)
    if avg_per_job is not None and avg_per_job > 100:
        per_gpu_slot = avg_per_job / max(gpus_needed, 1)
        return per_gpu_slot, "runtime-average fallback"

    return None, "no profiled/runtime baseline"


def can_fit_another_job_on_gpu(
    gpu_index: int,
    state: "State",
    safety_margin: float = DEFAULT_MEMORY_SAFETY_MARGIN,
    gpus_needed: int = 1,
    job: Optional[Job] = None,
) -> Tuple[bool, str]:
    """Check if *gpu_index* has enough free memory for one more job.

    Uses scheduler profiling memory first; runtime averages are fallback.
    Returns (can_fit, reason_string).
    """
    free = get_gpu_free_memory(gpu_index)
    if free is None:
        return True, "Cannot query GPU memory — assuming OK"

    expected_per_gpu, source = _resolve_profiled_memory_per_gpu_for_job(
        job=job,
        state=state,
        gpus_needed=gpus_needed,
    )
    if expected_per_gpu is None or expected_per_gpu < 100:
        return True, (
            f"GPU {gpu_index}: no reliable profiled memory baseline yet "
            f"(source={source}), allowing launch"
        )

    needed = expected_per_gpu * (1.0 + safety_margin)

    if free >= needed:
        return True, (
            f"GPU {gpu_index}: {free:.0f} MiB free >= "
            f"{needed:.0f} MiB needed "
            f"(expected_per_gpu={expected_per_gpu:.0f}, source={source}, "
            f"gpus_needed={gpus_needed}) — OK"
        )
    else:
        return False, (
            f"GPU {gpu_index}: {free:.0f} MiB free < "
            f"{needed:.0f} MiB needed "
            f"(expected_per_gpu={expected_per_gpu:.0f}, source={source}, "
            f"gpus_needed={gpus_needed}) — full"
        )


def can_place_job_on_gpu_group(
    gpu_ids: List[int],
    state: "State",
    safety_margin: float,
    gpus_needed: int,
    job: Optional[Job],
) -> Tuple[bool, List[str]]:
    """Return whether a job may launch on a GPU group.

    With overcommit disabled, any overlap with a running job makes the group
    unavailable regardless of reported free VRAM. Otherwise, use the normal
    per-GPU profiled-memory admission checks.
    """
    if not state.daemon.overcommit:
        requested = set(gpu_ids)
        conflicts = [
            running.name
            for running in state.jobs
            if running.status == "running"
            and running.gpu_ids
            and requested.intersection(running.gpu_ids)
        ]
        if conflicts:
            return False, [
                f"GPU group {gpu_ids}: exclusive placement blocked by "
                f"running job(s): {', '.join(conflicts)}"
            ]
        return True, [f"GPU group {gpu_ids}: exclusive placement available"]

    reasons: List[str] = []
    can_fit = True
    for gpu_id in gpu_ids:
        fits, reason = can_fit_another_job_on_gpu(
            gpu_id,
            state,
            safety_margin,
            gpus_needed=gpus_needed,
            job=job,
        )
        reasons.append(f"GPU {gpu_id}: {reason}")
        if not fits:
            can_fit = False
    return can_fit, reasons


# ---------------------------------------------------------------------------
# Inbox watcher
# ---------------------------------------------------------------------------

def scan_inbox(inbox_dir: str) -> List[str]:
    """Return YAML files in inbox sorted by modification time (FIFO)."""
    os.makedirs(inbox_dir, exist_ok=True)
    files = []
    for f in os.listdir(inbox_dir):
        full = os.path.join(inbox_dir, f)
        if f.endswith((".yaml", ".yml")) and os.path.isfile(full):
            files.append(full)
    files.sort(key=os.path.getmtime)
    return files


def parse_input_yaml(path: str) -> Tuple[List[Job], Dict[str, Any]]:
    """Parse an input YAML file into a list of Job objects.

    Returns:
        (jobs_list, defaults_dict)
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not data or "jobs" not in data:
        _log(f"Warning: {path} has no 'jobs' key, skipping")
        return [], {}

    defaults = data.get("defaults", {})
    default_gpus = defaults.get("gpus", 1)
    default_retries = defaults.get("max_retries", DEFAULT_MAX_RETRIES)
    default_working_dir = defaults.get("working_dir", None)

    jobs = []
    for entry in data["jobs"]:
        name = entry.get("name", f"job_{len(jobs)}")
        cmd = entry.get("cmd", "").strip()
        if not cmd:
            _log(f"Warning: job '{name}' has no cmd, skipping")
            continue
        model_key = entry.get("model_key", None)
        mem_per_gpu = entry.get("memory_mib_per_gpu", None)
        jobs.append(Job(
            name=name,
            cmd=cmd,
            gpus_needed=entry.get("gpus", default_gpus),
            max_retries=entry.get("max_retries", default_retries),
            working_dir=entry.get("working_dir", default_working_dir),
            submitted_at=datetime.now().isoformat(),
            model_key=model_key,
            estimated_memory_mib_per_gpu=mem_per_gpu,
        ))
    return jobs, defaults


def move_to_processed(file_path: str) -> None:
    """Move consumed inbox file to inbox/processed/ with timestamp prefix."""
    processed_dir = os.path.join(os.path.dirname(file_path), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.basename(file_path)
    dest = os.path.join(processed_dir, f"{ts}_{basename}")
    shutil.move(file_path, dest)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def render_status(state: State, compact: bool = False) -> None:
    """Render a status table to stdout."""
    running = [j for j in state.jobs if j.status == "running"]
    pending = [j for j in state.jobs if j.status == "pending"]
    completed = [j for j in state.jobs if j.status == "completed"]
    failed = [j for j in state.jobs if j.status == "failed"]
    cancelled = [j for j in state.jobs if j.status == "cancelled"]

    gpu_str = ",".join(str(g) for g in state.daemon.gpus) if state.daemon.gpus else "none"
    paused_str = " [PAUSED]" if state.daemon.paused else ""
    if state.daemon.burst_phase:
        phase_str = f" [BURST {state.daemon.burst_jobs_launched} launched]"
    elif state.daemon.saturated:
        phase_str = " [FIFO - timed VRAM polling, saturated]"
    else:
        phase_str = " [FIFO - timed VRAM polling]"
    overcommit_str = f" | OVERCOMMIT{phase_str}"

    print(f"\n{'=' * 70}")
    print(f" Job Daemon | GPUs: {gpu_str}"
          f" | Concurrent: {state.daemon.max_concurrent}{paused_str}{overcommit_str}")
    print(f"{'=' * 70}")

    # ETA calculation
    avg_duration = _avg_job_duration(state)
    eta_str = ""
    if avg_duration and pending:
        remaining_batches = len(pending) / max(1, state.daemon.max_concurrent)
        eta_seconds = remaining_batches * avg_duration
        eta_str = f" | ETA: {_format_duration(eta_seconds)}"

    print(f" ✅ {len(completed)} completed | ⏳ {len(pending)} pending"
          f" | 🔄 {len(running)} running | ❌ {len(failed)} failed"
          f" | 🚫 {len(cancelled)} cancelled{eta_str}")
    print(f"{'-' * 70}")

    # Running jobs
    if running:
        print(" RUNNING:")
        for j in running:
            runtime = _runtime_str(j.started_at)
            gpu_s = ",".join(str(g) for g in (j.gpu_ids or []))
            print(f"   {j.name:<40} GPUs=[{gpu_s}] PID={j.pid} "
                  f"{runtime} retry {j.retries}/{j.max_retries}")

    # Pending (show next 5)
    if pending:
        print(f" PENDING ({len(pending)} total):")
        for j in pending[:5]:
            print(f"   {j.name:<40} needs {j.gpus_needed} GPU(s)")
        if len(pending) > 5:
            print(f"   ... and {len(pending) - 5} more")

    # Failed (show all)
    if failed:
        print(f" FAILED ({len(failed)}):")
        for j in failed:
            snippet = f" | {j.error_snippet[:60]}..." if j.error_snippet else ""
            print(f"   {j.name:<40} exit={j.exit_code} "
                  f"retries={j.retries}/{j.max_retries}{snippet}")

    print(f"{'=' * 70}\n")


def _runtime_str(started_at: Optional[str]) -> str:
    if not started_at:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
        elapsed = (datetime.now() - start).total_seconds()
        return _format_duration(elapsed)
    except Exception:
        return ""


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def _avg_job_duration(state: State) -> Optional[float]:
    """Average duration of completed jobs in seconds."""
    durations = []
    for j in state.jobs:
        if j.status == "completed" and j.started_at and j.completed_at:
            try:
                start = datetime.fromisoformat(j.started_at)
                end = datetime.fromisoformat(j.completed_at)
                durations.append((end - start).total_seconds())
            except Exception:
                pass
    return sum(durations) / len(durations) if durations else None


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def _kill_job_tree(job: Job, sig: int = signal.SIGTERM) -> None:
    """Send a signal to a job's entire process tree.

    Uses psutil to find and signal the full process tree (more reliable
    than os.killpg which only works if the child kept the same pgid).
    Falls back to os.killpg if psutil is not available.
    """
    if HAS_PSUTIL and job.pid:
        try:
            parent = psutil.Process(job.pid)
            children = parent.children(recursive=True)
            # Signal children first (bottom-up), then parent
            for child in children:
                try:
                    child.send_signal(sig)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                parent.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            # Fall back to pgid-based kill
            if job.pgid:
                try:
                    os.killpg(job.pgid, sig)
                except ProcessLookupError:
                    pass
    elif job.pgid:
        try:
            os.killpg(job.pgid, sig)
        except ProcessLookupError:
            pass


def graceful_shutdown(state: State, timeout: int = 30) -> None:
    """Send SIGTERM to all running jobs, wait, then SIGKILL stragglers.

    Uses psutil for reliable process tree management when available.
    """
    running = [j for j in state.jobs if j.status == "running"]
    if not running:
        save_state(state)
        return

    _log(f"Stopping {len(running)} running job(s)...")

    # Phase 1: SIGTERM entire process trees
    for job in running:
        _kill_job_tree(job, signal.SIGTERM)

    # Phase 2: Wait for processes to exit
    if HAS_PSUTIL:
        # Use psutil.wait_procs for reliable multi-process waiting
        procs_to_wait = []
        for job in running:
            if job.pid:
                try:
                    procs_to_wait.append(psutil.Process(job.pid))
                except psutil.NoSuchProcess:
                    pass
        if procs_to_wait:
            _, alive = psutil.wait_procs(procs_to_wait, timeout=timeout)
            if alive:
                _log(f"  {len(alive)} process(es) still alive after {timeout}s")
    else:
        deadline = time.time() + timeout
        while time.time() < deadline:
            still_alive = [j for j in running if is_process_alive(j.pid)]
            if not still_alive:
                break
            time.sleep(1)

    # Phase 3: SIGKILL stragglers (entire process trees)
    for job in running:
        if is_process_alive(job.pid):
            _kill_job_tree(job, signal.SIGKILL)
            _log(f"Force-killed {job.name}")

    # Phase 4: Mark interrupted jobs as pending (will retry on restart)
    for job in running:
        job.status = "pending"
        job.pid = None
        job.pgid = None
        job.gpu_ids = None
        job.master_port = None
        if job._log_fh:
            try:
                job._log_fh.close()
            except Exception:
                pass
            job._log_fh = None

    save_state(state)
    _log("Shutdown complete. State saved.")


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------

def daemon_main(args: argparse.Namespace) -> None:
    """Run the daemon main loop (blocking, runs forever)."""
    gpus = [int(g) for g in args.gpus.split(",")]

    # Load or create state
    existing = load_state(args.state_dir)
    if existing:
        state = existing
        _log(f"Loaded existing state with {len(state.jobs)} job(s)")
        # Re-validate running jobs
        for job in state.jobs:
            if job.status == "running":
                if not is_process_alive(job.pid):
                    _log(f"Job {job.name} was 'running' but PID {job.pid} is dead, requeueing")
                    job.status = "pending"
                    job.retries += 1
                    job.pid = None
                    job.pgid = None
                    job.gpu_ids = None
                    job.master_port = None
    else:
        state = State(state_dir=args.state_dir)

    state.daemon.started_at = datetime.now().isoformat()
    state.daemon.pid = os.getpid()
    state.daemon.gpus = gpus
    state.daemon.max_concurrent = len(gpus)  # each job gets 1 GPU by default; overcommit shares all
    state.daemon.overcommit = bool(getattr(args, "overcommit", True))
    state.daemon.round_robin = bool(getattr(args, "round_robin", True))
    state.daemon.saturated = False
    state.daemon.saturation_time = None
    state.daemon.burst_phase = True
    state.daemon.burst_jobs_launched = 0
    state.daemon.saturation_checkpoints_passed = 0
    state.daemon.last_fifo_launch_time = None
    state.daemon.master_port_range = list(
        range(args.port_range[0], args.port_range[1] + 1)
    )
    # Store as [lo, hi] pair
    state.daemon.master_port_range = [args.port_range[0], args.port_range[1]]

    save_state(state)

    _log(f"Daemon started (PID={os.getpid()})")
    _log(f"  GPUs: {gpus}")
    _log(f"  GPU backend: {'pynvml (nvidia-ml-py)' if HAS_PYNVML else 'nvidia-smi subprocess (install pynvml for better performance)'}")
    _log(f"  Process backend: {'psutil' if HAS_PSUTIL else 'os.kill + /proc (install psutil for better reliability)'}")
    _log(f"  ⚡ DYNAMIC SCHEDULING: Health-gated burst with saturation checkpoints")
    _log(
        "  GPU sharing: "
        + ("overcommit enabled" if state.daemon.overcommit else "exclusive (one job per GPU group)")
    )
    _log(f"  Saturation checkpoint: every {len(gpus)} jobs (= num GPUs), "
         f"progressive wait = N × {args.stabilization_wait}s "
         f"(1st={args.stabilization_wait}s, 2nd={2*args.stabilization_wait}s, ...)")
    _log(f"  Burst GPU wait: {args.burst_gpu_wait}s (wait for GPU presence before next launch)")
    _log(f"  Burst delay: {args.burst_delay}s (settling time after GPU confirmed)")
    _log(f"  FIFO VRAM poll interval: {args.fifo_launch_cooldown}s")
    _log(f"  GPU check delay: {args.gpu_check_delay}s grace period")
    _log(f"  OOM threshold: {args.oom_threshold}s")
    _log(f"  Saturation cooldown: {args.saturation_cooldown}s")
    _log(f"  🧠 OOM prediction: safety_margin={args.memory_safety_margin*100:.0f}%, "
         f"wait_timeout={args.oom_wait_timeout:.0f}s "
         f"(avg memory/job + {args.memory_safety_margin*100:.0f}% must fit in free VRAM)")
    _log(f"  Port range: {args.port_range[0]}-{args.port_range[1]}")
    _log(f"  Inbox: {args.inbox_dir}")
    _log(f"  State: {args.state_dir}")
    _log(f"  Logs: {args.log_dir}")
    _log(f"  Working dir: {args.working_dir}")


    # Signal handlers
    def _handle_shutdown(signum, frame):
        _log(f"Received signal {signum}, shutting down...")
        graceful_shutdown(state)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # Reap zombie children
    def _handle_sigchld(signum, frame):
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break

    signal.signal(signal.SIGCHLD, _handle_sigchld)

    scheduler_log_times: Dict[str, float] = {}

    def _log_scheduler_event(
        key: str,
        msg: str,
        every: float = 60.0,
    ) -> None:
        now = time.time()
        last = scheduler_log_times.get(key)
        if last is None or (now - last) >= every:
            _log(msg)
            scheduler_log_times[key] = now

    # Main loop
    while True:
        changed = False

        # Step 1: Check inbox
        new_files = scan_inbox(args.inbox_dir)
        for fpath in new_files:
            try:
                new_jobs, _defaults = parse_input_yaml(fpath)

                # Enforce one GPUs-per-job shape per daemon.
                if new_jobs:
                    incoming_gpus_needed = int(new_jobs[0].gpus_needed)
                    if state.daemon.fixed_gpus_needed is None:
                        state.daemon.fixed_gpus_needed = incoming_gpus_needed
                        _log(
                            f"🔒 Fixed gpus_needed set to {incoming_gpus_needed} "
                            f"for this daemon"
                        )
                    elif incoming_gpus_needed != state.daemon.fixed_gpus_needed:
                        _log(
                            f"❌ Rejecting {os.path.basename(fpath)}: "
                            f"gpus_needed={incoming_gpus_needed} but daemon enforces "
                            f"{state.daemon.fixed_gpus_needed}"
                        )
                        move_to_processed(fpath)
                        changed = True
                        continue

                    mixed = [j.name for j in new_jobs if int(j.gpus_needed) != incoming_gpus_needed]
                    if mixed:
                        _log(
                            f"❌ Rejecting {os.path.basename(fpath)}: mixed gpus_needed in one file. "
                            f"Offending jobs: {mixed[:5]}"
                        )
                        move_to_processed(fpath)
                        changed = True
                        continue

                # Check for duplicate names
                existing_names = {j.name for j in state.jobs}
                for job in new_jobs:
                    if job.name in existing_names:
                        # Auto-suffix
                        suffix = 2
                        while f"{job.name}_{suffix}" in existing_names:
                            suffix += 1
                        old_name = job.name
                        job.name = f"{job.name}_{suffix}"
                        _log(f"Renamed duplicate '{old_name}' -> '{job.name}'")
                    existing_names.add(job.name)
                    state.jobs.append(job)
                _log(f"Ingested {len(new_jobs)} job(s) from {os.path.basename(fpath)}")
                pending_after_ingest = sum(
                    1 for j in state.jobs if j.status == "pending")
                running_after_ingest = sum(
                    1 for j in state.jobs if j.status == "running")
                next_pending_name = next(
                    (j.name for j in state.jobs if j.status == "pending"),
                    "none",
                )
                _log(
                    f"📥 Queue update: {pending_after_ingest} pending, "
                    f"{running_after_ingest} running after ingest; "
                    f"next up: {next_pending_name}")
                move_to_processed(fpath)
                changed = True
            except Exception as e:
                _log(f"Error parsing {fpath}: {e}")

        # Step 2: Check running jobs
        for job in state.jobs:
            if job.status != "running":
                continue

            # Check via Popen object first (more reliable), then PID
            alive = False
            exit_code = None
            if job._proc is not None:
                ret = job._proc.poll()
                if ret is None:
                    alive = True
                else:
                    exit_code = ret
            else:
                alive = is_process_alive(job.pid)
                if not alive:
                    # Try to get exit code
                    try:
                        _, status = os.waitpid(job.pid, os.WNOHANG)
                        if os.WIFEXITED(status):
                            exit_code = os.WEXITSTATUS(status)
                        elif os.WIFSIGNALED(status):
                            exit_code = -os.WTERMSIG(status)
                    except ChildProcessError:
                        exit_code = -1

            if alive:
                continue

            # Close log file handle
            if job._log_fh:
                try:
                    job._log_fh.close()
                except Exception:
                    pass
                job._log_fh = None



            # Validate exit_code == 0: check log for crash indicators.
            # torchrun can sometimes exit 0 even when workers failed
            # (e.g., elastic agent cleanup succeeds after worker crash).
            if exit_code == 0:
                crash_snippet = tail_log(job.log_file, lines=50)
                if crash_snippet:
                    crash_indicators = [
                        "ChildFailedError", "failed (exitcode:",
                        "SIGTERM", "SIGKILL",
                        "torch.distributed.elastic.multiprocessing.errors",
                        "Traceback (most recent call last)",
                        "DistBackendError", "wait timeout",
                        "NCCL error", "CUDA error",
                    ]
                    for indicator in crash_indicators:
                        if indicator in crash_snippet:
                            _log(f"⚠️ {job.name} exited with code 0 but log "
                                 f"contains '{indicator}' — treating as failure")
                            exit_code = 1  # override to non-zero
                            break

            if exit_code == 0:
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.exit_code = 0
                _log(f"✅ {job.name} completed successfully")
            else:
                job.exit_code = exit_code
                job.error_snippet = tail_log(job.log_file, lines=5)

                # Check if this was an OOM crash
                if is_oom_crash(job, oom_threshold=getattr(args, 'oom_threshold', DEFAULT_OOM_CRASH_THRESHOLD)):
                    # OOM crash — requeue WITHOUT counting as a retry
                    _requeue_job(job, f"OOM-killed while running (exit={exit_code})")
                    state.daemon.saturated = True
                    state.daemon.saturation_time = datetime.now().isoformat()
                    _log(f"🛑 {job.name} OOM-killed while running. "
                         f"Requeued (NOT counting as retry). "
                         f"Entering drain mode.")

                    # If still in burst phase, cascade requeue post-checkpoint jobs
                    if state.daemon.burst_phase:
                        cascade_requeue_since_checkpoint(
                            state, job.name,
                            getattr(args, 'oom_threshold', DEFAULT_OOM_CRASH_THRESHOLD))
                        state.daemon.burst_phase = False
                        _log(f"🔀 BURST → FIFO: OOM detected in main loop. "
                             f"Switching to timed VRAM polling mode.")
                elif job.retries < job.max_retries:
                    job.retries += 1
                    job.status = "pending"
                    job.pid = None
                    job.pgid = None
                    job.gpu_ids = None
                    job.master_port = None
                    _log(f"⚠️  {job.name} failed (exit={exit_code}), "
                         f"requeueing (retry {job.retries}/{job.max_retries})")
                else:
                    job.status = "failed"
                    job.completed_at = datetime.now().isoformat()
                    _log(f"❌ {job.name} permanently failed after "
                         f"{job.max_retries} retries (exit={exit_code})")
            changed = True

        # Step 3: Launch pending jobs
        #
        # FILL-PER-GPU SCHEDULING (signal-based, deterministic):
        #   Launch a job on GPU group [X..X+N-1] → wait for sentinel
        #   → check ALL GPUs in group for free memory → if room, launch
        #   another on the same group → if full, advance fill_gpu_index
        #   by gpus_needed to the next group → repeat.
        #
        # Phase 1 - BURST: Fill GPUs one at a time until all GPUs are
        #   full or all pending jobs are launched.
        # Phase 2 - FIFO: poll GPU VRAM on the configured interval and
        #   launch the next pending job when a full GPU group fits.
        if not state.daemon.paused:
            pending = [j for j in state.jobs if j.status == "pending"]
            running_count = sum(1 for j in state.jobs if j.status == "running")

            all_gpus = list(state.daemon.gpus)
            num_gpus = len(all_gpus)
            safety_margin = getattr(args, 'memory_safety_margin', DEFAULT_MEMORY_SAFETY_MARGIN)
            sentinel_timeout = getattr(args, 'sentinel_timeout', DEFAULT_SENTINEL_TIMEOUT)

            if state.daemon.burst_phase and not pending and state.daemon.burst_jobs_launched > 0:
                # All pending jobs launched in burst — transition to FIFO
                _log(f"🏁 Burst complete: all jobs launched "
                     f"({state.daemon.burst_jobs_launched} total). "
                     f"Switching to FIFO mode.")
                state.daemon.burst_phase = False
                state.daemon.last_launched_job_name = None
                changed = True

            elif state.daemon.burst_phase and pending:
                # --- BURST PHASE (Fill-Per-GPU, Signal-Based) ---
                #
                # Algorithm:
                #   1. If we are waiting for a previously launched job's
                #      sentinel, check for it. If not ready yet, skip this
                #      cycle (the main poll loop will come back).
                #   2. Once the sentinel is confirmed (or this is the first
                #      job), check if the current GPU has room for another.
                #   3. If yes → launch on the same GPU.
                #   4. If no  → advance fill_gpu_index to next GPU and launch there.
                #   5. If all GPUs are full → switch to FIFO.

                pending = [j for j in state.jobs if j.status == "pending"]

                # ─── STEP A: Wait for last-launched job's sentinel ───
                waiting_for_sentinel = False
                if state.daemon.last_launched_job_name:
                    last_job = next(
                        (j for j in state.jobs
                         if j.name == state.daemon.last_launched_job_name
                         and j.status == "running"),
                        None,
                    )
                    if last_job is not None:
                        # Check if the last job died while we were waiting
                        if last_job._proc and last_job._proc.poll() is not None:
                            exit_code = last_job._proc.poll()
                            last_job.exit_code = exit_code
                            last_job.error_snippet = tail_log(last_job.log_file, lines=5)
                            if last_job._log_fh:
                                try:
                                    last_job._log_fh.close()
                                except Exception:
                                    pass
                                last_job._log_fh = None
                            _log(f"💀 {last_job.name} died while waiting "
                                 f"for sentinel (exit={exit_code})")
                            is_oom = is_oom_crash(
                                last_job,
                                oom_threshold=getattr(
                                    args, 'oom_threshold',
                                    DEFAULT_OOM_CRASH_THRESHOLD))
                            if is_oom:
                                _requeue_job(
                                    last_job,
                                    f"OOM before sentinel (exit={exit_code})")
                                state.daemon.saturated = True
                                state.daemon.saturation_time = (
                                    datetime.now().isoformat())
                                _log(f"🔀 BURST → FIFO: OOM on GPU "
                                     f"{all_gpus[state.daemon.fill_gpu_index]}. "
                                     f"Switching to timed VRAM polling mode.")
                                state.daemon.burst_phase = False
                            elif last_job.retries < last_job.max_retries:
                                last_job.retries += 1
                                last_job.status = "pending"
                                last_job.pid = None
                                last_job.pgid = None
                                last_job.gpu_ids = None
                                last_job.master_port = None
                                _log(f"↩ {last_job.name} requeueing "
                                     f"(retry {last_job.retries}/"
                                     f"{last_job.max_retries})")
                            else:
                                last_job.status = "failed"
                                last_job.completed_at = (
                                    datetime.now().isoformat())
                                _log(f"❌ {last_job.name} permanently failed")
                            state.daemon.last_launched_job_name = None
                            changed = True

                        elif check_training_started(last_job):
                            # Sentinel confirmed — snapshot memory and
                            # clear the wait so we can proceed.
                            _snapshot_peak_memory(state)
                            _log(f"✅ {last_job.name} training started "
                                 f"(sentinel confirmed on GPU "
                                 f"{all_gpus[state.daemon.fill_gpu_index]})")
                            state.daemon.last_launched_job_name = None
                            changed = True

                        elif last_job.started_at:
                            # Still waiting — check if we exceeded timeout
                            try:
                                started = datetime.fromisoformat(
                                    last_job.started_at)
                                elapsed = (
                                    datetime.now() - started
                                ).total_seconds()
                            except Exception:
                                elapsed = 0
                            if elapsed > sentinel_timeout:
                                _log(f"⚠️ {last_job.name} sentinel not "
                                     f"found after {elapsed:.0f}s — "
                                     f"proceeding (fallback to GPU presence)")
                                if check_gpu_presence(last_job):
                                    _log(f"  ✅ {last_job.name} has GPU "
                                         f"presence, treating as started")
                                else:
                                    _log(f"  ⚠️ {last_job.name} no GPU "
                                         f"presence either, proceeding "
                                         f"cautiously")
                                state.daemon.last_launched_job_name = None
                                changed = True
                            else:
                                # Still within timeout — wait for next
                                # poll cycle
                                waiting_for_sentinel = True
                                _log_scheduler_event(
                                    "burst_waiting_for_sentinel",
                                    f"⏸ BURST hold: waiting for {last_job.name} "
                                    f"to confirm startup before queuing more "
                                    f"jobs ({len(pending)} pending behind it, "
                                    f"elapsed {elapsed:.0f}s/{sentinel_timeout:.0f}s)",
                                    every=max(30.0, args.poll_interval * 6),
                                )

                if not waiting_for_sentinel and state.daemon.burst_phase and pending:
                    # ─── STEP B: Determine target GPU group ──────────
                    # For multi-GPU jobs (gpus_needed > 1), we treat
                    # consecutive GPUs as a "group". fill_gpu_index
                    # always points to the FIRST GPU in the current
                    # group.  We check ALL GPUs in the group for
                    # available memory before launching.
                    #
                    # When a group is full, we advance by gpus_needed
                    # (not by 1) so that groups don't overlap.
                    # E.g. with gpus_needed=2 and GPUs [0..7]:
                    #   group 0 = [0,1], group 1 = [2,3], ...
                    next_job = pending[0]
                    gpj = next_job.gpus_needed  # peek at next job
                    fill_idx = state.daemon.fill_gpu_index
                    # Build the GPU group for the current fill_idx
                    target_group = [
                        all_gpus[(fill_idx + g) % num_gpus]
                        for g in range(gpj)
                    ]

                    # Check ALL GPUs in the group — the group can only fit
                    # another job if EVERY GPU has room. Use scheduler
                    # profiling for the pending job (with runtime fallback).
                    group_can_fit, group_reasons = can_place_job_on_gpu_group(
                        target_group,
                        state,
                        safety_margin,
                        gpus_needed=gpj,
                        job=next_job,
                    )
                    if group_can_fit:
                        _log_scheduler_event(
                            f"burst_capacity_{'-'.join(str(g) for g in target_group)}",
                            f"✅ BURST capacity available: {next_job.name} can be queued on "
                            f"GPU group {target_group} ({len(pending)} pending total)",
                            every=max(30.0, args.poll_interval * 6),
                        )

                    if not group_can_fit:
                        # Current group is full — try next group(s).
                        # Advance by gpus_needed each step so groups
                        # don't overlap.
                        advanced = False
                        num_groups = num_gpus // max(gpj, 1)
                        for step in range(1, max(num_groups, 1)):
                            next_idx = (
                                fill_idx + step * gpj
                            ) % num_gpus
                            next_group = [
                                all_gpus[(next_idx + g) % num_gpus]
                                for g in range(gpj)
                            ]
                            # Check all GPUs in next group
                            next_ok, next_reasons = can_place_job_on_gpu_group(
                                next_group,
                                state,
                                safety_margin,
                                gpus_needed=gpj,
                                job=next_job,
                            )
                            if next_ok:
                                state.daemon.fill_gpu_index = next_idx
                                target_group = next_group
                                fill_idx = next_idx
                                _log(f"➡️ Advanced to GPU group {target_group}")
                                advanced = True
                                break

                        if not advanced:
                            _log_scheduler_event(
                                "burst_all_groups_full",
                                f"⏸ BURST hold: {next_job.name} remains queued because "
                                f"all {num_groups} GPU group(s) are at capacity",
                                every=max(30.0, args.poll_interval * 6),
                            )
                            # ALL GPU groups full — switch to FIFO
                            _log(f"🔀 BURST → FIFO: All "
                                 f"{num_groups} GPU group(s) full. "
                                 f"Switching to timed VRAM polling mode.")
                            state.daemon.burst_phase = False
                            state.daemon.saturated = True
                            state.daemon.saturation_time = (
                                datetime.now().isoformat())
                            changed = True

                    # ─── STEP C: Launch the job on target GPU group ─
                    if state.daemon.burst_phase and pending:
                        job = pending[0]

                        # Health gate: verify all running jobs alive
                        dead_jobs = []
                        for rj in state.jobs:
                            if rj.status != "running":
                                continue
                            proc_alive = False
                            rj_exit_code = None
                            if rj._proc is not None:
                                ret = rj._proc.poll()
                                if ret is None:
                                    proc_alive = True
                                else:
                                    rj_exit_code = ret
                            else:
                                proc_alive = is_process_alive(rj.pid)
                            if not proc_alive:
                                dead_jobs.append((rj, rj_exit_code))

                        if dead_jobs:
                            _log(f"🛑 HEALTH GATE: {len(dead_jobs)} "
                                 f"job(s) died:")
                            any_oom = False
                            for dj, dj_exit in dead_jobs:
                                _log(f"   💀 {dj.name} (exit={dj_exit})")
                                dj.exit_code = dj_exit
                                dj.error_snippet = tail_log(
                                    dj.log_file, lines=5)
                                if dj._log_fh:
                                    try:
                                        dj._log_fh.close()
                                    except Exception:
                                        pass
                                    dj._log_fh = None
                                if is_oom_crash(
                                    dj,
                                    oom_threshold=getattr(
                                        args, 'oom_threshold',
                                        DEFAULT_OOM_CRASH_THRESHOLD)):
                                    any_oom = True
                                    _requeue_job(
                                        dj,
                                        "Died (OOM) — health gate")
                                elif dj.retries < dj.max_retries:
                                    dj.retries += 1
                                    dj.status = "pending"
                                    dj.pid = None
                                    dj.pgid = None
                                    dj.gpu_ids = None
                                    dj.master_port = None
                                    _log(f"   ↩ {dj.name} requeueing "
                                         f"(retry {dj.retries}/"
                                         f"{dj.max_retries})")
                                else:
                                    dj.status = "failed"
                                    dj.completed_at = (
                                        datetime.now().isoformat())
                                    _log(f"   ❌ {dj.name} permanently "
                                         f"failed")

                            if any_oom:
                                cascade_requeue_since_checkpoint(
                                    state,
                                    dead_jobs[0][0].name,
                                    getattr(
                                        args, 'oom_threshold',
                                        DEFAULT_OOM_CRASH_THRESHOLD))
                            _log(f"🔀 BURST → FIFO: Job(s) died. "
                                 f"Switching to timed VRAM polling mode.")
                            state.daemon.burst_phase = False
                            state.daemon.saturated = True
                            state.daemon.saturation_time = (
                                datetime.now().isoformat())
                            changed = True
                        else:
                            # All healthy — launch on target GPU
                            try:
                                port = allocate_port(state)
                                wd = job.working_dir or args.working_dir
                                # Fill-per-GPU-group: pin to current
                                # group (target_group was computed in
                                # STEP B above).
                                pin_gpus = list(target_group)
                                # Also keep rr_index in sync for
                                # estimate_oom_risk compatibility
                                state.daemon.overcommit_rr_index = (
                                    (fill_idx + job.gpus_needed)
                                    % num_gpus
                                )
                                launch_job(
                                    job, pin_gpus, port,
                                    args.log_dir, wd)
                                state.daemon.burst_jobs_launched += 1
                                job.launched_at_burst_index = (
                                    state.daemon.burst_jobs_launched)
                                state.daemon.last_launched_job_name = (
                                    job.name)
                                changed = True

                                # Round-robin: after a successful launch,
                                # advance fill_gpu_index by gpus_needed so
                                # the next burst job starts on the next GPU
                                # group. Without this, fill_gpu_index stays
                                # put until the current group OOMs, which
                                # crams all jobs onto the first GPU(s).
                                # Use --no-round-robin to restore the
                                # legacy fill-first behavior.
                                if state.daemon.round_robin:
                                    state.daemon.fill_gpu_index = (
                                        (fill_idx + job.gpus_needed)
                                        % num_gpus
                                    )

                                _log(
                                    f"🚀 BURST [{state.daemon.burst_jobs_launched}] "
                                    f"Launched {job.name} on GPU "
                                    f"{pin_gpus} (fill_gpu_index="
                                    f"{fill_idx}->{state.daemon.fill_gpu_index}, "
                                    f"total running: "
                                    f"{running_count + 1})")

                                # Quick sanity: died immediately?
                                time.sleep(1)
                                if (job._proc
                                        and job._proc.poll() is not None):
                                    exit_code = job._proc.poll()
                                    job.exit_code = exit_code
                                    _log(f"⚡ {job.name} died immediately"
                                         f" (exit={exit_code})")
                                    is_oom = is_oom_crash(
                                        job,
                                        oom_threshold=getattr(
                                            args, 'oom_threshold',
                                            DEFAULT_OOM_CRASH_THRESHOLD))
                                    _requeue_job(
                                        job,
                                        f"Died immediately "
                                        f"(exit={exit_code})")
                                    state.daemon.last_launched_job_name = (
                                        None)
                                    if is_oom:
                                        _log(
                                            f"🔀 BURST → FIFO: OOM on "
                                            f"GPU group {pin_gpus}.")
                                        state.daemon.burst_phase = False
                                        state.daemon.saturated = True
                                        state.daemon.saturation_time = (
                                            datetime.now().isoformat())
                                    else:
                                        job.retries += 1
                                        if job.retries > job.max_retries:
                                            job.status = "failed"
                                            job.completed_at = (
                                                datetime.now().isoformat())
                                else:
                                    # Job launched OK — save state and
                                    # wait for sentinel on next poll cycle
                                    save_state(state)

                            except Exception as e:
                                _log(f"Error launching {job.name}: {e}")
                                job.error_snippet = str(e)

            elif not state.daemon.burst_phase:
                # --- FIFO PHASE (timed VRAM polling) ---
                # Poll GPU memory on the configured interval and launch at
                # most one pending job if a GPU group can fit it.
                fifo_interval = getattr(
                    args,
                    'fifo_launch_cooldown',
                    DEFAULT_FIFO_LAUNCH_COOLDOWN,
                )

                if pending:
                    job = pending[0]
                    fifo_ready = state.daemon.last_fifo_check_time is None
                    wait_remaining = 0.0

                    if not fifo_ready:
                        try:
                            last_check = datetime.fromisoformat(
                                state.daemon.last_fifo_check_time)
                            elapsed_check = (
                                datetime.now() - last_check
                            ).total_seconds()
                            fifo_ready = elapsed_check >= fifo_interval
                            if not fifo_ready:
                                wait_remaining = max(
                                    0.0,
                                    fifo_interval - elapsed_check,
                                )
                        except Exception:
                            fifo_ready = True

                    if not fifo_ready:
                        _log_scheduler_event(
                            "fifo_poll_wait",
                            f"⏸ FIFO poll wait: {job.name} will be checked again in "
                            f"{wait_remaining:.0f}s ({len(pending)} pending, "
                            f"{running_count} running)",
                            every=max(30.0, args.poll_interval * 6),
                        )
                    else:
                        state.daemon.last_fifo_check_time = (
                            datetime.now().isoformat())

                        gpj = job.gpus_needed
                        num_groups = max(1, num_gpus // max(gpj, 1))
                        start_idx = state.daemon.fill_gpu_index
                        selected_group = None

                        for step in range(num_groups):
                            probe_idx = (start_idx + step * gpj) % num_gpus
                            probe_group = [
                                all_gpus[(probe_idx + g) % num_gpus]
                                for g in range(gpj)
                            ]
                            probe_ok, probe_reasons = can_place_job_on_gpu_group(
                                probe_group,
                                state,
                                safety_margin,
                                gpus_needed=gpj,
                                job=job,
                            )
                            if probe_ok:
                                selected_group = probe_group
                                state.daemon.fill_gpu_index = (
                                    (probe_idx + gpj) % num_gpus
                                )
                                break

                        if selected_group is None:
                            _log_scheduler_event(
                                "fifo_no_capacity",
                                f"⏸ FIFO poll: no GPU group can fit {job.name} right now; "
                                f"retrying in {fifo_interval:.0f}s",
                                every=max(30.0, args.poll_interval * 6),
                            )
                            state.daemon.saturated = True
                            state.daemon.saturation_time = (
                                datetime.now().isoformat())
                        else:
                            _log_scheduler_event(
                                f"fifo_capacity_{'-'.join(str(g) for g in selected_group)}",
                                f"✅ FIFO capacity available: {job.name} can be queued on "
                                f"GPU group {selected_group}",
                                every=max(30.0, args.poll_interval * 6),
                            )
                            state.daemon.overcommit_rr_index = (
                                (all_gpus.index(selected_group[0]) + gpj)
                                % num_gpus
                            )

                            try:
                                port = allocate_port(state)
                                wd = job.working_dir or args.working_dir
                                launch_job(
                                    job, selected_group, port,
                                    args.log_dir, wd)
                                changed = True
                                state.daemon.last_fifo_launch_time = (
                                    datetime.now().isoformat())
                                state.daemon.saturated = False
                                _log(
                                    f"🚀 FIFO: Launched {job.name} on GPU "
                                    f"{selected_group} after VRAM poll "
                                    f"({fifo_interval:.0f}s interval)")

                                # Quick verify
                                time.sleep(2)
                                if (job._proc
                                        and job._proc.poll() is not None):
                                    exit_code = job._proc.poll()
                                    job.exit_code = exit_code
                                    is_oom = is_oom_crash(
                                        job,
                                        oom_threshold=getattr(
                                            args, 'oom_threshold',
                                            DEFAULT_OOM_CRASH_THRESHOLD))
                                    _requeue_job(
                                        job,
                                        f"Died immediately in FIFO "
                                        f"(exit={exit_code})")
                                    if is_oom:
                                        _log(
                                            f"🛑 FIFO: {job.name} OOM "
                                            f"→ waiting for next VRAM poll")
                                        state.daemon.saturated = True
                                        state.daemon.saturation_time = (
                                            datetime.now().isoformat())
                                    else:
                                        job.retries += 1
                                        if job.retries > job.max_retries:
                                            job.status = "failed"
                                            job.completed_at = (
                                                datetime.now().isoformat())
                            except Exception as e:
                                _log(f"Error launching {job.name}: {e}")
                                job.error_snippet = str(e)

                # Track running count for legacy state compatibility
                state.daemon.prev_running = sum(
                    1 for j in state.jobs if j.status == "running")

        # Step 4: Persist state
        if changed:
            save_state(state)

        _log_scheduler_event(
            "scheduler_summary",
            f"📊 Scheduler: {sum(1 for j in state.jobs if j.status == 'pending')} pending, "
            f"{sum(1 for j in state.jobs if j.status == 'running')} running, "
            f"phase={'BURST' if state.daemon.burst_phase else 'FIFO'}",
            every=max(60.0, args.poll_interval * 12),
        )

        # Step 5: Sleep
        time.sleep(args.poll_interval)


# ---------------------------------------------------------------------------
# Client commands
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    """Copy job YAML into the daemon's inbox."""
    inbox_dir = args.inbox_dir
    os.makedirs(inbox_dir, exist_ok=True)

    if args.yaml_file:
        src = args.yaml_file
        if not os.path.exists(src):
            print(f"Error: {src} not found")
            sys.exit(1)
        basename = os.path.basename(src)
        dest = os.path.join(inbox_dir, basename)
        if os.path.exists(dest):
            ts = datetime.now().strftime("%H%M%S")
            name, ext = os.path.splitext(basename)
            dest = os.path.join(inbox_dir, f"{name}_{ts}{ext}")
        shutil.copy2(src, dest)
        print(f"Submitted {src} -> {dest}")
        print(f"Daemon will pick it up on next poll cycle.")
    elif args.cmd:
        # Single command submission
        job_name = args.name or f"cmd_{datetime.now().strftime('%H%M%S')}"
        data = {
            "defaults": {"gpus": args.gpus_needed or 1, "max_retries": DEFAULT_MAX_RETRIES},
            "jobs": [{"name": job_name, "cmd": args.cmd}],
        }
        dest = os.path.join(inbox_dir, f"{job_name}.yaml")
        with open(dest, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        print(f"Submitted single job '{job_name}' -> {dest}")
    else:
        print("Error: provide a YAML file or --cmd")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Show current job status from state.yaml."""
    while True:
        state = load_state(args.state_dir)
        if state is None:
            print("No state file found. Is the daemon running?")
            sys.exit(1)

        # Clear screen for watch mode
        if args.watch:
            os.system("clear")

        render_status(state)

        if not args.watch:
            break
        time.sleep(2)


def cmd_cancel(args: argparse.Namespace) -> None:
    """Cancel job(s)."""
    state = load_state(args.state_dir)
    if state is None:
        print("No state file found.")
        sys.exit(1)

    cancelled = 0
    for job in state.jobs:
        should_cancel = False
        if args.all_pending and job.status == "pending":
            should_cancel = True
        elif args.all and job.status in ("pending", "running"):
            should_cancel = True
        elif args.job_name and job.name == args.job_name:
            should_cancel = True

        if should_cancel:
            if job.status == "running":
                _kill_job_tree(job, signal.SIGTERM)
            job.status = "cancelled"
            job.completed_at = datetime.now().isoformat()
            cancelled += 1

    save_state(state)
    print(f"Cancelled {cancelled} job(s).")


def cmd_pause(args: argparse.Namespace) -> None:
    """Pause the daemon (stop launching new jobs)."""
    state = load_state(args.state_dir)
    if state is None:
        print("No state file found.")
        sys.exit(1)
    state.daemon.paused = True
    save_state(state)
    print("Daemon paused. Running jobs will continue, no new jobs will launch.")


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume the daemon."""
    state = load_state(args.state_dir)
    if state is None:
        print("No state file found.")
        sys.exit(1)
    state.daemon.paused = False
    save_state(state)
    print("Daemon resumed.")


def cmd_retry(args: argparse.Namespace) -> None:
    """Retry failed job(s) by resetting them to pending."""
    state = load_state(args.state_dir)
    if state is None:
        print("No state file found.")
        sys.exit(1)

    retried = 0
    for job in state.jobs:
        should_retry = False
        if args.all_failed and job.status == "failed":
            should_retry = True
        elif args.job_name and job.name == args.job_name and job.status == "failed":
            should_retry = True

        if should_retry:
            job.status = "pending"
            job.exit_code = None
            job.completed_at = None
            job.pid = None
            job.pgid = None
            job.gpu_ids = None
            job.master_port = None
            retried += 1

    save_state(state)
    print(f"Reset {retried} failed job(s) to pending.")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _parse_port_range(s: str) -> Tuple[int, int]:
    """Parse '29500-29599' into (29500, 29599)."""
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid port range: {s} (expected LO-HI)")
    return int(parts[0]), int(parts[1])


def _resolve_schedule_dirs(args: argparse.Namespace) -> None:
    """Resolve state/inbox/log dirs with schedule + node isolation.

    Priority:
      1) If ``--schedule-name`` is set, base path is ``schedules/<name>``.
      2) If ``--node-index`` is set, append ``node_<index>`` under that base.
      3) Only override dirs still equal to defaults.

    This prevents race/collision on shared filesystems when running one daemon
    per node.
    """
    sname = getattr(args, "schedule_name", None)
    node_index = getattr(args, "node_index", None)

    if sname:
        base = os.path.join(SCHEDULES_ROOT, sname)
    else:
        base = None

    if node_index is not None:
        node_suffix = f"node_{int(node_index)}"
        if base is None:
            state_base = os.path.join(DEFAULT_STATE_DIR, node_suffix)
            inbox_base = os.path.join(DEFAULT_INBOX_DIR, node_suffix)
            log_base = os.path.join(DEFAULT_LOG_DIR, node_suffix)
        else:
            node_base = os.path.join(base, node_suffix)
            state_base = os.path.join(node_base, "state")
            inbox_base = os.path.join(node_base, "inbox")
            log_base = os.path.join(node_base, "logs")
    else:
        if base is None:
            return
        state_base = os.path.join(base, "state")
        inbox_base = os.path.join(base, "inbox")
        log_base = os.path.join(base, "logs")

    if getattr(args, "state_dir", DEFAULT_STATE_DIR) == DEFAULT_STATE_DIR:
        args.state_dir = state_base
    if getattr(args, "inbox_dir", DEFAULT_INBOX_DIR) == DEFAULT_INBOX_DIR:
        args.inbox_dir = inbox_base
    if getattr(args, "log_dir", DEFAULT_LOG_DIR) == DEFAULT_LOG_DIR:
        args.log_dir = log_base


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Job daemon for GPU training orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Top-level schedule-name flag (before subcommands)
    p.add_argument(
        "-s", "--schedule-name",
        default=None,
        help="Schedule name — isolates all state/inbox/logs under "
             "schedules/<name>/. Prevents stale state collisions "
             "between different experiment runs.",
    )
    p.add_argument(
        "--node-index",
        type=int,
        default=None,
        help="Node index for per-daemon directory isolation on shared FS. "
             "When set, default state/inbox/log dirs are mapped to node-specific paths.",
    )
    sub = p.add_subparsers(dest="command", help="Available commands")

    # -- start --
    sp = sub.add_parser("start", help="Start the daemon (run in tmux)")
    sp.add_argument("--gpus", required=True, type=str,
                    help="Comma-separated GPU IDs (e.g. 0,1,2,3,4,5,6,7)")
    sp.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                    help=f"Default max retries (default: {DEFAULT_MAX_RETRIES})")
    sp.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                    help=f"Seconds between checks (default: {DEFAULT_POLL_INTERVAL})")
    sp.add_argument("--port-range", type=_parse_port_range, default=DEFAULT_PORT_RANGE,
                    help="Port range for torchrun (default: 29500-29599)")
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                    help=f"State directory (default: {DEFAULT_STATE_DIR})")
    sp.add_argument("--inbox-dir", default=DEFAULT_INBOX_DIR,
                    help=f"Inbox directory (default: {DEFAULT_INBOX_DIR})")
    sp.add_argument("--log-dir", default=DEFAULT_LOG_DIR,
                    help=f"Job log directory (default: {DEFAULT_LOG_DIR})")
    sp.add_argument("--working-dir", default=".",
                    help="Working directory for subprocesses (default: cwd)")

    sp.add_argument("--oom-threshold", type=float, default=DEFAULT_OOM_CRASH_THRESHOLD,
                    help=f"If a job dies within this many seconds, treat as OOM "
                         f"(default: {DEFAULT_OOM_CRASH_THRESHOLD})")
    sp.add_argument("--saturation-cooldown", type=float, default=DEFAULT_SATURATION_COOLDOWN,
                    help=f"Seconds to wait after OOM before trying next launch "
                         f"(default: {DEFAULT_SATURATION_COOLDOWN})")
    sp.add_argument("--burst-delay", type=float, default=DEFAULT_BURST_DELAY,
                    help=f"Seconds between consecutive burst launches "
                         f"(default: {DEFAULT_BURST_DELAY})")
    sp.add_argument("--fifo-launch-cooldown", type=float, default=DEFAULT_FIFO_LAUNCH_COOLDOWN,
                    help=f"Seconds between FIFO VRAM polls / launch attempts after burst "
                         f"(default: {DEFAULT_FIFO_LAUNCH_COOLDOWN})")
    sp.add_argument("--gpu-check-delay", type=float, default=DEFAULT_GPU_CHECK_DELAY,
                    help=f"Grace period (seconds) before checking if PID is on GPU "
                         f"(default: {DEFAULT_GPU_CHECK_DELAY})")
    sp.add_argument("--burst-gpu-wait", type=float, default=DEFAULT_BURST_GPU_WAIT,
                    help=f"Max seconds to wait for a job to appear on GPU before "
                         f"launching the next one in burst mode "
                         f"(default: {DEFAULT_BURST_GPU_WAIT})")
    sp.add_argument("--stabilization-wait", type=float, default=DEFAULT_STABILIZATION_WAIT,
                    help=f"Base seconds for saturation checkpoint wait. Actual wait "
                         f"increases progressively: 1st checkpoint = 1×base, "
                         f"2nd = 2×base, 3rd = 3×base, etc. "
                         f"(default: {DEFAULT_STABILIZATION_WAIT}s = 5 min)")
    sp.add_argument("--memory-safety-margin", type=float, default=DEFAULT_MEMORY_SAFETY_MARGIN,
                    help=f"Safety margin for OOM prediction. If free GPU memory < "
                         f"avg_per_job × (1 + margin), predict OOM and wait/queue. "
                         f"E.g. 0.15 = 15%% headroom. "
                         f"(default: {DEFAULT_MEMORY_SAFETY_MARGIN})")
    sp.add_argument("--oom-wait-timeout", type=float, default=DEFAULT_OOM_WAIT_TIMEOUT,
                    help=f"Max seconds to wait for GPU memory headroom before "
                         f"switching to queue mode. "
                         f"(default: {DEFAULT_OOM_WAIT_TIMEOUT}s = 10 min)")
    sp.add_argument("--sentinel-timeout", type=float, default=DEFAULT_SENTINEL_TIMEOUT,
                    help=f"Max seconds to wait for a job's .training_started sentinel "
                         f"before falling back to GPU presence detection. "
                         f"(default: {DEFAULT_SENTINEL_TIMEOUT}s = 10 min)")
    sp.add_argument("--round-robin", dest="round_robin",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Round-robin GPU placement in burst phase: advance "
                         "fill_gpu_index by gpus_needed after every launch so "
                         "jobs spread evenly across all GPUs instead of "
                         "stacking on one GPU until it's full. Use "
                         "--no-round-robin for the legacy fill-first behavior "
                         "(fill GPU 0 to capacity, then GPU 1, etc.). "
                         "Default: on.")
    sp.add_argument(
        "--overcommit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow multiple jobs to share the same GPU group when memory "
             "permits (default: on). Use --no-overcommit for exclusive "
             "placement: at most one running job may occupy a GPU group, "
             "regardless of free VRAM.",
    )


    # -- submit --
    sp = sub.add_parser("submit", help="Submit job(s)")
    sp.add_argument("yaml_file", nargs="?", default=None,
                    help="Path to input YAML file with job definitions")
    sp.add_argument("--cmd", type=str, default=None,
                    help="Submit a single command")
    sp.add_argument("--name", type=str, default=None,
                    help="Job name (for --cmd mode)")
    sp.add_argument("--gpus-needed", type=int, default=None,
                    help="GPUs needed (for --cmd mode)")
    sp.add_argument("--inbox-dir", default=DEFAULT_INBOX_DIR)
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR)

    # -- status --
    sp = sub.add_parser("status", help="Show job status")
    sp.add_argument("--watch", action="store_true",
                    help="Auto-refresh every 2 seconds")
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR)

    # -- cancel --
    sp = sub.add_parser("cancel", help="Cancel job(s)")
    sp.add_argument("job_name", nargs="?", default=None,
                    help="Name of job to cancel")
    sp.add_argument("--all-pending", action="store_true",
                    help="Cancel all pending jobs")
    sp.add_argument("--all", action="store_true",
                    help="Cancel all pending and running jobs")
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR)

    # -- pause --
    sp = sub.add_parser("pause", help="Pause launching new jobs")
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR)

    # -- resume --
    sp = sub.add_parser("resume", help="Resume launching new jobs")
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR)

    # -- retry --
    sp = sub.add_parser("retry", help="Retry failed job(s)")
    sp.add_argument("job_name", nargs="?", default=None,
                    help="Name of failed job to retry")
    sp.add_argument("--all-failed", action="store_true",
                    help="Retry all failed jobs")
    sp.add_argument("--state-dir", default=DEFAULT_STATE_DIR)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Resolve schedule/node context → state/inbox/log dirs
    _resolve_schedule_dirs(args)

    dispatch = {
        "start": daemon_main,
        "submit": cmd_submit,
        "status": cmd_status,
        "cancel": cmd_cancel,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "retry": cmd_retry,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
