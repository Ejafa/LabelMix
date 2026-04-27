#!/usr/bin/env python3
"""Smart job scheduler with memory profiling and class-balanced distribution.

This scheduler keeps responsibilities minimal and explicit:

1. Analyze all jobs and identify unique model configurations (VRAM classes).
2. Profile each unique model for peak GPU memory on the LOCAL node
   (with cache reuse).
3. Statically distribute jobs class-balanced round-robin across ``N`` nodes
   so each node gets the same number of jobs per VRAM class (±1).  With
   3 seeds per (model, trial), each node naturally handles ~1 seed.
4. Attach model/memory metadata to each job in per-node YAML output.
5. Leave GPU placement and runtime admission control to ``jobdaemon.py``.

The scheduler does NOT talk to remote nodes: it only needs a node *count*.
The count is taken from ``--num-nodes`` or, if omitted, from the
``$HOST_NUM`` environment variable (falling back to 1).  Profiling is
always done on whichever machine runs the scheduler; the resulting
per-node YAML files are then copied/submitted to the individual nodes by
the user.

Generated artifacts (explicit contract)::

    1) <output-dir>/<input-basename>_node_<idx>_jobs.yaml
       - Purpose: runtime input for ``jobdaemon.py submit``.
       - Consumed by: ``jobdaemon.py`` (inbox ingestion path).
       - Filename is derived from the INPUT yaml's basename, so
         ``--input vit.yaml`` produces ``vit_node_0_jobs.yaml`` etc.

    2) schedules/<schedule>/memory_profile.yaml
       - Purpose: scheduler cache for model peak-memory profiling.
       - Consumed by: future ``job_scheduler.py`` runs only.
       - Not consumed by: ``jobdaemon.py``.

    3) schedules/<schedule>/debug/schedule_info.yaml
       - Purpose: human-readable audit/debug snapshot of one scheduling run.
       - Consumed by: nothing in runtime pipeline (debug-only metadata).
       - Not required for daemon ``start/submit/status``.

    4) schedules/<schedule>/debug/profile_logs/profile_<model_key>.log
       - Purpose: profiler stdout/stderr logs for troubleshooting.
       - Consumed by: humans only (debug-only).

Usage::

    # Explicit node count:
    python job_scheduler.py --input vit.yaml --num-nodes 3 --schedule-name in1k_main

    # Pick up $HOST_NUM from the environment (defaults to 1 if unset):
    HOST_NUM=3 python job_scheduler.py --input vit.yaml --schedule-name in1k_main

    # Preview without writing files
    python job_scheduler.py --input vit.yaml --num-nodes 3 \\
        --schedule-name in1k_main --dry-run

    # Then on each node, start daemon (dynamic GPU scheduling):
    #   python jobdaemon.py -s in1k_main --node-index 0 start --gpus 0,1,...,7
    #   python jobdaemon.py -s in1k_main --node-index 0 submit vit_node_0_jobs.yaml
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
from typing import Any, Dict, List, Optional, Tuple

import yaml

SCHEDULES_ROOT = "./schedules"
PROFILE_CACHE_FILENAME = "memory_profile.yaml"
DEBUG_SUBDIR_NAME = "debug"
PROFILE_LOGS_SUBDIR_NAME = "profile_logs"
SCHEDULE_INFO_FILENAME = "schedule_info.yaml"

# Batch size assumed by any legacy profile-cache entry that only stores a
# plain ``<model_key>`` (no ``__bs<N>`` suffix).  Jobs whose batch size
# differs from this default trigger a fresh profile run, while jobs at
# this batch size continue to read the legacy entries unchanged.
DEFAULT_BATCH_SIZE = 1024

# ---------------------------------------------------------------------------
# GPU memory query
# ---------------------------------------------------------------------------

def get_local_gpu_free_memory() -> List[Tuple[int, int, int]]:
    """Query local GPU memory via pynvml or nvidia-smi.

    Returns list of (gpu_index, total_mib, free_mib).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        result = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total = mem.total // (1024 * 1024)
            free = mem.free // (1024 * 1024)
            result.append((i, total, free))
        pynvml.nvmlShutdown()
        return result
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return []
        result = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                result.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return result
    except Exception:
        return []


def query_local_node_memory() -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """Query the LOCAL node's GPU memory.

    Returns (num_gpus, total_free_mib, per_gpu_info).  The scheduler only
    ever needs to know about the local machine's GPUs (because profiling
    is local); remote node introspection has been removed intentionally
    — the user supplies a node *count*, not addresses.
    """
    print(f"  📡 local node — querying GPU memory...")
    info = get_local_gpu_free_memory()

    if not info:
        print(f"  ❌ local node: could not query GPU memory")
        return 0, 0, []

    total_free = sum(free for _, _, free in info)
    return len(info), total_free, info


# ---------------------------------------------------------------------------
# Model extraction from job commands
# ---------------------------------------------------------------------------

def extract_model_key(job: Dict[str, Any]) -> str:
    """Extract a model identifier from a job's command string.

    The model key is derived from the ``--config`` path in the command.
    Jobs with the same config file will have the same memory footprint,
    so they can be grouped together.

    Falls back to the full command hash if no config is found.
    """
    cmd = job.get("cmd", "")
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    # Look for --config, --config-file or -c flag
    for i, tok in enumerate(tokens):
        if tok in ("--config", "--config-file", "-c") and i + 1 < len(tokens):
            config_path = tokens[i + 1]
            # Use the basename without extension as the model key
            return os.path.splitext(os.path.basename(config_path))[0]
        if tok.startswith("--config=") or tok.startswith("--config-file="):
            config_path = tok.split("=", 1)[1]
            return os.path.splitext(os.path.basename(config_path))[0]

    # Fallback: look for --model flag
    for i, tok in enumerate(tokens):
        if tok == "--model" and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith("--model="):
            return tok.split("=", 1)[1]

    # Last resort: hash of command to group identical commands
    import hashlib
    return f"unknown_{hashlib.md5(cmd.encode()).hexdigest()[:8]}"


def extract_num_steps(job: Dict[str, Any]) -> Optional[int]:
    """Extract the --num-steps value from a job's command string."""
    cmd = job.get("cmd", "")
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    for i, tok in enumerate(tokens):
        if tok == "--num-steps" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return None
        if tok.startswith("--num-steps="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def extract_batch_size(job: Dict[str, Any]) -> Optional[int]:
    """Extract the --batch-size value from a job's command string."""
    cmd = job.get("cmd", "")
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    for i, tok in enumerate(tokens):
        if tok in ("--batch-size", "--batch_size") and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return None
        if tok.startswith("--batch-size=") or tok.startswith("--batch_size="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return None
        # Detectron2 LazyConfig override form (positional key=value).
        if tok.startswith("dataloader.train.total_batch_size="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def extract_profile_key(job: Dict[str, Any]) -> str:
    """Return the cache key used for memory profiling.

    - For jobs at the default batch size (``DEFAULT_BATCH_SIZE``) or with
      an unparseable batch size, this is just the plain ``model_key``.
      This keeps all pre-existing ``memory_profile.yaml`` entries valid.
    - For jobs at any other batch size, we suffix ``__bs<N>`` so the
      scheduler treats them as a separate VRAM class and runs a fresh
      profile when the entry is missing from the cache.
    """
    model_key = extract_model_key(job)
    bs = extract_batch_size(job)
    if bs is None or bs == DEFAULT_BATCH_SIZE:
        return model_key
    return f"{model_key}__bs{bs}"


# ---------------------------------------------------------------------------
# Memory profiling via short test runs
# ---------------------------------------------------------------------------

def _pick_free_port() -> int:
    """Ask the kernel for an ephemeral TCP port that is currently free.

    Used to avoid rendezvous-endpoint collisions between consecutive (or
    concurrent) profile runs.  A previously-used port can linger in
    ``TIME_WAIT`` for ~60s, which causes the next ``torchrun`` to either
    fail to bind or, worse, silently attach to the stale server and then
    produce ``sendBytes failed ... Broken pipe`` /
    ``TCPStore server has shut down too early`` warnings.

    Note: there is a tiny TOCTOU window between closing this socket and
    ``torchrun`` binding the port.  In practice this is not an issue for
    short-lived profiling runs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_detectron2_cmd(cmd: str) -> bool:
    """Heuristic: does this job use the detectron2 ``train_net.py`` entry?

    The timm-based jobs use ``torchrun ... train.py --config ...`` with
    ``--num-steps`` / ``--batch-size`` dash-flags, whereas detectron2
    ViTDet jobs use ``python train_net.py --config-file ...`` with
    positional LazyConfig ``key=value`` overrides.  We detect the latter
    by the presence of ``--config-file`` / ``train_net.py`` / ``--num-gpus``
    which are unique to the detectron2 entry point.
    """
    return (
        "--config-file" in cmd
        or "train_net.py" in cmd
        or "--num-gpus" in cmd
    )


def _build_profile_cmd_detectron2(
    job_cmd: str,
    profile_steps: int,
    gpu_ids: List[int],
) -> Tuple[str, Dict[str, str]]:
    """Build a short profile run for a detectron2 ``train_net.py`` job.

    Strategy:
    - Resolve ``{gpus}`` / ``{port}`` placeholders the usual way.
    - Replace ``--num-gpus {gpus}`` with the chosen GPU count.
    - Rewrite / inject LazyConfig overrides to cap the run:
        * ``train.max_iter=<profile_steps>``
        * ``train.eval_period=0``
        * ``train.log_period=1``
        * ``train.checkpointer.period=10000000`` (effectively disabled)
        * ``train.output_dir=/tmp/profile_runs/<model_key>``
    - Keep the original ``train.init_checkpoint`` so profiling reflects
      realistic memory (the backbone weights get loaded).
    - Set ``CUDA_VISIBLE_DEVICES`` to the chosen GPUs and
      ``--num-gpus`` to ``len(gpu_ids)`` — detectron2's ``launch()``
      picks the first N visible devices.

    Unlike the timm profiler, we do not touch any CLI dash-flags beyond
    the placeholders — all ViTDet hyperparameters live in the LazyConfig
    and are adjusted via ``key=value`` overrides appended to the cmd.
    """
    # Resolve placeholders
    cmd = job_cmd.replace("{gpus}", str(len(gpu_ids)))
    port = _pick_free_port()
    cmd = cmd.replace("{port}", str(port))
    cmd = cmd.replace("{gpu_ids}", ",".join(str(g) for g in gpu_ids))
    cmd = cmd.replace("{job_name}", "profile_run")
    cmd = cmd.replace("{log_file}", "/dev/null")

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    # Drop (override) any key=value tokens the user set — we rebuild them
    # ourselves so the profile run is small and self-contained.  Everything
    # else (``python``, ``train_net.py``, ``--config-file FOO``,
    # ``--num-gpus N``, ``--dist-url ...``) is preserved.
    _SUPPRESS_KEYS = {
        "train.max_iter",
        "train.eval_period",
        "train.log_period",
        "train.checkpointer.period",
        "train.output_dir",
    }

    def _is_suppressed_override(tok: str) -> bool:
        if "=" not in tok:
            return False
        key = tok.split("=", 1)[0]
        return key in _SUPPRESS_KEYS

    filtered: List[str] = [t for t in tokens if not _is_suppressed_override(t)]

    # Derive a model-key fragment for the profile output dir.
    model_key = "profile"
    for i, tok in enumerate(filtered):
        if tok in ("--config-file", "--config") and i + 1 < len(filtered):
            model_key = os.path.splitext(os.path.basename(filtered[i + 1]))[0]
            break
        if tok.startswith("--config-file=") or tok.startswith("--config="):
            model_key = os.path.splitext(os.path.basename(tok.split("=", 1)[1]))[0]
            break

    profile_output_dir = f"/tmp/profile_runs/{model_key}"

    # Append short-run overrides.  These go LAST so they win against any
    # that might have been baked into the config file.
    filtered.extend([
        f"train.max_iter={profile_steps}",
        "train.eval_period=0",
        "train.log_period=1",
        "train.checkpointer.period=10000000",
        f"train.output_dir={profile_output_dir}",
    ])

    # Defensive Hydra-override quoting.
    #
    # Hydra's LazyConfig override parser rejects an un-quoted ``=`` inside
    # the VALUE of a ``key=value`` token with
    #   OverrideParseException: mismatched input '=' expecting <EOF>
    # This bites us for checkpoint paths generated by the backup pipeline
    # (e.g. ``…/baseline_seed=42__runid-xyz.pth``).  We re-wrap any such
    # value in double quotes so Hydra takes it as a literal string.
    # ``shlex.split``/``shlex.quote`` preserve the inner quotes intact
    # because the outer quoting they use is single-quotes.
    def _quote_bad_override(tok: str) -> str:
        if "=" not in tok or tok.startswith("--"):
            return tok
        key, _, value = tok.partition("=")
        # Only re-quote if the value still contains an '=' AND it isn't
        # already wrapped in matching quotes.
        if "=" not in value:
            return tok
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return tok
        escaped = value.replace('"', r"\"")
        return f'{key}="{escaped}"'

    filtered = [_quote_bad_override(t) for t in filtered]

    # Force the subprocess to use the SAME Python interpreter the scheduler
    # is running under, instead of resolving the bare word "python" via PATH.
    # Without this pin, a user who `conda activate det2`-ed in one shell and
    # launched the scheduler from a different one (or under any wrapper that
    # scrubs PATH) will see the child resolve `python` to the system binary
    # and fail with `ModuleNotFoundError: No module named 'detectron2'`,
    # even though the env itself is fully installed.
    if filtered and filtered[0] in ("python", "python3"):
        filtered[0] = sys.executable

    resolved_cmd = " ".join(shlex.quote(t) for t in filtered)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # Keep HF datasets offline to avoid surprise network hits, matching
    # the timm profile builder.
    env.setdefault("HF_DATASETS_OFFLINE", "1")

    return resolved_cmd, env


def _build_profile_cmd(
    job_cmd: str,
    profile_steps: int,
    gpu_ids: List[int],
) -> Tuple[str, Dict[str, str]]:
    """Build a short test-run command from a job's command template.

    Dispatches to the detectron2-dialect builder when the job uses
    ``train_net.py`` / ``--config-file`` / ``--num-gpus``, and otherwise
    falls back to the original timm-dialect builder below.

    Returns (resolved_cmd, env_dict).
    """
    if _is_detectron2_cmd(job_cmd):
        return _build_profile_cmd_detectron2(job_cmd, profile_steps, gpu_ids)
    return _build_profile_cmd_timm(job_cmd, profile_steps, gpu_ids)


def _build_profile_cmd_timm(
    job_cmd: str,
    profile_steps: int,
    gpu_ids: List[int],
) -> Tuple[str, Dict[str, str]]:
    """Build a short test-run command from a job's command template.

    Modifies the command to:
    - Run only ``profile_steps`` training steps (override --num-steps)
    - Disable wandb logging (--log-wandb → removed)
    - Disable checkpointing
    - Resolve {gpus} and {port} placeholders

    Returns (resolved_cmd, env_dict).
    """
    # Resolve placeholders
    cmd = job_cmd.replace("{gpus}", str(len(gpu_ids)))
    # Pick a free port at launch time to avoid rendezvous collisions
    # between consecutive/concurrent profile runs (see _pick_free_port).
    port = _pick_free_port()
    cmd = cmd.replace("{port}", str(port))
    cmd = cmd.replace("{gpu_ids}", ",".join(str(g) for g in gpu_ids))
    cmd = cmd.replace("{job_name}", "profile_run")
    cmd = cmd.replace("{log_file}", "/dev/null")

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    # Override/inject arguments
    new_tokens = []
    skip_next = False
    removed_flags = set()

    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        # Override --num-steps
        if tok == "--num-steps" and i + 1 < len(tokens):
            new_tokens.append("--num-steps")
            new_tokens.append(str(profile_steps))
            skip_next = True
            removed_flags.add("--num-steps")
            continue
        if tok.startswith("--num-steps="):
            new_tokens.append(f"--num-steps={profile_steps}")
            removed_flags.add("--num-steps")
            continue

        # Disable checkpoint saving (override num_saves to 0)
        if tok == "--num-saves" and i + 1 < len(tokens):
            new_tokens.append("--num-saves")
            new_tokens.append("0")
            skip_next = True
            removed_flags.add("--num-saves")
            continue
        if tok.startswith("--num-saves="):
            new_tokens.append("--num-saves=0")
            removed_flags.add("--num-saves")
            continue

        # Disable eval (override num_evals to 0)
        if tok == "--num-evals" and i + 1 < len(tokens):
            new_tokens.append("--num-evals")
            new_tokens.append("0")
            skip_next = True
            removed_flags.add("--num-evals")
            continue
        if tok.startswith("--num-evals="):
            new_tokens.append("--num-evals=0")
            removed_flags.add("--num-evals")
            continue

        # Disable recovery
        if tok == "--check-resume":
            continue
        if tok == "--check_resume":
            continue

        # Ignore/remove --log-wandb token
        if tok == "--log-wandb":
            continue
        if tok == "--log_wandb":
            continue

        # Disable EMA (Exponential Moving Average)
        if tok == "--model-ema":
            continue
        if tok == "--model_ema":
            continue
        if tok.startswith("--model-ema="):
            continue
        if tok.startswith("--model_ema="):
            continue

        # Set warmup_steps to 0
        if tok == "--warmup-steps" and i + 1 < len(tokens):
            new_tokens.append("--warmup-steps")
            new_tokens.append("0")
            skip_next = True
            continue
        if tok.startswith("--warmup-steps="):
            new_tokens.append("--warmup-steps=0")
            continue

        new_tokens.append(tok)

    # Inject --num-steps if not already present
    if "--num-steps" not in removed_flags:
        new_tokens.append("--num-steps")
        new_tokens.append(str(profile_steps))

    # Inject --no-log-wandb if we didn't see --log-wandb
    # (it might be in the config YAML, but CLI override takes precedence)
    if "--log-wandb" not in cmd and "--log_wandb" not in cmd:
        new_tokens.append("--no-log-wandb")

    # Inject --no-model-ema to ensure EMA is disabled
    # (even if it's enabled in the config YAML, CLI override takes precedence)

    # Inject num-saves=0 and num-evals=0 if not already handled
    if "--num-saves" not in removed_flags:
        new_tokens.append("--num-saves")
        new_tokens.append("0")
    if "--num-evals" not in removed_flags:
        new_tokens.append("--num-evals")
        new_tokens.append("0")

    # Inject warmup-steps=0 if not already handled
    if "--warmup-steps" not in removed_flags:
        new_tokens.append("--warmup-steps")
        new_tokens.append("0")

    # Use a temp output dir for profiling
    final_tokens = []
    skip_next2 = False
    for i, tok in enumerate(new_tokens):
        if skip_next2:
            skip_next2 = False
            continue
        if tok == "--output" and i + 1 < len(new_tokens):
            final_tokens.append("--output")
            final_tokens.append("/tmp/profile_runs")
            skip_next2 = True
            continue
        if tok.startswith("--output="):
            final_tokens.append("--output=/tmp/profile_runs")
            continue
            final_tokens.append(tok)

    # Mirror the detectron2 profiler: pin `python` to the scheduler's own
    # interpreter so we never depend on whatever `python` happens to be
    # first on PATH inside the subprocess.
    if final_tokens and final_tokens[0] in ("python", "python3"):
        final_tokens[0] = sys.executable

    resolved_cmd = " ".join(shlex.quote(t) for t in final_tokens)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["HF_DATASETS_OFFLINE"] = "1"

    return resolved_cmd, env


def _get_peak_gpu_memory(gpu_ids: List[int]) -> Dict[int, int]:
    """Query current GPU memory usage for the given GPU indices.

    Returns {gpu_index: used_mib}.
    """
    result = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for gid in gpu_ids:
            h = pynvml.nvmlDeviceGetHandleByIndex(gid)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            result[gid] = mem.used // (1024 * 1024)
        pynvml.nvmlShutdown()
        return result
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    idx = int(parts[0])
                    if idx in gpu_ids:
                        result[idx] = int(parts[1])
    except Exception:
        pass
    return result


def select_profile_gpu_ids(profile_gpu: int, gpus_needed: int) -> List[int]:
    """Select concrete local GPU indices for profiling.

    For multi-GPU jobs, returns ``gpus_needed`` indices starting from
    ``profile_gpu`` (wrapping within the locally detected GPU list).
    """
    if gpus_needed <= 0:
        return []

    local_info = get_local_gpu_free_memory()
    available = sorted(idx for idx, _total, _free in local_info)

    if available:
        if gpus_needed > len(available):
            return []
        if profile_gpu in available:
            start = available.index(profile_gpu)
        else:
            start = 0
        return [
            available[(start + i) % len(available)]
            for i in range(gpus_needed)
        ]

    # Fallback when we cannot query local GPUs: optimistic contiguous IDs.
    return [profile_gpu + i for i in range(gpus_needed)]


def profile_model_memory(
    representative_job: Dict[str, Any],
    use_gpus: List[int],
    profile_steps: int,
    working_dir: str,
    timeout: int = 600,
    profile_log_dir: Optional[str] = None,
) -> Optional[int]:
    """Run a short test of a model and measure peak GPU memory per GPU.

    Launches the job's command with --num-steps=profile_steps on the
    specified GPU group, waits for completion, and samples peak memory
    during the run.

    Returns peak memory in MiB per GPU, or None on failure.
    """
    cmd_template = representative_job.get("cmd", "")
    gpus_needed = int(representative_job.get("gpus", 1) or 1)

    if len(use_gpus) != gpus_needed:
        print(f"  ❌ Profiling GPU selection mismatch: model needs {gpus_needed} GPU(s), got {use_gpus}")
        return None

    cmd, env = _build_profile_cmd(cmd_template, profile_steps, use_gpus)

    model_key = extract_model_key(representative_job)
    print(f"  🔬 Profiling {model_key} ({profile_steps} steps on GPU {use_gpus})...")

    # Record baseline memory
    baseline = _get_peak_gpu_memory(use_gpus)

    # Launch the profile run
    if profile_log_dir:
        os.makedirs(profile_log_dir, exist_ok=True)
        log_file = os.path.join(profile_log_dir, f"profile_{model_key}.log")
    else:
        log_file = f"/tmp/profile_{model_key}.log"
    print(f"  📝 Profile log: {log_file}")
    try:
        fh = open(log_file, "w")
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=working_dir,
            start_new_session=True,
        )
    except Exception as e:
        print(f"  ❌ Failed to launch profile for {model_key}: {e}")
        return None

    # Poll memory usage during the run, track peak
    peak_per_gpu: Dict[int, int] = {g: 0 for g in use_gpus}
    start_time = time.time()

    try:
        while proc.poll() is None:
            if time.time() - start_time > timeout:
                print(f"  ⚠️  Profile run for {model_key} timed out after {timeout}s")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
                break

            current = _get_peak_gpu_memory(use_gpus)
            for gid in use_gpus:
                if gid in current:
                    # Subtract baseline to get model-only memory
                    model_mem = current[gid] - baseline.get(gid, 0)
                    peak_per_gpu[gid] = max(peak_per_gpu[gid], model_mem)
            time.sleep(3)

        # One final measurement
        current = _get_peak_gpu_memory(use_gpus)
        for gid in use_gpus:
            if gid in current:
                model_mem = current[gid] - baseline.get(gid, 0)
                peak_per_gpu[gid] = max(peak_per_gpu[gid], model_mem)

    finally:
        fh.close()
        # Ensure process is dead
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        proc.wait()

    exit_code = proc.returncode
    if exit_code != 0 and all(v < 100 for v in peak_per_gpu.values()):
        print(f"  ❌ Profile for {model_key} failed (exit={exit_code}), "
              f"no meaningful memory recorded")
        print(f"     Log: {log_file}")
        return None

    # Average peak across GPUs used
    valid_peaks = [v for v in peak_per_gpu.values() if v > 0]
    if not valid_peaks:
        print(f"  ❌ No memory readings for {model_key}")
        return None

    avg_peak = int(sum(valid_peaks) / len(valid_peaks))
    print(f"  ✅ {model_key}: peak {avg_peak} MiB/GPU "
          f"(per GPU: {dict(peak_per_gpu)})")
    print(f"  📄 Profile log saved: {log_file}")

    # Wait for GPU memory to settle back to baseline
    print(f"  ⏳ Waiting for GPU memory to release...")
    for _ in range(30):  # up to 30 seconds
        time.sleep(1)
        current = _get_peak_gpu_memory(use_gpus)
        settled = all(
            abs(current.get(g, 0) - baseline.get(g, 0)) < 200
            for g in use_gpus
        )
        if settled:
            break

    return avg_peak


# ---------------------------------------------------------------------------
# Profile cache
# ---------------------------------------------------------------------------

def load_profile_cache(schedule_dir: str) -> Optional[Dict[str, Any]]:
    """Load scheduler-internal memory profile cache from a schedule directory.

    Returns the full profile dict or None if not found.
    This cache is used only by ``job_scheduler.py`` (not by ``jobdaemon.py``).
    """
    path = os.path.join(schedule_dir, PROFILE_CACHE_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data and "models" in data:
            return data
        return None
    except Exception:
        return None


def save_profile_cache(
    schedule_dir: str,
    profiles: Dict[str, Dict[str, Any]],
    gpu_total_mib: int,
) -> None:
    """Save scheduler-only memory profiles to ``schedules/<name>/memory_profile.yaml``.

    This artifact is reused by future scheduler runs and is not consumed by
    runtime daemon commands.

    Format::

        gpu_total_mib: 81920
        models:
          vit-wee:
            peak_memory_mib: 4500
            gpus_needed: 1
          vit-base:
            peak_memory_mib: 12000
            gpus_needed: 1
    """
    os.makedirs(schedule_dir, exist_ok=True)
    path = os.path.join(schedule_dir, PROFILE_CACHE_FILENAME)

    data = {
        "gpu_total_mib": gpu_total_mib,
        "models": profiles,
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  💾 Saved memory profile cache → {path}")


# ---------------------------------------------------------------------------
# Job metadata annotation
# ---------------------------------------------------------------------------

def annotate_jobs_with_profile_metadata(
    jobs: List[Dict[str, Any]],
    model_profiles: Dict[str, Dict[str, Any]],
    fallback_memory_mib: int,
) -> List[Dict[str, Any]]:
    """Attach model/memory metadata to jobs without assigning concrete GPUs."""
    annotated: List[Dict[str, Any]] = []
    for job in jobs:
        model_key = extract_model_key(job)
        profile_key = extract_profile_key(job)
        # Look up by the (model, batch-size) composite key first, then
        # fall back to the plain model_key for backward compatibility
        # with legacy memory_profile.yaml files that predate batch-size
        # awareness (those implicitly assume ``DEFAULT_BATCH_SIZE``).
        profile = model_profiles.get(profile_key) or model_profiles.get(model_key)
        if profile:
            mem_per_gpu = int(profile["peak_memory_mib"])
        else:
            mem_per_gpu = int(fallback_memory_mib)
            print(
                f"  ⚠️  No profile for {profile_key}, using conservative estimate: "
                f"{mem_per_gpu} MiB/GPU"
            )

        job_copy = dict(job)
        job_copy["model_key"] = model_key
        job_copy["memory_mib_per_gpu"] = mem_per_gpu
        annotated.append(job_copy)

    return annotated


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_jobs_yaml(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a jobs.yaml and return (defaults_dict, jobs_list)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not data or "jobs" not in data:
        print(f"Error: {path} has no 'jobs' key")
        sys.exit(1)
    defaults = data.get("defaults", {})
    jobs = data["jobs"]
    return defaults, jobs


def write_node_yaml(
    path: str,
    defaults: Dict[str, Any],
    jobs: List[Dict[str, Any]],
) -> None:
    """Write a per-node jobs.yaml for dynamic scheduling by jobdaemon."""
    clean_jobs = []
    for job in jobs:
        clean = dict(job)
        clean.pop("assigned_gpus", None)
        clean_jobs.append(clean)

    defaults_copy = dict(defaults)
    defaults_copy.pop("pre_grouped", None)
    defaults_copy.pop("static_gpu_budget_mib", None)

    output = {"defaults": defaults_copy, "jobs": clean_jobs}
    with open(path, "w") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)


def schedule(
    input_path: str,
    num_nodes: int,
    output_dir: str = ".",
    dry_run: bool = False,
    schedule_name: Optional[str] = None,
    profile_steps: int = 20,
    profile_gpu: int = 0,
    working_dir: Optional[str] = None,
    skip_profile: bool = False,
) -> None:
    """Profile models locally, annotate jobs, and split across ``num_nodes``.

    Parameters
    ----------
    input_path : str
        Path to the input jobs.yaml.  Its basename (without the ``.yaml``
        / ``.yml`` extension) is used as the prefix for every output file
        so ``--input vit.yaml`` produces ``vit_node_0_jobs.yaml`` etc.
    num_nodes : int
        Number of nodes the job set should be split across.  The scheduler
        does NOT contact these nodes — it only needs the count to perform
        the round-robin distribution.
    output_dir : str
        Directory to write per-node YAML files.
    dry_run : bool
        If True, print the plan but don't write files.
    schedule_name : str | None
        Name for caching profiles under schedules/<name>/.
    profile_steps : int
        Number of training steps for each profile test run.
    profile_gpu : int
        Starting GPU index for profiling selection (default: 0).
        Multi-GPU jobs use consecutive indices from this start.
    working_dir : Optional[str]
        Working directory for profile runs.
    skip_profile : bool
        If True, skip profiling even if no cache exists (use conservative estimates).
    """
    if num_nodes < 1:
        print(f"Error: num_nodes must be >= 1, got {num_nodes}")
        sys.exit(1)

    # Derive the output filename prefix from the input YAML's basename so
    # ``--input vit.yaml`` -> ``vit_node_<i>_jobs.yaml``.  We strip any
    # ``.yaml`` / ``.yml`` extension but keep the rest of the basename
    # verbatim (hyphens and underscores alike), so e.g.
    # ``vit-wee-jobs.yaml`` -> ``vit-wee-jobs_node_<i>_jobs.yaml``.
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    file_prefix = input_stem

    # Load jobs
    defaults, jobs = load_jobs_yaml(input_path)
    total_jobs = len(jobs)
    default_gpus = defaults.get("gpus", 2)

    # Apply defaults to jobs
    for job in jobs:
        if "gpus" not in job:
            job["gpus"] = default_gpus

    # Resolve the effective working directory for profile subprocesses.
    # Precedence (highest first):
    #   1. CLI --working-dir (explicit user override)
    #   2. defaults.working_dir in the input YAML (set by generate_vitdet_jobs
    #      et al. so scripts living in sub-packages like ``detectron2_vitdet/``
    #      are invoked from their own directory)
    #   3. current working directory (``.``)
    # Per-job ``working_dir`` overrides are honoured further down when the
    # representative job is actually profiled.
    cli_working_dir = working_dir
    yaml_default_working_dir = defaults.get("working_dir")
    effective_working_dir = (
        cli_working_dir
        or yaml_default_working_dir
        or "."
    )

    print(f"\n📋 Loaded {total_jobs} job(s) from {input_path}")
    print(f"   Defaults: gpus={default_gpus}, "
          f"max_retries={defaults.get('max_retries', 3)}")
    if yaml_default_working_dir and not cli_working_dir:
        print(f"   Profile cwd: {effective_working_dir}  "
              f"(from defaults.working_dir)")
    elif cli_working_dir:
        print(f"   Profile cwd: {effective_working_dir}  (from --working-dir)")
    print(f"   Output prefix: {file_prefix}_node_<i>_jobs.yaml")

    # ── Step 1: Identify unique models ──────────────────
    # Group by (model_config, batch_size) composite key — jobs with the
    # same model but different batch sizes are distinct VRAM classes and
    # must be profiled separately.  The composite key degrades to the
    # plain model key for the default batch size so existing caches keep
    # working without migration.
    model_jobs: Dict[str, List[Dict[str, Any]]] = {}
    for job in jobs:
        key = extract_profile_key(job)
        model_jobs.setdefault(key, []).append(job)

    print(f"\n🔍 Found {len(model_jobs)} unique model config(s):")
    for key, mjobs in model_jobs.items():
        num_steps = extract_num_steps(mjobs[0])
        steps_str = f", {num_steps} steps" if num_steps else ""
        bs = extract_batch_size(mjobs[0])
        bs_str = f", bs={bs}" if bs is not None else ""
        print(f"   {key}: {len(mjobs)} job(s), "
              f"{mjobs[0].get('gpus', 1)} GPU(s)/job{steps_str}{bs_str}")

    # ── Step 2: Query LOCAL GPU memory (once) ──────────────
    # The scheduler profiles only on the local node.  All target nodes are
    # assumed to have identical hardware to the local node, so we
    # replicate the locally-observed GPU info across ``num_nodes`` slots
    # purely for reporting/debug purposes.  No remote SSH queries are
    # performed and no node addresses are required.
    print(f"\n�️  Querying local node GPU memory (will replicate across {num_nodes} logical node(s))...")
    local_num_gpus, local_total_free, local_per_gpu = query_local_node_memory()
    local_gpu_total = local_per_gpu[0][1] if local_per_gpu else 0

    if local_per_gpu:
        print(f"  ✅ local: {local_num_gpus} GPUs, "
              f"{local_total_free:,} MiB total free")
        for idx, total, free in local_per_gpu:
            print(f"       GPU {idx}: {free:,}/{total:,} MiB")
    else:
        print(f"  ❌ local: no GPU info detected")

    node_info: List[Dict[str, Any]] = [
        {
            "num_gpus": local_num_gpus,
            "total_free_mib": local_total_free,
            "per_gpu": local_per_gpu,
            "gpu_total_mib": local_gpu_total,
        }
        for _ in range(num_nodes)
    ]

    # Determine gpu_total_mib (from the local query)
    gpu_total_mib = local_gpu_total
    if gpu_total_mib == 0:
        print("❌ Could not determine local GPU total memory")
        sys.exit(1)

    # ── Step 3: Memory profiling ────────────────────────────────────
    schedule_dir = None
    schedule_debug_dir = None
    profile_log_dir = None
    if schedule_name:
        schedule_dir = os.path.join(SCHEDULES_ROOT, schedule_name)
        schedule_debug_dir = os.path.join(schedule_dir, DEBUG_SUBDIR_NAME)
        profile_log_dir = os.path.join(schedule_debug_dir, PROFILE_LOGS_SUBDIR_NAME)

    model_profiles: Dict[str, Dict[str, Any]] = {}
    cache = load_profile_cache(schedule_dir) if schedule_dir else None

    if cache:
        print(f"\n💾 Found cached memory profile in {schedule_dir}/")
        cached_models = cache.get("models", {})
        all_cached = True
        for key in model_jobs:
            if key in cached_models:
                model_profiles[key] = cached_models[key]
                print(f"   ✅ {key}: {cached_models[key]['peak_memory_mib']} MiB/GPU (cached)")
            else:
                all_cached = False
                print(f"   ⚠️  {key}: not in cache, needs profiling")

        if all_cached:
            print("   All models cached — skipping profiling! 🎉")
    else:
        print(f"\n🔬 No cached profile found. Running memory profiling...")

    # Profile any models not in cache
    models_to_profile = [k for k in model_jobs if k not in model_profiles]

    if models_to_profile and not skip_profile:
        print(f"\n🔬 Profiling {len(models_to_profile)} model(s) "
              f"({profile_steps} steps each)...")

        for key in models_to_profile:
            representative = model_jobs[key][0]
            gpus_needed = int(representative.get("gpus", 1) or 1)
            profile_gpu_ids = select_profile_gpu_ids(profile_gpu, gpus_needed)
            if len(profile_gpu_ids) != gpus_needed:
                print(f"  ⚠️  Cannot allocate {gpus_needed} local GPU(s) for profiling {key}; "
                      f"got {profile_gpu_ids}. Using conservative estimate.")
                peak = None
            else:
                job_working_dir = (
                    representative.get("working_dir")
                    or effective_working_dir
                )
                peak = profile_model_memory(
                    representative_job=representative,
                    use_gpus=profile_gpu_ids,
                    profile_steps=profile_steps,
                    working_dir=job_working_dir,
                    profile_log_dir=profile_log_dir,
                )
            if peak is not None:
                model_profiles[key] = {
                    "peak_memory_mib": peak,
                    "gpus_needed": gpus_needed,
                }
            else:
                # Use conservative 50% of GPU total
                conservative = gpu_total_mib // 2
                model_profiles[key] = {
                    "peak_memory_mib": conservative,
                    "gpus_needed": gpus_needed,
                }
                print(f"  ⚠️  Using conservative estimate for {key}: "
                      f"{conservative} MiB/GPU")

        # Save updated cache
        if schedule_dir:
            save_profile_cache(schedule_dir, model_profiles, gpu_total_mib)

    elif models_to_profile and skip_profile:
        print(f"\n⏭️  Skipping profiling (--skip-profile). "
              f"Using conservative estimates for: {models_to_profile}")
        for key in models_to_profile:
            representative = model_jobs[key][0]
            conservative = gpu_total_mib // 2
            model_profiles[key] = {
                "peak_memory_mib": conservative,
                "gpus_needed": representative.get("gpus", 1),
            }

    # ── Step 4: Class-balanced round-robin distribution ────────────
    #
    # Strategy: group jobs by VRAM class (model_key), then deal each
    # class's jobs round-robin across nodes.  This ensures every node
    # gets the same number of jobs per class (±1 when not evenly
    # divisible).  Because generate_jobs.py emits N seeds per
    # (model, trial), each node naturally ends up owning ~1 seed per
    # configuration — giving a balanced workload without bin-packing.
    #
    conservative_fallback = gpu_total_mib // 2

    # First annotate ALL jobs with profile metadata so we can group them.
    all_annotated = annotate_jobs_with_profile_metadata(
        jobs, model_profiles, fallback_memory_mib=conservative_fallback,
    )

    # Group by VRAM class (model_key).
    class_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for job in all_annotated:
        mk = job.get("model_key", "unknown")
        class_buckets.setdefault(mk, []).append(job)

    # Round-robin each class across nodes.
    node_jobs: List[List[Dict[str, Any]]] = [[] for _ in range(num_nodes)]
    print(f"\n🔄 Class-balanced round-robin distribution across {num_nodes} node(s):")
    for mk in sorted(class_buckets):
        bucket = class_buckets[mk]
        mem = bucket[0].get("memory_mib_per_gpu", "?")
        per_node_counts = [0] * num_nodes
        for idx, job in enumerate(bucket):
            target = idx % num_nodes
            node_jobs[target].append(job)
            per_node_counts[target] += 1
        print(f"   {mk} ({len(bucket)} jobs, ~{mem} MiB/GPU): "
              f"per-node → {per_node_counts}")

    # Summary per node
    for i, info in enumerate(node_info):
        model_summary: Dict[str, int] = {}
        for j in node_jobs[i]:
            mk = j.get("model_key", "unknown")
            model_summary[mk] = model_summary.get(mk, 0) + 1
        node_label = f"{file_prefix}_node_{i}"
        print(f"\n  {node_label}: {len(node_jobs[i])} jobs")
        print(f"    Models: {model_summary}")

    # ── Step 5: Print plan ──────────────────────────────────
    weights = [n["total_free_mib"] for n in node_info]
    print(f"\n📊 Job Distribution Plan:")
    print(f"{'─' * 65}")
    print(f"{'Node':<30} {'Free Memory':>15} {'Jobs':>8}")
    print(f"{'─' * 65}")
    for i, (info, node_list) in enumerate(zip(node_info, node_jobs)):
        node_label = f"{file_prefix}_node_{i}"
        print(f"  {node_label:<28} "
              f"{info['total_free_mib']:>10,} MiB "
              f"{len(node_list):>6}")
    print(f"{'─' * 65}")
    total_split = sum(len(p) for p in node_jobs)
    print(f"  {'TOTAL':<28} {sum(weights):>10,} MiB {total_split:>6}")

    if total_split != total_jobs:
        print(f"\n⚠️  WARNING: split total ({total_split}) != "
              f"total jobs ({total_jobs})")

    # ── Step 6: Write output files ──────────────────────────
    if dry_run:
        print(f"\n🔍 DRY RUN — no files written.")
        for i, (info, node_list) in enumerate(zip(node_info, node_jobs)):
            node_label = f"{file_prefix}_node_{i}"
            print(f"\n  {node_label}: {len(node_list)} job(s)")
            for j in node_list[:5]:
                print(
                    f"    - {j['name']} "
                    f"(model={j.get('model_key')}, mem={j.get('memory_mib_per_gpu')} MiB/GPU)"
                )
            if len(node_list) > 5:
                print(f"    ... and {len(node_list) - 5} more")
        return

    os.makedirs(output_dir, exist_ok=True)

    written_files = []
    for i, (info, node_list) in enumerate(zip(node_info, node_jobs)):
        fname = f"{file_prefix}_node_{i}_jobs.yaml"
        fpath = os.path.join(output_dir, fname)
        write_node_yaml(
            fpath,
            defaults,
            node_list,
        )
        written_files.append((fpath, len(node_list), i))

    # Also save schedule debug metadata if schedule context is set
    if schedule_dir:
        schedule_info = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": input_path,
            "total_jobs": total_jobs,
            "profile_steps": profile_steps,
            "model_profiles": model_profiles,
            "num_nodes": num_nodes,
            "nodes": [],
        }
        for i, (info, node_list) in enumerate(zip(node_info, node_jobs)):
            node_entry = {
                "index": i,
                "num_gpus": info["num_gpus"],
                "total_free_mib": info["total_free_mib"],
                "num_jobs": len(node_list),
                "jobs": [
                    {
                        "name": j.get("name"),
                        "model_key": j.get("model_key"),
                        "gpus_needed": j.get("gpus", defaults.get("gpus", 1)),
                        "memory_per_gpu_mib": j.get("memory_mib_per_gpu"),
                    }
                    for j in node_list
                ],
            }
            schedule_info["nodes"].append(node_entry)

        os.makedirs(schedule_debug_dir, exist_ok=True)
        info_path = os.path.join(schedule_debug_dir, SCHEDULE_INFO_FILENAME)

        with open(info_path, "w") as f:
            yaml.safe_dump(schedule_info, f, default_flow_style=False, sort_keys=False)
        print(f"\n🧪 Debug metadata saved → {info_path}")

    # Print summary and next steps
    print(f"\n✅ Written {len(written_files)} runtime file(s):")
    for fpath, count, node_index in written_files:
        print(f"   {fpath}  ({count} jobs, node_index={node_index})")

    print(f"\n📁 Scheduler artifacts generated:")
    print("   Runtime-consumed (required):")
    for fpath, _count, _node_index in written_files:
        print(f"   - {fpath}  # submit this file to jobdaemon")

    if schedule_name:
        profile_cache_path = os.path.join(schedule_dir, PROFILE_CACHE_FILENAME)
        print("   Scheduler-only cache (not consumed by jobdaemon):")
        print(f"   - {profile_cache_path}  # reused by future scheduler runs")

        debug_info_path = os.path.join(schedule_debug_dir, SCHEDULE_INFO_FILENAME)
        debug_profile_logs_path = os.path.join(schedule_debug_dir, PROFILE_LOGS_SUBDIR_NAME)
        print("   Debug-only metadata/logs (not consumed by runtime):")
        print(f"   - {debug_info_path}  # audit snapshot for humans")
        print(f"   - {debug_profile_logs_path}/profile_<model>.log  # profiling logs")

    print(f"\n🚀 Next steps — on each node, run daemon with matching --node-index:")
    print(f"{'─' * 70}")
    sched_flag = f" -s {schedule_name}" if schedule_name else ""
    for fpath, count, node_index in written_files:
        node_label = f"{file_prefix}_node_{node_index}"
        if count == 0:
            print(f"\n  # {node_label}: 0 jobs — skip")
            continue
        fname = os.path.basename(fpath)
        print(f"\n  # {node_label}:")
        print(f"  # (copy {fname} to node {node_index} if running remotely)")
        print(f"  tmux new -s daemon")
        print(f"  python jobdaemon.py{sched_flag} --node-index {node_index} start --gpus 0,1,2,3,4,5,6,7")
        print(f"  # (in another terminal)")
        print(f"  python jobdaemon.py{sched_flag} --node-index {node_index} submit {fname}")
    print(f"{'─' * 70}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart job scheduler with profiling cache and class-balanced round-robin splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Explicit node count:\n"
            "  python job_scheduler.py --input vit.yaml --num-nodes 3 \\\n"
            "      --schedule-name imagenet_sweep --profile-steps 20\n"
            "\n"
            "  # Pick up node count from $HOST_NUM:\n"
            "  HOST_NUM=3 python job_scheduler.py --input vit.yaml \\\n"
            "      --schedule-name imagenet_sweep\n"
            "\n"
            "  # Reuse cached profile (skip profiling):\n"
            "  python job_scheduler.py --input vit.yaml --num-nodes 3 \\\n"
            "      --schedule-name imagenet_sweep --skip-profile\n"
            "\n"
            "  # Preview without writing:\n"
            "  python job_scheduler.py --input vit.yaml --num-nodes 3 --dry-run\n"
            "\n"
            "Output filenames are derived from the input basename:\n"
            "  --input vit.yaml  →  vit_node_0_jobs.yaml, vit_node_1_jobs.yaml, ...\n"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input jobs.yaml. Its basename (minus extension) is "
             "used as the prefix for every output file, e.g. 'vit.yaml' → "
             "'vit_node_<i>_jobs.yaml'.",
    )
    # Default is resolved at parse time: env $HOST_NUM takes precedence when
    # --num-nodes is omitted, falling back to 1 when neither is given.
    _default_num_nodes_env = os.environ.get("HOST_NUM")
    try:
        _default_num_nodes = int(_default_num_nodes_env) if _default_num_nodes_env else 1
    except ValueError:
        print(f"Error: $HOST_NUM is not an integer: {_default_num_nodes_env!r}")
        sys.exit(1)
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=_default_num_nodes,
        help=f"Number of nodes to split the job set across. "
             f"Falls back to $HOST_NUM (currently: {_default_num_nodes_env or 'unset'}) "
             f"or 1 when neither is provided. "
             f"The scheduler does NOT contact these nodes — it only needs "
             f"the count to perform the round-robin distribution.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to write per-node YAML files (default: cwd)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing files",
    )
    parser.add_argument(
        "--schedule-name",
        default=None,
        help="Schedule name namespace. Generates scheduler artifacts under "
             "schedules/<name>/ (memory_profile.yaml cache) and "
             "schedules/<name>/debug/ (schedule_info.yaml + profile logs). "
             "Only per-node *_jobs.yaml files are consumed by jobdaemon.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Number of training steps for each memory profile test run "
             "(default: 20)",
    )
    parser.add_argument(
        "--profile-gpu",
        type=int,
        default=0,
        help="Starting GPU index for profiling (default: 0). "
             "Multi-GPU jobs use consecutive indices from this start.",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Working directory for profile runs. If omitted, the scheduler "
             "uses the input YAML's defaults.working_dir (or per-job "
             "working_dir), falling back to cwd.",
    )

    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip profiling even if no cache exists "
             "(use conservative 50%% GPU estimates)",
    )
    args = parser.parse_args()

    # Validate
    if args.num_nodes < 1:
        print(f"Error: --num-nodes must be >= 1, got {args.num_nodes}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    schedule(
        input_path=args.input,
        num_nodes=args.num_nodes,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        schedule_name=args.schedule_name,
        profile_steps=args.profile_steps,
        profile_gpu=args.profile_gpu,
        working_dir=args.working_dir,
        skip_profile=args.skip_profile,
    )


if __name__ == "__main__":
    main()
