#!/usr/bin/env python3
"""Smart job scheduler with memory profiling and model-aware GPU grouping.

Instead of letting the jobdaemon discover memory usage at runtime (which
causes slow sequential starts and suboptimal GPU utilization), this
scheduler:

1. Analyzes all jobs and identifies unique model configurations.
2. Runs a short test (N steps) for each unique model to measure peak
   GPU memory usage.
3. Bin-packs jobs onto GPUs to maximize total memory utilization,
   grouping by model so that similar-memory jobs share GPUs optimally.
4. Saves the profiled memory info and GPU assignments as a cached
   schedule YAML under ``schedules/<name>/`` so subsequent runs skip
   the profiling step.
5. Outputs per-node YAML files with ``pre_grouped: true`` so the
   jobdaemon launches everything immediately without waiting/fitting.

Usage::

    # Profile models and generate optimized schedule
    python job_scheduler.py --input jobs.yaml \\
        --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\
        --schedule-name imagenet_sweep \\
        --profile-steps 200

    # Reuse cached profile (skip profiling)
    python job_scheduler.py --input jobs.yaml \\
        --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\
        --schedule-name imagenet_sweep

    # Preview the plan without writing files
    python job_scheduler.py --input jobs.yaml \\
        --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\
        --schedule-name imagenet_sweep \\
        --dry-run

    # Then on each node, start daemon with --pre-grouped:
    #   python jobdaemon.py -s imagenet_sweep start --gpus 0,1,...,7 --pre-grouped
    #   python jobdaemon.py -s imagenet_sweep submit node_0_jobs.yaml
"""
from __future__ import annotations

import argparse
import math
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


def get_remote_gpu_free_memory(host: str, timeout: int = 30) -> List[Tuple[int, int, int]]:
    """Query GPU memory on a remote node via SSH + nvidia-smi.

    Returns list of (gpu_index, total_mib, free_mib).
    """
    cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout={timeout} "
        f"root@{host} "
        f"\"nvidia-smi --query-gpu=index,memory.total,memory.free "
        f"--format=csv,noheader,nounits\""
    )
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout + 10,
        )
        if r.returncode != 0:
            print(f"  ⚠️  SSH to {host} failed: {r.stderr.strip()}")
            return []
        result = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                result.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return result
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  SSH to {host} timed out after {timeout}s")
        return []
    except Exception as e:
        print(f"  ⚠️  SSH to {host} error: {e}")
        return []


def is_local(host: str) -> bool:
    """Check if a hostname/IP refers to this machine."""
    local_hostname = socket.gethostname()
    try:
        local_ips = set()
        for info in socket.getaddrinfo(local_hostname, None):
            local_ips.add(info[4][0])
        # Also add common loopback
        local_ips.add("127.0.0.1")
        local_ips.add("::1")
        local_ips.add(local_hostname)

        # Resolve the target host
        target_ips = set()
        for info in socket.getaddrinfo(host, None):
            target_ips.add(info[4][0])
        target_ips.add(host)

        return bool(local_ips & target_ips)
    except Exception:
        return host in ("localhost", "127.0.0.1", local_hostname)


def query_node_memory(host: str) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """Query a node's GPU memory.

    Returns (num_gpus, total_free_mib, per_gpu_info).
    """
    if is_local(host):
        print(f"  📡 {host} (local) — querying GPU memory...")
        info = get_local_gpu_free_memory()
    else:
        print(f"  📡 {host} (remote) — querying GPU memory via SSH...")
        info = get_remote_gpu_free_memory(host)

    if not info:
        print(f"  ❌ {host}: could not query GPU memory")
        return 0, 0, []

    total_free = sum(free for _, _, free in info)
    return len(info), total_free, info


# ---------------------------------------------------------------------------
# Job distribution (Hamilton's largest-remainder method)
# ---------------------------------------------------------------------------

def distribute_jobs_proportionally(
    total_jobs: int,
    node_weights: List[float],
) -> List[int]:
    """Distribute total_jobs across nodes proportionally to their weights.

    Uses Hamilton's largest-remainder method for fair integer allocation:
    1. Compute ideal (fractional) allocation per node.
    2. Give each node floor(ideal) jobs.
    3. Distribute remaining jobs to nodes with the largest remainders.

    This ensures no jobs are dropped and the split closely follows the
    memory proportions.
    """
    if not node_weights or total_jobs == 0:
        return [0] * len(node_weights)

    total_weight = sum(node_weights)
    if total_weight == 0:
        # Equal split if all weights are zero
        base = total_jobs // len(node_weights)
        remainder = total_jobs % len(node_weights)
        result = [base] * len(node_weights)
        for i in range(remainder):
            result[i] += 1
        return result

    # Step 1: Compute fractional allocations
    fractions = [(w / total_weight) * total_jobs for w in node_weights]

    # Step 2: Floor allocations
    floors = [int(f) for f in fractions]
    remainders = [f - int(f) for f in fractions]

    # Step 3: Distribute leftovers to largest remainders
    leftover = total_jobs - sum(floors)
    # Get indices sorted by remainder (descending)
    sorted_indices = sorted(range(len(remainders)),
                            key=lambda i: remainders[i], reverse=True)
    for i in range(leftover):
        floors[sorted_indices[i]] += 1

    return floors


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

    # Look for --config or -c flag
    for i, tok in enumerate(tokens):
        if tok in ("--config", "-c") and i + 1 < len(tokens):
            config_path = tokens[i + 1]
            # Use the basename without extension as the model key
            return os.path.splitext(os.path.basename(config_path))[0]
        if tok.startswith("--config="):
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


# ---------------------------------------------------------------------------
# Memory profiling via short test runs
# ---------------------------------------------------------------------------

def _build_profile_cmd(
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
    cmd = job_cmd.replace("{gpus}", str(len(gpu_ids)))
    # Use a fixed port for profiling
    cmd = cmd.replace("{port}", "29400")
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
    """Load cached memory profile from a schedule directory.

    Returns the full profile dict or None if not found.
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
    """Save memory profiles to the schedule directory.

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
# GPU bin-packing
# ---------------------------------------------------------------------------

def bin_pack_jobs_to_gpus(
    jobs: List[Dict[str, Any]],
    model_profiles: Dict[str, Dict[str, Any]],
    gpu_total_mib: int,
    num_gpus: int,
    safety_margin: float = 0.10,
) -> List[Dict[str, Any]]:
    """Assign GPU indices to jobs using first-fit-decreasing bin packing.

    Groups jobs by model (same model → same memory footprint), then
    assigns them to GPUs to maximize utilization while respecting memory
    limits.

    Each job in the returned list gets an extra ``assigned_gpus`` field
    (list of GPU indices).

    Strategy:
    - Sort models by memory usage (descending) for better packing.
    - For each job, find the GPU(s) with the most remaining capacity
      that can still fit the job.
    - Uses safety_margin (10% default) to avoid edge-case OOMs.
    """
    usable_per_gpu = int(gpu_total_mib * (1.0 - safety_margin))
    gpu_remaining = [usable_per_gpu] * num_gpus  # remaining MiB per GPU

    # Build list of (job_dict, model_key, memory_per_gpu)
    job_entries = []
    for job in jobs:
        model_key = extract_model_key(job)
        gpus_needed = job.get("gpus", 1)
        profile = model_profiles.get(model_key)
        if profile:
            mem_per_gpu = profile["peak_memory_mib"]
        else:
            # No profile — use a conservative estimate (50% of GPU)
            mem_per_gpu = usable_per_gpu // 2
            print(f"  ⚠️  No profile for {model_key}, using conservative "
                  f"estimate: {mem_per_gpu} MiB/GPU")
        job_entries.append((job, model_key, mem_per_gpu, gpus_needed))

    # Sort by memory descending (first-fit-decreasing → better packing)
    job_entries.sort(key=lambda x: x[2], reverse=True)

    # Assign GPUs
    assigned_jobs = []
    unassigned = []

    for job, model_key, mem_per_gpu, gpus_needed in job_entries:
        if gpus_needed == 1:
            # Single-GPU job: find the GPU with most remaining that can fit
            best_gpu = None
            best_remaining = -1
            for g in range(num_gpus):
                if gpu_remaining[g] >= mem_per_gpu and gpu_remaining[g] > best_remaining:
                    best_gpu = g
                    best_remaining = gpu_remaining[g]

            if best_gpu is not None:
                gpu_remaining[best_gpu] -= mem_per_gpu
                job_copy = dict(job)
                job_copy["assigned_gpus"] = [best_gpu]
                job_copy["_model_key"] = model_key
                job_copy["_memory_per_gpu"] = mem_per_gpu
                assigned_jobs.append(job_copy)
            else:
                unassigned.append((job, model_key, mem_per_gpu, gpus_needed))
        else:
            # Multi-GPU job: find gpus_needed consecutive or best-fit GPUs
            # Try to find gpus_needed GPUs all with enough remaining memory
            # Sort GPUs by remaining capacity (descending)
            gpu_order = sorted(
                range(num_gpus),
                key=lambda g: gpu_remaining[g],
                reverse=True,
            )
            fit_gpus = [g for g in gpu_order if gpu_remaining[g] >= mem_per_gpu]

            if len(fit_gpus) >= gpus_needed:
                chosen = fit_gpus[:gpus_needed]
                for g in chosen:
                    gpu_remaining[g] -= mem_per_gpu
                job_copy = dict(job)
                job_copy["assigned_gpus"] = sorted(chosen)
                job_copy["_model_key"] = model_key
                job_copy["_memory_per_gpu"] = mem_per_gpu
                assigned_jobs.append(job_copy)
            else:
                unassigned.append((job, model_key, mem_per_gpu, gpus_needed))

    if unassigned:
        print(f"\n  ⚠️  {len(unassigned)} job(s) could not fit in {num_gpus} GPUs:")
        for job, mk, mem, gn in unassigned:
            print(f"     {job['name']} ({mk}, {mem} MiB/GPU, needs {gn} GPU(s))")
        # Add unassigned jobs with round-robin GPU assignment as fallback
        rr = 0
        for job, model_key, mem_per_gpu, gpus_needed in unassigned:
            chosen = [(rr + g) % num_gpus for g in range(gpus_needed)]
            rr = (rr + gpus_needed) % num_gpus
            job_copy = dict(job)
            job_copy["assigned_gpus"] = sorted(chosen)
            job_copy["_model_key"] = model_key
            job_copy["_memory_per_gpu"] = mem_per_gpu
            assigned_jobs.append(job_copy)

    return assigned_jobs


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
    gpu_total_mib: int,
    safety_margin: float,
) -> None:
    """Write a per-node jobs.yaml with pre_grouped flag."""
    # Clean internal keys from jobs before writing, but keep scheduler metadata
    clean_jobs = []
    for job in jobs:
        clean = {k: v for k, v in job.items() if not k.startswith("_")}
        clean["memory_mib_per_gpu"] = job.get("_memory_per_gpu")
        clean["model_key"] = job.get("_model_key")
        clean_jobs.append(clean)

    defaults_copy = dict(defaults)
    defaults_copy["pre_grouped"] = True
    defaults_copy["static_gpu_budget_mib"] = int(gpu_total_mib * (1.0 - safety_margin))

    output = {"defaults": defaults_copy, "jobs": clean_jobs}
    with open(path, "w") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)


def schedule(
    input_path: str,
    nodes: List[str],
    output_dir: str = ".",
    output_prefix: str = "node",
    free_memory_override: Optional[List[int]] = None,
    dry_run: bool = False,
    schedule_name: Optional[str] = None,
    profile_steps: int = 20,
    profile_gpu: int = 0,
    working_dir: str = ".",
    safety_margin: float = 0.05,
    skip_profile: bool = False,
) -> None:
    """Profile models, bin-pack jobs onto GPUs, split across nodes.

    Parameters
    ----------
    input_path : str
        Path to the input jobs.yaml.
    nodes : list[str]
        List of node IPs/hostnames.
    output_dir : str
        Directory to write per-node YAML files.
    output_prefix : str
        Prefix for output files (e.g. "node" → node_0_jobs.yaml).
    free_memory_override : list[int] | None
        If provided, skip GPU queries and use these values (MiB per node).
    dry_run : bool
        If True, print the plan but don't write files.
    schedule_name : str | None
        Name for caching profiles under schedules/<name>/.
    profile_steps : int
        Number of training steps for each profile test run.
    profile_gpu : int
        Starting GPU index for profiling selection (default: 0).
        Multi-GPU jobs use consecutive indices from this start.
    working_dir : str
        Working directory for profile runs.
    safety_margin : float
        Safety margin for bin packing (default: 0.05 = 5%).
    skip_profile : bool
        If True, skip profiling even if no cache exists (use conservative estimates).
    """
    # Load jobs
    defaults, jobs = load_jobs_yaml(input_path)
    total_jobs = len(jobs)
    default_gpus = defaults.get("gpus", 1)

    # Apply defaults to jobs
    for job in jobs:
        if "gpus" not in job:
            job["gpus"] = default_gpus

    print(f"\n📋 Loaded {total_jobs} job(s) from {input_path}")
    print(f"   Defaults: gpus={default_gpus}, "
          f"max_retries={defaults.get('max_retries', 3)}")

    # ── Step 1: Identify unique models ──────────────────────────────
    model_jobs: Dict[str, List[Dict[str, Any]]] = {}
    for job in jobs:
        key = extract_model_key(job)
        model_jobs.setdefault(key, []).append(job)

    print(f"\n🔍 Found {len(model_jobs)} unique model config(s):")
    for key, mjobs in model_jobs.items():
        num_steps = extract_num_steps(mjobs[0])
        steps_str = f", {num_steps} steps" if num_steps else ""
        print(f"   {key}: {len(mjobs)} job(s), "
              f"{mjobs[0].get('gpus', 1)} GPU(s)/job{steps_str}")

    # ── Step 2: Query GPU memory on nodes ───────────────────────────
    print(f"\n🖥️  Querying {len(nodes)} node(s)...")
    node_info: List[Dict[str, Any]] = []
    for i, host in enumerate(nodes):
        if free_memory_override and i < len(free_memory_override):
            total_free = free_memory_override[i]
            node_info.append({
                "host": host,
                "num_gpus": 8,  # assume 8 GPUs when overriding
                "total_free_mib": total_free,
                "per_gpu": [],
                "gpu_total_mib": total_free // 8,
            })
            print(f"  📡 {host} — using override: {total_free:,} MiB free")
        else:
            num_gpus, total_free, per_gpu = query_node_memory(host)
            gpu_total = per_gpu[0][1] if per_gpu else 0  # total per GPU
            node_info.append({
                "host": host,
                "num_gpus": num_gpus,
                "total_free_mib": total_free,
                "per_gpu": per_gpu,
                "gpu_total_mib": gpu_total,
            })
            if per_gpu:
                gpu_strs = [f"GPU {idx}: {free:,}/{total:,} MiB"
                            for idx, total, free in per_gpu]
                print(f"  ✅ {host}: {num_gpus} GPUs, "
                      f"{total_free:,} MiB total free")
                for gs in gpu_strs:
                    print(f"       {gs}")
            else:
                print(f"  ❌ {host}: no GPU info (will get 0 jobs)")

    # Determine gpu_total_mib (from first node that has info)
    gpu_total_mib = 0
    for ni in node_info:
        if ni["gpu_total_mib"] > 0:
            gpu_total_mib = ni["gpu_total_mib"]
            break
    if gpu_total_mib == 0:
        print("❌ Could not determine GPU total memory from any node")
        sys.exit(1)

    # ── Step 3: Memory profiling ────────────────────────────────────
    schedule_dir = None
    if schedule_name:
        schedule_dir = os.path.join(SCHEDULES_ROOT, schedule_name)

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
                peak = profile_model_memory(
                    representative_job=representative,
                    use_gpus=profile_gpu_ids,
                    profile_steps=profile_steps,
                    working_dir=working_dir,
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

    # ── Step 4: Distribute jobs across nodes (by free memory weight) ─
    weights = [n["total_free_mib"] for n in node_info]
    allocation = distribute_jobs_proportionally(total_jobs, weights)

    # Split jobs into per-node lists (preserving order)
    splits: List[List[Dict[str, Any]]] = []
    offset = 0
    for count in allocation:
        splits.append(jobs[offset:offset + count])
        offset += count

    # ── Step 5: Bin-pack each node's jobs onto its GPUs ─────────────
    print(f"\n📦 Bin-packing jobs onto GPUs (safety_margin={safety_margin*100:.0f}%)...")

    node_packed: List[List[Dict[str, Any]]] = []
    for i, (info, split) in enumerate(zip(node_info, splits)):
        if not split:
            node_packed.append([])
            continue

        num_gpus = info["num_gpus"]
        # Use the node's actual GPU total if available, else the global
        node_gpu_total = info["gpu_total_mib"] or gpu_total_mib

        packed = bin_pack_jobs_to_gpus(
            split, model_profiles, node_gpu_total, num_gpus, safety_margin,
        )
        node_packed.append(packed)

        # Print GPU utilization summary for this node
        gpu_usage: Dict[int, int] = {g: 0 for g in range(num_gpus)}
        gpu_job_count: Dict[int, int] = {g: 0 for g in range(num_gpus)}
        for pj in packed:
            for g in pj.get("assigned_gpus", []):
                gpu_usage[g] += pj.get("_memory_per_gpu", 0)
                gpu_job_count[g] += 1

        usable = int(node_gpu_total * (1.0 - safety_margin))
        node_label = f"{schedule_name}_node_{i}" if schedule_name else f"node_{i}"
        print(f"\n  {node_label} ({info['host']}): {len(packed)} jobs on "
              f"{num_gpus} GPUs")
        for g in range(num_gpus):
            pct = (gpu_usage[g] / usable * 100) if usable > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    GPU {g}: {gpu_job_count[g]} jobs, "
                  f"~{gpu_usage[g]:,}/{usable:,} MiB "
                  f"[{bar}] {pct:.0f}%")

    # ── Step 6: Print plan ──────────────────────────────────────────
    print(f"\n📊 Job Distribution Plan:")
    print(f"{'─' * 65}")
    print(f"{'Node':<25} {'Free Memory':>15} {'Jobs':>8} {'Weight':>10}")
    print(f"{'─' * 65}")
    total_weight = sum(weights) or 1
    for i, (info, packed) in enumerate(zip(node_info, node_packed)):
        pct = (info['total_free_mib'] / total_weight) * 100 if total_weight else 0
        node_label = f"{schedule_name}_node_{i}" if schedule_name else f"node_{i}"
        print(f"  {node_label} ({info['host']:<15}) "
              f"{info['total_free_mib']:>10,} MiB "
              f"{len(packed):>6} "
              f"{pct:>8.1f}%")
    print(f"{'─' * 65}")
    total_packed = sum(len(p) for p in node_packed)
    print(f"  {'TOTAL':<25} {sum(weights):>10,} MiB {total_packed:>6}")

    if total_packed != total_jobs:
        print(f"\n⚠️  WARNING: packed total ({total_packed}) != "
              f"total jobs ({total_jobs})")

    # ── Step 7: Write output files ──────────────────────────────────
    if dry_run:
        print(f"\n🔍 DRY RUN — no files written.")
        for i, (info, packed) in enumerate(zip(node_info, node_packed)):
            node_label = f"{schedule_name}_node_{i}" if schedule_name else f"node_{i}"
            print(f"\n  {node_label} ({info['host']}): {len(packed)} job(s)")
            for j in packed[:5]:
                gpu_str = ",".join(str(g) for g in j.get("assigned_gpus", []))
                print(f"    - {j['name']} → GPU [{gpu_str}]")
            if len(packed) > 5:
                print(f"    ... and {len(packed) - 5} more")
        return

    os.makedirs(output_dir, exist_ok=True)
    written_files = []
    for i, (info, packed) in enumerate(zip(node_info, node_packed)):
        fname = f"{output_prefix}_{i}_jobs.yaml"
        fpath = os.path.join(output_dir, fname)
        node_gpu_total = info["gpu_total_mib"] or gpu_total_mib
        write_node_yaml(
            fpath,
            defaults,
            packed,
            gpu_total_mib=node_gpu_total,
            safety_margin=safety_margin,
        )
        written_files.append((fpath, info["host"], len(packed)))

    # Also save the schedule info to the schedule dir if specified
    if schedule_dir:
        schedule_info = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": input_path,
            "total_jobs": total_jobs,
            "profile_steps": profile_steps,
            "safety_margin": safety_margin,
            "model_profiles": model_profiles,
            "nodes": [],
        }
        for i, (info, packed) in enumerate(zip(node_info, node_packed)):
            node_entry = {
                "index": i,
                "host": info["host"],
                "num_gpus": info["num_gpus"],
                "total_free_mib": info["total_free_mib"],
                "num_jobs": len(packed),
            }
            # Per-GPU summary
            gpu_summary = {}
            for pj in packed:
                for g in pj.get("assigned_gpus", []):
                    gpu_summary.setdefault(g, {"jobs": 0, "memory_mib": 0})
                    gpu_summary[g]["jobs"] += 1
                    gpu_summary[g]["memory_mib"] += pj.get("_memory_per_gpu", 0)
            node_entry["gpu_assignments"] = gpu_summary

            # Explicit per-job static assignments for daemon launch policy
            node_entry["job_assignments"] = [
                {
                    "name": pj.get("name"),
                    "model_key": pj.get("_model_key"),
                    "gpus_needed": pj.get("gpus", defaults.get("gpus", 1)),
                    "assigned_gpus": pj.get("assigned_gpus", []),
                    "memory_per_gpu_mib": pj.get("_memory_per_gpu"),
                    "static_gpu_budget_mib": int((info["gpu_total_mib"] or gpu_total_mib) * (1.0 - safety_margin)),
                }
                for pj in packed
            ]
            schedule_info["nodes"].append(node_entry)

        info_path = os.path.join(schedule_dir, "schedule_info.yaml")
        os.makedirs(schedule_dir, exist_ok=True)

        with open(info_path, "w") as f:
            yaml.safe_dump(schedule_info, f, default_flow_style=False, sort_keys=False)
        print(f"\n💾 Schedule info saved → {info_path}")

    # Print summary and next steps
    print(f"\n✅ Written {len(written_files)} file(s):")
    for fpath, host, count in written_files:
        print(f"   {fpath}  ({count} jobs for {host})")

    print(f"\n🚀 Next steps — on each node, start the daemon with --pre-grouped:")
    print(f"{'─' * 70}")
    sched_flag = f" -s {schedule_name}" if schedule_name else ""
    for i, (fpath, host, count) in enumerate(written_files):
        if count == 0:
            node_label = f"{schedule_name}_node_{i}" if schedule_name else f"node_{i}"
            print(f"\n  # {node_label} ({host}): 0 jobs — skip")
            continue
        fname = os.path.basename(fpath)
        node_label = f"{schedule_name}_node_{i}" if schedule_name else f"node_{i}"
        if is_local(host):
            print(f"\n  # {node_label} ({host}) — LOCAL:")
            print(f"  tmux new -s daemon")
            print(f"  python jobdaemon.py{sched_flag} start --gpus 0,1,2,3,4,5,6,7 --pre-grouped")
            print(f"  # (in another terminal)")
            print(f"  python jobdaemon.py{sched_flag} submit {fname}")
        else:
            print(f"\n  # {node_label} ({host}) — REMOTE:")
            print(f"  ssh root@{host}")
            print(f"  cd {os.getcwd()}")
            print(f"  tmux new -s daemon")
            print(f"  python jobdaemon.py{sched_flag} start --gpus 0,1,2,3,4,5,6,7 --pre-grouped")
            print(f"  # (in another terminal)")
            print(f"  python jobdaemon.py{sched_flag} submit {fname}")
    print(f"{'─' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart job scheduler with memory profiling and model-aware GPU grouping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Profile models and generate optimized schedule:\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\\n"
            "      --schedule-name imagenet_sweep --profile-steps 20\n"
            "\n"
            "  # Reuse cached profile (skip profiling):\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\\n"
            "      --schedule-name imagenet_sweep\n"
            "\n"
            "  # Manual memory override (MiB):\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\\n"
            "      --free-memory 320000 320000 310000\n"
            "\n"
            "  # Preview without writing:\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\\n"
            "      --dry-run\n"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input jobs.yaml",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        required=True,
        help="Node IPs or hostnames (e.g. 28.12.129.140 28.12.25.40 28.12.130.213)",
    )
    parser.add_argument(
        "--free-memory",
        nargs="+",
        type=int,
        default=None,
        help="Override free GPU memory (MiB) per node. "
             "Must match --nodes count. Skips SSH queries.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to write per-node YAML files (default: cwd)",
    )
    parser.add_argument(
        "--prefix",
        default="node",
        help="Filename prefix (default: 'node' → node_0_jobs.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing files",
    )
    parser.add_argument(
        "--schedule-name",
        default=None,
        help="Schedule name for caching profiles under schedules/<name>/. "
             "If a cached profile exists, profiling is skipped.",
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
        default=".",
        help="Working directory for profile runs (default: cwd)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.10,
        help="Safety margin for GPU memory bin packing (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip profiling even if no cache exists "
             "(use conservative 50%% GPU estimates)",
    )
    args = parser.parse_args()

    # Validate
    if args.free_memory and len(args.free_memory) != len(args.nodes):
        print(f"Error: --free-memory count ({len(args.free_memory)}) "
              f"!= --nodes count ({len(args.nodes)})")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    schedule(
        input_path=args.input,
        nodes=args.nodes,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        free_memory_override=args.free_memory,
        dry_run=args.dry_run,
        schedule_name=args.schedule_name,
        profile_steps=args.profile_steps,
        profile_gpu=args.profile_gpu,
        working_dir=args.working_dir,
        safety_margin=args.safety_margin,
        skip_profile=args.skip_profile,
    )


if __name__ == "__main__":
    main()
