# Running LabelMix Experiments

End-to-end guide for generating experiment jobs, distributing them across nodes, managing the GPU job daemon, and monitoring training runs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Step 0 — Copy Data to RAM](#step-0--copy-data-to-ram)
4. [Step 1 — Generate jobs.yaml](#step-1--generate-jobsyaml)
5. [Step 2 — Distribute Jobs Across Nodes](#step-2--distribute-jobs-across-nodes)
6. [Step 3 — Launch Everything](#step-3--launch-everything)
7. [Step 4 — Monitor & Manage](#step-4--monitor--manage)
8. [Step 5 — Stop All Nodes](#step-5--stop-all-nodes)
9. [jobs.yaml Format Reference](#jobsyaml-format-reference)
10. [Schedule Isolation (`-s`)](#schedule-isolation--s)
11. [Daemon Advanced Options](#daemon-advanced-options)
12. [Training-Started Sentinel](#training-started-sentinel)
13. [GPU Monitoring](#gpu-monitoring)
14. [Troubleshooting](#troubleshooting)
15. [Complete End-to-End Example](#complete-end-to-end-example)

---

## Prerequisites

- Python 3.8+
- PyYAML (`pip install pyyaml`)
- `pynvml` (optional, for GPU memory monitoring): `pip install pynvml`
- `psutil` (optional, for process monitoring): `pip install psutil`
- tmux installed on all nodes
- Passwordless SSH between nodes (on the custom port `$JIZHI_SSH_PORT`)
- `NODE_IP_LIST` environment variable set (format: `IP1:GPU_COUNT,IP2:GPU_COUNT,...`)

---

## Architecture Overview

The experiment pipeline follows this flow:

```
generate_jobs.py  →  jobs.yaml  →  job_scheduler.py  →  node_X_jobs.yaml
                                                              ↓
                                              launch_all.sh (one command)
                                                     ↓
                                    ┌────────────────┼────────────────┐
                                    ▼                ▼                ▼
                               node_0            node_1           node_2
                            (local tmux)     (remote tmux)    (remote tmux)
                              daemon             daemon          daemon
                           -s schedule        -s schedule     -s schedule
                           8 GPUs each        8 GPUs each     8 GPUs each
```

**Key design principles:**
- Each node runs its own independent `jobdaemon.py` instance
- The `-s` (schedule name) flag isolates state per experiment run under `schedules/<name>/`
- `launch_all.sh` automates the entire cluster lifecycle (launch / status / jobs / stop)
- Node IPs and GPU counts come from the `NODE_IP_LIST` environment variable

---

## Step 0 — Copy Data to RAM

Before running experiments, copy the ImageNet-1K dataset to `/dev/shm` (RAM-backed tmpfs) to eliminate filesystem I/O bottlenecks. This only needs to be done **once per machine reboot** on **every node**.

```bash
# Copy to RAM (default: /dev/shm/imagenet-1k)
python copy_data_to_ram.py

# Preview what would be copied without doing it
python copy_data_to_ram.py --dry-run

# Force overwrite an existing copy
python copy_data_to_ram.py --force

# Check status of existing copy + /dev/shm usage
python copy_data_to_ram.py --status

# Verify integrity of an existing RAM copy
python copy_data_to_ram.py --verify

# Clean up — remove the RAM copy to free /dev/shm
python copy_data_to_ram.py --cleanup
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | `/apdcephfs_fsgm/.../data/imagenet-1k` | Source data directory on disk |
| `--dst` | `/dev/shm/imagenet-1k` | Destination in RAM |
| `--dry-run` | — | Show what would be copied without doing it |
| `--force` | — | Overwrite existing destination |
| `--verify` | — | Verify an existing RAM copy against source |
| `--cleanup` | — | Remove the RAM copy to free /dev/shm |
| `--status` | — | Show current copy status and /dev/shm usage |

> **Note:** After copying, `generate_jobs.py` is already configured to use `/dev/shm/imagenet-1k` as the `data_dir`. If you use a different `--dst`, update the `data_dir` in `generate_jobs.py` accordingly.

---

## Step 1 — Generate jobs.yaml

The `generate_jobs.py` script expands the experiment grid (K × alpha × loss × model) into a `jobs.yaml` file.

### Basic Usage

```bash
# Generate with defaults (2 GPUs/job, output to jobs.yaml)
python experiments/labelmix_imagenet1k/generate_jobs.py

# With 2 GPUs per job (explicit)
python experiments/labelmix_imagenet1k/generate_jobs.py --gpus-per-job 2

# Custom output path and training output root
python experiments/labelmix_imagenet1k/generate_jobs.py \
    --gpus-per-job 2 \
    -o jobs.yaml \
    --output-root ./output_runs/daemon

# Filter to specific models only
python experiments/labelmix_imagenet1k/generate_jobs.py \
    --model-filter vit-wee vit-little
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gpus-per-job` | `1` | GPUs per job. Affects per-GPU batch size (1024 ÷ gpus). |
| `-o`, `--output` | `jobs.yaml` | Output YAML file path |
| `--output-root` | `./output_runs/daemon` | Root directory for training outputs |
| `--max-retries` | `3` | Default max retries per job |
| `--model-filter` | all active models | Only generate jobs for these model tags |

### Current Phase 1 Grid

| Parameter | Values |
|-----------|--------|
| K (`labelmix_mix_k`) | 7, 8, 9, 10 |
| Alpha (`labelmix_alpha_min`) | 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0 |
| Loss (`labelmix_loss`) | `pl_loss`, `soft_ce` |
| Model | `vit-wee` (Phase 1 only) |

Total: 4 × 8 × 2 = **64 jobs** per model.

### Available Model Configs

| Tag | Config Path | Status |
|-----|-------------|--------|
| `vit-wee` | `configs/vit-wee.yaml` | ✅ Active |
| `vit-little` | `configs/vit-little.yaml` | Commented out |
| `vit-medium` | `configs/vit-medium.yaml` | Commented out |
| `vit-base` | `configs/vit-base.yaml` | Commented out |
| `mnv4-conv-medium` | `configs/mnv4-conv-medium.yaml` | Commented out |
| `mnv4-conv-large` | `configs/mnv4-conv-large.yaml` | Commented out |
| `mnv4-hybrid-medium` | `configs/mnv4-hybrid-medium.yaml` | Commented out |
| `mnv4-hybrid-large` | `configs/mnv4-hybrid-large.yaml` | Commented out |
| `resnet-50` | `configs/resnet-50.yaml` | Commented out |
| `resnet-101` | `configs/resnet-101.yaml` | Commented out |

To activate more models, uncomment them in `generate_jobs.py` under `MODEL_CONFIGS`.

---

## Step 2 — Distribute Jobs Across Nodes

The `job_scheduler.py` script splits a `jobs.yaml` into per-node YAML files based on available GPU memory. Nodes with more free memory get proportionally more jobs (Hamilton's largest-remainder method).

### Basic Usage

```bash
# Auto-discover GPU memory on all nodes (queries via SSH)
python job_scheduler.py --input jobs.yaml \
    --nodes 28.12.129.140 28.12.25.40 28.12.130.213

# Override with manual free-memory values (MiB per node)
python job_scheduler.py --input jobs.yaml \
    --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \
    --free-memory 320000 320000 310000

# Preview the split without writing files
python job_scheduler.py --input jobs.yaml \
    --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \
    --dry-run
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | **(required)** | Path to the input `jobs.yaml` |
| `--nodes` | **(required)** | Space-separated list of node IPs/hostnames |
| `--free-memory` | auto-query | Override free GPU memory (MiB) per node. Must match `--nodes` count. |
| `-o`, `--output-dir` | `.` | Directory to write per-node YAML files |
| `--prefix` | `node` | Filename prefix (e.g. `node` → `node_0_jobs.yaml`) |
| `--dry-run` | — | Print the distribution plan without writing files |

### Output

```
📋 Loaded 64 job(s) from jobs.yaml
📊 Job Distribution Plan:
─────────────────────────────────────────────────────────────────
Node                       Free Memory     Jobs    Weight
─────────────────────────────────────────────────────────────────
  node_0 (28.12.129.140)     320,000 MiB     22     33.7%
  node_1 (28.12.25.40)   320,000 MiB     22     33.7%
  node_2 (28.12.130.213)      310,000 MiB     20     32.6%
─────────────────────────────────────────────────────────────────
  TOTAL                      950,000 MiB     64

✅ Written 3 file(s):
   node_0_jobs.yaml  (22 jobs for 28.12.129.140)
   node_1_jobs.yaml  (22 jobs for 28.12.25.40)
   node_2_jobs.yaml  (20 jobs for 28.12.130.213)
```

---

## Step 3 — Launch Everything

The `launch_all.sh` script handles the entire cluster lifecycle in one command. It reads `NODE_IP_LIST` from the environment, creates tmux sessions on each node (local + remote via SSH), starts the job daemon, and submits the corresponding per-node job YAML.

### Environment Variable

```bash
# NODE_IP_LIST must be set. Format: IP1:GPU_COUNT,IP2:GPU_COUNT,...
# First entry = local (master) node; remaining = remote (SSH)
export NODE_IP_LIST="28.12.129.140:8,28.12.25.40:8,28.12.130.213:8"
```

### Configuration

Edit the top of `launch_all.sh` to customize:

```bash
PROJECT_DIR="/path/to/LabelMix"           # project root (on shared filesystem)
SSH_USER="root"                            # SSH user for remote nodes
SSH_PORT="${JIZHI_SSH_PORT:-36000}"         # SSH port (from env or default)
TMUX_SESSION="daemon"                      # tmux session name on each node
SCHEDULE_NAME="imagenet_sweep"             # isolates state under schedules/<name>/
```

### Commands

```bash
# Launch daemons and submit jobs on all nodes
./launch_all.sh

# Check tmux session and daemon health
./launch_all.sh --status

# Query job progress (pending/running/completed/failed) on all nodes
./launch_all.sh --jobs

# Attach to a specific node's tmux session
./launch_all.sh --logs 0    # local node
./launch_all.sh --logs 1    # remote node (SSH + tmux attach)

# Stop all daemons and kill tmux sessions
./launch_all.sh --stop

# Show help
./launch_all.sh --help
```

### What `./launch_all.sh` Does (Under the Hood)

1. Parses `NODE_IP_LIST` to get node IPs and GPU counts
2. For each node:
   - Verifies `node_X_jobs.yaml` exists
   - Creates a tmux session running `python jobdaemon.py -s <SCHEDULE_NAME> start --gpus <gpu_list>`
   - Opens a second tmux window that submits `node_X_jobs.yaml` after a 5s delay
3. For remote nodes, all commands are executed via SSH

### `--status` Output

```
  NODE     HOST               TMUX       DAEMON
  ──────── ────────────────── ────────── ──────────────
  node_0   28.12.129.140      alive      running
  node_1   28.12.25.40    alive      running
  node_2   28.12.130.213       alive      running
```

### `--jobs` Output

```
━━━ node_0 (28.12.129.140) ━━━
======================================================================
 Job Daemon | GPUs: 0,1,2,3,4,5,6,7 | Concurrent: 8 | OVERCOMMIT
======================================================================
 ✅ 12 completed | ⏳ 6 pending | 🔄 4 running | ❌ 0 failed
----------------------------------------------------------------------
 RUNNING:
   k8_a0.2_pl-loss     GPUs=[0,1] PID=12345 15m
   k8_a0.4_soft-ce     GPUs=[2,3] PID=12346 12m
   ...
======================================================================
```

---

## Step 4 — Monitor & Manage

### Check Status (Single Node)

If you're attached to a node's tmux session, you can query the local daemon directly:

```bash
# One-shot status table
python jobdaemon.py -s imagenet_sweep status

# Live auto-refresh (every 2 seconds)
python jobdaemon.py -s imagenet_sweep status --watch
```

### Check Status (All Nodes)

```bash
./launch_all.sh --jobs
```

### Pause / Resume

```bash
python jobdaemon.py -s imagenet_sweep pause
python jobdaemon.py -s imagenet_sweep resume
```

### Cancel Jobs

```bash
# Cancel a specific job
python jobdaemon.py -s imagenet_sweep cancel <job_name>

# Cancel all pending jobs
python jobdaemon.py -s imagenet_sweep cancel --all-pending

# Cancel all pending + running jobs
python jobdaemon.py -s imagenet_sweep cancel --all
```

### Retry Failed Jobs

```bash
# Retry a specific failed job
python jobdaemon.py -s imagenet_sweep retry <job_name>

# Retry all failed jobs
python jobdaemon.py -s imagenet_sweep retry --all-failed
```

---

## Step 5 — Stop All Nodes

```bash
# Stop daemons and kill tmux sessions on all nodes
./launch_all.sh --stop
```

This kills the tmux session on each node and any stray `jobdaemon.py` processes.

---

## jobs.yaml Format Reference

The `generate_jobs.py` script outputs a YAML file with this structure:

```yaml
defaults:
  gpus: 2              # GPUs per job
  max_retries: 3       # Max retries before marking failed
  working_dir: /path/to/LabelMix

jobs:
  - name: k7_a0.2_pl-loss
    cmd: >-
      torchrun --nproc_per_node={gpus} --master_port={port}
      train.py --config path/to/vit-wee.yaml
      --labelmix --labelmix-mix-k 7 --labelmix-alpha-min 0.2
      --labelmix-loss pl_loss ...

  - name: k7_a0.2_soft-ce
    cmd: >-
      torchrun --nproc_per_node={gpus} --master_port={port}
      train.py --config path/to/vit-wee.yaml
      --labelmix --labelmix-mix-k 7 --labelmix-alpha-min 0.2
      --labelmix-loss soft_ce ...
```

**Placeholders** resolved at launch time by the daemon:
- `{gpus}` → number of GPUs assigned (from `defaults.gpus`)
- `{port}` → auto-assigned free port from the port range

---

## Schedule Isolation (`-s`)

The `-s`/`--schedule-name` flag on `jobdaemon.py` isolates all daemon state for a given experiment run. This prevents stale state from previous runs interfering with new ones.

### How It Works

```bash
# Without -s: uses default shared directories
python jobdaemon.py start --gpus 0-7
#   state → ./state/
#   inbox → ./inbox/
#   logs  → ./logs/jobs/

# With -s: everything under schedules/<name>/
python jobdaemon.py -s imagenet_sweep start --gpus 0-7
#   state → ./schedules/imagenet_sweep/state/
#   inbox → ./schedules/imagenet_sweep/inbox/
#   logs  → ./schedules/imagenet_sweep/logs/
```

### Directory Structure

```
schedules/
├── imagenet_sweep/        ← current experiment
│   ├── state/             ← daemon state (job queue, GPU assignments)
│   ├── inbox/             ← submitted job files
│   └── logs/              ← per-job log files
└── next_experiment/       ← future experiment (completely isolated)
    ├── state/
    ├── inbox/
    └── logs/
```

### Key Benefits

- **No stale state:** Changing `SCHEDULE_NAME` in `launch_all.sh` ensures a fresh start
- **Resumable:** Reuse the same schedule name to resume where you left off
- **Inspectable:** Logs and state for each run are neatly organized
- **Backward compatible:** Omitting `-s` uses the old default directories

> **Note:** `launch_all.sh` always passes `-s ${SCHEDULE_NAME}` to every `jobdaemon.py` invocation. Change `SCHEDULE_NAME` at the top of the script for each new experiment batch.

---

## Daemon Advanced Options

### Starting the Daemon Directly

While `launch_all.sh` is the recommended way to start daemons, you can also start them manually:

```bash
python jobdaemon.py -s imagenet_sweep start \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-retries 3 \
    --poll-interval 5 \
    --port-range 29500-29599 \
    --burst-delay 15 \
    --stabilization-wait 300 \
    --memory-safety-margin 0.15 \
    --oom-wait-timeout 600 \
    --training-started-timeout 600 \
    --training-started-poll 5
```

### `start` Options

| Flag | Default | Description |
|------|---------|-------------|
| `-s`, `--schedule-name` | `None` | Schedule name — isolates state/inbox/logs under `schedules/<name>/` |
| `--gpus` | **(required)** | Comma-separated GPU IDs (e.g. `0,1,2,3,4,5,6,7`) |
| `--max-retries` | `3` | Max retries per job before marking as failed |
| `--poll-interval` | `5` | Seconds between daemon poll cycles |
| `--port-range` | `29500-29599` | Port range for `torchrun --master_port` |
| `--state-dir` | `./state` | Directory for daemon persistent state (overridden by `-s`) |
| `--inbox-dir` | `./inbox` | Directory the daemon watches for new job submissions (overridden by `-s`) |
| `--log-dir` | `./logs/jobs` | Directory for per-job log files (overridden by `-s`) |
| `--working-dir` | `.` | Working directory for spawned subprocesses |
| `--oom-threshold` | `120` | If a job dies within this many seconds, treat as OOM crash |
| `--saturation-cooldown` | `30` | Seconds to wait after OOM before trying next launch |
| `--burst-delay` | `15` | Seconds of settling time after training-started confirmed |
| `--gpu-check-delay` | `30` | Grace period before checking if a PID appears on GPU |
| `--burst-gpu-wait` | `300` | Max seconds to wait for a job to appear on GPU (fallback) |
| `--stabilization-wait` | `300` | Base seconds for saturation checkpoint (progressive: 1st=1×, 2nd=2×, ...) |
| `--memory-safety-margin` | `0.15` | OOM prediction safety margin (0.15 = 15% headroom) |
| `--oom-wait-timeout` | `600` | Max seconds to wait for GPU memory headroom before queueing |
| `--training-started-timeout` | `600` | Max seconds to wait for the `.training_started` sentinel per job |
| `--training-started-poll` | `5` | Seconds between polls for the `.training_started` sentinel |

### GPU-Sequential Launch with Training-Started Gate

The daemon uses a two-phase launching strategy designed to pack as many small
experiments as possible onto each GPU:

```
┌──────────────────────────────────────────────────────────────┐
│                     BURST PHASE (GPU-by-GPU)                 │
│                                                              │
│  1. Probe free memory on all GPUs at startup                 │
│  2. Distribute total jobs proportionally to free memory      │
│  3. Start with GPU 0:                                        │
│     a) Launch job → GPU 0                                    │
│     b) Wait for .training_started sentinel (not log parse!)  │
│     c) Training confirmed → launch NEXT job → same GPU 0    │
│     d) Repeat until GPU 0's allocation full or OOM predicted │
│  4. Advance to GPU 1, repeat                                 │
│  5. Continue until all GPUs filled                           │
│                                                              │
│  Saturation checkpoints every num_gpus jobs (progressive     │
│  wait: 1st=1×base, 2nd=2×base, ...) to catch late OOM       │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                     FIFO PHASE (1-out-1-in)                  │
│                                                              │
│  Once all GPUs filled or OOM exhausts all GPUs:              │
│  - When a running job completes → launch next pending job    │
│  - OOM prediction gate still active before each launch       │
│  - Round-robin GPU assignment for replacement jobs           │
└──────────────────────────────────────────────────────────────┘
```

### Memory-Proportional Job Distribution

At daemon startup, free memory is probed on every GPU via NVML. Jobs are then
distributed proportionally using Hamilton's largest-remainder method:

```
📊 GPU free memory at startup: GPU 0: 40960 MiB, GPU 1: 40960 MiB, GPU 2: 38912 MiB, GPU 3: 40960 MiB
📊 Job distribution by memory: GPU 0: 17 jobs, GPU 1: 17 jobs, GPU 2: 13 jobs, GPU 3: 17 jobs (total=64/64)
```

### OOM Prediction

The daemon monitors GPU memory usage and predicts whether launching a new job would cause OOM:
- It tracks average per-job memory usage from running jobs
- Before each launch, it checks if the target GPU has enough free memory (avg per GPU-slot + safety margin)
- If not enough memory, it waits briefly for headroom
- If still no room → **advance to next GPU** (not immediate FIFO switch)
- Only switches to FIFO when **all GPUs** are exhausted

### Tuning Tips

| Scenario | Adjustment |
|----------|------------|
| Jobs are small, want more overcommit | Lower `--memory-safety-margin` (e.g. `0.05`) |
| Getting false OOM predictions | Lower `--memory-safety-margin` |
| OOM crashes happening | Raise `--memory-safety-margin` (e.g. `0.25`) |
| Checkpoint waits too long | Lower `--stabilization-wait` (e.g. `120`) |
| Models slow to load/compile | Raise `--training-started-timeout` (e.g. `900`) |
| Want faster sentinel polling | Lower `--training-started-poll` (e.g. `2`) |
| Want faster burst launches | Lower `--burst-delay` (e.g. `5`) |
| Sentinel not reliable for some jobs | Raise `--training-started-step` in train.py (e.g. `5`) |
| Want to disable sentinel entirely | Set `--training-started-step 0` in train.py cmd |

---

## Training-Started Sentinel

The daemon does **not** parse training logs to detect whether a job has started training. Instead, `train.py` writes a lightweight sentinel file once training begins.

### How It Works

1. `train.py` accepts `--training-started-step N` (default: `2`)
2. After completing N training steps, the primary rank writes a `.training_started` file to the experiment's output directory (`<output>/<experiment>/`)
3. The daemon polls for this file to confirm the job is actively training (not just loading model weights, compiling, or initializing NCCL)
4. Once confirmed, the daemon knows GPU memory has stabilized and proceeds to launch the next job

### Sentinel File Location

```
<output>/<experiment>/.training_started
```

For example, with `--output ./output_runs/daemon --experiment k7_a0.2_pl-loss`:
```
./output_runs/daemon/k7_a0.2_pl-loss/.training_started
```

### Sentinel File Content

```
2
2026-03-16T03:22:46.123456
```

Line 1 is the global step number, line 2 is the ISO timestamp.

### train.py Options

| Flag | Default | Description |
|------|---------|-------------|
| `--training-started-step` | `2` | Write `.training_started` after this many training steps. Set `0` to disable. |

### How the Daemon Uses It

The daemon extracts `--output` and `--experiment` from each job's command line to derive the sentinel path. For each job launched during burst:

1. **Launch** → start the subprocess
2. **Poll** → check for `.training_started` every `--training-started-poll` seconds
3. **Confirmed** → log success, apply a short settling delay, then launch next job
4. **Timeout** → if job is still alive but sentinel missing after `--training-started-timeout` seconds, proceed cautiously
5. **Job died** → check for OOM, requeue or advance to next GPU

**Fallback:** If the daemon can't determine the output directory from the command (e.g., non-standard cmd), it falls back to GPU presence detection via NVML.

---

## GPU Monitoring

The `gpu_monitor.sh` script provides cluster-wide GPU visibility. It reads `NODE_IP_LIST` automatically.

### Usage

```bash
# One-shot summary across all nodes (utilization, memory, temperature)
./gpu_monitor.sh

# Launch interactive nvitop on a specific node
./gpu_monitor.sh nvitop 0    # node 0 (local)
./gpu_monitor.sh nvitop 1    # node 1 (remote, via SSH)
./gpu_monitor.sh nvitop 2    # node 2 (remote, via SSH)
```

### Example Output

```
========================================
  Cluster GPU Monitor
  Nodes: 3  |  GPUs/node: 8  |  GPU: H20
  SSH Port: 36000
========================================

=== Node 0 (launcher) - 28.12.129.140 ===
0, 85 %, 18432 MiB, 98304 MiB, 52
1, 72 %, 16384 MiB, 98304 MiB, 50
...

=== Node 1 (worker-0) - 28.12.25.40 ===
0, 90 %, 19456 MiB, 98304 MiB, 54
...
```

---

## Troubleshooting

### Daemon loading old/stale jobs

If the daemon picks up jobs from a previous experiment:
- You're likely missing the `-s` flag, so it reads from the shared `./state/` directory
- Fix: always use `-s <schedule_name>` or use `launch_all.sh` which passes it automatically
- Change `SCHEDULE_NAME` in `launch_all.sh` for each new experiment batch

### `--jobs` shows "Could not query status" on remote nodes

The remote SSH query timed out or the daemon isn't running:
```bash
# Check if daemon is alive
./launch_all.sh --status

# If tmux is alive but daemon died, stop and relaunch
./launch_all.sh --stop
./launch_all.sh
```

### Daemon stopped launching but jobs could fit

The OOM prediction gate may be too conservative. Check the daemon logs for messages like:
```
⚠️ OOM PREDICTION on GPU 3: k7_a0.2_pl-loss at risk — ...
```
Try lowering `--memory-safety-margin`:
```bash
python jobdaemon.py -s imagenet_sweep start --gpus 0-7 --memory-safety-margin 0.05
```

### Jobs dying as OOM shortly after launch

Raise the safety margin:
```bash
python jobdaemon.py -s imagenet_sweep start --gpus 0-7 --memory-safety-margin 0.25
```

### Training-started sentinel never written

If the daemon logs show timeouts waiting for `.training_started`:

1. **Check the step threshold:** By default, train.py writes the sentinel at step 2. If your jobs resume from a later checkpoint, `start_step` might already be past the threshold. The sentinel writes at `start_step + training_started_step`.
2. **Check the output directory:** Ensure `--output` and `--experiment` are present in the job command. The daemon derives the sentinel path from these.
3. **Check permissions:** The training process must have write access to the output directory.
4. **Increase the step count:** If model compilation takes very long, set `--training-started-step 5` or higher in the train.py command.
5. **Disable sentinel:** Set `--training-started-step 0` and the daemon will fall back to GPU presence detection.

```bash
# Check if sentinel exists for a specific experiment
ls -la ./output_runs/daemon/k7_a0.2_pl-loss/.training_started
```

### Job logs location

Per-job logs are stored under the schedule's log directory:
```bash
# With -s imagenet_sweep:
tail -f schedules/imagenet_sweep/logs/k7_a0.2_pl-loss.log

# Without -s (default):
tail -f logs/jobs/k7_a0.2_pl-loss.log
```

### Data not found errors

Ensure you've run `copy_data_to_ram.py` first and that `/dev/shm/imagenet-1k` exists:
```bash
python copy_data_to_ram.py --status
```

### Recovering from a daemon crash

The daemon persists state. Restart the daemon with the same schedule name and it will resume where it left off:
```bash
python jobdaemon.py -s imagenet_sweep start --gpus 0,1,2,3,4,5,6,7
```

Or simply re-run `./launch_all.sh` — it will detect existing tmux sessions and skip them (use `--stop` first for a clean restart).

---

## Complete End-to-End Example

```bash
# 0. Set environment (usually already set by the cluster)
export NODE_IP_LIST="28.12.129.140:8,28.12.25.40:8,28.12.130.213:8"

# 1. Copy data to RAM on all nodes (once per reboot)
python copy_data_to_ram.py

# 2. Generate the experiment grid
python experiments/labelmix_imagenet1k/generate_jobs.py \
    --gpus-per-job 2 \
    -o jobs.yaml

# 3. Distribute jobs across nodes (auto-queries GPU memory via SSH)
python job_scheduler.py --input jobs.yaml \
    --nodes 28.12.129.140 28.12.25.40 28.12.130.213

# 4. (Optional) Edit SCHEDULE_NAME in launch_all.sh for this batch
#    Default: SCHEDULE_NAME="imagenet_sweep"

# 5. Launch everything (one command!)
./launch_all.sh

# 6. Monitor
./launch_all.sh --status   # daemon health
./launch_all.sh --jobs     # job progress on all nodes
./gpu_monitor.sh           # GPU utilization across cluster
./gpu_monitor.sh nvitop 0  # interactive GPU monitor on node 0

# 7. When done
./launch_all.sh --stop

# 8. Clean up RAM (optional)
python copy_data_to_ram.py --cleanup
```

### Starting a New Experiment Batch

To run a different set of experiments without interference:

```bash
# 1. Change SCHEDULE_NAME in launch_all.sh (e.g. "phase2_sweep")
# 2. Generate new jobs.yaml
# 3. Distribute across nodes
# 4. ./launch_all.sh
```

Previous experiment state remains intact under `schedules/imagenet_sweep/` for inspection.
