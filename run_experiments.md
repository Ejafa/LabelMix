
# Running LabelMix Experiments

End-to-end guide for generating experiment jobs, managing the GPU job daemon, and monitoring training runs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 0 — Copy Data to RAM](#step-0--copy-data-to-ram)
3. [Step 1 — Generate jobs.yaml](#step-1--generate-jobsyaml)
4. [Step 2 — Start the Job Daemon](#step-2--start-the-job-daemon)
5. [Step 3 — Submit Jobs](#step-3--submit-jobs)
6. [Step 4 — Monitor & Manage](#step-4--monitor--manage)
7. [jobs.yaml Format Reference](#jobsyaml-format-reference)
8. [Daemon Advanced Options](#daemon-advanced-options)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.8+
- PyYAML (`pip install pyyaml`)
- `pynvml` (optional, for GPU memory monitoring): `pip install pynvml`
- `psutil` (optional, for process monitoring): `pip install psutil`
- A tmux session (recommended for daemon persistence)

---

## Step 0 — Copy Data to RAM

Before running experiments, copy the ImageNet-1K dataset to `/dev/shm` (RAM-backed tmpfs) to eliminate filesystem I/O bottlenecks. This only needs to be done **once per machine reboot**.

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

The `generate_jobs.py` script expands the experiment grid (K × alpha × loss × model) into a `jobs.yaml` file that the daemon consumes.

### Basic Usage

```bash
# Generate with defaults (1 GPU/job, output to jobs.yaml)
python experiments/labelmix_imagenet1k/generate_jobs.py

# With 2 GPUs per job
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

## Step 2 — Start the Job Daemon

The daemon runs persistently (ideally in tmux), manages GPU assignment, process lifecycle, auto-retry, and OOM protection.

### Quick Start

```bash
# Open a tmux session
tmux new -s daemon

# Start daemon on all 8 GPUs
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7

# Or on specific GPUs
python jobdaemon.py start --gpus 0,1,2,3
```

### Full Example with All Options

```bash
python jobdaemon.py start \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-retries 3 \
    --poll-interval 5 \
    --port-range 29500-29599 \
    --state-dir ./state \
    --inbox-dir ./inbox \
    --log-dir ./logs/jobs \
    --working-dir . \
    --burst-delay 15 \
    --stabilization-wait 300 \
    --memory-safety-margin 0.15 \
    --oom-wait-timeout 600
```

### `start` Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gpus` | **(required)** | Comma-separated GPU IDs (e.g. `0,1,2,3,4,5,6,7`) |
| `--max-retries` | `3` | Max retries per job before marking as failed |
| `--poll-interval` | `5` | Seconds between daemon poll cycles |
| `--port-range` | `29500-29599` | Port range for `torchrun --master_port` |
| `--state-dir` | `./state` | Directory for daemon persistent state |
| `--inbox-dir` | `./inbox` | Directory the daemon watches for new job submissions |
| `--log-dir` | `./logs/jobs` | Directory for per-job log files |
| `--working-dir` | `.` | Working directory for spawned subprocesses |
| `--oom-threshold` | `120` | If a job dies within this many seconds, treat as OOM crash |
| `--saturation-cooldown` | `30` | Seconds to wait after OOM before trying next launch |
| `--burst-delay` | `15` | Seconds between consecutive launches in burst mode |
| `--gpu-check-delay` | `30` | Grace period before checking if a PID appears on GPU |
| `--burst-gpu-wait` | `300` | Max seconds to wait for a job to appear on GPU in burst mode |
| `--stabilization-wait` | `300` | Base seconds for saturation checkpoint (progressive: 1st=1×, 2nd=2×, ...) |
| `--memory-safety-margin` | `0.15` | OOM prediction safety margin (0.15 = 15% headroom) |
| `--oom-wait-timeout` | `600` | Max seconds to wait for GPU memory headroom before queueing |

---

## Step 3 — Submit Jobs

### Submit from a YAML File

```bash
# Submit all jobs from the generated YAML
python jobdaemon.py submit jobs.yaml
```

### Submit a Single Ad-hoc Command

```bash
python jobdaemon.py submit \
    --cmd "torchrun --nproc_per_node=2 train.py --config path/to/config.yaml" \
    --name "my_experiment" \
    --gpus-needed 2
```

### Submit Options

| Flag | Default | Description |
|------|---------|-------------|
| `yaml_file` | — | Path to `jobs.yaml` with job definitions |
| `--cmd` | — | Submit a single command directly |
| `--name` | — | Job name (for `--cmd` mode) |
| `--gpus-needed` | — | GPUs needed (for `--cmd` mode) |
| `--inbox-dir` | `./inbox` | Inbox directory for the daemon |
| `--state-dir` | `./state` | State directory |

---

## Step 4 — Monitor & Manage

### Check Status

```bash
# One-shot status table
python jobdaemon.py status

# Live auto-refresh (every 2 seconds)
python jobdaemon.py status --watch
```

### Pause / Resume

```bash
# Pause launching new jobs (running jobs continue)
python jobdaemon.py pause

# Resume launching
python jobdaemon.py resume
```

### Cancel Jobs

```bash
# Cancel a specific job
python jobdaemon.py cancel <job_name>

# Cancel all pending jobs
python jobdaemon.py cancel --all-pending

# Cancel all pending + running jobs
python jobdaemon.py cancel --all
```

### Retry Failed Jobs

```bash
# Retry a specific failed job
python jobdaemon.py retry <job_name>

# Retry all failed jobs
python jobdaemon.py retry --all-failed
```

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

## Daemon Advanced Options

### Overcommit & Burst Mode

The daemon uses a two-phase launching strategy:

1. **Burst phase**: Jobs are launched rapidly in round-robin across GPUs. After every `num_gpus` jobs, a **saturation checkpoint** pauses and waits to verify all jobs survive (progressive wait: 1st=5min, 2nd=10min, etc.).

2. **FIFO phase**: Once the burst detects instability (OOM, crash), it switches to conservative one-at-a-time launching with memory checks between each.

### OOM Prediction

The daemon monitors GPU memory usage and predicts whether launching a new job would cause OOM:
- It tracks average per-job memory usage from running jobs
- Before each launch, it checks if the target GPU(s) have enough free memory (avg per GPU-slot + safety margin)
- If not enough memory, it waits up to `--oom-wait-timeout` seconds for jobs to finish and free memory
- If still not enough, remaining jobs are queued for later

### Tuning Tips

| Scenario | Adjustment |
|----------|------------|
| Jobs are small, want more overcommit | Lower `--memory-safety-margin` (e.g. `0.05`) |
| Getting false OOM predictions | Lower `--memory-safety-margin` |
| OOM crashes happening | Raise `--memory-safety-margin` (e.g. `0.25`) |
| Checkpoint waits too long | Lower `--stabilization-wait` (e.g. `120`) |
| Jobs need more startup time | Raise `--burst-gpu-wait` (e.g. `600`) |
| Want faster burst launches | Lower `--burst-delay` (e.g. `5`) |

---

## Troubleshooting

### Daemon stopped launching but jobs could fit

The OOM prediction gate may be too conservative. Check the daemon logs for messages like:
```
GPU X: YYYY MiB free < ZZZZ MiB needed
```
Try lowering `--memory-safety-margin`:
```bash
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7 --memory-safety-margin 0.05
```

### Jobs dying as OOM shortly after launch

Raise the safety margin and/or lower the number of GPUs per job:
```bash
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7 --memory-safety-margin 0.25
```

### Jobs logs location

Per-job logs are stored at `./logs/jobs/<job_name>.log` by default. Check them for training errors:
```bash
tail -f logs/jobs/k7_a0.2_pl-loss.log
```

### Data not found errors

Ensure you've run `copy_data_to_ram.py` first and that `/dev/shm/imagenet-1k` exists:
```bash
python copy_data_to_ram.py --status
```

### Recovering from a daemon crash

The daemon persists state to `./state/`. Just restart the daemon with the same `--state-dir` and it will resume where it left off:
```bash
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7 --state-dir ./state
```

---

## Complete End-to-End Example

```bash
# 1. Copy data to RAM (once per reboot)
python copy_data_to_ram.py

# 2. Generate the experiment grid
python experiments/labelmix_imagenet1k/generate_jobs.py \
    --gpus-per-job 2 \
    -o jobs.yaml

# 3. Start the daemon in tmux
tmux new -s daemon
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7

# 4. In another terminal, submit jobs
python jobdaemon.py submit jobs.yaml

# 5. Monitor progress
python jobdaemon.py status --watch

# 6. When done, clean up RAM
python copy_data_to_ram.py --cleanup
```

# Running LabelMix Experiments

End-to-end guide for generating experiment jobs, managing the GPU job daemon, and monitoring training runs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 0 — Copy Data to RAM](#step-0--copy-data-to-ram)
3. [Step 1 — Generate jobs.yaml](#step-1--generate-jobsyaml)
4. [Step 2 — Start the Job Daemon](#step-2--start-the-job-daemon)
5. [Step 3 — Submit Jobs](#step-3--submit-jobs)
6. [Step 4 — Monitor & Manage](#step-4--monitor--manage)
7. [jobs.yaml Format Reference](#jobsyaml-format-reference)
8. [Daemon Advanced Options](#daemon-advanced-options)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.8+
- PyYAML (`pip install pyyaml`)
- `pynvml` (optional, for GPU memory monitoring): `pip install pynvml`
- `psutil` (optional, for process monitoring): `pip install psutil`
- A tmux session (recommended for daemon persistence)

---

## Step 0 — Copy Data to RAM

Before running experiments, copy the ImageNet-1K dataset to `/dev/shm` (RAM-backed tmpfs) to eliminate filesystem I/O bottlenecks. This only needs to be done **once per machine reboot**.

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

The `generate_jobs.py` script expands the experiment grid (K × alpha × loss × model) into a `jobs.yaml` file that the daemon consumes.

### Basic Usage

```bash
# Generate with defaults (1 GPU/job, output to jobs.yaml)
python experiments/labelmix_imagenet1k/generate_jobs.py

# With 2 GPUs per job
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
| `--gpus-per-job` | `1` | GPUs per job. Affects per-GPU batch size (1024 / gpus). |
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

Total: 4 x 8 x 2 = **64 jobs** per model.

### Available Model Configs

| Tag | Config Path | Status |
|-----|-------------|--------|
| `vit-wee` | `configs/vit-wee.yaml` | Active |
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

## Step 2 — Start the Job Daemon

The daemon runs persistently (ideally in tmux), manages GPU assignment, process lifecycle, auto-retry, and OOM protection.

### Quick Start

```bash
# Open a tmux session
tmux new -s daemon

# Start daemon on all 8 GPUs
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7

# Or on specific GPUs
python jobdaemon.py start --gpus 0,1,2,3
```

### Full Example with All Options

```bash
python jobdaemon.py start \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-retries 3 \
    --poll-interval 5 \
    --port-range 29500-29599 \
    --state-dir ./state \
    --inbox-dir ./inbox \
    --log-dir ./logs/jobs \
    --working-dir . \
    --burst-delay 15 \
    --stabilization-wait 300 \
    --memory-safety-margin 0.15 \
    --oom-wait-timeout 600
```

### `start` Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gpus` | **(required)** | Comma-separated GPU IDs (e.g. `0,1,2,3,4,5,6,7`) |
| `--max-retries` | `3` | Max retries per job before marking as failed |
| `--poll-interval` | `5` | Seconds between daemon poll cycles |
| `--port-range` | `29500-29599` | Port range for `torchrun --master_port` |
| `--state-dir` | `./state` | Directory for daemon persistent state |
| `--inbox-dir` | `./inbox` | Directory the daemon watches for new job submissions |
| `--log-dir` | `./logs/jobs` | Directory for per-job log files |
| `--working-dir` | `.` | Working directory for spawned subprocesses |
| `--oom-threshold` | `120` | If a job dies within this many seconds, treat as OOM crash |
| `--saturation-cooldown` | `30` | Seconds to wait after OOM before trying next launch |
| `--burst-delay` | `15` | Seconds between consecutive launches in burst mode |
| `--gpu-check-delay` | `30` | Grace period before checking if a PID appears on GPU |
| `--burst-gpu-wait` | `300` | Max seconds to wait for a job to appear on GPU in burst mode |
| `--stabilization-wait` | `300` | Base seconds for saturation checkpoint (progressive: 1st=1x, 2nd=2x, ...) |
| `--memory-safety-margin` | `0.15` | OOM prediction safety margin (0.15 = 15% headroom) |
| `--oom-wait-timeout` | `600` | Max seconds to wait for GPU memory headroom before queueing |

---

## Step 3 — Submit Jobs

### Submit from a YAML File

```bash
# Submit all jobs from the generated YAML
python jobdaemon.py submit jobs.yaml
```

### Submit a Single Ad-hoc Command

```bash
python jobdaemon.py submit \
    --cmd "torchrun --nproc_per_node=2 train.py --config path/to/config.yaml" \
    --name "my_experiment" \
    --gpus-needed 2
```

### Submit Options

| Flag | Default | Description |
|------|---------|-------------|
| `yaml_file` | — | Path to `jobs.yaml` with job definitions |
| `--cmd` | — | Submit a single command directly |
| `--name` | — | Job name (for `--cmd` mode) |
| `--gpus-needed` | — | GPUs needed (for `--cmd` mode) |
| `--inbox-dir` | `./inbox` | Inbox directory for the daemon |
| `--state-dir` | `./state` | State directory |

---

## Step 4 — Monitor & Manage

### Check Status

```bash
# One-shot status table
python jobdaemon.py status

# Live auto-refresh (every 2 seconds)
python jobdaemon.py status --watch
```

### Pause / Resume

```bash
# Pause launching new jobs (running jobs continue)
python jobdaemon.py pause

# Resume launching
python jobdaemon.py resume
```

### Cancel Jobs

```bash
# Cancel a specific job
python jobdaemon.py cancel <job_name>

# Cancel all pending jobs
python jobdaemon.py cancel --all-pending

# Cancel all pending + running jobs
python jobdaemon.py cancel --all
```

### Retry Failed Jobs

```bash
# Retry a specific failed job
python jobdaemon.py retry <job_name>

# Retry all failed jobs
python jobdaemon.py retry --all-failed
```

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
- `{gpus}` — number of GPUs assigned (from `defaults.gpus`)
- `{port}` — auto-assigned free port from the port range

---

## Daemon Advanced Options

### Overcommit & Burst Mode

The daemon uses a two-phase launching strategy:

1. **Burst phase**: Jobs are launched rapidly in round-robin across GPUs. After every `num_gpus` jobs, a **saturation checkpoint** pauses and waits to verify all jobs survive (progressive wait: 1st=5min, 2nd=10min, etc.).

2. **FIFO phase**: Once the burst detects instability (OOM, crash), it switches to conservative one-at-a-time launching with memory checks between each.

### OOM Prediction

The daemon monitors GPU memory usage and predicts whether launching a new job would cause OOM:
- It tracks average per-job memory usage from running jobs
- Before each launch, it checks if the target GPU(s) have enough free memory (avg per GPU-slot + safety margin)
- If not enough memory, it waits up to `--oom-wait-timeout` seconds for jobs to finish and free memory
- If still not enough, remaining jobs are queued for later

### Tuning Tips

| Scenario | Adjustment |
|----------|------------|
| Jobs are small, want more overcommit | Lower `--memory-safety-margin` (e.g. `0.05`) |
| Getting false OOM predictions | Lower `--memory-safety-margin` |
| OOM crashes happening | Raise `--memory-safety-margin` (e.g. `0.25`) |
| Checkpoint waits too long | Lower `--stabilization-wait` (e.g. `120`) |
| Jobs need more startup time | Raise `--burst-gpu-wait` (e.g. `600`) |
| Want faster burst launches | Lower `--burst-delay` (e.g. `5`) |

---

## Troubleshooting

### Daemon stopped launching but jobs could fit

The OOM prediction gate may be too conservative. Check the daemon logs for messages like:
```
GPU X: YYYY MiB free < ZZZZ MiB needed
```
Try lowering `--memory-safety-margin`:
```bash
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7 --memory-safety-margin 0.05
```

### Jobs dying as OOM shortly after launch

Raise the safety margin:
```bash
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7 --memory-safety-margin 0.25
```

### Job logs location

Per-job logs are stored at `./logs/jobs/<job_name>.log` by default. Check them for training errors:
```bash
tail -f logs/jobs/k7_a0.2_pl-loss.log
```

### Data not found errors

Ensure you've run `copy_data_to_ram.py` first and that `/dev/shm/imagenet-1k` exists:
```bash
python copy_data_to_ram.py --status
```

### Recovering from a daemon crash

The daemon persists state to `./state/`. Just restart with the same `--state-dir` and it resumes:
```bash
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7 --state-dir ./state
```

---

## Complete End-to-End Example

```bash
# 1. Copy data to RAM (once per reboot)
python copy_data_to_ram.py

# 2. Generate the experiment grid
python experiments/labelmix_imagenet1k/generate_jobs.py \
    --gpus-per-job 2 \
    -o jobs.yaml

# 3. Start the daemon in tmux
tmux new -s daemon
python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7

# 4. In another terminal, submit jobs
python jobdaemon.py submit jobs.yaml

# 5. Monitor progress
python jobdaemon.py status --watch

# 6. When done, clean up RAM
python copy_data_to_ram.py --cleanup
```
