# Baselines Runner (Ray)

This directory contains the baseline experiment launcher and Ray scheduler used to run `train.py` jobs in parallel.

## What is here

- `run_experiments.py`: Builds experiment definitions, fills defaults from `--project-root`, and hands jobs to the Ray runner.
- `ray_execution.py`: Ray orchestration loop (placement groups, launch, wait, terminate, cleanup).
- `vram_estimation.py`: Dedicated VRAM estimation module (`estimate_job_vram_gb`) with model/optimizer-based heuristics used by the scheduler.
- `ray_scheduler_logging.py`: Small scheduler logger that writes events to a separate log file.
- `configs/`: Per-model YAML config files.

## High-level flow

1. Parse CLI args and set defaults (`output-root`, `log-dir`, `status-file`) relative to `project-root`.
2. Build experiments from `configs/` and learning-rate sweep values.
3. Call `vram_estimation.estimate_job_vram_gb(...)` and attach VRAM metadata (`estimated_vram_gb`, factors, log path, status path).
4. In Ray mode, sort jobs by estimated VRAM (largest first), reserve a placement group on one eligible GPU group, then launch `torchrun`.
5. Emit launch/termination scheduler events and release placement groups when jobs finish.

## Key properties

- Group scheduling is based on per-group custom Ray resources:
  - `<prefix>_<group_idx>_SLOTS` (group concurrency cap)
  - `<prefix>_<group_idx>_VRAM_GB` (group VRAM budget, integer units)
- Resource prefix is fixed to `GPU_GROUP`; group size is controlled by `--ray-gpus-per-group`.
- `--max-experiments-per-group` limits concurrent jobs per group independently of VRAM.
- VRAM estimation logic is intentionally isolated in `vram_estimation.py` so scheduling heuristics can be tuned without changing launcher/scheduler flow.
- Placement groups are **single-bundle per experiment** in this setup (single-node experiments).
- Finished jobs are skipped using:
  - global status file (default: `<project-root>/experiment_status.yaml`)
  - per-experiment status file (`<output-root>/<exp_name>/run_status.yaml`)
- Scheduler logs are written to:
  - `<log-dir>/ray_scheduler/ray_scheduler_<timestamp>.log`
- Training stdout/stderr for each job goes to:
  - `<log-dir>/<exp_name>_<timestamp>.txt`

## Notes

- `run_experiments.py` currently includes both ImageNet and CIFAR100 argument blocks; the active experiment list in `main()` determines which one is used.
- VRAM estimation is heuristic (not measured GPU memory), so tune `vram_estimation.py` factors as needed for your hardware/models.
