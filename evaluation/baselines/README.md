# Baselines Runner (Ray)

This directory contains the baseline experiment launcher and Ray scheduler used to run `train.py` jobs in parallel.

## What is here

- `run_experiments.py`: Builds experiment definitions, fills defaults from `--project-root`, and hands jobs to the Ray runner.
- `ray_execution.py`: Ray orchestration loop (placement groups, launch, wait, terminate, cleanup).
- `ray_scheduler_logging.py`: Small scheduler logger that writes events to a separate log file.
- `configs/`: Per-model YAML config files.

## High-level flow

1. Parse CLI args and set defaults (`output-root`, `log-dir`, `status-file`) relative to `project-root`.
2. Build experiments from `configs/` and learning-rate sweep values.
3. In Ray mode, reserve a placement group on one eligible GPU group (slot resource), then launch `torchrun`.
4. Emit launch/termination scheduler events and release placement groups when jobs finish.

## Key properties

- Group scheduling is based on per-group custom Ray resources:
  - `<prefix>_<group_idx>_SLOTS` (group concurrency cap)
- Resource prefix is fixed to `GPU_GROUP`; group size is controlled by `--ray-gpus-per-group`.
- `--max-experiments-per-group` limits concurrent jobs per group.
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
