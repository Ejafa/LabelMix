#!/usr/bin/env python3
"""Generate jobs.yaml for the LabelMix ImageNet-1K experiment family.

Phase 1 — Fixed-K × Fixed-alpha grid search on vit-wee
========================================================
Goal: determine the best (K, alpha) pair before exploring scheduling.

Grid:
    K     = [7, 8, 9, 10] # 3, 4, 5, 6,
    alpha = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    loss  = [pl_loss, soft_ce]
    => 8 × 8 × 2 = 128 full training runs, no early stopping.

All 10 baseline model configs are available under ``configs/`` but only
vit-wee is active for Phase 1. Uncomment others when ready.

Usage::

    # From the project root:
    python experiments/labelmix_imagenet1k/generate_jobs.py

    # With options:
    python experiments/labelmix_imagenet1k/generate_jobs.py \\
        --gpus-per-job 8 \\
        --output jobs.yaml \\
        --output-root ./output_runs/daemon \\
        --max-retries 3 \\
        --model-filter vit-wee

    # Then submit to the daemon:
    python jobdaemon.py submit jobs.yaml
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any, Dict, List

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIGS_DIR = os.path.join(_THIS_DIR, "configs")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# ---------------------------------------------------------------------------
# Experiment family metadata
# ---------------------------------------------------------------------------

FAMILY_NAME = "labelmix_in1k"
METRIC = "top1"
MODE = "max"

# ---------------------------------------------------------------------------
# Model configs: each entry maps a short tag to a YAML config path
# relative to this package's ``configs/`` directory.
#
# These are the full baseline configs from evaluation/baselines/configs.
# Each YAML contains all model-specific defaults (model name, epochs,
# lr_base, optimizer, weight_decay, augmentation strength, EMA, etc.).
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, str] = {
    # --- Phase 1: K-scheduling study on vit-wee only ---
    "vit-wee":           os.path.join(_CONFIGS_DIR, "vit-wee.yaml"),
    # "vit-little":        os.path.join(_CONFIGS_DIR, "vit-little.yaml"),
    # "vit-medium":        os.path.join(_CONFIGS_DIR, "vit-medium.yaml"),
    # "vit-base":          os.path.join(_CONFIGS_DIR, "vit-base.yaml"),
    # "mnv4-conv-medium":  os.path.join(_CONFIGS_DIR, "mnv4-conv-medium.yaml"),
    # "mnv4-conv-large":   os.path.join(_CONFIGS_DIR, "mnv4-conv-large.yaml"),
    # "mnv4-hybrid-medium": os.path.join(_CONFIGS_DIR, "mnv4-hybrid-medium.yaml"),
    # "mnv4-hybrid-large": os.path.join(_CONFIGS_DIR, "mnv4-hybrid-large.yaml"),
    # "resnet-50":         os.path.join(_CONFIGS_DIR, "resnet-50.yaml"),
    # "resnet-101":        os.path.join(_CONFIGS_DIR, "resnet-101.yaml"),
}

# ---------------------------------------------------------------------------
# Common overrides — broken into logical groups.  All values use argparse
# dest names (underscored) since they are passed through as CLI args.
#
# Note: model-specific settings (model name, lr_base, optimizer,
# weight_decay, epochs, drop_path, etc.) come from the YAML config.
# ---------------------------------------------------------------------------

BASE_BATCH_SIZE = 1024


def build_common_overrides(nproc_per_experiment: int) -> Dict[str, Any]:
    """Build the merged common-overrides dict for one experiment family.

    Parameters
    ----------
    nproc_per_experiment : int
        Number of GPUs per trial.  Used to derive ``batch_size`` from
        ``BASE_BATCH_SIZE``.

    Returns
    -------
    Dict[str, Any]
        Merged overrides dict ready for CLI arg generation.
    """
    if BASE_BATCH_SIZE % nproc_per_experiment != 0:
        raise ValueError(
            f"Base batch size {BASE_BATCH_SIZE} must be divisible by per-experiment "
            f"GPU count ({nproc_per_experiment})."
        )
    per_gpu_batch_size = BASE_BATCH_SIZE // nproc_per_experiment

    # -- Base training common (precision, batch, scheduling, logging) --
    base_train_common: Dict[str, Any] = {
        "amp_dtype": "bfloat16",
        "batch_size": per_gpu_batch_size,
        "warmup_prefix": True,
        "aug_repeats": 0,
        "img_size": 256,
        "sched_on_updates": True,
        "num_logs": 1000,
        "num_evals": 100,
        "num_saves": 20,
        "wandb_project": "labelmix_ejafa",
        "log_wandb": True,
        "workers": 4,
        "loader_prefetch_factor": 2,
        "balanced_buffer_steps": 4,
        "balanced_cache_threshold_steps": 3,
        "pin_mem": True,
    }

    # -- ImageNet-1K dataset args --
    imagenet_args: Dict[str, Any] = {
        "dataset": "hfds/ILSVRC/imagenet-1k",
        #"data_dir": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k",
        "data_dir":"/dev/shm/imagenet-1k",
        "train_split": "train",
        "val_split": "validation",
        "input_key": "image",
        "target_key": "label",
        "balanced_mode": 1280,
        "num_classes": 1000,
        "num_steps": 125000,
        "warmup_steps": 12500,
    }

    # -- LabelMix common args --
    labelmix_common_args: Dict[str, Any] = {
        "labelmix": True,
        "labelmix_schedule": "fixed",
        "labelmix_k_schedule": "fixed",
        "labelmix_step_mode": "total",
        # Disable timm mixup/cutmix (LabelMix replaces these)
        "mixup": 0,
        "cutmix": 0,
        "mixup_prob": 0.0,
    }

    # -- Checkpoint / resume --
    checkpoint_args: Dict[str, Any] = {
        "check_resume": True,
        "recovery_interval": 5000,
    }

    # Merge all sections (later dicts override earlier ones)
    merged: Dict[str, Any] = {}
    merged.update(base_train_common)
    merged.update(imagenet_args)
    merged.update(labelmix_common_args)
    merged.update(checkpoint_args)
    return merged


def get_model_configs() -> Dict[str, str]:
    """Return available model config paths for this family."""
    return dict(MODEL_CONFIGS)


# ---------------------------------------------------------------------------
# Search space — Phase 1: Fixed-K × Fixed-alpha × Loss grid search
#
# Goal: determine whether K scheduling or alpha scheduling matters more,
#        and compare pl_loss vs soft_ce.
# Step 1: sweep fixed K, fixed alpha, and loss on vit-wee.
#   K     = [3, 4, 5, 6, 7, 8, 9, 10]
#   alpha = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
#   loss  = [pl_loss, soft_ce]
#   => 8 × 8 × 2 = 128 training runs (full grid, no pruning)
# ---------------------------------------------------------------------------

_K_VALUES = [7, 8, 9, 10] # 3, 4, 5, 6,
_ALPHA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
_LOSS_VALUES = ["pl_loss", "soft_ce"]

SEARCH_SPACE: Dict[str, Any] = {
    "labelmix_mix_k": _K_VALUES,
    "labelmix_alpha_min": _ALPHA_VALUES,
    "labelmix_loss": _LOSS_VALUES,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_experiment_name(
    trial_overrides: Dict[str, Any],
    common_overrides: Dict[str, Any],
) -> str:
    """Build a human-readable experiment name from trial hyperparameters.

    Example output: ``k3_a0.3_pl-loss``
    """
    parts: List[str] = []

    if "labelmix_mix_k" in trial_overrides:
        parts.append(f"k{trial_overrides['labelmix_mix_k']}")

    if "labelmix_alpha_min" in trial_overrides:
        v = trial_overrides["labelmix_alpha_min"]
        parts.append(f"a{v:g}")

    if "labelmix_loss" in trial_overrides:
        parts.append(str(trial_overrides["labelmix_loss"]).replace("_", "-"))

    # Include schedule info only when it differs from the default ("fixed")
    k_sched = trial_overrides.get("labelmix_k_schedule") or common_overrides.get(
        "labelmix_k_schedule"
    )
    if k_sched and k_sched != "fixed":
        parts.append(f"ks-{k_sched}")

    a_sched = trial_overrides.get("labelmix_schedule") or common_overrides.get(
        "labelmix_schedule"
    )
    if a_sched and a_sched != "fixed":
        parts.append(f"as-{a_sched}")

    # Catch-all for any extra overrides not already covered
    _covered = {
        "labelmix_mix_k",
        "labelmix_k_min",
        "labelmix_k_max",
        "labelmix_alpha_min",
        "labelmix_alpha_max",
        "labelmix_loss",
        "labelmix_k_schedule",
        "labelmix_schedule",
    }
    for k, v in sorted(trial_overrides.items()):
        if k not in _covered and not k.startswith("_"):
            parts.append(f"{k}={v}")

    return "_".join(parts) if parts else "trial"


def dict_to_cli_args(d: Dict[str, Any]) -> str:
    """Convert a dict of overrides to a CLI argument string.

    Conversion rules:
    - Keys with underscores become hyphens: ``labelmix_mix_k`` → ``--labelmix-mix-k``
    - ``True``  → bare flag (``--labelmix``)
    - ``False`` → omitted
    - Lists/tuples → space-separated (``--scale 0.08 1.0``)
    - Everything else → ``--key value``
    """
    parts: List[str] = []
    for k, v in d.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        elif isinstance(v, (list, tuple)):
            parts.append(f"{flag} {' '.join(str(x) for x in v)}")
        else:
            parts.append(f"{flag} {v}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate(
    gpus_per_job: int = 1,
    output_path: str = "jobs.yaml",
    output_root: str = "./output_runs/daemon",
    max_retries: int = 3,
    model_filter: List[str] | None = None,
) -> None:
    """Expand the experiment grid and write ``jobs.yaml``.

    Parameters
    ----------
    gpus_per_job : int
        GPUs allocated to each training job.
    output_path : str
        Path for the generated YAML file.
    output_root : str
        Root directory for training outputs (``--output`` flag).
    max_retries : int
        Default max retries per job.
    model_filter : list[str] | None
        If provided, only generate jobs for these model tags.
    """
    # ---- Load experiment config -------------------------------------------
    model_configs = get_model_configs()
    common_overrides = build_common_overrides(nproc_per_experiment=gpus_per_job)

    # ---- Expand search space into grid axes --------------------------------
    grid_axes: Dict[str, List[Any]] = {}
    for key, value in SEARCH_SPACE.items():
        if isinstance(value, (list, tuple)):
            grid_axes[key] = list(value)
        else:
            grid_axes[key] = [value]  # single fixed value

    keys = list(grid_axes.keys())
    combinations = list(itertools.product(*[grid_axes[k] for k in keys]))

    # ---- Apply model filter ------------------------------------------------
    if model_filter:
        model_configs = {k: v for k, v in model_configs.items() if k in model_filter}
    if not model_configs:
        all_models = list(get_model_configs().keys())
        print(f"Error: No model configs matched filter. Available: {all_models}")
        sys.exit(1)

    # ---- Build job entries -------------------------------------------------
    jobs: List[Dict[str, str]] = []
    for model_tag, config_path in model_configs.items():
        for combo in combinations:
            trial_overrides = dict(zip(keys, combo))

            # Sync k params: k_min == k_max == mix_k (fixed K per trial)
            if "labelmix_mix_k" in trial_overrides:
                k = trial_overrides["labelmix_mix_k"]
                trial_overrides.setdefault("labelmix_k_min", k)
                trial_overrides.setdefault("labelmix_k_max", k)

            # Sync alpha: alpha_max == alpha_min (fixed alpha per trial)
            if "labelmix_alpha_min" in trial_overrides:
                trial_overrides.setdefault(
                    "labelmix_alpha_max", trial_overrides["labelmix_alpha_min"]
                )

            # Build human-readable experiment name
            exp_name = build_experiment_name(trial_overrides, common_overrides)

            # Merge all overrides (common < trial-specific < runtime)
            all_overrides: Dict[str, Any] = {}
            all_overrides.update(common_overrides)
            all_overrides.update(trial_overrides)
            all_overrides["experiment"] = exp_name
            all_overrides["output"] = output_root

            cli_args = dict_to_cli_args(all_overrides)

            # {gpus} and {port} are resolved at launch time by the daemon
            cmd = (
                f"torchrun --nproc_per_node={{gpus}} --master_port={{port}} "
                f"train.py --config {config_path} {cli_args}"
            )

            # Prefix with model tag when multiple models are active
            name = (
                f"{model_tag}__{exp_name}"
                if len(model_configs) > 1
                else exp_name
            )
            jobs.append({"name": name, "cmd": cmd})

    # ---- Write output YAML -------------------------------------------------
    output: Dict[str, Any] = {
        "defaults": {
            "gpus": gpus_per_job,
            "max_retries": max_retries,
            "working_dir": _PROJECT_ROOT,
        },
        "jobs": jobs,
    }

    with open(output_path, "w") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)

    # ---- Summary -----------------------------------------------------------
    grid_desc = " × ".join(f"{k}={len(v)}" for k, v in grid_axes.items())
    print(f"Generated {len(jobs)} job(s) -> {output_path}")
    print(f"  Models: {list(model_configs.keys())}")
    print(f"  Grid:   {grid_desc}")
    print(f"  GPUs/job: {gpus_per_job}")
    print(f"  Output root: {output_root}")
    print()
    print("Next steps:")
    print(f"  python jobdaemon.py submit {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate jobs.yaml for LabelMix ImageNet-1K experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiments/labelmix_imagenet1k/generate_jobs.py\n"
            "  python experiments/labelmix_imagenet1k/generate_jobs.py "
            "--gpus-per-job 8 -o jobs.yaml\n"
            "  python experiments/labelmix_imagenet1k/generate_jobs.py "
            "--model-filter vit-wee vit-little\n"
        ),
    )
    parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=1,
        help="GPUs per job (default: 1). Affects --batch-size via BASE_BATCH_SIZE.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="jobs.yaml",
        help="Output YAML path (default: jobs.yaml)",
    )
    parser.add_argument(
        "--output-root",
        default="./output_runs/daemon",
        help="Root directory for training outputs (default: ./output_runs/daemon)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Default max retries per job (default: 3)",
    )
    parser.add_argument(
        "--model-filter",
        nargs="+",
        default=None,
        help="Only generate jobs for these model config tags (e.g. vit-wee)",
    )
    args = parser.parse_args()

    generate(
        gpus_per_job=args.gpus_per_job,
        output_path=args.output,
        output_root=args.output_root,
        max_retries=args.max_retries,
        model_filter=args.model_filter,
    )


if __name__ == "__main__":
    main()
