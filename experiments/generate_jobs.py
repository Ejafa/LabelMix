#!/usr/bin/env python3
"""Generate jobs.yaml for LabelMix experiments with configurable datasets.

This script supports multiple datasets (ImageNet-1K, Places365) and allows
for easy swapping between them via the --dataset argument.

Usage::

    python experiments/generate_jobs.py --dataset in1k
    python experiments/generate_jobs.py --dataset places365
    python experiments/generate_jobs.py --dataset in1k --gpus-per-job 8 --output jobs.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIGS_DIR = os.path.join(_THIS_DIR, "labelmix_imagenet1k", "configs")
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# ---------------------------------------------------------------------------
# Experiment family metadata
# ---------------------------------------------------------------------------

FAMILY_NAME = "labelmix"
EXPERIMENT_NAME_PREFIX = "model_ablation"
METRIC = "top1"
MODE = "max"

# Seeds to sweep over — each trial is replicated once per seed.
# SEEDS: List[int] = [42, 43, 44]
SEEDS: List[int] = [42, 43]

# Image sizes to sweep over — each trial is replicated once per image size.
IMAGE_SIZES: List[int] = [256]

# Extra tags to include in experiment names (e.g. ["v2", "debug"]).
# Leave empty for no extra tags.
TAGS: List[str] = []

# ---------------------------------------------------------------------------
# Dataset configs — add new datasets here; the active one is selected via
# ACTIVE_DATASET.  Each entry maps a short identifier (used in experiment
# names and wandb tags) to its training overrides.
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """All dataset-specific CLI overrides plus metadata."""
    dataset_id: str                        # short tag embedded in exp name & wandb tags
    dataset: str                           # --dataset value
    data_dir: str                          # --data-dir value
    train_split: str
    val_split: str
    input_key: str
    target_key: str
    num_classes: int
    num_steps: int
    warmup_steps: int
    balanced_mode: int
    extra: Dict[str, Any] = field(default_factory=dict)  # any additional overrides

    def to_overrides(self) -> Dict[str, Any]:
        """Return a flat dict of CLI overrides for this dataset."""
        d: Dict[str, Any] = {
            "dataset": self.dataset,
            "data_dir": self.data_dir,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "input_key": self.input_key,
            "target_key": self.target_key,
            "balanced_mode": self.balanced_mode,
            "num_classes": self.num_classes,
            "num_steps": self.num_steps,
            "warmup_steps": self.warmup_steps,
        }
        d.update(self.extra)
        return d


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "in1k": DatasetConfig(
        dataset_id="in1k",
        dataset="hfds/ILSVRC/imagenet-1k",
        data_dir="/dev/shm/imagenet-1k",  # Arrow cache destination in RAM; use copy_data_to_ram.py to pre-warm
        train_split="train",
        val_split="validation",
        input_key="image",
        target_key="label",
        num_classes=1000,
        num_steps=125000,
        warmup_steps=12500,
        balanced_mode=1280,
    ),
    "places365": DatasetConfig(
        dataset_id="places365",
        # "hfds/" prefix routes to ReaderHFDS (parquet-based); the path after the prefix
        # is the local directory that HF auto-detects as a parquet dataset.
        # data_dir is used as cache_dir in ReaderHfds, so point it at the same location.
        dataset="hfds//dev/shm/places365",
        data_dir="/dev/shm/places365",  # RAM destination; use copy_data_to_ram.py --dataset places365 to populate
        train_split="train",
        val_split="validation",
        input_key="image",
        target_key="labels",  # README: field is 'labels' (plural)
        num_classes=365,
        num_steps=170702, # ~ 100 epochs (balanced_mode * 365 (amount of classes)) * 100 (epochs) / 1024 (batch size)
        warmup_steps=17070, # ~ 10 epochs
        balanced_mode=4789,  # (1_839_960 (total images) * 0.95 (% of train images) / 365 (number of classes))
    ),
    # Add more datasets here as needed
}

# ---------------------------------------------------------------------------
# Model configs: active model variants for this sweep.
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, str] = {
    # "vit-medium": os.path.join(_CONFIGS_DIR, "vit-medium.yaml"),
    # "vit-wee": os.path.join(_CONFIGS_DIR, "vit-wee.yaml"),
    # "vit-little": os.path.join(_CONFIGS_DIR, "vit-little.yaml"),
    # "vit-base": os.path.join(_CONFIGS_DIR, "vit-base.yaml"),
    "vit-betwixt": os.path.join(_CONFIGS_DIR, "vit-betwixt"),
    "vit-betwixt-rope": os.path.join(_CONFIGS_DIR, "vit-betwixt-rope.yaml"),
    # "convnextv2-base": os.path.join(_CONFIGS_DIR, "convnextv2-base.yaml"),
    # "convnextv2-tiny": os.path.join(_CONFIGS_DIR, "convnextv2-tiny.yaml"),
}

# Models whose constructors require img_size to be passed explicitly via
# --model-kwargs.  ViT / Eva models hardcode img_size in their entrypoint's
# model_args dict; the CLI --img-size flag only affects the data pipeline
# (resolve_data_config), NOT the model constructor.  Without this, the model
# builds its patch-embed grid and positional embeddings for the wrong
# resolution.  Fully-convolutional models (ConvNeXt, MobileNet, ResNet, etc.)
# do NOT need this.
MODELS_REQUIRING_IMG_SIZE_KWARG: set[str] = {
    "vit-base",
    "vit-betwixt",
    "vit-betwixt-rope",
    "vit-medium",
    "vit-little",
    "vit-wee",
}

# ---------------------------------------------------------------------------
# Common overrides — broken into logical groups.
# ---------------------------------------------------------------------------

BASE_BATCH_SIZE = 1024


def build_common_overrides(nproc_per_experiment: int, dataset_cfg: Optional[DatasetConfig] = None) -> Dict[str, Any]:
    """Build the merged common-overrides dict for one experiment family."""
    if dataset_cfg is None:
        raise ValueError("Dataset config must be provided")
    
    if BASE_BATCH_SIZE % nproc_per_experiment != 0:
        raise ValueError(
            f"Base batch size {BASE_BATCH_SIZE} must be divisible by per-experiment "
            f"GPU count ({nproc_per_experiment})."
        )
    per_gpu_batch_size = BASE_BATCH_SIZE // nproc_per_experiment

    # Base training common (precision, batch, scheduling, logging)
    base_train_common: Dict[str, Any] = {
        "amp_dtype": "bfloat16",
        "batch_size": per_gpu_batch_size,
        "warmup_prefix": True,
        "aug_repeats": 0,
        "img_size": None,  # set per-run from IMAGE_SIZES
        "sched_on_updates": True,
        "num_logs": 1000,
        "num_evals": 100,
        "num_saves": 20,
        "wandb_project": "labelmix",
        "wandb_tags": dataset_cfg.dataset_id,
        "log_wandb": True,
        "workers": 8,
        "loader_prefetch_factor": 2,
        "balanced_buffer_steps": 4,
        "balanced_cache_threshold_steps": 3,
        "pin_mem": True,
    }

    # Dataset args (sourced from DatasetConfig)
    dataset_args: Dict[str, Any] = dataset_cfg.to_overrides()

    # LabelMix common args
    labelmix_common_args: Dict[str, Any] = {
        "labelmix": True,
        "labelmix_schedule": "fixed",
        "labelmix_k_schedule": "fixed",
        "labelmix_step_mode": "total",
        # K schedule params
        "labelmix_k_reverse": False,
        "labelmix_k_warmup_epochs": 0,
        "labelmix_k_total_epochs": None,
        "labelmix_k_cooldown_epochs": None,
        # Alpha schedule params
        "labelmix_reverse": False,
        "labelmix_warmup_steps": 0,
        "labelmix_total_epochs": None,
        "labelmix_total_steps": None,
        # Sampling params
        "labelmix_sampling": True,
        "labelmix_sampling_min_side_px": 0,
        "labelmix_sampling_max_aspect": 20.0,
        "labelmix_sampling_bins": 16,
        "labelmix_sampling_pool_size": 128,
        "labelmix_sampling_low_watermark": 32,
        "labelmix_sampling_max_attempts": 200,
        # Producer params
        "labelmix_producer_rank": -1,
        "labelmix_producer_workers": 0,
        # Disable timm mixup/cutmix (LabelMix replaces these)
        "mixup": 0,
        "cutmix": 0,
        "mixup_prob": 0.0,
    }

    # Checkpoint / resume
    checkpoint_args: Dict[str, Any] = {
        "check_resume": True,
        "recovery_interval": 5000,
    }

    # Merge all sections (later dicts override earlier ones)
    merged: Dict[str, Any] = {}
    merged.update(base_train_common)
    merged.update(dataset_args)       # dataset overrides may override batch/logging defaults
    merged.update(labelmix_common_args)
    merged.update(checkpoint_args)
    return merged


def get_model_configs() -> Dict[str, str]:
    """Return available model config paths for this family."""
    return dict(MODEL_CONFIGS)


# ---------------------------------------------------------------------------
# Candidate trial configurations.
# ---------------------------------------------------------------------------

CANDIDATE_TRIALS: List[Dict[str, Any]] = [
    {
        "labelmix_schedule": "cosine",
        "labelmix_k_schedule": "fixed",
        "labelmix_k_warmup_epochs": 0,
        "labelmix_mix_k": 4,
        "labelmix_k_min": 4,
        "labelmix_k_max": 4,
        "labelmix_alpha_min": 0.1,
        "labelmix_alpha_max": 0.5,
        "labelmix_loss": "soft_ce",
    },
    {
        "labelmix_schedule": "cosine",
        "labelmix_k_schedule": "fixed",
        "labelmix_k_warmup_epochs": 0,
        "labelmix_mix_k": 6,
        "labelmix_k_min": 6,
        "labelmix_k_max": 6,
        "labelmix_alpha_min": 0.1,
        "labelmix_alpha_max": 0.5,
        "labelmix_loss": "pl_loss",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_experiment_name(
    trial_overrides: Dict[str, Any],
    common_overrides: Dict[str, Any],
) -> str:
    """Build a human-readable experiment name from trial hyperparameters."""
    parts: List[str] = []
    labelmix_enabled = trial_overrides.get(
        "labelmix", common_overrides.get("labelmix", False)
    )

    if not labelmix_enabled:
        parts.append("baseline")
    else:
        k_min = trial_overrides.get("labelmix_k_min")
        k_max = trial_overrides.get("labelmix_k_max")
        if k_min is not None and k_max is not None:
            parts.append(f"k{k_min}-{k_max}")
        elif "labelmix_mix_k" in trial_overrides:
            parts.append(f"k{trial_overrides['labelmix_mix_k']}")

        if "labelmix_k_cooldown_epochs" in trial_overrides and trial_overrides["labelmix_k_cooldown_epochs"] is not None:
            parts.append(f"kcd{trial_overrides['labelmix_k_cooldown_epochs']}")

        k_reverse = trial_overrides.get(
            "labelmix_k_reverse",
            common_overrides.get("labelmix_k_reverse", False),
        )
        if k_reverse:
            parts.append("krev")

        alpha_min = trial_overrides.get(
            "labelmix_alpha_min",
            common_overrides.get("labelmix_alpha_min"),
        )
        alpha_max = trial_overrides.get(
            "labelmix_alpha_max",
            common_overrides.get("labelmix_alpha_max"),
        )
        if alpha_min is not None and alpha_max is not None:
            if alpha_min == alpha_max:
                parts.append(f"a{alpha_min:g}")
            else:
                parts.append(f"a{alpha_min:g}-{alpha_max:g}")
        elif alpha_min is not None:
            parts.append(f"a{alpha_min:g}")
        elif alpha_max is not None:
            parts.append(f"a{alpha_max:g}")

        alpha_reverse = trial_overrides.get(
            "labelmix_reverse",
            common_overrides.get("labelmix_reverse", False),
        )
        if alpha_reverse:
            parts.append("arev")

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

        # Include scheduling tag when LabelMix sampling is enabled
        sampling = trial_overrides.get(
            "labelmix_sampling",
            common_overrides.get("labelmix_sampling", False),
        )
        if sampling:
            parts.append("scheduling")

    # Catch-all for any extra overrides not already covered
    _covered = {
        "labelmix",
        "labelmix_mix_k",
        "labelmix_k_min",
        "labelmix_k_max",
        "labelmix_alpha_min",
        "labelmix_alpha_max",
        "labelmix_loss",
        "labelmix_k_schedule",
        "labelmix_k_reverse",
        "labelmix_k_warmup_epochs",
        "labelmix_k_cooldown_epochs",
        "labelmix_schedule",
        "labelmix_sampling",
        "_sampling_config",
        "labelmix_sampling_min_side_px",
        "labelmix_sampling_max_aspect",
        "labelmix_reverse",
    }
    for k, v in sorted(trial_overrides.items()):
        if k not in _covered and not k.startswith("_"):
            parts.append(f"{k}={v}")

    return "_".join(parts) if parts else "trial"


def _read_config_model_kwargs(config_path: str) -> Dict[str, Any]:
    """Read the ``model_kwargs`` dict from a YAML config file.

    Returns a *copy* so callers can safely mutate the result without
    affecting cached data.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg.get("model_kwargs", {}))


def dict_to_cli_args(d: Dict[str, Any]) -> str:
    """Convert a dict of overrides to a CLI argument string."""
    parts: List[str] = []
    for k, v in d.items():
        if v is None:
            continue
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
    dataset: str = "in1k",
    gpus_per_job: int = 1,
    output_path: str = "jobs.yaml",
    output_root: str = "./output_runs/daemon",
    max_retries: int = 3,
    model_filter: List[str] | None = None,
    baseline: bool = False,
) -> None:
    """Expand the experiment grid and write ``jobs.yaml``.

    Parameters
    ----------
    dataset : str
        Dataset identifier (in1k, places365, etc.)
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
    baseline : bool
        If True, generate baseline jobs with labelmix=false.
    """
    # Validate dataset selection
    if dataset not in DATASET_CONFIGS:
        available_datasets = list(DATASET_CONFIGS.keys())
        print(f"Error: Dataset '{dataset}' not found. Available datasets: {available_datasets}")
        sys.exit(1)
    
    # Load experiment config
    model_configs = get_model_configs()
    dataset_cfg = DATASET_CONFIGS[dataset]
    common_overrides = build_common_overrides(nproc_per_experiment=gpus_per_job, dataset_cfg=dataset_cfg)
    candidate_trials = list(CANDIDATE_TRIALS)

    # Add baseline trial if requested
    if baseline:
        candidate_trials = [{"labelmix": False}]
        print("Generating baseline jobs with labelmix=false")

    # Apply model filter
    if model_filter:
        model_configs = {k: v for k, v in model_configs.items() if k in model_filter}
    if not model_configs:
        all_models = list(get_model_configs().keys())
        print(f"Error: No model configs matched filter. Available: {all_models}")
        sys.exit(1)

    # Build job entries
    jobs: List[Dict[str, str]] = []
    seen_experiment_names: set[str] = set()
    expected_seeds: set[int] = set(SEEDS)
    
    if len(expected_seeds) != len(SEEDS):
        raise ValueError(f"SEEDS contains duplicate values: {SEEDS}")
    if not expected_seeds:
        raise ValueError("SEEDS must not be empty.")

    # Ensure every (model, trial) emits exactly one run per configured seed.
    seed_coverage: Dict[tuple[str, str], set[int]] = {}

    image_sizes = list(IMAGE_SIZES)
    if not image_sizes:
        raise ValueError("IMAGE_SIZES must not be empty.")

    tags_suffix = "__".join(TAGS) if TAGS else ""

    for model_tag, config_path in model_configs.items():
        for trial_overrides in candidate_trials:
            for img_size in image_sizes:
                for seed in SEEDS:
                    trial_overrides = dict(trial_overrides)

                    # Sync k params: k_min == k_max == mix_k (fixed K per trial)
                    if "labelmix_mix_k" in trial_overrides:
                        k = trial_overrides["labelmix_mix_k"]
                        trial_overrides.setdefault("labelmix_k_min", k)
                        trial_overrides.setdefault("labelmix_k_max", k)

                    base_exp_name = build_experiment_name(trial_overrides, common_overrides)
                    # Build experiment name: prefix__model__dataset__img{size}[__tags]__trial__seed=N
                    name_parts = [
                        EXPERIMENT_NAME_PREFIX,
                        model_tag,
                        dataset_cfg.dataset_id,
                        f"img{img_size}",
                    ]
                    if tags_suffix:
                        name_parts.append(tags_suffix)
                    name_parts.append(base_exp_name)
                    name_parts.append(f"seed={seed}")
                    exp_name = "__".join(name_parts)

                    if exp_name in seen_experiment_names:
                        raise ValueError(
                            f"Duplicate experiment name generated: {exp_name}. "
                            "Ensure all active sweep axes are encoded in build_experiment_name()."
                        )
                    seen_experiment_names.add(exp_name)

                    group_key = (model_tag, base_exp_name, img_size)
                    seen_group_seeds = seed_coverage.setdefault(group_key, set())
                    if seed in seen_group_seeds:
                        raise ValueError(
                            f"Duplicate seed run generated for {model_tag}/{base_exp_name}/img{img_size}: seed={seed}"
                        )
                    seen_group_seeds.add(seed)

                    # Merge all overrides (common < trial-specific < runtime)
                    all_overrides: Dict[str, Any] = {}
                    all_overrides.update(common_overrides)
                    all_overrides.update(trial_overrides)
                    all_overrides["img_size"] = img_size
                    all_overrides["seed"] = seed
                    all_overrides["experiment"] = exp_name
                    all_overrides["output"] = output_root

                    cli_args = dict_to_cli_args(all_overrides)

                    # For ViT / Eva models the constructor needs img_size via
                    # --model-kwargs so the patch-embed grid and positional
                    # embeddings match the actual input resolution.  The CLI
                    # --img-size only affects the data pipeline.
                    #
                    # IMPORTANT: --model-kwargs on the CLI *replaces* (not
                    # merges with) the model_kwargs from the YAML config.
                    # We therefore read the config's existing model_kwargs,
                    # merge in the runtime img_size, and emit the full set
                    # so nothing (e.g. fix_init) is lost.
                    model_kwargs_args = ""
                    if model_tag in MODELS_REQUIRING_IMG_SIZE_KWARG:
                        cfg_model_kwargs = _read_config_model_kwargs(config_path)
                        cfg_model_kwargs["img_size"] = img_size
                        kw_parts = [f"{k}={v}" for k, v in cfg_model_kwargs.items()]
                        model_kwargs_args = " --model-kwargs " + " ".join(kw_parts)

                    # {gpus} and {port} are resolved at launch time by the daemon
                    cmd = (
                        f"torchrun --nproc_per_node={{gpus}} --master_port={{port}} "
                        f"train.py --config {config_path} {cli_args}{model_kwargs_args}"
                    )

                    if f"--seed {seed}" not in cmd:
                        raise ValueError(
                            f"Internal error: generated command for {exp_name} is missing '--seed {seed}'."
                        )

                    jobs.append({"name": exp_name, "cmd": cmd})

    # Check seed coverage
    missing_seed_errors: List[str] = []
    for (model_tag, base_exp_name, img_size), seen_seeds in sorted(seed_coverage.items()):
        missing = sorted(expected_seeds - seen_seeds)
        if missing:
            missing_seed_errors.append(
                f"{model_tag}/{base_exp_name}/img{img_size}: missing seeds {missing}"
            )

    if missing_seed_errors:
        raise ValueError(
            "Seed coverage check failed. Each model/trial must include one run per seed in SEEDS.\n"
            + "\n".join(missing_seed_errors)
        )

    # Write output YAML
    output: Dict[str, Any] = {
        "defaults": {
            "gpus": gpus_per_job,
            "max_retries": max_retries,
            "working_dir": _PROJECT_ROOT,
        },
        "jobs": jobs,
    }

    # Check if output file already exists
    if os.path.exists(output_path):
        raise FileExistsError(f"Output file {output_path} already exists. Please rename the file or delete it before proceeding.")

    with open(output_path, "w") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)

    # Summary
    print(f"Generated {len(jobs)} job(s) -> {output_path}")
    print(f"  Dataset: {dataset}")
    print(f"  Models: {list(model_configs.keys())}")
    print(f"  Image sizes: {image_sizes}")
    print(f"  Candidate configs/model: {len(candidate_trials)}")
    print(f"  Tags: {TAGS if TAGS else '(none)'}")
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
        description="Generate jobs.yaml for LabelMix experiments with configurable datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiments/generate_jobs.py --dataset in1k\n"
            "  python experiments/generate_jobs.py --dataset places365\n"
            "  python experiments/generate_jobs.py --dataset in1k --gpus-per-job 8 -o jobs.yaml\n"
            "  python experiments/generate_jobs.py --dataset places365 --model-filter vit-medium vit-base\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="in1k",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use for training (default: in1k)",
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
        help="Only generate jobs for these model config tags",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Generate baseline jobs with labelmix=false",
    )
    args = parser.parse_args()

    generate(
        dataset=args.dataset,
        gpus_per_job=args.gpus_per_job,
        output_path=args.output,
        output_root=args.output_root,
        max_retries=args.max_retries,
        model_filter=args.model_filter,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()