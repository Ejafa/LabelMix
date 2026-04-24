#!/usr/bin/env python3
"""Generate jobs.yaml for LabelMix experiments with configurable datasets.

This script supports multiple datasets (ImageNet-1K, Places365, CIFAR-100) and
allows for easy swapping between them via the --dataset argument.

Usage::

    python experiments/generate_jobs.py --dataset in1k
    python experiments/generate_jobs.py --dataset places365
    python experiments/generate_jobs.py --dataset cifar100
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
EXPERIMENT_NAME_PREFIX = ""
METRIC = "top1"
MODE = "max"

# ---------------------------------------------------------------------------
# Active sweep axes.
#
# This generator sweeps over exactly three dimensions:
#   1. models          -- via get_model_configs() / --models filter
#   2. seeds           -- the SEEDS list below
#   3. learning rates  -- the LEARNING_RATES list below (empty == use YAML default)
#
# Every other hyperparameter (image size, layer-decay, mosaic knobs,
# labelmix mixed-alpha, ...) is pinned to a single value below.  If you
# ever need to re-enable one of those sweeps, wrap the corresponding
# scalar in a loop at the job-expansion site.
# ---------------------------------------------------------------------------

# Seeds to sweep over — each trial is replicated once per seed.
SEEDS: List[int] = [42, 43, 44]

# Image size used for every generated job (pinned, no sweep).
IMG_SIZE: int = 256

# Extra tags to include in experiment names (e.g. ["v2", "debug"]).
# Leave empty for no extra tags.
# Note: when --baseline is passed, the tag "baseline" is automatically
# appended regardless of this list.
TAGS: List[str] = []

# Learning rates to sweep over — each trial is replicated once per LR.
# If empty, the default lr_base from the config file is used and no LR tag
# is added to the experiment name.
LEARNING_RATES: List[float] = []

# ---------------------------------------------------------------------------
# Mosaic configuration (pinned, no sweep).
#
# Mosaic is a 4-image fusion augmentation that's incompatible with
# Mixup/CutMix, so the generator zeroes `mixup`, `cutmix`, `mixup_prob`
# and omits `cutmix_minmax` on every Mosaic job.
# ---------------------------------------------------------------------------

# Close-mosaic schedule: disable Mosaic during the last N epochs.
MOSAIC_CLOSE_EPOCHS: int = 10

# Center-ratio range for the Mosaic quadrant boundary: (lo, hi) means the
# center point is sampled at S * uniform(lo, hi) along each axis.
MOSAIC_CENTER_RATIO: tuple[float, float] = (0.5, 1.5)

# Post-mosaic affine zoom range (applied to the final S x S crop).
MOSAIC_SCALE_RANGE: tuple[float, float] = (0.8, 1.2)

# Mosaic application probability.
MOSAIC_PROB: float = 0.5

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
    "cifar100": DatasetConfig(
        dataset_id="cifar100",
        # "hfds/" prefix routes to ReaderHFDS (parquet-based); the path after
        # the prefix is the local directory that HF auto-detects as a parquet
        # dataset.  data_dir is used as cache_dir in ReaderHfds.
        dataset="hfds//dev/shm/cifar100",
        data_dir="/dev/shm/cifar100",  # RAM destination; use copy_data_to_ram.py --dataset cifar100 to populate
        train_split="train",
        val_split="test",
        input_key="img",
        target_key="fine_label",
        num_classes=100,
        # 50 000 train images / 1024 batch * 200 epochs ≈ 9 766 steps
        num_steps=9766,
        warmup_steps=977,   # ~10 epochs
        balanced_mode=500,  # 500 images per class (50 000 / 100 classes)
    ),
    # Add more datasets here as needed
}

# ---------------------------------------------------------------------------
# Model configs: active model variants for this sweep.
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, str] = {
    "vit-medium": os.path.join(_CONFIGS_DIR, "vit-medium.yaml"),
    "vit-wee": os.path.join(_CONFIGS_DIR, "vit-wee.yaml"),
    "vit-little": os.path.join(_CONFIGS_DIR, "vit-little.yaml"),
    "vit-betwixt": os.path.join(_CONFIGS_DIR, "vit-betwixt.yaml"),
#    "convnextv2-base": os.path.join(_CONFIGS_DIR, "convnextv2-base.yaml"),
#    "convnextv2-tiny": os.path.join(_CONFIGS_DIR, "convnextv2-tiny.yaml"),
#    "resnet50": os.path.join(_CONFIGS_DIR, "resnet-50.yaml"),
#    "resnet101": os.path.join(_CONFIGS_DIR, "resnet-101.yaml"),
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
        "img_size": None,  # set per-run from IMG_SIZE
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
#
# LabelMix trials used to be hard-coded in a ``CANDIDATE_TRIALS`` list.  The
# list has been removed in favour of :func:`build_labelmix_trial`, which
# produces a standard LabelMix trial dict for a given loss variant.  The CLI
# option ``--labelmix {mixed,pl,sce}`` (repeatable) selects which variants to
# emit; ``--all`` is shorthand for every variant.
# ---------------------------------------------------------------------------

# Mapping from the public ``--labelmix`` choice to
#   (wandb tag, train.py labelmix_loss value, mix-k, experiment-name-friendly key).
# Keeping it declarative makes it trivial to add a new loss variant later.
LABELMIX_LOSS_SPECS: Dict[str, Dict[str, Any]] = {
    "mixed": {"tag": "labelmix-mixed", "loss": "mixed",   "mix_k": 4},
    "pl":    {"tag": "labelmix-pl",    "loss": "pl_loss", "mix_k": 6},
    "sce":   {"tag": "labelmix-sce",   "loss": "soft_ce", "mix_k": 4},
}


def build_labelmix_trial(variant: str) -> Dict[str, Any]:
    """Return a standard LabelMix trial dict for the given variant.

    ``variant`` must be one of ``LABELMIX_LOSS_SPECS`` (``mixed``, ``pl``,
    ``sce``).  The returned dict carries a private ``_tags`` marker so the
    job-emission loop can attach the correct wandb tag (e.g.
    ``labelmix-mixed``) per trial.
    """
    try:
        spec = LABELMIX_LOSS_SPECS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown labelmix variant {variant!r}; expected one of "
            f"{sorted(LABELMIX_LOSS_SPECS)}"
        ) from exc

    mix_k = spec["mix_k"]
    return {
        "labelmix_schedule": "cosine",
        "labelmix_k_schedule": "fixed",
        "labelmix_k_warmup_epochs": 0,
        "labelmix_mix_k": mix_k,
        "labelmix_k_min": mix_k,
        "labelmix_k_max": mix_k,
        "labelmix_alpha_min": 0.1,
        "labelmix_alpha_max": 0.5,
        "labelmix_loss": spec["loss"],
        # Private bookkeeping key (stripped before CLI emission).
        "_tags": [spec["tag"]],
    }


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
    mosaic_enabled = trial_overrides.get("mosaic", False)

    if mosaic_enabled:
        # For mosaic experiments, just return "mosaic" without hyperparameter details
        return "mosaic"
    elif not labelmix_enabled:
        # The `_baseline_variant` marker distinguishes the four
        # LabelMix-disabled flavors (baseline/mixup/cutmix/bare) in the
        # experiment name.  If no marker is present we fall back to the
        # generic "baseline" token for backwards compatibility.
        variant = trial_overrides.get("_baseline_variant", "baseline")
        parts.append(str(variant))
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

        if "labelmix_mixed_alpha" in trial_overrides:
            parts.append(f"ma{trial_overrides['labelmix_mixed_alpha']:g}")

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
        "labelmix_mixed_alpha",
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
        # Mosaic keys (already encoded explicitly above)
        "mosaic",
        "mosaic_prob",
        "mosaic_center_ratio",
        "mosaic_scale_range",
        "mosaic_close_epochs",
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
    gpus_per_job: int = 2,
    output_path: str = "jobs.yaml",
    output_root: str = "./output_runs/daemon",
    max_retries: int = 3,
    model_filter: List[str] | None = None,
    baseline: bool = False,
    mosaic: bool = False,
    mixup: bool = False,
    cutmix: bool = False,
    bare: bool = False,
    noaug: bool = False,
    labelmix_variants: List[str] | None = None,
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
    mixup : bool
        If True, generate baseline jobs with mixup only.
    cutmix : bool
        If True, generate baseline jobs with cutmix only.
    bare : bool
        If True, generate baseline jobs with no mixup/cutmix.
    noaug : bool
        If True, generate baseline jobs with *all* heavy augmentations
        disabled.  Only simple geometric preprocessing remains:
        horizontal flip + random-resized-crop (scale/ratio).  Everything
        else -- RandAugment, color jitter, random erasing, mixup, cutmix,
        labelmix, aug_repeats, aug_splits -- is turned off.
    labelmix_variants : list[str] | None
        List of LabelMix loss variants to emit.  Each entry must be a key of
        ``LABELMIX_LOSS_SPECS`` (``mixed``, ``pl``, ``sce``).  ``None`` or the
        empty list means no LabelMix trials are emitted.
    """
    # Validate dataset selection
    if dataset not in DATASET_CONFIGS:
        available_datasets = list(DATASET_CONFIGS.keys())
        print(f"Error: Dataset '{dataset}' not found. Available datasets: {available_datasets}")
        sys.exit(1)

    # Enforce the `output_runs/` prefix on output_root.  Users can pass any
    # shorthand (e.g. `xy`, `./xy`, `runs/xy`) and we will rewrite it to
    # `output_runs/xy`.  If the path already lives under `output_runs/`
    # (including the default `./output_runs/daemon`), leave it untouched.
    _normalized_root = os.path.normpath(output_root)
    _root_parts = _normalized_root.split(os.sep)
    # Strip a single leading '.' component introduced by `./xy` style paths
    # so we compare against the first *meaningful* segment.
    if _root_parts and _root_parts[0] == ".":
        _root_parts = _root_parts[1:]
    if not _root_parts or _root_parts[0] != "output_runs":
        # Drop any leading "./" before prepending so we don't end up with
        # awkward "output_runs/./xy" strings.
        _stripped = output_root[2:] if output_root.startswith("./") else output_root
        output_root = os.path.join("output_runs", _stripped)
        print(f"Prefixing output_root with 'output_runs/': {output_root}")

    # Load experiment config
    model_configs = get_model_configs()
    dataset_cfg = DATASET_CONFIGS[dataset]
    common_overrides = build_common_overrides(nproc_per_experiment=gpus_per_job, dataset_cfg=dataset_cfg)

    # Build LabelMix trials from the CLI-selected variants.  If none were
    # selected this starts empty; baseline / mosaic flavors below may
    # replace or extend it.
    labelmix_variants = list(labelmix_variants or [])
    # Preserve user-supplied order but deduplicate.
    _seen: set[str] = set()
    _ordered_variants: List[str] = []
    for _v in labelmix_variants:
        if _v not in _seen:
            _seen.add(_v)
            _ordered_variants.append(_v)
    labelmix_variants = _ordered_variants
    candidate_trials: List[Dict[str, Any]] = [
        build_labelmix_trial(v) for v in labelmix_variants
    ]

    # Add baseline trials if requested.  The four LabelMix-disabled flavors
    # (baseline / mixup / cutmix / bare) differ only in which parts of the
    # timm mixup/cutmix pipeline are left active.  Values that are *not*
    # set here fall through to whatever the per-model YAML config specifies
    # (e.g. mixup=0.8, cutmix=1.0, mixup_prob=1.0 for most ViT configs).
    #
    # Note on `mixup_switch_prob`: in timm's Mixup module this controls
    # whether a given *mixing event* picks CutMix (prob=switch_prob) or
    # MixUp (prob=1-switch_prob).  It is orthogonal to `mixup_prob`
    # (probability of mixing firing at all).  We deliberately do NOT touch
    # `mixup_prob` here -- it stays at whatever the per-model config sets
    # -- but we DO pin `mixup_switch_prob` for the mixup-only / cutmix-only
    # variants so that whenever mixing fires it always uses the requested
    # augmentation.
    #
    # Multiple baseline flavors can be combined in a single invocation
    # (e.g. --baseline --bare); each one contributes its own trial dict to
    # `candidate_trials`, and the `_baseline_variant` marker makes each
    # variant addressable in the experiment name.
    baseline_trials: List[Dict[str, Any]] = []
    if baseline:
        baseline_trials.append({"labelmix": False, "_baseline_variant": "baseline"})
        print("Adding baseline jobs with labelmix=false (config mixup+cutmix intact)")
    if mixup:
        # Mixup only: force the Mixup/CutMix switch to always pick MixUp.
        # `mixup`, `cutmix`, `mixup_prob` are left to the config.
        baseline_trials.append({
            "labelmix": False,
            "mixup_switch_prob": 0.0,
            "_baseline_variant": "mixup",
        })
        print("Adding mixup-only baseline jobs (mixup_switch_prob=0.0)")
    if cutmix:
        # Cutmix only: force the Mixup/CutMix switch to always pick CutMix.
        # `mixup`, `cutmix`, `mixup_prob` are left to the config.
        baseline_trials.append({
            "labelmix": False,
            "mixup_switch_prob": 1.0,
            "_baseline_variant": "cutmix",
        })
        print("Adding cutmix-only baseline jobs (mixup_switch_prob=1.0)")
    if bare:
        # Bare: no mixup, no cutmix at all.  We zero the strengths and
        # mixup_prob so the Mixup module is effectively a no-op; the switch
        # probability is irrelevant in this case.
        baseline_trials.append({
            "labelmix": False,
            "mixup": 0,
            "cutmix": 0,
            "mixup_prob": 0.0,
            "cutmix_minmax": None,
            "_baseline_variant": "bare",
        })
        print("Adding bare baseline jobs (no mixup, no cutmix)")
    if noaug:
        # NoAug: flip timm's master `--no-aug` switch.  This forces the data
        # pipeline into a deterministic resize + center-crop + normalize
        # path, bypassing RandAugment, mixup/cutmix, random-erasing, color
        # jitter, random-resized-crop, hflip, and everything else in one go.
        # We also disable labelmix to keep the baseline truly free of augmentation.
        baseline_trials.append({
            "labelmix": False,
            "no_aug": True,
            "_baseline_variant": "noaug",
        })
        print("Adding noaug baseline jobs (no_aug=True: all train-time augmentation disabled)")

    # Tag each baseline trial with its variant name so the job-emission
    # loop can attach the corresponding wandb tag (e.g. ``baseline``,
    # ``mixup``, ``cutmix``, ``bare``, ``noaug``).
    for _trial in baseline_trials:
        _variant = _trial.get("_baseline_variant")
        if _variant:
            _trial.setdefault("_tags", []).append(_variant)

    # Baseline flavors are *additive* to any LabelMix variants: each emits
    # its own trial dict so a single invocation can sweep LabelMix AND the
    # four/five augmentation-disabled baselines in one go.
    if baseline_trials:
        candidate_trials = candidate_trials + baseline_trials

    # Mosaic trial.  Uses the pinned scalar Mosaic constants -- no sweep.
    # Side-by-side with LabelMix / baseline trials (each job picks a
    # single trial dict, so mosaic+labelmix coexist as *separate* trials).
    if mosaic:
        mosaic_trial: Dict[str, Any] = {
            # Disable LabelMix on the data path.
            "labelmix": False,
            # Enable Mosaic.
            "mosaic": True,
            "mosaic_prob": float(MOSAIC_PROB),
            "mosaic_center_ratio": [float(MOSAIC_CENTER_RATIO[0]), float(MOSAIC_CENTER_RATIO[1])],
            "mosaic_scale_range": [float(MOSAIC_SCALE_RANGE[0]), float(MOSAIC_SCALE_RANGE[1])],
            "mosaic_close_epochs": int(MOSAIC_CLOSE_EPOCHS),
            "_tags": ["mosaic"],
        }
        candidate_trials = candidate_trials + [mosaic_trial]
        print("Generating 1 Mosaic trial per model/seed/lr combination")

    # Guard against silently emitting zero jobs when the user neither
    # requested a baseline/mosaic/mixup/cutmix/bare/noaug sweep nor a
    # LabelMix variant via --labelmix / --all.
    if not candidate_trials:
        raise ValueError(
            "No candidate trials to generate. Pass --baseline, --mosaic, "
            "--mixup, --cutmix, --bare, --noaug, or --labelmix {mixed,pl,sce} "
            "(or --all to enable everything)."
        )

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

    img_size: int = int(IMG_SIZE)

    learning_rates: List[Optional[float]] = list(LEARNING_RATES) if LEARNING_RATES else [None]

    # Compose tag list.  All augmentation-specific tags (baseline, mixup,
    # cutmix, bare, noaug, mosaic, labelmix-*) are attached *per trial* via
    # the trial's ``_tags`` marker and emitted both into the experiment
    # name and into ``--wandb-tags`` below -- so we deliberately do NOT
    # hoist them into the global tag list here.
    effective_tags = list(TAGS)
    tags_suffix = "__".join(effective_tags) if effective_tags else ""

    for model_tag, config_path in model_configs.items():
        for trial_overrides in candidate_trials:
            for lr in learning_rates:
                for seed in SEEDS:
                    trial_overrides = dict(trial_overrides)

                    # Sync k params: k_min == k_max == mix_k (fixed K per trial)
                    if "labelmix_mix_k" in trial_overrides:
                        k = trial_overrides["labelmix_mix_k"]
                        trial_overrides.setdefault("labelmix_k_min", k)
                        trial_overrides.setdefault("labelmix_k_max", k)

                    base_exp_name = build_experiment_name(trial_overrides, common_overrides)
                    # Build experiment name: [prefix__]model__dataset__img{size}[__tags][__lr=X]__trial__seed=N
                    name_parts: List[str] = []
                    if EXPERIMENT_NAME_PREFIX:
                        name_parts.append(EXPERIMENT_NAME_PREFIX)
                    name_parts.extend([
                        model_tag,
                        dataset_cfg.dataset_id,
                        f"img{img_size}",
                    ])
                    if tags_suffix:
                        name_parts.append(tags_suffix)
                    if lr is not None:
                        name_parts.append(f"lr={lr:g}")
                    name_parts.append(base_exp_name)
                    name_parts.append(f"seed={seed}")
                    exp_name = "__".join(name_parts)

                    if exp_name in seen_experiment_names:
                        raise ValueError(
                            f"Duplicate experiment name generated: {exp_name}. "
                            "Ensure all active sweep axes are encoded in build_experiment_name()."
                        )
                    seen_experiment_names.add(exp_name)

                    group_key = (model_tag, base_exp_name, lr)
                    seen_group_seeds = seed_coverage.setdefault(group_key, set())
                    if seed in seen_group_seeds:
                        raise ValueError(
                            f"Duplicate seed run generated for {model_tag}/{base_exp_name}/lr={lr}: seed={seed}"
                        )
                    seen_group_seeds.add(seed)

                    # Merge all overrides (common < trial-specific < runtime)
                    all_overrides: Dict[str, Any] = {}
                    all_overrides.update(common_overrides)
                    all_overrides.update(trial_overrides)

                    # Determine active augmentations for this trial.
                    labelmix_enabled = trial_overrides.get(
                        "labelmix",
                        common_overrides.get("labelmix", True)
                    )
                    mosaic_enabled = trial_overrides.get("mosaic", False)

                    # Mosaic is an *alternative* to LabelMix: the two cannot
                    # coexist on the same trial.  Raise before we silently
                    # zero any knobs so configuration mistakes are loud.
                    if mosaic_enabled and labelmix_enabled:
                        raise ValueError(
                            f"Mosaic and LabelMix cannot both be active on the same trial "
                            f"(exp={exp_name}). When --mosaic is used, every trial must set "
                            f"labelmix=False (and train.py will refuse to run otherwise)."
                        )

                    # Mosaic / LabelMix both ship their own target format,
                    # so the timm mixup/cutmix pipeline must be disabled.
                    if labelmix_enabled or mosaic_enabled:
                        all_overrides["mixup"] = 0
                        all_overrides["cutmix"] = 0
                        all_overrides["mixup_prob"] = 0.0
                        all_overrides["cutmix_minmax"] = None

                    # When LabelMix is disabled for this trial (baseline or
                    # Mosaic sweep), drop *every* labelmix_* key that leaked
                    # in via ``labelmix_common_args``.  Otherwise the CLI
                    # would still contain flags like ``--labelmix-sampling``
                    # (a bool ON!) or ``--labelmix-sampling-*`` knobs, which
                    # on a Mosaic run would spin up the LabelMix sampling
                    # producer and/or confuse train.py's argument parsing.
                    if not labelmix_enabled:
                        for _k in list(all_overrides.keys()):
                            if _k.startswith("labelmix"):
                                all_overrides.pop(_k, None)

                    # Strip private marker keys (e.g. "_baseline_variant")
                    # that are used purely for internal bookkeeping and
                    # must not be emitted as CLI flags.
                    for _k in list(all_overrides.keys()):
                        if _k.startswith("_"):
                            all_overrides.pop(_k, None)

                    # Belt-and-suspenders: every Mosaic job must have the
                    # substring "mosaic" in its experiment name so it is
                    # trivially greppable in run logs, dashboards and
                    # checkpoint directories.
                    if mosaic_enabled and "mosaic" not in exp_name:
                        raise ValueError(
                            f"Mosaic trial produced an experiment name without the 'mosaic' token: "
                            f"{exp_name}. Check build_experiment_name() / effective_tags."
                        )

                    # Every baseline-variant job must have its variant
                    # token (baseline/mixup/cutmix/bare) embedded in the
                    # experiment name so different flavors are trivially
                    # distinguishable in run logs and checkpoint dirs.
                    # Note: we check the trial's own marker, not the CLI
                    # flags, because multiple flags may be active at once.
                    variant_marker = trial_overrides.get("_baseline_variant")
                    if variant_marker and variant_marker not in exp_name:
                        raise ValueError(
                            f"Baseline trial (variant={variant_marker}) produced an experiment name "
                            f"without the '{variant_marker}' token: {exp_name}. "
                            f"Check build_experiment_name()."
                        )

                    all_overrides["img_size"] = img_size
                    all_overrides["seed"] = seed
                    all_overrides["experiment"] = exp_name
                    all_overrides["output"] = output_root
                    if lr is not None:
                        all_overrides["lr_base"] = lr

                    # Augment wandb_tags with the per-trial augmentation
                    # tags (baseline/mixup/cutmix/bare/noaug/mosaic/
                    # labelmix-*).  ``wandb_tags`` from common_overrides
                    # is a scalar (dataset id); we fold everything into a
                    # space-delimited list so train.py's argparse
                    # (``nargs='+'``) picks up every tag.
                    _trial_tags: List[str] = list(trial_overrides.get("_tags", []) or [])
                    _base_tags_raw = all_overrides.get("wandb_tags", "")
                    if isinstance(_base_tags_raw, (list, tuple)):
                        _base_tags = [str(t) for t in _base_tags_raw if t]
                    elif _base_tags_raw:
                        _base_tags = [str(_base_tags_raw)]
                    else:
                        _base_tags = []
                    _merged_tags: List[str] = []
                    for _t in _base_tags + _trial_tags:
                        if _t and _t not in _merged_tags:
                            _merged_tags.append(_t)
                    if _merged_tags:
                        all_overrides["wandb_tags"] = _merged_tags

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
    for (model_tag, base_exp_name, lr), seen_seeds in sorted(
        seed_coverage.items(),
        key=lambda kv: (kv[0][0], kv[0][1],
                        (kv[0][2] is None, kv[0][2])),
    ):
        missing = sorted(expected_seeds - seen_seeds)
        if missing:
            lr_tag = f"/lr={lr:g}" if lr is not None else ""
            missing_seed_errors.append(
                f"{model_tag}/{base_exp_name}{lr_tag}: missing seeds {missing}"
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
    print(f"  Image size: {img_size}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Learning rates: {LEARNING_RATES if LEARNING_RATES else '(config default)'}")
    print(f"  Candidate configs/model: {len(candidate_trials)}")
    print(f"  Tags: {effective_tags if effective_tags else '(none)'}")
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
            "  python experiments/generate_jobs.py --dataset cifar100\n"
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
    parser.add_argument(
        "--mosaic",
        action="store_true",
        help="Generate Mosaic sweep jobs (replaces LabelMix trials; "
             "mixup/cutmix auto-zeroed; mutually exclusive with the four "
             "baseline flavors).",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Add mixup-only baseline jobs (labelmix=false, mixup_switch_prob=0.0). "
             "Combinable with --baseline/--cutmix/--bare.",
    )
    parser.add_argument(
        "--cutmix",
        action="store_true",
        help="Add cutmix-only baseline jobs (labelmix=false, mixup_switch_prob=1.0). "
             "Combinable with --baseline/--mixup/--bare.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Add bare baseline jobs (labelmix=false, mixup=0, cutmix=0). "
             "Combinable with --baseline/--mixup/--cutmix.",
    )
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Add noaug baseline jobs: sets timm's --no-aug master switch, "
             "which disables ALL train-time augmentation (RandAugment, "
             "color jitter, random erasing, mixup, cutmix, hflip, "
             "random-resized-crop, etc.) and forces a deterministic "
             "resize + center-crop pipeline. LabelMix is also disabled. "
             "Combinable with --baseline/--mixup/--cutmix/--bare.",
    )
    parser.add_argument(
        "--labelmix",
        dest="labelmix_variants",
        nargs="+",
        choices=sorted(LABELMIX_LOSS_SPECS.keys()),
        default=[],
        help="LabelMix loss variant(s) to generate. Choices: "
             f"{sorted(LABELMIX_LOSS_SPECS.keys())}. "
             "Each variant adds the corresponding wandb tag "
             "(mixed -> labelmix-mixed, pl -> labelmix-pl, "
             "sce -> labelmix-sce). Combinable with the baseline flavors.",
    )
    parser.add_argument(
        "--all",
        dest="all_configs",
        action="store_true",
        help="Generate every augmentation configuration at once: all "
             "LabelMix variants (mixed, pl, sce) PLUS baseline, mixup, "
             "cutmix, bare, noaug and mosaic. Overrides / unions with any "
             "individually-set flags.",
    )
    args = parser.parse_args()

    # --all is shorthand for turning on every augmentation flavor.  We
    # *union* with any individually-set flags rather than overwrite so
    # that ``--all --mixup`` behaves identically to ``--all``.
    if args.all_configs:
        args.baseline = True
        args.mixup = True
        args.cutmix = True
        args.bare = True
        args.noaug = True
        args.mosaic = True
        _all_variants = sorted(LABELMIX_LOSS_SPECS.keys())
        _existing = list(args.labelmix_variants or [])
        for _v in _all_variants:
            if _v not in _existing:
                _existing.append(_v)
        args.labelmix_variants = _existing

    # Every flag (--baseline, --mixup, --cutmix, --bare, --noaug, --mosaic,
    # --labelmix, --all) is additive: each contributes its own trial dict(s)
    # to the job list.  There is no mutual-exclusion check any more -- the
    # per-job guard inside ``generate()`` still rejects nonsensical
    # combinations on a *single* trial (e.g. mosaic+labelmix on the same
    # trial), but trials themselves can be freely mixed.

    generate(
        dataset=args.dataset,
        gpus_per_job=args.gpus_per_job,
        output_path=args.output,
        output_root=args.output_root,
        max_retries=args.max_retries,
        model_filter=args.model_filter,
        baseline=args.baseline,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cutmix=args.cutmix,
        bare=args.bare,
        noaug=args.noaug,
        labelmix_variants=args.labelmix_variants,
    )


if __name__ == "__main__":
    main()