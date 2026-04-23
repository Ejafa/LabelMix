#!/usr/bin/env python3
r"""Inspect images returned by the (balanced) dataset loader.

This tool mirrors the dataset/loader construction used in ``train.py`` so that
you can visually inspect what the model actually sees for any of the three
primary training configurations:

  * ``--inspect-mode normal``   :: plain (balanced or not) loader, optionally
                                    with Mixup / CutMix applied in-batch.
  * ``--inspect-mode labelmix`` :: :class:`BalancedBucketDataset` with
                                    ``labelmix=True`` (K / alpha schedules).
  * ``--inspect-mode mosaic``   :: :class:`BalancedBucketDataset` (labelmix
                                    disabled) wrapped in :class:`MosaicDataset`.

Quick start
-----------
Two axes of presets are wired up so you don't need to think about flags:

  * ``--dataset-preset {in1k,places365}`` hardcodes the dataset spec string,
    ``data_dir`` (matching :mod:`copy_data_to_ram`), ``train_split``,
    ``input_key`` / ``target_key`` and ``num_classes``.
  * ``--preset {normal,labelmix,mosaic}`` hardcodes the augmentation /
    balanced-loader flags for each of the three training configurations.

Pick a config, pick both presets, and run::

    # (1) Normal training config (mixup + cutmix) on ImageNet-1k
    python experiments/inspect_dataset_loader.py \
        -c experiments/labelmix_imagenet1k/configs/convnextv2-base.yaml \
        --dataset-preset in1k --preset normal \
        --output-dir output_inspect/in1k_normal

    # (2) LabelMix config on ImageNet-1k
    python experiments/inspect_dataset_loader.py \
        -c experiments/labelmix_imagenet1k/configs/convnextv2-base.yaml \
        --dataset-preset in1k --preset labelmix \
        --output-dir output_inspect/in1k_labelmix

    # (3) Mosaic config on Places365
    python experiments/inspect_dataset_loader.py \
        -c experiments/labelmix_imagenet1k/configs/convnextv2-base.yaml \
        --dataset-preset in1k --preset mosaic \
        --output-dir output_inspect/in1k_mosaic

Note: if you copy a command from a shell transcript that uses ``\\`` instead
of ``\`` for line continuation, bash will treat the ``\\`` as a *literal*
backslash rather than a newline escape, and every flag after the first line
will be silently dropped.  Use single ``\`` or put everything on one line.

Anything you pass on the CLI still wins over the presets, the presets win
over the YAML, and the YAML wins over argparse defaults.  Unknown YAML keys
are harmlessly ignored via ``parse_known_args``.

Output layout
-------------
By default the inspector writes **one PNG per sample** (plus a tiny JSON
sidecar per sample with the caption / batch index), so you can scroll a
file browser through ``output_inspect/<run>/`` instead of opening a dense
grid.  Filenames are flat-numbered (``img_00000_b001_s00.png``) so sorting
by name reproduces the order the loader returned them.

The old behaviour (one tiled grid PNG per batch) is still available:

  * ``--grid``                 :: also save the batch-grid PNG.
  * ``--no-individual``        :: skip per-sample PNGs.
  * ``--grid --no-individual`` :: grids only (== pre-refactor default).

The dataset presets set a small ``batch_size`` (8) tuned for inspection,
since we only pull ``--num-batches`` (default 4) batches and per-sample
writes would otherwise dump a lot of files when a YAML training config
carries ``batch_size: 1024``.

DataLoader workers
------------------
The inspector *forces* ``--workers 0`` by default, even when the YAML sets
it to e.g. ``workers: 8``.  Rationale: an HF Arrow dataset sharded across
hundreds of files (ImageNet-1k = 267 shards) keeps one memory-mapped FD per
shard open in the parent process; when PyTorch's DataLoader spawns workers
via ``forkserver`` on Python >= 3.12, it tries to pass those FDs through
``SCM_RIGHTS`` and the kernel refuses once the count exceeds ``SCM_MAX_FD``
(~253), yielding::

    ValueError: too many fds

Since the inspector only materializes a handful of batches, worker
parallelism is pointless anyway.  Pass an explicit ``--workers N`` on the
CLI if you really need to exercise the multi-worker path.
"""
import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# Make sibling ``timm`` package importable when running the script directly
# via ``python experiments/inspect_dataset_loader.py``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from timm import utils  # noqa: E402
from timm.data import (  # noqa: E402
    BalancedBucketDataset,
    MosaicDataset,
    Mixup,
    create_dataset,
    create_transform,
    resolve_data_config,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_logger = logging.getLogger("inspect_dataset")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Source of truth = ``experiments/generate_jobs.DATASET_CONFIGS``.  Importing
# it here guarantees that what the inspector visualizes (dataset spec string,
# data_dir, splits, input/target keys, num_classes) is *exactly* what the
# training pipeline consumes -- no silent drift.
#
# ``balanced_mode`` is deliberately NOT lifted out of the dataset config: for
# the inspector it's an augmentation-preset concern (the ``normal`` aug
# preset disables balancing by setting ``balanced_mode=""``, while
# ``labelmix`` / ``mosaic`` want the balanced loader active).

from experiments.generate_jobs import DATASET_CONFIGS as _TRAIN_DATASET_CONFIGS  # noqa: E402

# Subset of DatasetConfig fields that the inspect script consumes.  Adding a
# new dataset to generate_jobs.py automatically makes it available here.
_DATASET_PRESET_FIELDS: Tuple[str, ...] = (
    "dataset",
    "data_dir",
    "train_split",
    "input_key",
    "target_key",
    "num_classes",
)

# Per-dataset inspection batch size.  Kept small on purpose: the script only
# materializes a handful of batches and now (see --individual/--no-grid) writes
# one PNG per sample, so B=8 gives a comfortable 8*num_batches images without
# dumping hundreds of files.  Overridable via --batch-size or YAML.
_DATASET_INSPECT_BATCH_SIZE: dict = {
    "in1k": 8,
    "places365": 8,
}

DATASETS: dict = {
    tag: {
        **{k: getattr(cfg, k) for k in _DATASET_PRESET_FIELDS},
        "batch_size": _DATASET_INSPECT_BATCH_SIZE.get(tag, 8),
    }
    for tag, cfg in _TRAIN_DATASET_CONFIGS.items()
}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


PRESETS: dict = {
    "normal": {
        "inspect_mode": "normal",
        "balanced_mode": "",
        "labelmix": False,
        "mosaic": False,
        "mixup": 0.8,
        "cutmix": 1.0,
        "mixup_prob": 1.0,
        "mixup_switch_prob": 0.5,
        "mixup_mode": "batch",
        "smoothing": 0.1,
    },
    "labelmix": {
        "inspect_mode": "labelmix",
        "balanced_mode": "max",
        "labelmix": True,
        "mosaic": False,
        "labelmix_mix_k": 4,
        "labelmix_alpha_min": 0.1,
        "labelmix_alpha_max": 0.5,
        "labelmix_schedule": "cosine",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    "mosaic": {
        "inspect_mode": "mosaic",
        "balanced_mode": "max",
        "labelmix": False,
        "mosaic": True,
        # Mosaic hyperparameters (mirrors train.py defaults 1:1).
        "mosaic_prob": 1.0,
        "mosaic_center_ratio": [0.8, 1.2],     # YOLOv5: xc ~ U(S/2, 3S/2)
        "mosaic_scale_range": [0.8, 1.2],             # e.g. [0.8, 1.2] for post-mosaic zoom
        "mosaic_close_epochs": 0,               # disable Mosaic in the last N epochs
        "mosaic_fill_value": 114.0 / 255.0,     # YOLOv5 gray padding (in [0, 1])
        # Mixup / CutMix must be off (Mosaic already produces soft targets).
        "mixup": 0.0,
        "cutmix": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect images returned by the (balanced) dataset loader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- config / dataset ------------------------------------------------
    parser.add_argument("-c", "--config", default="", type=str, metavar="FILE",
                        help="YAML config file (same format as train.py configs).")
    parser.add_argument("--preset", default=None, type=str,
                        choices=sorted(PRESETS.keys()),
                        help="Apply a ready-to-run preset of augmentation flags (overridable by CLI).")
    parser.add_argument("--dataset-preset", default=None, type=str,
                        choices=sorted(DATASETS.keys()),
                        help="Hardcoded dataset: fills in --dataset, --data-dir, "
                             "--train-split, --input-key, --target-key, --num-classes.")
    parser.add_argument("--inspect-mode", default="labelmix", type=str,
                        choices=["normal", "labelmix", "mosaic"],
                        help="Which dataset/loader pipeline to inspect.")
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--data-dir", default=None, type=str,
                        help="Dataset root override (maps to --data-dir / data_dir in train.py).")
    parser.add_argument("--data", default=None, type=str,
                        help="Alias for --data-dir (for yaml compat).")
    parser.add_argument("--train-split", default="train", type=str)
    parser.add_argument("--dataset-download", action="store_true", default=False)
    parser.add_argument("--dataset-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--input-img-mode", default=None, type=str)
    parser.add_argument("--input-key", default=None, type=str)
    parser.add_argument("--target-key", default=None, type=str)
    parser.add_argument("--class-map", default="", type=str)

    # ---- output / inspection budget -------------------------------------
    parser.add_argument("--output-dir", default="output_inspect", type=str)
    parser.add_argument("--num-batches", default=4, type=int,
                        help="Number of batches to pull from the loader.")
    parser.add_argument("--samples-per-batch", default=16, type=int,
                        help="Max samples to save per batch.")
    parser.add_argument("--grid-cols", default=4, type=int,
                        help="Columns in the saved sample grid (ignored if --no-grid).")
    # Output mode: default is now one PNG *per sample* (easier to open, pan,
    # and diff).  The old batch-grid behaviour is still available, just opt-in.
    parser.add_argument("--individual", dest="individual",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Save each sample as its own PNG (default). "
                             "Disable with --no-individual.")
    parser.add_argument("--grid", dest="grid",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Also save a tiled grid per batch. "
                             "Disabled by default; enable with --grid.")
    # Back-compat alias for the old flag name.
    parser.add_argument("--save-individual", dest="individual",
                        action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--epoch", default=0, type=int,
                        help="Epoch index set on the dataset (affects LabelMix K/alpha schedules).")

    # ---- data config (timm standard) ------------------------------------
    parser.add_argument("--input-size", nargs=3, default=None, type=int, metavar=("C", "H", "W"))
    parser.add_argument("--img-size", default=None, type=int)
    parser.add_argument("--in-chans", default=None, type=int)
    parser.add_argument("--chans", default=None, type=int)
    parser.add_argument("--mean", nargs="*", default=None, type=float)
    parser.add_argument("--std", nargs="*", default=None, type=float)
    parser.add_argument("--interpolation", default="", type=str)
    parser.add_argument("--train-interpolation", default="random", type=str)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--crop-pct", default=None, type=float)

    # ---- augmentation knobs (matches train.py names) --------------------
    parser.add_argument("--no-aug", action="store_true", default=False)
    parser.add_argument("--train-crop-mode", default=None, type=str)
    parser.add_argument("--scale", nargs="+", default=[0.08, 1.0], type=float)
    parser.add_argument("--ratio", nargs="+", default=[3.0 / 4.0, 4.0 / 3.0], type=float)
    parser.add_argument("--hflip", default=0.5, type=float)
    parser.add_argument("--vflip", default=0.0, type=float)
    parser.add_argument("--color-jitter", default=0.4, type=float)
    parser.add_argument("--color-jitter-prob", default=None, type=float)
    parser.add_argument("--grayscale-prob", default=None, type=float)
    parser.add_argument("--gaussian-blur-prob", default=None, type=float)
    parser.add_argument("--aa", default=None, type=str)
    parser.add_argument("--reprob", default=0.0, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=1, type=int)
    parser.add_argument("--resplit", action="store_true", default=False)

    # ---- mixup / cutmix -------------------------------------------------
    parser.add_argument("--mixup", default=0.0, type=float)
    parser.add_argument("--cutmix", default=0.0, type=float)
    parser.add_argument("--cutmix-minmax", nargs="+", default=None, type=float)
    parser.add_argument("--mixup-prob", default=1.0, type=float)
    parser.add_argument("--mixup-switch-prob", default=0.5, type=float)
    parser.add_argument("--mixup-mode", default="batch", type=str)
    parser.add_argument("--smoothing", default=0.1, type=float)

    # ---- loader ---------------------------------------------------------
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--workers", default=0, type=int,
                        help="DataLoader workers.  Defaults to 0 (see module "
                             "docstring): sharded HF datasets + Python 3.12+ "
                             "forkserver trigger 'too many fds'.  The inspect "
                             "script forces 0 unless you pass --workers N "
                             "explicitly on the CLI, overriding any YAML value.")

    # ---- balanced loading -----------------------------------------------
    parser.add_argument("--balanced-mode", default="", type=str)
    parser.add_argument("--balanced-buffer", default=256, type=int)
    parser.add_argument("--balanced-cache-path", default="", type=str)
    parser.add_argument("--balanced-cache-threshold", default=256, type=int)
    parser.add_argument("--balanced-input-key", default=None, type=str)
    parser.add_argument("--balanced-target-key", default=None, type=str)

    # ---- labelmix -------------------------------------------------------
    parser.add_argument("--labelmix", action="store_true", default=False)
    parser.add_argument("--labelmix-mix-k", default=5, type=int)
    parser.add_argument("--labelmix-k-min", default=None, type=int)
    parser.add_argument("--labelmix-k-max", default=None, type=int)
    parser.add_argument("--labelmix-k-schedule", default="linear", type=str,
                        choices=["fixed", "linear", "cosine"])
    parser.add_argument("--labelmix-k-reverse", action="store_true", default=False)
    parser.add_argument("--labelmix-k-warmup-epochs", default=0, type=int)
    parser.add_argument("--labelmix-k-total-epochs", default=None, type=int)
    parser.add_argument("--labelmix-k-cooldown-epochs", default=None, type=int)
    parser.add_argument("--labelmix-alpha-min", default=0.1, type=float)
    parser.add_argument("--labelmix-alpha-max", default=1.0, type=float)
    parser.add_argument("--labelmix-schedule", default="linear", type=str,
                        choices=["fixed", "linear", "cosine"])
    parser.add_argument("--labelmix-reverse", action="store_true", default=False)
    parser.add_argument("--labelmix-step-mode", default="total", type=str,
                        choices=["epoch", "total"])
    parser.add_argument("--labelmix-warmup-steps", default=0, type=int)
    parser.add_argument("--labelmix-total-epochs", default=None, type=int)
    parser.add_argument("--labelmix-total-steps", default=None, type=int)
    parser.add_argument("--labelmix-sampling", action="store_true", default=False)
    parser.add_argument("--labelmix-sampling-min-side-px", default=6, type=int)
    parser.add_argument("--labelmix-sampling-max-aspect", default=10.0, type=float)
    parser.add_argument("--labelmix-sampling-bins", default=16, type=int)
    parser.add_argument("--labelmix-sampling-pool-size", default=128, type=int)
    parser.add_argument("--labelmix-sampling-low-watermark", default=32, type=int)
    parser.add_argument("--labelmix-sampling-max-attempts", default=200, type=int)
    parser.add_argument("--epochs", default=None, type=int,
                        help="Used to populate LabelMix/Mosaic total_epochs when not explicit.")

    # ---- mosaic ---------------------------------------------------------
    # Mirrors the `Mosaic augmentation` argparse group in train.py; additionally
    # exposes `--mosaic-fill-value`, which MosaicDataset supports but train.py
    # leaves at its default (114/255, the YOLOv5 gray).
    parser.add_argument("--mosaic", action="store_true", default=False,
                        help="Enable Mosaic augmentation (requires --balanced-mode, "
                             "mutually exclusive with --labelmix).")
    parser.add_argument("--mosaic-prob", default=1.0, type=float,
                        help="Probability of applying Mosaic per sample (default: 1.0).")
    parser.add_argument("--mosaic-center-ratio", nargs=2, default=[0.5, 1.5], type=float,
                        metavar=("LO", "HI"),
                        help="Center-ratio range for Mosaic quadrant boundary. "
                             "Default (0.5, 1.5) matches YOLOv5 with "
                             "mosaic_border=-s//2, i.e. xc ~ U(s/2, 3s/2).")
    parser.add_argument("--mosaic-scale-range", nargs=2, default=None, type=float,
                        metavar=("LO", "HI"),
                        help="Optional post-mosaic affine zoom range, e.g. 0.8 1.2. "
                             "Applied to the full 2S x 2S canvas around the sampled "
                             "center BEFORE the S x S crop (YOLOv5 / MMYOLO style). "
                             "Disabled if not set.")
    parser.add_argument("--mosaic-close-epochs", default=0, type=int,
                        help="Disable Mosaic during the last N epochs "
                             "(close-mosaic schedule, default: 0). "
                             "Requires --epochs to take effect.")
    parser.add_argument("--mosaic-fill-value", default=114.0 / 255.0, type=float,
                        help="Gray padding value (in [0, 1]) for uncovered canvas "
                             "regions. Default 114/255 matches YOLOv5.")

    parser.add_argument("--seed", default=42, type=int)

    # Track which args came from the CLI so we know what's safe to override.
    cli_tokens = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            cli_tokens.add(tok.split("=", 1)[0].lstrip("-").replace("-", "_"))
        elif tok == "-c":
            cli_tokens.add("config")

    args, _ = parser.parse_known_args()
    defaults = {a.dest: parser.get_default(a.dest) for a in parser._actions}

    # 1a) Apply the dataset preset FIRST (only for flags the user didn't set).
    if args.dataset_preset:
        _logger.info("Applying --dataset-preset=%s", args.dataset_preset)
        for k, v in DATASETS[args.dataset_preset].items():
            if k in cli_tokens:
                continue
            setattr(args, k, v)

    # 1b) Apply the augmentation preset (only for flags the user didn't set).
    if args.preset:
        _logger.info("Applying --preset=%s", args.preset)
        for k, v in PRESETS[args.preset].items():
            if k in cli_tokens:
                continue
            setattr(args, k, v)

    # 2) Then apply the YAML config (only overrides values still at default).
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if not hasattr(args, key):
                continue  # ignore unknown yaml keys (e.g. optimizer settings)
            if key in cli_tokens:
                continue
            # Don't let yaml clobber preset-injected values.
            if args.preset and key in PRESETS[args.preset]:
                continue
            if args.dataset_preset and key in DATASETS[args.dataset_preset]:
                continue
            if getattr(args, key, defaults.get(key)) == defaults.get(key):
                setattr(args, key, v)

    # Harmonize --data / --data-dir.
    if args.data_dir is None and getattr(args, "data", None):
        args.data_dir = args.data

    # 3) Force workers=0 for the inspector unless the user explicitly asked
    #    otherwise.  YAMLs lifted from training configs routinely set
    #    workers=8 which crashes this script on many-shard HF datasets
    #    (forkserver + SCM_RIGHTS FD cap, see module docstring).
    if "workers" not in cli_tokens and int(getattr(args, "workers", 0)) != 0:
        _logger.info(
            "Forcing --workers=0 for inspection (was %d from YAML). "
            "Pass --workers N explicitly to override.",
            int(args.workers),
        )
        args.workers = 0

    return args


def _ensure_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _denorm_to_uint8(t: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """CHW float tensor (possibly normalized) -> HWC uint8 tensor on CPU."""
    assert t.ndim == 3, f"expected CHW, got shape={tuple(t.shape)}"
    t = t.detach().float().cpu()
    c = t.shape[0]
    mean_t = torch.tensor(list(mean)[:c], dtype=t.dtype).view(c, 1, 1)
    std_t = torch.tensor(list(std)[:c], dtype=t.dtype).view(c, 1, 1)

    # Heuristic: if tensor already sits in [0, 1] we don't denormalize.
    if float(t.min()) >= -0.05 and float(t.max()) <= 1.05 and float(t.max()) > 0.6:
        out = t.clamp(0.0, 1.0)
    else:
        out = (t * std_t + mean_t).clamp(0.0, 1.0)
    out = (out * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous()
    if c == 1:
        out = out.expand(-1, -1, 3).contiguous()
    return out


def _format_target(target: Any) -> str:
    if isinstance(target, torch.Tensor):
        if target.ndim == 0:
            return str(int(target.item()))
        if target.ndim == 1 and target.numel() <= 16:
            # Dense soft target
            vals = target.detach().float().cpu().tolist()
            top = sorted(enumerate(vals), key=lambda kv: -kv[1])[:3]
            return ", ".join(f"{i}:{p:.2f}" for i, p in top)
        # class ids tensor (Labelmix/Mosaic)
        return str(target.detach().cpu().tolist())
    if isinstance(target, (list, tuple)):
        return str(target)
    return str(target)


def _save_grid(
    path: str,
    imgs: torch.Tensor,  # (N, H, W, 3) uint8
    captions: Sequence[str],
    cols: int,
    title: str,
) -> None:
    n = int(imgs.shape[0])
    if n == 0:
        return
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))

    if plt is None:
        # Fallback: just paste into a simple canvas via PIL.
        h, w = int(imgs.shape[1]), int(imgs.shape[2])
        canvas = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
        for i in range(n):
            r, c = divmod(i, cols)
            tile = Image.fromarray(imgs[i].numpy())
            canvas.paste(tile, (c * w, r * h))
        canvas.save(path)
        return

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows + 0.6))
    if rows * cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n:
            ax.imshow(imgs[i].numpy())
            ax.set_title(captions[i], fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset / loader builders
# ---------------------------------------------------------------------------


def _build_transform(args: argparse.Namespace, data_config: dict) -> Any:
    """Train-time transform matching the one in train.py (balanced path)."""
    return create_transform(
        input_size=data_config["input_size"],
        is_training=True,
        no_aug=_ensure_bool(getattr(args, "no_aug", False)),
        train_crop_mode=getattr(args, "train_crop_mode", None),
        scale=tuple(args.scale),
        ratio=tuple(args.ratio),
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        interpolation=args.train_interpolation or data_config.get("interpolation", "bicubic"),
        mean=data_config["mean"],
        std=data_config["std"],
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=0,
        use_prefetcher=False,
        separate=False,
    )


def _build_labelmix_kwargs(args: argparse.Namespace) -> dict:
    default_train_epochs = getattr(args, "epochs", None)
    if default_train_epochs is None:
        default_train_epochs = getattr(args, "labelmix_total_epochs", None)
    if default_train_epochs is None:
        default_train_epochs = 1

    k_min = args.labelmix_k_min if args.labelmix_k_min is not None else args.labelmix_mix_k
    k_max = args.labelmix_k_max if args.labelmix_k_max is not None else args.labelmix_mix_k
    k_total = (
        args.labelmix_k_total_epochs
        if args.labelmix_k_total_epochs is not None
        else args.labelmix_total_epochs
    )
    if k_total is None:
        k_total = default_train_epochs

    return {
        "mix_k": int(args.labelmix_mix_k),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "k_schedule": str(args.labelmix_k_schedule),
        "k_reverse": _ensure_bool(args.labelmix_k_reverse),
        "k_warmup_epochs": int(args.labelmix_k_warmup_epochs),
        "k_total_epochs": k_total,
        "labelmix_k_cooldown_epochs": args.labelmix_k_cooldown_epochs,
        "train_epochs": default_train_epochs,
        "alpha_min": float(args.labelmix_alpha_min),
        "alpha_max": float(args.labelmix_alpha_max),
        "schedule": str(args.labelmix_schedule),
        "reverse": _ensure_bool(args.labelmix_reverse),
        "step_mode": str(args.labelmix_step_mode),
        "warmup_steps": int(args.labelmix_warmup_steps),
        "total_epochs": args.labelmix_total_epochs,
        "total_steps": args.labelmix_total_steps,
        "batch_size": int(args.batch_size),
        "sampling": _ensure_bool(args.labelmix_sampling),
        "sampling_min_side_px": int(args.labelmix_sampling_min_side_px),
        "sampling_max_aspect": float(args.labelmix_sampling_max_aspect),
        "sampling_bins": int(args.labelmix_sampling_bins),
        "sampling_pool_size": int(args.labelmix_sampling_pool_size),
        "sampling_low_watermark": int(args.labelmix_sampling_low_watermark),
        "sampling_max_attempts": int(args.labelmix_sampling_max_attempts),
    }


def _inspection_collate(batch: List[Any]) -> Tuple[torch.Tensor, Any]:
    """Collate that preserves tuple/tensor LabelMix/Mosaic targets."""
    if not batch:
        return torch.empty(0), torch.empty(0)

    sample = batch[0]
    if not (isinstance(sample, tuple) and len(sample) >= 2):
        # Fallback: let torch default_collate handle it.
        return torch.utils.data.dataloader.default_collate(batch)

    imgs = torch.stack([b[0] for b in batch], dim=0).contiguous()
    target = sample[1]

    if isinstance(target, (tuple, list)) and len(target) >= 2 and isinstance(target[0], torch.Tensor):
        # LabelMix / Mosaic: (labels[K], weights[K]) (+ optional sym_id)
        labels = torch.stack([b[1][0] for b in batch], dim=0)
        weights = torch.stack([b[1][1] for b in batch], dim=0)
        if len(sample[1]) >= 3:
            sym_ids = torch.tensor([int(b[1][2]) for b in batch], dtype=torch.int64)
            return imgs, (labels, weights, sym_ids)
        return imgs, (labels, weights)

    # Scalar int labels
    labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.int64)
    return imgs, labels


def _build_dataset_and_loader(args: argparse.Namespace, data_config: dict) -> Tuple[Any, torch.utils.data.DataLoader]:
    mode = args.inspect_mode
    _logger.info("Building dataset for inspect-mode=%s", mode)

    if not args.dataset:
        raise ValueError(
            "--dataset must be specified (either via CLI, YAML, or a "
            "--dataset-preset). Available dataset presets: "
            f"{sorted(DATASETS.keys())}. "
            "Tip: if you copied a multi-line command using '\\\\' for "
            "continuation, bash will drop every flag after the first line \u2014 "
            "use single '\\' or put everything on one line."
        )

    base_dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=_ensure_bool(args.dataset_download),
        input_img_mode=args.input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        trust_remote_code=_ensure_bool(args.dataset_trust_remote_code),
    )

    transform = _build_transform(args, data_config)

    # --- Normal (non-balanced) path -----------------------------------------
    if mode == "normal" and not args.balanced_mode:
        if hasattr(base_dataset, "transform"):
            base_dataset.transform = transform
        loader = torch.utils.data.DataLoader(
            base_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            collate_fn=_inspection_collate,
            drop_last=True,
            pin_memory=False,
        )
        return base_dataset, loader

    # --- Balanced path (used by labelmix / mosaic / balanced-normal) --------
    if not args.balanced_mode:
        raise ValueError(
            f"--inspect-mode={mode} requires --balanced-mode to be set "
            f"(e.g. 'max' or 'min')."
        )

    balanced_input_key = args.balanced_input_key or (args.input_key or "image")
    balanced_target_key = args.balanced_target_key or (args.target_key or "label")

    cache_path = args.balanced_cache_path
    if not cache_path:
        cache_path = os.path.join(args.output_dir, "class_buckets.pkl")
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    want_labelmix = mode == "labelmix" or (_ensure_bool(args.labelmix) and mode != "mosaic")
    labelmix_kwargs = _build_labelmix_kwargs(args) if want_labelmix else None

    dataset = BalancedBucketDataset(
        base_dataset=base_dataset,
        transform=transform,
        mode=args.balanced_mode,
        buffer_size=int(args.balanced_buffer),
        cache_path=cache_path,
        cache_small_classes_threshold=int(args.balanced_cache_threshold),
        input_key=balanced_input_key,
        target_key=balanced_target_key,
        labelmix=want_labelmix,
        labelmix_kwargs=labelmix_kwargs,
    )

    if mode == "mosaic":
        if want_labelmix:
            raise ValueError("Mosaic is mutually exclusive with LabelMix.")
        # --- Mosaic hyperparameter validation (mirrors train.py) ------------
        if not (0.0 <= float(args.mosaic_prob) <= 1.0):
            raise ValueError("--mosaic-prob must be in [0, 1]")
        if (len(args.mosaic_center_ratio) != 2
                or args.mosaic_center_ratio[0] > args.mosaic_center_ratio[1]):
            raise ValueError("--mosaic-center-ratio must be 'LO HI' with LO <= HI")
        if args.mosaic_scale_range is not None:
            if (len(args.mosaic_scale_range) != 2
                    or args.mosaic_scale_range[0] <= 0
                    or args.mosaic_scale_range[0] > args.mosaic_scale_range[1]):
                raise ValueError("--mosaic-scale-range must be 'LO HI' with 0 < LO <= HI")
        if int(args.mosaic_close_epochs) < 0:
            raise ValueError("--mosaic-close-epochs must be >= 0")

        _, h, w = data_config["input_size"]
        dataset = MosaicDataset(
            base_dataset=dataset,
            output_size=(int(h), int(w)),
            prob=float(args.mosaic_prob),
            center_ratio_range=tuple(args.mosaic_center_ratio),
            post_scale_range=tuple(args.mosaic_scale_range) if args.mosaic_scale_range else None,
            close_epochs=int(args.mosaic_close_epochs),
            total_epochs=getattr(args, "epochs", None),
            fill_value=float(args.mosaic_fill_value),
            seed=int(args.seed),
        )
        _logger.info(
            "Mosaic enabled: prob=%.3f center_ratio=%s scale=%s close_epochs=%d "
            "fill_value=%.4f (total_epochs=%s)",
            float(args.mosaic_prob), tuple(args.mosaic_center_ratio),
            tuple(args.mosaic_scale_range) if args.mosaic_scale_range else None,
            int(args.mosaic_close_epochs), float(args.mosaic_fill_value),
            getattr(args, "epochs", None),
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=int(args.workers),
        collate_fn=_inspection_collate,
        drop_last=True,
        pin_memory=False,
    )
    return dataset, loader


# ---------------------------------------------------------------------------
# Inspection loop
# ---------------------------------------------------------------------------


def _caption_for(
    target: Any,
    idx_in_batch: int,
) -> str:
    """Produce a short caption string for the i-th sample in a batch."""
    if isinstance(target, (tuple, list)) and len(target) >= 2 and isinstance(target[0], torch.Tensor):
        labels_i = target[0][idx_in_batch]
        weights_i = target[1][idx_in_batch]
        pairs = [
            f"{int(l)}:{float(w):.2f}"
            for l, w in zip(labels_i.tolist(), weights_i.tolist())
        ]
        return " ".join(pairs)

    if isinstance(target, torch.Tensor):
        if target.ndim == 1:
            return f"y={int(target[idx_in_batch].item())}"
        if target.ndim >= 2:
            row = target[idx_in_batch].detach().float().cpu()
            top = sorted(enumerate(row.tolist()), key=lambda kv: -kv[1])[:3]
            return " ".join(f"{i}:{p:.2f}" for i, p in top)
    return _format_target(target)


def _label_for(
    target: Any,
    idx_in_batch: int,
    topk: int = 5,
) -> dict:
    """Return a JSON-friendly structured label record for the i-th sample.

    The record always contains a ``kind`` tag so downstream tooling can
    dispatch without having to re-infer the augmentation mode:

    * ``kind="hard"``  -- ``{"class": int}`` (vanilla training, no mixup).
    * ``kind="soft"``  -- ``{"topk": [{"class": int, "prob": float}, ...],
                              "num_classes": int}``
                          (mixup / cutmix produced a soft one-hot-ish vector).
    * ``kind="mix"``   -- ``{"components": [{"class": int, "weight": float},
                                              ...]}``
                          (labelmix / mosaic produced a ``(labels, weights)``
                          tuple with ``K`` mixing components per sample).
    * ``kind="raw"``   -- fallback string repr.
    """
    # labelmix / mosaic: tuple of (labels[B, K], weights[B, K])
    if (
        isinstance(target, (tuple, list))
        and len(target) >= 2
        and isinstance(target[0], torch.Tensor)
        and isinstance(target[1], torch.Tensor)
    ):
        labels_i = target[0][idx_in_batch].detach().cpu().tolist()
        weights_i = target[1][idx_in_batch].detach().float().cpu().tolist()
        # Normalize to scalar lists (K=1 comes through as a 0-d tensor).
        if not isinstance(labels_i, list):
            labels_i = [labels_i]
        if not isinstance(weights_i, list):
            weights_i = [weights_i]
        return {
            "kind": "mix",
            "components": [
                {"class": int(l), "weight": float(w)}
                for l, w in zip(labels_i, weights_i)
            ],
        }

    if isinstance(target, torch.Tensor):
        # Vanilla hard labels: shape [B].
        if target.ndim == 1:
            return {"kind": "hard", "class": int(target[idx_in_batch].item())}
        # Soft labels from Mixup: shape [B, num_classes].
        if target.ndim >= 2:
            row = target[idx_in_batch].detach().float().cpu().tolist()
            top = sorted(enumerate(row), key=lambda kv: -kv[1])[:max(1, int(topk))]
            return {
                "kind": "soft",
                "num_classes": int(target.shape[1]),
                "topk": [
                    {"class": int(i), "prob": float(p)}
                    for i, p in top
                ],
            }

    return {"kind": "raw", "repr": _format_target(target)}


def _iter_batches(
    dataset: Any,
    loader: torch.utils.data.DataLoader,
    num_batches: int,
    epoch: int,
) -> Iterable[Tuple[int, Any, Any]]:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    it = iter(loader)
    for step in range(1, num_batches + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        imgs, target = batch
        yield step, imgs, target


def inspect(args: argparse.Namespace) -> None:
    data_config = resolve_data_config(vars(args), model=None)
    _logger.info("Resolved data_config=%s", data_config)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "inspect_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    dataset, loader = _build_dataset_and_loader(args, data_config)

    mixup_fn: Optional[Mixup] = None
    mixup_active = (args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None)
    if args.inspect_mode == "normal" and mixup_active:
        if args.num_classes is None:
            raise ValueError("Mixup/CutMix require --num-classes to be set.")
        mixup_fn = Mixup(
            mixup_alpha=float(args.mixup),
            cutmix_alpha=float(args.cutmix),
            cutmix_minmax=tuple(args.cutmix_minmax) if args.cutmix_minmax else None,
            prob=float(args.mixup_prob),
            switch_prob=float(args.mixup_switch_prob),
            mode=str(args.mixup_mode),
            label_smoothing=float(args.smoothing),
            num_classes=int(args.num_classes),
        )
        _logger.info(
            "Applying Mixup(mixup=%.2f, cutmix=%.2f, prob=%.2f, switch=%.2f, mode=%s)",
            args.mixup, args.cutmix, args.mixup_prob, args.mixup_switch_prob, args.mixup_mode,
        )

    mean = data_config["mean"]
    std = data_config["std"]
    samples_per_batch = max(1, int(args.samples_per_batch))

    save_individual = bool(getattr(args, "individual", True))
    save_grid = bool(getattr(args, "grid", False))
    if not (save_individual or save_grid):
        _logger.warning(
            "Both --no-individual and --no-grid requested; forcing --individual "
            "so at least *something* gets written."
        )
        save_individual = True

    total_images = 0
    for step, imgs, target in _iter_batches(dataset, loader, args.num_batches, args.epoch):
        # Apply mixup/cutmix AFTER collation, matching train.py behaviour.
        if mixup_fn is not None and imgs.shape[0] % 2 == 0:
            imgs, target = mixup_fn(imgs.float(), target)

        n = min(samples_per_batch, int(imgs.shape[0]))
        to_save = torch.stack(
            [_denorm_to_uint8(imgs[i], mean=mean, std=std) for i in range(n)],
            dim=0,
        )
        captions = [_caption_for(target, i) for i in range(n)]
        labels = [_label_for(target, i) for i in range(n)]

        title = (
            f"[{args.inspect_mode}] batch={step} "
            f"B={int(imgs.shape[0])} HxW={int(imgs.shape[-2])}x{int(imgs.shape[-1])}"
        )

        if save_grid:
            grid_path = os.path.join(args.output_dir, f"batch_{step:03d}.png")
            _save_grid(grid_path, to_save, captions,
                       cols=int(args.grid_cols), title=title)
            _logger.info("Saved grid %s (%d samples)", grid_path, n)

        if save_individual:
            for i in range(n):
                # Flat numbering makes sorting by filename == sorting by order
                # seen; the (step, sample) pair is preserved in the filename.
                img_idx = total_images + i
                out_path = os.path.join(
                    args.output_dir,
                    f"img_{img_idx:05d}_b{step:03d}_s{i:02d}.png",
                )
                Image.fromarray(to_save[i].numpy()).save(out_path)
                # Per-sample sidecar JSON: caption + batch/sample indices so
                # downstream tooling can recover the label(s) without having
                # to re-parse the grid.
                sidecar = {
                    "image": os.path.basename(out_path),
                    "batch": int(step),
                    "sample": int(i),
                    "mode": args.inspect_mode,
                    "caption": captions[i],
                    "label": labels[i],
                }
                sidecar_path = os.path.splitext(out_path)[0] + ".json"
                with open(sidecar_path, "w") as f:
                    json.dump(sidecar, f, indent=2, default=str)
            _logger.info(
                "Saved %d individual PNGs for batch %d (total so far: %d)",
                n, step, total_images + n,
            )

        total_images += n

        # Dump a small per-batch metadata record for scripting downstream.
        meta = {
            "step": int(step),
            "mode": args.inspect_mode,
            "batch_shape": list(imgs.shape),
            "captions": captions,
            "labels": labels,
        }
        if isinstance(target, (tuple, list)):
            meta["target_type"] = "mix_tuple"
            meta["target_shapes"] = [list(t.shape) for t in target if isinstance(t, torch.Tensor)]
        elif isinstance(target, torch.Tensor):
            meta["target_type"] = "tensor"
            meta["target_shape"] = list(target.shape)
        else:
            meta["target_type"] = type(target).__name__

        with open(os.path.join(args.output_dir, f"batch_{step:03d}.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    _logger.info("Done. Wrote %d images under %s", total_images, args.output_dir)


def main() -> None:
    utils.setup_default_logging()
    args = _parse_args()
    _logger.info("args=%s", json.dumps(vars(args), indent=2, default=str))
    inspect(args)


if __name__ == "__main__":
    main()
