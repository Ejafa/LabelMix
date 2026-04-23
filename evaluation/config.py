"""Evaluator configuration.

This module owns the dataclasses and default column sets consumed by the
offline evaluation runner.  It deliberately has no heavy dependencies so that
downstream tooling (processing, plots, notebooks) can import the same
constants without pulling in torch / timm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Default CSV column policy
# ---------------------------------------------------------------------------

#: ``args.yaml`` keys surfaced as CSV columns by default.  Override via
#: ``EvalConfig.args_columns``.
DEFAULT_ARGS_COLUMNS: Tuple[str, ...] = (
    "model",
    "dataset",
    "img_size",
    "num_classes",
    "layer_decay",
    "lr_base",
    "seed",
    "experiment",
    "labelmix",
    "labelmix_loss",
    "labelmix_mixed_alpha",
    "labelmix_k_min",
    "labelmix_k_max",
    "labelmix_alpha_min",
    "labelmix_alpha_max",
)

#: Fixed identifying columns that always lead the CSV.
DEFAULT_FIXED_COLUMNS: Tuple[str, ...] = ("name", "run_dir", "checkpoint")


@dataclass
class EvalConfig:
    """Knobs for the offline evaluator.

    Attributes:
        checkpoint_name: File name of the checkpoint inside every run dir.
        use_ema: Prefer EMA weights when present in the checkpoint.
        batch_size: Validation batch size.
        workers: DataLoader worker count.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        amp: Enable autocast during the forward pass.
        amp_dtype: Autocast dtype; one of ``"bfloat16"`` / ``"float16"``.
        data_dir_override: If set, overrides ``data_dir`` from ``args.yaml``.
        dataset_override: If set, overrides ``dataset`` from ``args.yaml``.
        val_split_override: If set, overrides ``val_split`` from ``args.yaml``.
        metrics: Subset of registered metric names to run.  ``None`` runs all.
        args_columns: ``args.yaml`` keys to surface as CSV columns.
        fixed_columns: Identifying columns that lead the CSV.
        save_raw_logits: If True, dump ``{run_dir_slug}.pt`` to ``raw_logits_dir``.
        raw_logits_dir: Where to write raw logits / targets tensors.
    """

    checkpoint_name: str = "model_best.pth.tar"
    use_ema: bool = True
    batch_size: int = 256
    workers: int = 4
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "bfloat16"
    data_dir_override: Optional[str] = None
    dataset_override: Optional[str] = None
    val_split_override: Optional[str] = None
    metrics: Optional[Sequence[str]] = None
    args_columns: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_ARGS_COLUMNS)
    )
    fixed_columns: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_FIXED_COLUMNS)
    )
    save_raw_logits: bool = False
    raw_logits_dir: Optional[str] = None
