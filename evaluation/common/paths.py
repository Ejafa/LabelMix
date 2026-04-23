"""Canonical filesystem layout for the evaluation package.

All evaluation artifacts live under ``evaluation/data`` and follow a strict
raw / processed / figures split:

* ``data/raw/``        -- outputs of the offline evaluator (eval CSVs, logits)
* ``data/processed/``  -- tidy tables ready for analysis / plotting
* ``data/processed/figures/`` -- final paper-ready PDFs / PNGs

Tools should never write outside these three directories, and plotting code
should never read from ``data/raw/``.
"""
from __future__ import annotations

from pathlib import Path


#: ``evaluation/`` package root.
PACKAGE_DIR: Path = Path(__file__).resolve().parent.parent

#: Evaluation data root (``evaluation/data``).
DATA_DIR: Path = PACKAGE_DIR / "data"

#: Raw artifacts written directly by the offline evaluator.
RAW_DIR: Path = DATA_DIR / "raw"

#: Raw logits / targets tensors (``.pt`` files) keyed by run name.
RAW_LOGITS_DIR: Path = RAW_DIR / "logits"

#: Raw per-run evaluation CSVs (one per ``run_eval`` invocation).
RAW_EVAL_DIR: Path = RAW_DIR / "eval_csv"

#: Raw W&B run materializations (one subdir per run_id).
WANDB_RAW_DIR: Path = RAW_DIR / "wandb"

#: Processed, tidy tables that plots and analyses consume.
PROCESSED_DIR: Path = DATA_DIR / "processed"

#: Rendered figures (one file per plot script).
FIGURES_DIR: Path = PROCESSED_DIR / "figures"


def ensure_dirs() -> None:
    """Create the canonical directory tree if it doesn't exist yet."""
    for d in (
        RAW_DIR,
        RAW_LOGITS_DIR,
        RAW_EVAL_DIR,
        WANDB_RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
