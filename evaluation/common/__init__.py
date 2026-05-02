"""Shared helpers (paths, logging) used across the evaluation package."""
from .paths import (
    DATA_DIR,
    DIAGNOSTIC_DIR,
    FIGURES_DIR,
    PACKAGE_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_EVAL_DIR,
    RAW_LOGITS_DIR,
    WANDB_RAW_DIR,
    ensure_dirs,
)
from .logging import setup_logging

__all__ = [
    "PACKAGE_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "RAW_LOGITS_DIR",
    "RAW_EVAL_DIR",
    "WANDB_RAW_DIR",
    "DIAGNOSTIC_DIR",
    "PROCESSED_DIR",
    "FIGURES_DIR",
    "ensure_dirs",
    "setup_logging",
]
