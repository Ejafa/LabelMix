"""Shared result dataclasses for the evaluation package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvalResult:
    """Everything produced by one offline evaluation of a single run dir.

    Attributes:
        name: Human-readable alias from the ``{name: run_dir}`` mapping.
        run_dir: Absolute / relative path to the checkpoint folder.
        checkpoint: Path to the checkpoint file actually loaded.
        args: Raw ``args.yaml`` contents from training time.
        metrics: ``{metric_name: value}`` returned by :data:`METRIC_REGISTRY`.
        eval_time: Wall-clock seconds spent evaluating this run.
    """

    name: str
    run_dir: str
    checkpoint: str
    args: Dict[str, Any]
    metrics: Dict[str, float]
    eval_time: float
