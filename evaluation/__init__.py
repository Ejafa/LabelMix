"""Offline evaluation, post-processing and plotting for LabelMix.

Subpackages:
    * :mod:`evaluation.offline_eval` -- run trained checkpoints on the val set
      and emit raw CSVs + (optional) logit dumps.
    * :mod:`evaluation.metrics`      -- plugin-style metric registry.
    * :mod:`evaluation.processing`   -- raw CSVs -> tidy analysis tables.
    * :mod:`evaluation.plots`        -- one paper-ready figure per file.
    * :mod:`evaluation.scripts`      -- thin CLI entry points.
    * :mod:`evaluation.common`       -- shared path / logging helpers.

Public API (stable) — legacy imports keep working:

    from evaluation import evaluate_mapping, METRIC_REGISTRY, register_metric
"""
from .config import EvalConfig
from .metrics import METRIC_REGISTRY, MetricResult, register_metric
from .offline_eval import evaluate_mapping, evaluate_run
from .types import EvalResult

__all__ = [
    "EvalConfig",
    "EvalResult",
    "METRIC_REGISTRY",
    "MetricResult",
    "register_metric",
    "evaluate_mapping",
    "evaluate_run",
]
