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


def __getattr__(name: str):
    """Lazily expose offline evaluation entry points.

    Plotting and processing modules should import without pulling in the
    checkpoint/data-loader stack.  The offline evaluator still remains
    available through the legacy ``from evaluation import evaluate_run`` API.
    """
    if name in {"evaluate_mapping", "evaluate_run"}:
        from .offline_eval import evaluate_mapping, evaluate_run

        return {
            "evaluate_mapping": evaluate_mapping,
            "evaluate_run": evaluate_run,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
