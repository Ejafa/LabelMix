"""Metric registry for the evaluation pipeline.

Adding a new metric:

    # evaluation/metrics/my_metric.py
    from .registry import register_metric

    @register_metric("my_metric")
    def my_metric(logits, targets):
        ...

Then import it in this ``__init__.py`` so it is auto-registered at package
import time.

Each registered metric receives:
  * ``logits``  : ``[N, C]`` float tensor of raw (pre-softmax) model outputs,
                  concatenated across the whole eval set (CPU, float32).
  * ``targets`` : ``[N]``    long tensor of integer class indices (CPU).

A metric may return either a single float (stored under its registered
name) or a ``dict[str, float]`` (emitted as ``"<metric_name>/<sub_key>"``).
"""
from .registry import METRIC_REGISTRY, MetricFn, MetricResult, register_metric

# Auto-register all built-in metrics by importing them for their side effects.
from . import accuracy  # noqa: F401  top1_acc, top5_acc
from . import brier     # noqa: F401  brier
from . import ece       # noqa: F401  ece
from . import nll       # noqa: F401  nll

__all__ = [
    "METRIC_REGISTRY",
    "MetricFn",
    "MetricResult",
    "register_metric",
]
