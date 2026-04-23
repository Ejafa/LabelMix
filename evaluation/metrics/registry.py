"""Metric registry primitives.

A metric is a pure function ``fn(logits, targets) -> float | dict[str, float]``.
Registering it via :func:`register_metric` makes it automatically available
to the offline evaluator and any CLI that introspects
:data:`METRIC_REGISTRY`.

The registry lives in its own module so that metric implementation files can
all depend on a single, circular-import-free source of truth.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Union

import torch


MetricResult = Union[float, Mapping[str, float]]
MetricFn = Callable[[torch.Tensor, torch.Tensor], MetricResult]

#: Global registry of ``{name: metric_fn}``.  Populated at import time by each
#: concrete metric module.
METRIC_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator: register ``fn`` in :data:`METRIC_REGISTRY` under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    def _wrap(fn: MetricFn) -> MetricFn:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' already registered.")
        METRIC_REGISTRY[name] = fn
        return fn
    return _wrap
