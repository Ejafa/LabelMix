"""Negative log-likelihood metric."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register_metric


@register_metric("nll")
def nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean per-sample negative log-likelihood of the true class."""
    return float(F.cross_entropy(logits, targets, reduction="mean").item())
