"""Multi-class Brier score."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register_metric


@register_metric("brier")
def brier(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Multi-class Brier score: :math:`\\text{mean}_i \\sum_c (p_{i,c} - y_{i,c})^2`.

    Ranges in :math:`[0, 2]`; 0 is perfect, lower is better.
    """
    probs = F.softmax(logits, dim=1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.view(-1, 1), 1.0)
    per_sample = (probs - one_hot).pow(2).sum(dim=1)
    return float(per_sample.mean().item())
