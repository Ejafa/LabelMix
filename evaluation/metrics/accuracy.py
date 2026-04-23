"""Top-k classification accuracy metrics."""
from __future__ import annotations

import torch

from .registry import register_metric


@register_metric("top1_acc")
def top1_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in percent."""
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item() * 100.0)


@register_metric("top5_acc")
def top5_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-5 accuracy in percent."""
    k = min(5, logits.shape[1])
    _, topk = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = topk.eq(targets.view(-1, 1)).any(dim=1)
    return float(correct.float().mean().item() * 100.0)
