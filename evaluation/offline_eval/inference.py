"""Forward-pass utilities: collect logits/targets and apply the metric registry."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..metrics import METRIC_REGISTRY


@torch.inference_mode()
def collect_logits(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_autocast,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` over the full loader and return ``(logits, targets)`` on CPU.

    Logits are accumulated in float32 for numerical stability of downstream
    metrics (NLL / Brier / ECE).  ``amp_autocast`` is a callable yielding a
    context manager (e.g. ``functools.partial(torch.autocast, ...)`` or
    :class:`contextlib.suppress`).
    """
    logits_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []

    for x, y in loader:
        # ``create_loader`` with ``use_prefetcher`` already places tensors on
        # device, but we guard against loaders that don't.
        if x.device != device:
            x = x.to(device, non_blocking=True)
        if y.device != device:
            y = y.to(device, non_blocking=True)
        with amp_autocast():
            out = model(x)
        logits_chunks.append(out.detach().float().cpu())
        target_chunks.append(y.detach().cpu())

    logits = torch.cat(logits_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0).long()
    return logits, targets


def run_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Apply every metric in ``metric_names`` (or all registered) to the tensors.

    Metric functions returning a dict get their sub-keys prefixed with
    ``"<metric_name>/"`` so the resulting flat dict can be serialized to CSV.
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    results: Dict[str, float] = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise KeyError(
                f"Metric '{name}' is not registered. "
                f"Known: {list(METRIC_REGISTRY)}"
            )
        value = METRIC_REGISTRY[name](logits, targets)
        if isinstance(value, Mapping):
            for sub_k, sub_v in value.items():
                results[f"{name}/{sub_k}"] = float(sub_v)
        else:
            results[name] = float(value)
    return results
