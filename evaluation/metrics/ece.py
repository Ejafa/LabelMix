"""Expected Calibration Error (Guo et al., 2017) with equal-width bins."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import register_metric


_DEFAULT_BIN_COUNTS = (5, 10, 15, 20)


@register_metric("ece")
def ece(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Expected Calibration Error with equal-width confidence bins.

    Uses the top-1 predicted probability as confidence (standard definition
    from Guo et al., 2017).  Returns one value per bin-count in
    :data:`_DEFAULT_BIN_COUNTS`, keyed as ``"n_bins=<k>"``.
    """
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()

    results: dict[str, float] = {}
    for n_bins in _DEFAULT_BIN_COUNTS:
        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
        total = conf.numel()
        ece_val = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            # include right edge only in the last bin
            if i == n_bins - 1:
                in_bin = (conf >= lo) & (conf <= hi)
            else:
                in_bin = (conf >= lo) & (conf < hi)
            n_in = int(in_bin.sum().item())
            if n_in == 0:
                continue
            acc_bin = float(correct[in_bin].mean().item())
            conf_bin = float(conf[in_bin].mean().item())
            ece_val += (n_in / total) * abs(acc_bin - conf_bin)
        results[f"n_bins={n_bins}"] = float(ece_val)
    return results
