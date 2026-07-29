"""Mixup-style composite LabelMix loss.

Combines soft-target cross-entropy and Plackett-Luce losses as:

    loss = alpha * soft_ce + (1 - alpha) * pl_loss

Both sub-losses are always evaluated so that the autograd graph is
identical on every rank (required for DDP correctness).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_entropy import LabelMixSoftTargetCrossEntropy
from .plackett_luce import LabelMixPlackettLuceLoss


# class LabelMixMixupLoss(nn.Module):
#     """Convex combination of LabelMix soft-CE and Plackett-Luce losses.

#     Parameters
#     ----------
#     alpha : float
#         Mixing coefficient in [0, 1].
#         Final loss = alpha * soft_ce + (1 - alpha) * pl_loss.
#     """

#     def __init__(self, alpha: float = 0.5) -> None:
#         super(LabelMixMixupLoss, self).__init__()
#         if not 0.0 <= alpha <= 1.0:
#             raise ValueError(
#                 f"LabelMixMixupLoss alpha must be in [0, 1], got {alpha}"
#             )
#         self.alpha = float(alpha)
#         self.soft_ce = LabelMixSoftTargetCrossEntropy()
#         self.pl_loss = LabelMixPlackettLuceLoss()

#     def forward(self, x: torch.Tensor, target) -> torch.Tensor:
#         # Always compute both sub-losses unconditionally so the autograd
#         # graph is identical across ranks (no data-dependent branches).
#         ce = self.soft_ce(x, target)
#         pl = self.pl_loss(x, target)
#         return self.alpha * ce + (1.0 - self.alpha) * pl

#     def extra_repr(self) -> str:
#         return f"alpha={self.alpha}"



class LabelMixMixupLoss(nn.Module):
    """Convex combination of LabelMix soft-CE and Plackett-Luce losses.

    Computed efficiently via the closed form:
    Term = (1 - alpha) * denom_log - log_p

    Parameters
    ----------
    alpha : float
        Mixing coefficient in [0, 1].
        Final loss = alpha * soft_ce + (1 - alpha) * pl_loss.
    eps : float
        Small constant to prevent log(0) numerical instability.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-12) -> None:
        super(LabelMixMixupLoss, self).__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"LabelMixMixupLoss alpha must be in [0, 1], got {alpha}"
            )
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, target) -> torch.Tensor:
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            raise TypeError("LabelMix target must be a (labels, weights) tuple.")
        
        labels, weights = target
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)

        batch_size, _ = x.shape
        if labels.shape[0] != batch_size or weights.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between logits, labels, and weights")

        idx = labels.to(device=x.device, dtype=torch.long)
        sample_weights = weights.to(device=x.device, dtype=x.dtype)
        k = idx.shape[1]

        if k == 0:
            return x.sum() * 0.0

        valid_positions = sample_weights > 0
        safe_idx = idx.masked_fill(~valid_positions, 0)
        logp = F.log_softmax(x, dim=1)
        ranked_logp = logp.gather(dim=1, index=safe_idx)
        ranked_p = ranked_logp.exp() * valid_positions.to(ranked_logp.dtype)
        unranked_mass = (1.0 - ranked_p.sum(dim=1, keepdim=True)).clamp_min(0.0)
        denom_log = torch.log(
            (unranked_mass + torch.cumsum(ranked_p, dim=1)).clamp_min(self.eps)
        )
        combined_per_position = (1.0 - self.alpha) * denom_log - ranked_logp
        per_sample_loss = (combined_per_position * sample_weights).sum(dim=1)
        valid_samples = valid_positions.any(dim=1)
        return per_sample_loss.sum() / valid_samples.sum().clamp_min(1)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, eps={self.eps}"
