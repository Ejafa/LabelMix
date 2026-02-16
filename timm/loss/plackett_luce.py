"""Plackett-Luce style losses for ranked LabelMix targets."""

import torch
import torch.nn as nn


def labelmix_plackett_luce_loss(
        logits: torch.Tensor,   # [B, C]
        labels: torch.Tensor,   # [B, K] ASC rank order: worst -> best
        weights: torch.Tensor,  # [B, K] ASC rank order: worst -> best
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, C], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be [B, K], got {tuple(labels.shape)}")
    if weights.ndim != 2:
        raise ValueError(f"weights must be [B, K], got {tuple(weights.shape)}")

    batch_size, num_classes = logits.shape
    if labels.shape[0] != batch_size or weights.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch: logits B={batch_size}, labels B={labels.shape[0]}, weights B={weights.shape[0]}"
        )
    if labels.shape[1] != weights.shape[1]:
        raise ValueError(f"K mismatch: labels K={labels.shape[1]} vs weights K={weights.shape[1]}")

    idx = labels.to(dtype=torch.long)
    k = idx.shape[1]
    if k == 0:
        # No ranked terms: return differentiable zero.
        return logits.sum() * 0.0

    ranked_logits = logits.gather(dim=1, index=idx)

    unranked_mask = torch.ones((batch_size, num_classes), dtype=torch.bool, device=logits.device)
    unranked_mask.scatter_(1, idx, False)

    neg_inf = torch.finfo(logits.dtype).min
    unranked_logits = logits.masked_fill(~unranked_mask, neg_inf)
    unranked_mass = torch.logsumexp(unranked_logits, dim=1)

    ranked_prefix_lse = torch.logcumsumexp(ranked_logits, dim=1)
    denom = torch.logaddexp(unranked_mass.unsqueeze(1), ranked_prefix_lse)

    per_position_loss = denom - ranked_logits
    per_position_loss = torch.nan_to_num(per_position_loss, nan=0.0, posinf=0.0, neginf=0.0)

    sample_weights = weights.to(dtype=per_position_loss.dtype)
    per_sample_loss = (per_position_loss * sample_weights).sum(dim=1)
    return per_sample_loss.mean()


class LabelMixPlackettLuceLoss(nn.Module):
    """LabelMix top-K Plackett-Luce/ListMLE loss (ASC rank order: worst -> best)."""

    def __init__(self) -> None:
        super(LabelMixPlackettLuceLoss, self).__init__()

    def forward(self, x: torch.Tensor, target) -> torch.Tensor:
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            raise TypeError("LabelMix target must be a (labels, weights) tuple.")
        labels, weights = target
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        return labelmix_plackett_luce_loss(x, labels, weights)
