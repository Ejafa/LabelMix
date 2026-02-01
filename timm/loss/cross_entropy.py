""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class LabelMixSoftTargetCrossEntropy(nn.Module):
    """Soft CE for LabelMix targets: target = (labels[B,K], weights[B,K])."""

    def __init__(self) -> None:
        super(LabelMixSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target) -> torch.Tensor:
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            raise TypeError("LabelMix target must be a (labels, weights) tuple.")
        labels, weights = target
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)

        labels = labels.to(dtype=torch.long)
        logprobs = F.log_softmax(x, dim=-1)
        nll = -logprobs.gather(dim=-1, index=labels)
        weights = weights.to(dtype=nll.dtype)
        loss = (nll * weights).sum(dim=-1)
        return loss.mean()
