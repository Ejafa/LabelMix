""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class ECEMeter:
    """Computes Expected Calibration Error (ECE) using pure PyTorch."""
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        self.confidences = []
        self.predictions = []
        self.targets = []

    @torch.no_grad()
    def update(self, output, target):
        """Accumulate logits and targets for ECE computation.

        Args:
            output (torch.Tensor): Logits (batch_size, num_classes)
            target (torch.Tensor): Targets (batch_size) or soft targets
        """
        if target.ndim > 1:
            target = target.argmax(dim=1)

        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

        # Store on CPU to avoid GPU OOM, keep as torch tensors
        self.confidences.append(conf.detach().cpu())
        self.predictions.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())

    @torch.no_grad()
    def compute(self):
        if not self.confidences:
            return 0.0

        confidences = torch.cat(self.confidences)
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)

        accuracies = predictions.eq(targets)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        ece = torch.tensor(0.0)

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            if i == 0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (avg_confidence_in_bin - accuracy_in_bin).abs() * prop_in_bin

        return (ece * 100.0).item()
