""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
import torch.distributed as dist


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
        self.confidences.append(conf.detach().float().cpu())
        self.predictions.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())

    @torch.no_grad()
    def compute(self):
        if not self.confidences:
            return 0.0

        confidences = torch.cat(self.confidences).float()
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)

        accuracies = predictions.eq(targets).float()

        bin_indices = (confidences * self.n_bins).long().clamp(max=self.n_bins - 1)

        bin_counts = torch.bincount(bin_indices, minlength=self.n_bins).float()
        bin_conf_sum = torch.zeros(self.n_bins, dtype=torch.float32).scatter_add(0, bin_indices, confidences)
        bin_acc_sum = torch.zeros(self.n_bins, dtype=torch.float32).scatter_add(0, bin_indices, accuracies)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            backend = dist.get_backend()
            device = torch.device('cuda', torch.cuda.current_device()) if backend == 'nccl' else torch.device('cpu')
            bin_counts = bin_counts.to(device)
            bin_conf_sum = bin_conf_sum.to(device)
            bin_acc_sum = bin_acc_sum.to(device)

            dist.all_reduce(bin_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_conf_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_acc_sum, op=dist.ReduceOp.SUM)

        total_count = bin_counts.sum()
        if total_count == 0:
            return 0.0

        mask = bin_counts > 0
        if not mask.any():
            return 0.0

        avg_acc = bin_acc_sum[mask] / bin_counts[mask]
        avg_conf = bin_conf_sum[mask] / bin_counts[mask]
        prop_in_bin = bin_counts[mask] / total_count

        ece = torch.sum(torch.abs(avg_conf - avg_acc) * prop_in_bin)

        return (ece * 100.0).item()
