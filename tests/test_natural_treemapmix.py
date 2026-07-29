import torch
import torch.nn.functional as F
from PIL import Image
from unittest.mock import patch

from timm.data.balanced_dataset import BalancedBucketDataset
from timm.loss.cross_entropy import LabelMixSoftTargetCrossEntropy
from timm.loss.mixup_loss import LabelMixMixupLoss
from timm.loss.plackett_luce import labelmix_plackett_luce_loss
from train import LabelMixBroadcastLoader


class _LongTailDataset:
    def __init__(self, counts):
        self.targets = []
        for label, count in enumerate(counts):
            self.targets.extend([label] * count)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.new("RGB", (8, 8), color=int(index)), self.targets[index]


def _build_natural(counts=(5, 4, 3, 2), seed=42, epoch=0, k=3):
    dataset = BalancedBucketDataset(
        _LongTailDataset(counts),
        mode="natural",
        cache_path="",
        labelmix=True,
        labelmix_kwargs={
            "mix_k": k,
            "k_min": k,
            "k_max": k,
            "k_schedule": "fixed",
            "labelmix_k_cooldown_epochs": 0,
        },
        seed=seed,
    )
    dataset.set_epoch(epoch)
    levels = torch.arange(dataset.M, dtype=torch.int64)
    return dataset, dataset._build_natural_schedule(levels)


def _schedule_values(rows):
    return [row.tolist() for row in rows]


def _single_targets(targets, k=4):
    labels = torch.zeros(len(targets), k, dtype=torch.long)
    weights = torch.zeros(len(targets), k)
    labels[:, -1] = targets
    weights[:, -1] = 1.0
    return labels, weights


def test_natural_schedule_coverage_rows_and_statistics():
    counts = [5, 4, 3, 2]
    dataset, rows = _build_natural(counts=counts)

    assert [int(row.numel()) for row in rows] == [4, 4, 3, 2, 1]
    assert dataset.natural_stats["mixed_primary"] == 11
    assert dataset.natural_stats["single_primary"] == 3
    assert dataset.natural_stats["total_primary"] == sum(counts)

    all_indices = torch.cat(rows).tolist()
    assert len(all_indices) == sum(counts)
    assert len(set(all_indices)) == sum(counts)
    for row in rows:
        row_labels = [dataset.base_dataset.targets[index] for index in row.tolist()]
        assert len(row_labels) == len(set(row_labels))
        if len(row) >= dataset.mix_k and len(row) % dataset.mix_k:
            usable = (len(row) // dataset.mix_k) * dataset.mix_k
            remainder = len(row) - usable
            padded = torch.cat([row[usable:], row[: dataset.mix_k - remainder]])
            padded_labels = [dataset.base_dataset.targets[index] for index in padded.tolist()]
            assert len(padded_labels) == dataset.mix_k
            assert len(set(padded_labels)) == dataset.mix_k


def test_natural_schedule_reproducibility_and_epoch_variation():
    _, rows_a = _build_natural(seed=42, epoch=0)
    _, rows_b = _build_natural(seed=42, epoch=0)
    _, rows_c = _build_natural(seed=42, epoch=1)
    _, rows_d = _build_natural(seed=43, epoch=0)

    assert _schedule_values(rows_a) == _schedule_values(rows_b)
    assert _schedule_values(rows_a) != _schedule_values(rows_c)
    assert _schedule_values(rows_a) != _schedule_values(rows_d)


def test_natural_length_respects_centralized_override():
    counts = [5, 4, 3, 2]
    dataset, _ = _build_natural(counts=counts)
    dataset.dist_world_size_override = 1
    assert len(dataset) == sum(counts)


def test_k_above_class_count_uses_only_fixed_shape_single_targets():
    dataset, _ = _build_natural(k=6)
    dataset.buffer_size = 3
    with patch.object(
        torch.distributions.Dirichlet,
        "sample",
        side_effect=AssertionError("single rows must not draw Dirichlet weights"),
    ):
        outputs = list(dataset)

    assert len(outputs) == len(dataset.base_dataset)
    for _, (labels, weights) in outputs:
        assert labels.shape == (6,)
        assert weights.shape == (6,)
        assert weights[-1] == 1
        assert weights[:-1].sum() == 0


def test_mixed_and_single_rows_keep_exact_primary_output_count():
    dataset, _ = _build_natural(k=3)
    dataset.buffer_size = 4
    outputs = list(dataset)

    assert len(outputs) == len(dataset.base_dataset) == 14
    assert all(labels.shape == weights.shape == (3,) for _, (labels, weights) in outputs)
    single_outputs = sum(
        int(torch.equal(weights, torch.tensor([0.0, 0.0, 1.0])))
        for _, (_, weights) in outputs
    )
    assert single_outputs == 3


def test_single_target_losses_equal_cross_entropy():
    torch.manual_seed(7)
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    labels, weights = _single_targets(targets)
    expected = F.cross_entropy(logits, targets)

    losses = (
        labelmix_plackett_luce_loss(logits, labels, weights),
        LabelMixSoftTargetCrossEntropy()(logits, (labels, weights)),
        LabelMixMixupLoss(alpha=0.37)(logits, (labels, weights)),
    )
    for loss in losses:
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)


def test_zero_weight_dummy_entries_do_not_change_losses():
    torch.manual_seed(11)
    logits_real = torch.randn(5, 10)
    logits_dummy = torch.randn(3, 10)
    targets = torch.randint(0, 10, (5,))
    labels, weights = _single_targets(targets)
    padded_labels = torch.cat([labels, torch.zeros(3, 4, dtype=torch.long)])
    padded_weights = torch.cat([weights, torch.zeros(3, 4)])
    padded_logits = torch.cat([logits_real, logits_dummy])

    criteria = (
        lambda x, target: labelmix_plackett_luce_loss(x, *target),
        LabelMixSoftTargetCrossEntropy(),
        LabelMixMixupLoss(alpha=0.37),
    )
    for criterion in criteria:
        before = criterion(logits_real, (labels, weights))
        after = criterion(padded_logits, (padded_labels, padded_weights))
        torch.testing.assert_close(before, after)


def test_centralized_partial_batch_is_padded_with_zero_weight_targets():
    loader = LabelMixBroadcastLoader.__new__(LabelMixBroadcastLoader)
    loader.batch_size = 4
    loader.world_size = 2
    inputs = torch.randn(5, 3, 8, 8)
    targets = torch.randint(0, 10, (5,))
    labels, weights = _single_targets(targets)

    padded_inputs, padded_labels, padded_weights = loader._pad_global_batch(
        inputs, labels, weights
    )

    assert padded_inputs.shape[0] == 8
    assert padded_labels.shape == padded_weights.shape == (8, 4)
    torch.testing.assert_close(padded_inputs[5:], inputs[:3])
    assert torch.count_nonzero(padded_labels[5:]) == 0
    assert torch.count_nonzero(padded_weights[5:]) == 0
