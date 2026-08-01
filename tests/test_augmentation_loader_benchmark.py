import pytest

from experiments.benchmark_augmentation_loader import CONFIGURATIONS, percentile


def test_required_augmentation_benchmark_configurations():
    assert list(CONFIGURATIONS) == [
        "no_augmentation",
        "single_image",
        "mixup_cutmix",
        "treemapmix_sce",
        "treemapmix_pl",
    ]
    assert CONFIGURATIONS["treemapmix_sce"]["mix_k"] == 4
    assert CONFIGURATIONS["treemapmix_sce"]["loss"] == "soft_ce"
    assert CONFIGURATIONS["treemapmix_pl"]["mix_k"] == 6
    assert CONFIGURATIONS["treemapmix_pl"]["loss"] == "pl_loss"


def test_percentile_interpolates_and_validates_input():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0], 0.95) == pytest.approx(1.95)
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.1)
