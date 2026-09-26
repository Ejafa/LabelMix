import pandas as pd
import pytest

from evaluation.scripts.aggregate_per_experiment import (
    aggregate_by_type,
    is_long_horizon,
    sanitize_dataframe,
    sanitize_name,
)


@pytest.mark.parametrize("horizon", ["", "300__"])
def test_disabled_mixing_parameters_remain_single_image(horizon):
    name = (
        f"vit-wee__in1k__img256__{horizon}"
        "bare_cutmix=0_cutmix_minmax=None_mixup=0_mixup_prob=0.0__seed=42"
    )
    assert sanitize_name(name) == "bare"
    assert is_long_horizon(name) == bool(horizon)
    assert sanitize_name(name.replace("__in1k__", "__unbalanced__")) == "unbalanced-bare"


@pytest.mark.parametrize("recipe,expected", [
    ("bare", "bare"),
    ("cutmix_mixup_switch_prob=1.0", "cutmix"),
    ("mixup_mixup_switch_prob=0.0", "mixup"),
    ("baseline", "baseline"),
    ("openmixup-fmix", "fmix"),
])
def test_other_recipe_names_keep_their_identity(recipe, expected):
    assert sanitize_name(f"vit-wee__in1k__img256__{recipe}__seed=42") == expected


def test_single_image_and_cutmix_are_aggregated_separately():
    rows = []
    for seed, single_accuracy, cutmix_accuracy in [(42, 77., 75.), (43, 78., 76.), (44, 79., 77.)]:
        for recipe, accuracy in [
            ("bare_cutmix=0_cutmix_minmax=None_mixup=0_mixup_prob=0.0", single_accuracy),
            ("cutmix_mixup_switch_prob=1.0", cutmix_accuracy),
        ]:
            rows.append(dict(name=f"vit-wee__in1k__{recipe}__seed={seed}",
                             model="vit-wee", dataset="in1k", seed=seed, top1_acc=accuracy))
    clean = sanitize_dataframe(pd.DataFrame(rows), ["top1_acc"])
    assert not clean.duplicated(["type", "model", "dataset", "long_horizon", "seed"]).any()
    results = aggregate_by_type(clean, ["top1_acc"]).set_index("type")
    assert set(results.index) == {"bare", "cutmix"}
    assert results.loc["bare", "top1_acc_mean"] == 78.
    assert results.loc["cutmix", "top1_acc_mean"] == 76.
    assert (results["top1_acc_std"] == 1.).all()
    assert (results["n_seeds"] == 3).all()
