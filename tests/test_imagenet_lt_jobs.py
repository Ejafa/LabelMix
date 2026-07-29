from collections import Counter

import yaml

from experiments.generate_jobs import DATASET_CONFIGS, compute_schedule, generate
from imagenet_lt_adapter import (
    BUNDLED_OFFICIAL_MANIFEST,
    OFFICIAL_TRAIN_SAMPLES,
    assign_rank_counts_to_classes,
    build_exponential_class_counts,
    parse_imagenet_lt_manifest,
    resolve_official_manifest,
    select_long_tail_indices,
)
from train import build_args, validate_args


def test_noncanonical_exponential_profile_and_selection_are_reproducible():
    counts = build_exponential_class_counts()
    assert len(counts) == 1000
    assert counts[0] == 1280
    assert counts[-1] == 5
    assert sum(counts) == 230_341
    assert all(left >= right for left, right in zip(counts, counts[1:]))

    labels = [label for label, count in enumerate([8, 7, 6, 5]) for _ in range(count)]
    class_counts, rank_order = assign_rank_counts_to_classes(
        labels, [8, 6, 4, 2], seed=42
    )
    selected_a, actual = select_long_tail_indices(labels, class_counts, seed=42)
    selected_b, _ = select_long_tail_indices(labels, class_counts, seed=42)

    assert selected_a == selected_b
    assert len(selected_a) == len(set(selected_a)) == sum(actual)
    observed = Counter(labels[index] for index in selected_a)
    assert [observed[label] for label in range(4)] == class_counts
    assert sorted(rank_order) == [0, 1, 2, 3]


def test_official_manifest_parser_preserves_image_identities_and_counts(tmp_path):
    manifest = tmp_path / "ImageNet_LT_train.txt"
    manifest.write_text(
        "".join(f"train/class{label}/image{label}.JPEG {label}\n" for label in range(1000))
    )
    filenames, labels, counts = parse_imagenet_lt_manifest(str(manifest))
    assert len(filenames) == len(labels) == 1000
    assert counts == [1] * 1000
    assert filenames[42].endswith("image42.JPEG")


def test_bundled_official_manifest_is_the_default():
    resolved = resolve_official_manifest("/path/that/does/not/exist")
    assert resolved == str(BUNDLED_OFFICIAL_MANIFEST)

    filenames, labels, counts = parse_imagenet_lt_manifest(resolved)
    assert len(filenames) == len(labels) == OFFICIAL_TRAIN_SAMPLES
    assert max(counts) == 1280
    assert min(counts) == 5


def test_imagenet_lt_schedule_uses_exact_natural_train_size():
    config = DATASET_CONFIGS["imagenet-lt"]
    num_steps, warmup_steps, steps_per_epoch = compute_schedule(
        config, epochs=100, warmup_epochs=10, batch_size=1024
    )
    assert config.train_samples == OFFICIAL_TRAIN_SAMPLES == 115_846
    assert steps_per_epoch == 114
    assert num_steps == 11_314
    assert warmup_steps == 1_132


def test_imagenet_lt_requested_matrix_generates_32_jobs(tmp_path):
    output = tmp_path / "imagenet-lt.yaml"
    models = ["vit-wee", "vit-little", "vit-medium", "vit-betwixt"]
    generate(
        dataset="imagenet-lt",
        gpus_per_job=1,
        output_path=str(output),
        output_root="output_runs/imagenet-lt-test",
        model_filter=models,
        baseline=True,
        mosaic=True,
        labelmix_variants=["pl", "sce"],
        seeds=[42, 43],
    )

    with output.open() as handle:
        jobs = yaml.safe_load(handle)["jobs"]
    assert len(jobs) == 32

    model_counts = Counter(job["name"].split("__", 1)[0] for job in jobs)
    assert model_counts == {model: 8 for model in models}
    assert sum("--seed 42" in job["cmd"] for job in jobs) == 16
    assert sum("--seed 43" in job["cmd"] for job in jobs) == 16

    variants = Counter()
    for job in jobs:
        command = job["cmd"]
        assert f"--data-dir {DATASET_CONFIGS['imagenet-lt'].data_dir}" in command
        if "--labelmix-loss pl_loss" in command:
            variants["labelmix-pl"] += 1
            assert "--balanced-mode natural" in command
        elif "--labelmix-loss soft_ce" in command:
            variants["labelmix-sce"] += 1
            assert "--balanced-mode natural" in command
        elif "--mosaic" in command:
            variants["mosaic"] += 1
            assert "--balanced-mode natural" in command
            assert "--labelmix" not in command
        else:
            variants["baseline"] += 1
            assert "--balanced-mode" not in command
            assert "--labelmix" not in command
    assert variants == {
        "baseline": 8,
        "labelmix-pl": 8,
        "labelmix-sce": 8,
        "mosaic": 8,
    }


def test_single_process_natural_mosaic_configuration_is_valid():
    args, _ = build_args(
        cli_overrides=[
            "--balanced-mode", "natural",
            "--mosaic",
            "--mixup", "0",
            "--cutmix", "0",
            "--mixup-prob", "0",
        ]
    )
    validate_args(args)
