from collections import Counter
from types import SimpleNamespace

import yaml

from copy_data_to_ram import resolve_imagenet_lt_source
from experiments.generate_jobs import DATASET_CONFIGS, compute_schedule, generate
import imagenet_lt_adapter
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


def test_imagenet_lt_source_prefers_existing_local_dataset(tmp_path, monkeypatch):
    local_source = tmp_path / "imagenet-1k-ram"
    local_source.mkdir()
    shared_source = tmp_path / "imagenet-1k-shared"
    config = {
        "src": str(shared_source),
        "local_src": str(local_source),
    }

    assert resolve_imagenet_lt_source(None, config) == str(local_source)

    environment_source = tmp_path / "imagenet-1k-environment"
    monkeypatch.setenv("IMAGENET1K_DATA_DIR", str(environment_source))
    assert resolve_imagenet_lt_source(None, config) == str(environment_source)

    explicit_source = tmp_path / "imagenet-1k-explicit"
    assert resolve_imagenet_lt_source(str(explicit_source), config) == str(explicit_source)


def test_imagenet_lt_loads_local_arrow_split_without_hf_download(tmp_path, monkeypatch):
    arrow_split = tmp_path / "arrow" / "train"
    arrow_split.mkdir(parents=True)
    expected_dataset = object()
    calls = []
    fake_datasets = SimpleNamespace(
        load_from_disk=lambda path: calls.append(path) or expected_dataset,
    )
    monkeypatch.setattr(imagenet_lt_adapter, "_import_datasets", lambda: fake_datasets)

    observed = imagenet_lt_adapter._load_source_split(str(tmp_path), "train")

    assert observed is expected_dataset
    assert calls == [str(arrow_split)]


def test_imagenet_lt_loads_nested_builder_shards_in_numeric_order(
    tmp_path,
    monkeypatch,
):
    cache_dir = (
        tmp_path
        / "ilsvrc___imagenet-1k"
        / "default"
        / "0.0.0"
        / "fingerprint"
    )
    cache_dir.mkdir(parents=True)
    for index in (2, 0, 1):
        (cache_dir / f"imagenet-1k-train-{index:05d}-of-00003.arrow").touch()

    loaded = []

    class FakeDataset:
        @staticmethod
        def from_file(path):
            loaded.append(path)
            return path

    fake_datasets = SimpleNamespace(
        Dataset=FakeDataset,
        concatenate_datasets=lambda shards: tuple(shards),
    )
    monkeypatch.setattr(imagenet_lt_adapter, "_import_datasets", lambda: fake_datasets)

    observed = imagenet_lt_adapter._load_source_split(str(tmp_path), "train")

    expected = [
        str(cache_dir / f"imagenet-1k-train-{index:05d}-of-00003.arrow")
        for index in range(3)
    ]
    assert loaded == expected
    assert observed == tuple(expected)


def test_imagenet_lt_rejects_incomplete_builder_shard_set(tmp_path):
    cache_dir = tmp_path / "nested" / "fingerprint"
    cache_dir.mkdir(parents=True)
    for index in (0, 2):
        (cache_dir / f"imagenet-1k-validation-{index:05d}-of-00003.arrow").touch()

    try:
        imagenet_lt_adapter._load_source_split(str(tmp_path), "validation")
    except FileNotFoundError as exc:
        message = str(exc)
        assert "no complete shard set" in message
        assert "missing 1" in message
    else:
        raise AssertionError("incomplete Arrow cache should fail")


def test_manifest_count_fallback_is_deterministic_when_source_names_are_replaced(
    monkeypatch,
):
    source_labels = [label for label in range(1000) for _ in range(3)]
    source_images = [
        {"bytes": b"image", "path": f"generated-row-{index}"}
        for index in range(len(source_labels))
    ]

    class FakeSourceDataset:
        def cast_column(self, _name, _feature):
            return self

        def __getitem__(self, key):
            if key == "label":
                return source_labels
            if key == "image":
                return source_images
            raise KeyError(key)

    fake_datasets = SimpleNamespace(Image=lambda decode: ("image", decode))
    monkeypatch.setattr(imagenet_lt_adapter, "_import_datasets", lambda: fake_datasets)

    manifest_labels = [0, 0] + list(range(1, 1000))
    manifest_filenames = [
        f"train/class-{label}/official-{index}.JPEG"
        for index, label in enumerate(manifest_labels)
    ]
    source = FakeSourceDataset()

    selected_a, strategy_a = imagenet_lt_adapter.resolve_manifest_indices(
        source,
        manifest_filenames,
        manifest_labels,
        seed=42,
    )
    selected_b, strategy_b = imagenet_lt_adapter.resolve_manifest_indices(
        source,
        manifest_filenames,
        manifest_labels,
        seed=42,
    )

    assert strategy_a == strategy_b == "deterministic-manifest-counts"
    assert selected_a == selected_b
    assert len(selected_a) == len(set(selected_a)) == len(manifest_labels)
    observed = Counter(source_labels[index] for index in selected_a)
    assert observed[0] == 2
    assert all(observed[label] == 1 for label in range(1, 1000))


def test_permanent_selection_cache_replays_and_validates_indices(tmp_path):
    cache_path = tmp_path / "shared" / "selection.json"
    source_labels = [0, 0, 1, 1]
    selected = [1, 2]
    expected_counts = [1, 1]

    imagenet_lt_adapter._write_selection_cache(
        cache_path,
        profile="official",
        seed=42,
        manifest_sha256="manifest-hash",
        source_rows=4,
        source_fingerprint="source-hash",
        source_label_sha256=imagenet_lt_adapter._labels_sha256(source_labels),
        class_counts=expected_counts,
        selected=selected,
        selection_strategy="deterministic-manifest-counts",
        selected_labels=[source_labels[index] for index in selected],
    )

    loaded = imagenet_lt_adapter._load_selection_cache(
        cache_path,
        profile="official",
        seed=42,
        manifest_sha256="manifest-hash",
        source_rows=4,
        source_fingerprint="source-hash",
        source_label_sha256=imagenet_lt_adapter._labels_sha256(source_labels),
        expected_counts=expected_counts,
        source_labels=source_labels,
    )

    assert loaded == (selected, "deterministic-manifest-counts")

    try:
        imagenet_lt_adapter._load_selection_cache(
            cache_path,
            profile="official",
            seed=42,
            manifest_sha256="manifest-hash",
            source_rows=4,
            source_fingerprint="different-source",
            source_label_sha256=imagenet_lt_adapter._labels_sha256(source_labels),
            expected_counts=expected_counts,
            source_labels=source_labels,
        )
    except ValueError as exc:
        assert "source_fingerprint" in str(exc)
    else:
        raise AssertionError("a stale selection cache must be rejected")


def test_imagenet_lt_missing_local_arrow_split_fails_without_importing_hf(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        imagenet_lt_adapter,
        "_import_datasets",
        lambda: (_ for _ in ()).throw(AssertionError("must not import datasets")),
    )

    try:
        imagenet_lt_adapter._load_source_split(str(tmp_path), "train")
    except FileNotFoundError as exc:
        assert "does not download ImageNet" in str(exc)
        assert "--dataset imagenet-1k" in str(exc)
    else:
        raise AssertionError("missing local source should fail")


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
        gpus_per_job=2,
        output_path=str(output),
        output_root="output_runs/imagenet-lt-test",
        model_filter=models,
        baseline=True,
        mosaic=True,
        labelmix_variants=["pl", "sce"],
        seeds=[42, 43],
    )

    with output.open() as handle:
        generated = yaml.safe_load(handle)
        jobs = generated["jobs"]
    assert len(jobs) == 32
    assert generated["defaults"]["gpus"] == 2

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
            assert "--labelmix-producer-rank 0" in command
        elif "--labelmix-loss soft_ce" in command:
            variants["labelmix-sce"] += 1
            assert "--balanced-mode natural" in command
            assert "--labelmix-producer-rank 0" in command
        elif "--mosaic" in command:
            variants["mosaic"] += 1
            assert "--balanced-mode natural" in command
            assert " --labelmix " not in f" {command} "
            assert "--labelmix-producer-rank 0" in command
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
