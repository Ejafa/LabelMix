"""Create the official long-tailed ImageNet Arrow dataset in RAM.

The adapter starts from a locally cached Hugging Face ImageNet-1K dataset,
selects the published ImageNet-LT training subset using the manifest bundled
with this repository, and saves it plus the unchanged validation split in the
``arrow/<split>`` layout consumed by :mod:`timm.data.readers.reader_hfds`.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


IMAGENET_HF_NAME = "ILSVRC/imagenet-1k"
METADATA_FILENAME = "imagenet_lt_metadata.json"
OFFICIAL_TRAIN_SAMPLES = 115_846
OFFICIAL_PARETO_POWER = 6
OFFICIAL_MANIFEST_URL = (
    "https://raw.githubusercontent.com/facebookresearch/classifier-balancing/"
    "main/data/ImageNet_LT/ImageNet_LT_train.txt"
)
OFFICIAL_MANIFEST_SHA256 = "efdbdad4f050237c310b2f354cf95a8b1d7c8d57a63c4ea4bb6bf2bcb012f37f"
BUNDLED_OFFICIAL_MANIFEST = (
    Path(__file__).resolve().parent
    / "data"
    / "ImageNet_LT"
    / "ImageNet_LT_train.txt"
)

_BUILDER_ARROW_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<split>train|validation)-"
    r"(?P<index>\d+)-of-(?P<total>\d+)\.arrow$"
)


def build_exponential_class_counts(
    num_classes: int = 1000,
    max_samples: int = 1280,
    min_samples: int = 5,
) -> list[int]:
    """Return monotonically decreasing exponential per-class counts."""
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    if min_samples < 1 or max_samples < min_samples:
        raise ValueError("Require 1 <= min_samples <= max_samples")
    if num_classes == 1:
        return [max_samples]

    ratio = min_samples / max_samples
    counts = [
        int(round(max_samples * ratio ** (rank / (num_classes - 1))))
        for rank in range(num_classes)
    ]
    counts[0] = max_samples
    counts[-1] = min_samples
    return [max(min_samples, min(max_samples, count)) for count in counts]


def select_long_tail_indices(
    labels: Sequence[int],
    requested_counts: Sequence[int],
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Select deterministic per-class indices and return them globally shuffled."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[int(label)].append(index)

    expected_labels = list(range(len(requested_counts)))
    missing = [label for label in expected_labels if label not in buckets]
    if missing:
        raise ValueError(f"Source training split is missing class labels: {missing[:20]}")

    selected: list[int] = []
    actual_counts: list[int] = []
    for label, requested in enumerate(requested_counts):
        available = buckets[label]
        if len(available) < requested:
            raise ValueError(
                f"Class {label} has {len(available)} source samples, "
                f"but the long-tail profile requests {requested}"
            )
        class_rng = random.Random(int(seed) + 104_729 * (label + 1))
        class_indices = list(available)
        class_rng.shuffle(class_indices)
        selected.extend(class_indices[:requested])
        actual_counts.append(requested)

    random.Random(int(seed) + 15_485_863).shuffle(selected)
    return selected, actual_counts


def assign_rank_counts_to_classes(
    labels: Sequence[int],
    rank_counts: Sequence[int],
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Assign descending counts to random classes that can satisfy them.

    ImageNet-1K source classes are not perfectly equal-sized. Assigning the
    largest requested counts greedily to randomly chosen eligible classes
    avoids coupling head/tail status to semantic label order while ensuring
    the requested profile is feasible.
    """
    availability = Counter(int(label) for label in labels)
    remaining = set(range(len(rank_counts)))
    rng = random.Random(int(seed) + 32_452_843)
    class_counts = [0] * len(rank_counts)
    class_rank_order: list[int] = []
    for requested in rank_counts:
        eligible = sorted(label for label in remaining if availability[label] >= requested)
        if not eligible:
            largest_remaining = max((availability[label] for label in remaining), default=0)
            raise ValueError(
                f"No remaining source class can satisfy requested count {requested}; "
                f"largest remaining class has {largest_remaining} samples"
            )
        label = eligible[rng.randrange(len(eligible))]
        remaining.remove(label)
        class_counts[label] = int(requested)
        class_rank_order.append(label)
    return class_counts, class_rank_order


def parse_imagenet_lt_manifest(path: str) -> tuple[list[str], list[int], list[int]]:
    """Parse the official ``ImageNet_LT_train.txt`` split manifest."""
    filenames: list[str] = []
    labels: list[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                filename, label_text = line.rsplit(maxsplit=1)
                label = int(label_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed ImageNet-LT manifest line {line_number}: {line!r}"
                ) from exc
            if label < 0 or label >= 1000:
                raise ValueError(
                    f"Manifest label out of range at line {line_number}: {label}"
                )
            filenames.append(filename)
            labels.append(label)

    counts_by_label = Counter(labels)
    missing = [label for label in range(1000) if counts_by_label[label] == 0]
    if missing:
        raise ValueError(f"ImageNet-LT manifest is missing classes: {missing[:20]}")
    class_counts = [counts_by_label[label] for label in range(1000)]
    return filenames, labels, class_counts


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_sha256(filenames: Sequence[str], labels: Sequence[int]) -> str:
    """Hash an ordered filename/label selection without platform dependence."""
    if len(filenames) != len(labels):
        raise ValueError("Selection filenames and labels must have equal length")
    digest = hashlib.sha256()
    for filename, label in zip(filenames, labels):
        normalized = str(filename).replace("\\", "/")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _download_official_manifest(cache_dir: str) -> str:
    manifest_dir = Path(cache_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    destination = manifest_dir / "ImageNet_LT_train.txt"
    if destination.is_file():
        if _sha256(str(destination)) == OFFICIAL_MANIFEST_SHA256:
            return str(destination)
        raise ValueError(
            f"Cached ImageNet-LT manifest has an unexpected hash: {destination}. "
            "Remove it and rerun to download a clean copy."
        )

    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    print(f"Downloading official ImageNet-LT manifest from {OFFICIAL_MANIFEST_URL}...")
    request = urllib.request.Request(
        OFFICIAL_MANIFEST_URL,
        headers={"User-Agent": "LabelMix-ImageNet-LT-adapter"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        observed_hash = _sha256(str(temporary))
        if observed_hash != OFFICIAL_MANIFEST_SHA256:
            raise ValueError(
                "Downloaded ImageNet-LT manifest failed SHA-256 verification: "
                f"{observed_hash} != {OFFICIAL_MANIFEST_SHA256}"
            )
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            "Could not download the official ImageNet-LT manifest. Check remote "
            "network access or pass --lt-train-list explicitly."
        ) from exc
    return str(destination)


def resolve_official_manifest(
    src: str,
    explicit_path: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Locate the published ImageNet-LT training manifest."""
    candidates = [
        explicit_path,
        os.environ.get("IMAGENET_LT_TRAIN_LIST"),
        os.path.join(src, "ImageNet_LT_train.txt"),
        os.path.join(src, "ImageNet_LT", "ImageNet_LT_train.txt"),
        str(BUNDLED_OFFICIAL_MANIFEST),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            resolved = os.path.abspath(candidate)
            if Path(resolved) == BUNDLED_OFFICIAL_MANIFEST:
                observed_hash = _sha256(resolved)
                if observed_hash != OFFICIAL_MANIFEST_SHA256:
                    raise ValueError(
                        "Bundled ImageNet-LT manifest failed SHA-256 verification: "
                        f"{observed_hash} != {OFFICIAL_MANIFEST_SHA256}"
                    )
            return resolved
    if cache_dir:
        return _download_official_manifest(cache_dir)
    raise FileNotFoundError(
        "Could not find the bundled official ImageNet_LT_train.txt manifest. "
        "Pass --lt-train-list, set IMAGENET_LT_TRAIN_LIST, or place the file at "
        f"{os.path.join(src, 'ImageNet_LT_train.txt')}. The manifest is "
        "distributed with the OLTR ImageNet-LT dataset files."
    )


def select_manifest_indices(dataset, filenames: Sequence[str], labels: Sequence[int]) -> list[int]:
    """Map official manifest filenames to rows in a Hugging Face split."""
    datasets = _import_datasets()
    encoded = dataset.cast_column("image", datasets.Image(decode=False))
    source_labels = dataset["label"]
    source_by_basename: dict[str, int] = {}
    for index, image in enumerate(encoded["image"]):
        path = image.get("path") if isinstance(image, dict) else None
        if not path:
            raise ValueError(
                "Source ImageNet rows do not expose image filenames, so they "
                "cannot be matched to the official ImageNet-LT manifest."
            )
        basename = os.path.basename(path)
        if basename in source_by_basename:
            raise ValueError(f"Duplicate source image basename: {basename}")
        source_by_basename[basename] = index

    selected: list[int] = []
    missing: list[str] = []
    for filename, expected_label in zip(filenames, labels):
        basename = os.path.basename(filename)
        index = source_by_basename.get(basename)
        if index is None:
            missing.append(filename)
            continue
        actual_label = int(source_labels[index])
        if actual_label != int(expected_label):
            raise ValueError(
                f"Manifest/source label mismatch for {filename}: "
                f"manifest={expected_label}, source={actual_label}"
            )
        selected.append(index)
    if missing:
        raise ValueError(
            f"Could not match {len(missing)} official manifest images to the "
            f"source dataset; examples: {missing[:10]}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("Official ImageNet-LT manifest contains duplicate source images")
    return selected


def _import_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "ImageNet-LT preparation requires Hugging Face datasets. "
            "Install the repository requirements first."
        ) from exc
    return datasets


def _find_builder_arrow_shards(src: str, split: str) -> list[Path]:
    """Find one complete HF builder-cache shard set in deterministic order."""
    root = Path(src)
    groups: dict[tuple[Path, str, int], dict[int, Path]] = defaultdict(dict)
    for path in sorted(root.rglob("*.arrow"), key=lambda value: str(value)):
        match = _BUILDER_ARROW_SHARD_RE.match(path.name)
        if match is None or match.group("split") != split:
            continue
        index = int(match.group("index"))
        total = int(match.group("total"))
        if total < 1 or index < 0 or index >= total:
            raise ValueError(f"Invalid Arrow shard name: {path}")
        key = (path.parent, match.group("prefix"), total)
        if index in groups[key]:
            raise ValueError(
                f"Duplicate Arrow shard index {index} for {split!r} under {path.parent}"
            )
        groups[key][index] = path

    complete: list[list[Path]] = []
    incomplete: list[str] = []
    for (parent, prefix, total), indexed in sorted(
        groups.items(),
        key=lambda item: (str(item[0][0]), item[0][1], item[0][2]),
    ):
        missing = sorted(set(range(total)) - set(indexed))
        if missing:
            preview = ", ".join(str(index) for index in missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            incomplete.append(
                f"{parent} ({prefix}, {len(indexed)}/{total}; missing {preview}{suffix})"
            )
            continue
        complete.append([indexed[index] for index in range(total)])

    if len(complete) > 1:
        locations = ", ".join(str(paths[0].parent) for paths in complete)
        raise ValueError(
            f"Found multiple complete ImageNet-1K {split!r} Arrow caches under "
            f"{src!r}: {locations}. Pass --src pointing at one fingerprint directory."
        )
    if complete:
        return complete[0]
    if incomplete:
        raise FileNotFoundError(
            f"Found ImageNet-1K {split!r} Arrow shards, but no complete shard set: "
            + "; ".join(incomplete)
        )
    return []


def _load_source_split(src: str, split: str):
    arrow_split = Path(src) / "arrow" / split
    if arrow_split.is_dir():
        datasets = _import_datasets()
        print(f"Loading local ImageNet-1K Arrow split: {arrow_split}")
        return datasets.load_from_disk(str(arrow_split))

    direct_split = Path(src) / split
    if (direct_split / "dataset_info.json").is_file():
        datasets = _import_datasets()
        print(f"Loading local ImageNet-1K Arrow split: {direct_split}")
        return datasets.load_from_disk(str(direct_split))

    builder_shards = _find_builder_arrow_shards(src, split)
    if builder_shards:
        datasets = _import_datasets()
        print(
            f"Loading local ImageNet-1K builder cache: split={split} "
            f"shards={len(builder_shards)} directory={builder_shards[0].parent}"
        )
        shard_datasets = [
            datasets.Dataset.from_file(str(path))
            for path in builder_shards
        ]
        if len(shard_datasets) == 1:
            return shard_datasets[0]
        return datasets.concatenate_datasets(shard_datasets)

    raise FileNotFoundError(
        f"Could not find the local ImageNet-1K {split!r} Arrow split under "
        f"{src!r}. Expected {arrow_split}, a saved split at {direct_split}, "
        "or a nested Hugging Face builder cache containing numbered Arrow "
        "shards. First prepare the regular RAM copy with "
        "`python copy_data_to_ram.py --dataset imagenet-1k`, set "
        "IMAGENET1K_DATA_DIR, or pass --src explicitly. The ImageNet-LT "
        "adapter does not download ImageNet."
    )


def _ensure_embedded_images(dataset, image_key: str = "image"):
    """Ensure saved Arrow rows contain image bytes instead of source paths."""
    datasets = _import_datasets()
    encoded = dataset.cast_column(image_key, datasets.Image(decode=False))
    if not len(encoded):
        return encoded
    probe = encoded[0][image_key]
    if isinstance(probe, dict) and probe.get("bytes"):
        return encoded

    print("Embedding source image bytes so the RAM dataset is self-contained...")

    def embed_batch(batch):
        embedded = []
        for image in batch[image_key]:
            if image.get("bytes"):
                embedded.append(image)
                continue
            path = image.get("path")
            if not path:
                raise ValueError("Image row contains neither bytes nor a path")
            with open(path, "rb") as handle:
                embedded.append({"bytes": handle.read(), "path": os.path.basename(path)})
        return {image_key: embedded}

    return encoded.map(
        embed_batch,
        batched=True,
        batch_size=64,
        load_from_cache_file=False,
        desc="Embedding images",
    )


def _replace_saved_split(dataset, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; use --force to replace it"
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        dataset.save_to_disk(str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_imagenet_lt(
    src: str,
    dst: str,
    splits: Sequence[str],
    seed: int = 42,
    max_samples: int = 1280,
    min_samples: int = 5,
    profile: str = "official",
    train_list: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create ImageNet-LT train/validation Arrow splits under ``dst``."""
    requested_splits = list(dict.fromkeys(splits))
    invalid = sorted(set(requested_splits) - {"train", "validation"})
    if invalid:
        raise ValueError(f"Unsupported ImageNet-LT splits: {invalid}")

    profile = str(profile).strip().lower()
    if profile not in {"official", "synthetic-exponential"}:
        raise ValueError(f"Unsupported ImageNet-LT profile: {profile}")
    manifest_filenames: list[str] | None = None
    manifest_labels: list[int] | None = None
    official_manifest: str | None = None
    if profile == "official":
        official_manifest = resolve_official_manifest(src, train_list, cache_dir=dst)
        manifest_filenames, manifest_labels, class_counts = parse_imagenet_lt_manifest(
            official_manifest
        )
        if len(manifest_filenames) != OFFICIAL_TRAIN_SAMPLES:
            raise ValueError(
                f"Official manifest has {len(manifest_filenames)} rows; expected "
                f"{OFFICIAL_TRAIN_SAMPLES}"
            )
        if max(class_counts) != 1280 or min(class_counts) != 5:
            raise ValueError(
                "Official manifest class-count endpoints do not match "
                f"ImageNet-LT (observed max={max(class_counts)}, "
                f"min={min(class_counts)}; expected max=1280, min=5)"
            )
        rank_counts = sorted(class_counts, reverse=True)
        max_samples = max(class_counts)
        min_samples = min(class_counts)
    else:
        rank_counts = build_exponential_class_counts(
            num_classes=1000,
            max_samples=max_samples,
            min_samples=min_samples,
        )
        class_counts = []
    metadata_path = Path(dst) / METADATA_FILENAME
    previous_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous_metadata = json.load(handle)

    metadata: dict[str, Any] = {
        "version": 2,
        "source": IMAGENET_HF_NAME,
        "source_cache": os.path.abspath(src),
        "profile": profile,
        "distribution": "pareto" if profile == "official" else "exponential",
        "pareto_power": OFFICIAL_PARETO_POWER if profile == "official" else None,
        "seed": int(seed),
        "deterministic_selection": True,
        "num_classes": 1000,
        "max_samples": int(max_samples),
        "min_samples": int(min_samples),
        "imbalance_factor": float(max_samples / min_samples),
        "rank_counts": rank_counts,
        "train_samples": int(sum(rank_counts)),
        "splits": sorted(set(previous_metadata.get("splits", [])) | set(requested_splits)),
    }
    if official_manifest is not None:
        metadata["manifest"] = official_manifest
        metadata["manifest_sha256"] = _sha256(official_manifest)
        metadata["selection_order"] = "official-manifest"
        metadata["selection_sha256"] = _selection_sha256(
            manifest_filenames or [],
            manifest_labels or [],
        )
        metadata["class_counts"] = class_counts
    same_previous_selection = (
        previous_metadata.get("profile") == profile
        and int(previous_metadata.get("seed", seed)) == int(seed)
        and int(previous_metadata.get("max_samples", max_samples)) == int(max_samples)
        and int(previous_metadata.get("min_samples", min_samples)) == int(min_samples)
    )
    if "train" not in requested_splits and same_previous_selection:
        for retained_key in (
            "class_counts",
            "class_rank_order",
            "selection_order",
            "selection_sha256",
        ):
            if retained_key in previous_metadata:
                metadata[retained_key] = previous_metadata[retained_key]
    if "validation" not in requested_splits and "validation_samples" in previous_metadata:
        metadata["validation_samples"] = previous_metadata["validation_samples"]

    print(
        "ImageNet-LT profile: "
        f"profile={profile} "
        f"classes=1000 train_samples={sum(rank_counts)} "
        f"head={rank_counts[0]} tail={rank_counts[-1]} "
        f"imbalance_factor={max_samples / min_samples:g} seed={seed}"
    )
    if dry_run:
        print(f"DRY RUN: would create {requested_splits} under {dst}/arrow")
        return metadata

    existing = [
        Path(dst) / "arrow" / split
        for split in requested_splits
        if (Path(dst) / "arrow" / split).exists()
    ]
    if existing and not force:
        raise FileExistsError(
            "ImageNet-LT destination split(s) already exist: "
            + ", ".join(str(path) for path in existing)
            + "; use --force to replace them"
        )

    arrow_root = Path(dst) / "arrow"
    if "train" in requested_splits:
        print("Loading source ImageNet-1K train split...")
        train = _load_source_split(src, "train")
        if "label" not in train.column_names:
            raise ValueError(f"Source train split has no 'label' column: {train.column_names}")
        if profile == "official":
            assert manifest_filenames is not None and manifest_labels is not None
            selected = select_manifest_indices(train, manifest_filenames, manifest_labels)
            actual_counts = class_counts
        else:
            class_counts, class_rank_order = assign_rank_counts_to_classes(
                train["label"], rank_counts, seed=seed
            )
            metadata["class_counts"] = class_counts
            metadata["class_rank_order"] = class_rank_order
            selected, actual_counts = select_long_tail_indices(
                train["label"], class_counts, seed=seed
            )
            source_labels = train["label"]
            selected_labels = [int(source_labels[index]) for index in selected]
            metadata["selection_order"] = "seeded-class-selection-and-global-shuffle"
            metadata["selection_sha256"] = _selection_sha256(
                [str(index) for index in selected],
                selected_labels,
            )
        subset = train.select(selected)
        observed = Counter(int(label) for label in subset["label"])
        if [observed[label] for label in range(1000)] != actual_counts:
            raise RuntimeError("ImageNet-LT class-count verification failed before save")
        subset = _ensure_embedded_images(subset)
        print(f"Saving {len(subset)} long-tail train samples to RAM...")
        _replace_saved_split(subset, arrow_root / "train", force=force)

    if "validation" in requested_splits:
        print("Loading unchanged ImageNet-1K validation split...")
        validation = _load_source_split(src, "validation")
        metadata["validation_samples"] = len(validation)
        validation = _ensure_embedded_images(validation)
        print(f"Saving {len(validation)} validation samples to RAM...")
        _replace_saved_split(validation, arrow_root / "validation", force=force)

    Path(dst).mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"ImageNet-LT ready at {dst}")
    print(f"Metadata: {metadata_path}")
    return metadata


def verify_imagenet_lt(dst: str) -> dict[str, Any]:
    """Verify saved split sizes and train class counts against metadata."""
    metadata_path = Path(dst) / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing ImageNet-LT metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    datasets = _import_datasets()
    train_path = Path(dst) / "arrow" / "train"
    if train_path.is_dir():
        train = datasets.load_from_disk(str(train_path))
        observed = Counter(int(label) for label in train["label"])
        expected = [int(value) for value in metadata["class_counts"]]
        if len(train) != sum(expected):
            raise RuntimeError(f"Train size mismatch: {len(train)} != {sum(expected)}")
        if [observed[label] for label in range(len(expected))] != expected:
            raise RuntimeError("Saved ImageNet-LT class counts do not match metadata")
        encoded_train = train.cast_column("image", datasets.Image(decode=False))
        if len(encoded_train) and not encoded_train[0]["image"].get("bytes"):
            raise RuntimeError("Saved train images still reference external filesystem paths")

    validation_path = Path(dst) / "arrow" / "validation"
    if validation_path.is_dir() and "validation_samples" in metadata:
        validation = datasets.load_from_disk(str(validation_path))
        if len(validation) != int(metadata["validation_samples"]):
            raise RuntimeError("Validation size does not match metadata")
        encoded_validation = validation.cast_column("image", datasets.Image(decode=False))
        if len(encoded_validation) and not encoded_validation[0]["image"].get("bytes"):
            raise RuntimeError("Saved validation images still reference external filesystem paths")

    print(
        f"ImageNet-LT verification passed: train={metadata['train_samples']} "
        f"classes={metadata['num_classes']} seed={metadata['seed']}"
    )
    return metadata


def print_imagenet_lt_status(dst: str) -> None:
    """Print adapter metadata and the presence of each saved Arrow split."""
    metadata_path = Path(dst) / METADATA_FILENAME
    print(f"ImageNet-LT destination: {dst}")
    print(f"Metadata: {'present' if metadata_path.is_file() else 'missing'}")
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        print(
            f"Profile: {metadata.get('distribution')} "
            f"train={metadata.get('train_samples')} "
            f"head={metadata.get('max_samples')} tail={metadata.get('min_samples')} "
            f"seed={metadata.get('seed')}"
        )
    for split in ("train", "validation"):
        path = Path(dst) / "arrow" / split
        print(f"{split}: {'present' if path.is_dir() else 'missing'} ({path})")
