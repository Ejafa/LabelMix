"""Create the official long-tailed ImageNet Arrow dataset in RAM.

The adapter starts from a locally cached Hugging Face ImageNet-1K dataset. It
selects the published ImageNet-LT identities when original JPEG names remain
available; builder caches that replaced those names use a deterministic subset
with the manifest's exact per-class counts. The selected training data and the
unchanged validation split are saved in the ``arrow/<split>`` layout consumed
by :mod:`timm.data.readers.reader_hfds`.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
import re
import shutil
import time
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
DEFAULT_SELECTION_CACHE_DIR = Path(
    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/ImageNet_LT"
)
SELECTION_CACHE_VERSION = 2

_BUILDER_ARROW_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<split>train|validation)-"
    r"(?P<index>\d+)-of-(?P<total>\d+)\.arrow$"
)


def _log(message: str) -> None:
    """Print a timestamped ImageNet-LT diagnostic message."""
    print(f"[ImageNet-LT {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _fmt_bytes(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def _path_size_and_files(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    total_size = 0
    total_files = 0
    for child in path.rglob("*"):
        if child.is_file():
            total_files += 1
            total_size += child.stat().st_size
    return total_size, total_files


def _log_path_status(label: str, path: Path) -> None:
    exists = path.exists()
    kind = "directory" if path.is_dir() else "file" if path.is_file() else "missing"
    size, files = _path_size_and_files(path)
    _log(
        f"{label}: path={path} exists={exists} kind={kind} "
        f"files={files} size={_fmt_bytes(size)}"
    )


def _log_dataset_summary(label: str, dataset) -> None:
    fingerprint = str(
        getattr(
            dataset,
            "_labelmix_source_fingerprint",
            getattr(dataset, "_fingerprint", ""),
        )
    )
    _log(
        f"{label}: rows={len(dataset)} columns={list(dataset.column_names)} "
        f"fingerprint={fingerprint}"
    )
    try:
        features = getattr(dataset, "features", None)
        if features is not None:
            feature_summary = {
                name: feature.__class__.__name__
                for name, feature in features.items()
            }
            _log(f"{label}: feature_types={feature_summary}")
    except Exception as exc:
        _log(f"{label}: could not print features: {exc}")


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


def _labels_sha256(labels: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for label in labels:
        digest.update(int(label).to_bytes(2, byteorder="big", signed=False))
    return digest.hexdigest()


def _default_selection_cache_path(profile: str, seed: int) -> Path:
    return DEFAULT_SELECTION_CACHE_DIR / (
        f"ImageNet_LT_selection_{profile}_seed{int(seed)}.json"
    )


def _shuffle_selection_order(selected: Sequence[int], seed: int) -> list[int]:
    """Return the selected source indices in a deterministic global order."""
    shuffled = [int(index) for index in selected]
    random.Random(int(seed) + 15_485_863).shuffle(shuffled)
    return shuffled


def _resolve_selection_cache_path(
    profile: str,
    seed: int,
    explicit_path: str | None,
) -> Path:
    configured = explicit_path or os.environ.get("IMAGENET_LT_SELECTION_CACHE")
    return Path(configured).expanduser().resolve() if configured else _default_selection_cache_path(profile, seed)


def _load_selection_cache(
    path: Path,
    *,
    profile: str,
    seed: int,
    manifest_sha256: str | None,
    source_rows: int,
    source_fingerprint: str,
    source_label_sha256: str,
    expected_counts: Sequence[int],
    source_labels: Sequence[int],
) -> tuple[list[int], str] | None:
    if not path.is_file():
        _log(f"Selection cache not found: {path}")
        return None
    _log(f"Loading ImageNet-LT selection cache: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)
    _log("Selection cache JSON loaded; validating metadata fields")

    expected_fields = {
        "version": SELECTION_CACHE_VERSION,
        "profile": profile,
        "seed": int(seed),
        "manifest_sha256": manifest_sha256,
        "source_rows": int(source_rows),
        "source_fingerprint": source_fingerprint,
        "source_label_sha256": source_label_sha256,
        "class_counts": [int(value) for value in expected_counts],
    }
    mismatches = [
        key for key, expected in expected_fields.items()
        if cache.get(key) != expected
    ]
    if mismatches:
        _log(
            f"Selection cache metadata mismatch; ignoring stale cache {path} "
            f"(mismatched: {', '.join(mismatches)})"
        )
        return None
    _log("Selection cache metadata fields validated")

    selected = [int(index) for index in cache.get("selected_indices", [])]
    _log(f"Selection cache contains {len(selected)} selected indices")
    if len(selected) != sum(expected_counts) or len(selected) != len(set(selected)):
        raise ValueError(f"Invalid ImageNet-LT selection indices in cache: {path}")
    if selected and (min(selected) < 0 or max(selected) >= source_rows):
        raise ValueError(f"Out-of-range ImageNet-LT selection index in cache: {path}")
    _log("Selection cache index bounds validated; checking per-class counts")
    observed = Counter()
    selected_labels: list[int] = []
    last_log = time.time()
    for offset, index in enumerate(selected, start=1):
        selected_label = int(source_labels[index])
        selected_labels.append(selected_label)
        observed[selected_label] += 1
        now = time.time()
        if offset == 1 or offset == len(selected) or now - last_log >= 5:
            _log(
                f"Selection cache label validation progress: "
                f"{offset}/{len(selected)} indices checked"
            )
            last_log = now
    if [observed[label] for label in range(len(expected_counts))] != list(expected_counts):
        raise ValueError(f"ImageNet-LT cache class counts do not match: {path}")
    _log("Selection cache per-class counts validated; checking selection hash")
    observed_selection_sha256 = _selection_sha256(
        [str(index) for index in selected],
        selected_labels,
    )
    if cache.get("selection_sha256") != observed_selection_sha256:
        raise ValueError(f"ImageNet-LT cache selection hash does not match: {path}")

    strategy = str(cache.get("selection_strategy", "cached-selection"))
    _log(f"Selection cache fully validated: strategy={strategy}")
    return selected, strategy


def _write_selection_cache(
    path: Path,
    *,
    profile: str,
    seed: int,
    manifest_sha256: str | None,
    source_rows: int,
    source_fingerprint: str,
    source_label_sha256: str,
    class_counts: Sequence[int],
    selected: Sequence[int],
    selection_strategy: str,
    selected_labels: Sequence[int],
) -> None:
    cache = {
        "version": SELECTION_CACHE_VERSION,
        "profile": profile,
        "seed": int(seed),
        "manifest_sha256": manifest_sha256,
        "source_rows": int(source_rows),
        "source_fingerprint": source_fingerprint,
        "source_label_sha256": source_label_sha256,
        "class_counts": [int(value) for value in class_counts],
        "selection_strategy": selection_strategy,
        "selection_sha256": _selection_sha256(
            [str(index) for index in selected],
            selected_labels,
        ),
        "selected_indices": [int(index) for index in selected],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def resolve_manifest_indices(
    dataset,
    filenames: Sequence[str],
    labels: Sequence[int],
    seed: int = 42,
) -> tuple[list[int], str]:
    """Resolve the manifest exactly, or deterministically match its counts.

    Some Hugging Face ImageNet builder caches retain image bytes and labels but
    replace the original JPEG path. Exact manifest identity matching is then
    impossible. In that case, select a seeded subset with the manifest's exact
    per-class counts. This preserves the published long-tail distribution while
    recording that the image identities are a deterministic local equivalent.
    """
    datasets = _import_datasets()
    encoded = dataset.cast_column("image", datasets.Image(decode=False))
    _log("Materializing source labels for manifest resolution")
    source_labels = [int(label) for label in dataset["label"]]
    _log(f"Materialized {len(source_labels)} source labels for manifest resolution")
    desired_basenames = {os.path.basename(filename) for filename in filenames}
    source_by_basename: dict[str, int] = {}
    for index, image in enumerate(encoded["image"]):
        path = image.get("path") if isinstance(image, dict) else None
        if not path:
            continue
        basename = os.path.basename(path)
        if basename not in desired_basenames:
            continue
        if basename in source_by_basename:
            raise ValueError(f"Duplicate source image basename: {basename}")
        source_by_basename[basename] = index

    if not source_by_basename:
        manifest_counts = Counter(int(label) for label in labels)
        requested_counts = [manifest_counts[label] for label in range(1000)]
        selected, _ = select_long_tail_indices(
            source_labels,
            requested_counts,
            seed=seed,
        )
        return selected, "deterministic-manifest-counts"

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
    return _shuffle_selection_order(selected, seed), "exact-official-manifest-identities"


def select_manifest_indices(
    dataset,
    filenames: Sequence[str],
    labels: Sequence[int],
    seed: int = 42,
) -> list[int]:
    """Return deterministic source indices for the official manifest."""
    selected, _ = resolve_manifest_indices(dataset, filenames, labels, seed=seed)
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
    _log(f"Scanning for ImageNet-1K builder-cache Arrow shards: root={root} split={split}")
    groups: dict[tuple[Path, str, int], dict[int, Path]] = defaultdict(dict)
    scanned = 0
    matched = 0
    last_log = time.time()
    for path in root.rglob("*.arrow"):
        scanned += 1
        now = time.time()
        if scanned == 1 or now - last_log >= 5:
            _log(
                f"Shard scan progress for split={split}: "
                f"scanned_arrow_files={scanned} matched_split_shards={matched}"
            )
            last_log = now
        match = _BUILDER_ARROW_SHARD_RE.match(path.name)
        if match is None or match.group("split") != split:
            continue
        matched += 1
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
    _log(
        f"Shard scan complete for split={split}: "
        f"scanned_arrow_files={scanned} matched_split_shards={matched} groups={len(groups)}"
    )

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
        _log(
            f"Found complete builder-cache shard set for split={split}: "
            f"shards={len(complete[0])} directory={complete[0][0].parent}"
        )
        return complete[0]
    if incomplete:
        raise FileNotFoundError(
            f"Found ImageNet-1K {split!r} Arrow shards, but no complete shard set: "
            + "; ".join(incomplete)
        )
    _log(f"No builder-cache Arrow shards found for split={split} under {src}")
    return []


def _arrow_shard_fingerprint(paths: Sequence[Path]) -> str:
    """Fingerprint ordered shards without depending on their absolute paths."""
    digest = hashlib.sha256()
    sample_bytes = 4096
    for path in paths:
        size = path.stat().st_size
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            digest.update(handle.read(sample_bytes))
            if size > sample_bytes:
                handle.seek(-sample_bytes, os.SEEK_END)
                digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _set_source_fingerprint(dataset, fingerprint: str):
    try:
        dataset._labelmix_source_fingerprint = str(fingerprint)
    except Exception:
        pass
    return dataset


def _load_source_split(src: str, split: str):
    _log(f"Resolving source split: src={src} split={split}")
    _log_path_status("Source root", Path(src))
    arrow_split = Path(src) / "arrow" / split
    if arrow_split.is_dir():
        datasets = _import_datasets()
        _log_path_status(f"Saved Arrow split candidate ({split})", arrow_split)
        _log(f"Loading local ImageNet-1K Arrow split with load_from_disk: {arrow_split}")
        dataset = datasets.load_from_disk(str(arrow_split))
        dataset = _set_source_fingerprint(
            dataset,
            str(getattr(dataset, "_fingerprint", "")),
        )
        _log_dataset_summary(f"Loaded saved Arrow split ({split})", dataset)
        return dataset

    direct_split = Path(src) / split
    if (direct_split / "dataset_info.json").is_file():
        datasets = _import_datasets()
        _log_path_status(f"Direct saved split candidate ({split})", direct_split)
        _log(f"Loading local ImageNet-1K Arrow split with load_from_disk: {direct_split}")
        dataset = datasets.load_from_disk(str(direct_split))
        dataset = _set_source_fingerprint(
            dataset,
            str(getattr(dataset, "_fingerprint", "")),
        )
        _log_dataset_summary(f"Loaded direct saved split ({split})", dataset)
        return dataset

    _log(f"No saved split directory found for {split}; scanning nested builder-cache Arrow shards...")
    builder_shards = _find_builder_arrow_shards(src, split)
    if builder_shards:
        datasets = _import_datasets()
        total_bytes = sum(path.stat().st_size for path in builder_shards)
        _log(
            f"Loading local ImageNet-1K builder cache: split={split} "
            f"shards={len(builder_shards)} size={_fmt_bytes(total_bytes)} "
            f"directory={builder_shards[0].parent}"
        )
        _log(f"First shard: {builder_shards[0]}")
        _log(f"Last shard: {builder_shards[-1]}")
        shard_datasets = [
            datasets.Dataset.from_file(str(path))
            for path in builder_shards
        ]
        if len(shard_datasets) == 1:
            dataset = shard_datasets[0]
        else:
            dataset = datasets.concatenate_datasets(shard_datasets)
        dataset = _set_source_fingerprint(
            dataset,
            _arrow_shard_fingerprint(builder_shards),
        )
        _log_dataset_summary(f"Loaded builder-cache split ({split})", dataset)
        return dataset

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
    _log_dataset_summary("Image embedding input", encoded)
    if not len(encoded):
        _log("Image embedding skipped because dataset is empty")
        return encoded
    probe = encoded[0][image_key]
    if isinstance(probe, dict) and probe.get("bytes"):
        _log("Image embedding skipped because rows already contain image bytes")
        return encoded

    _log("Embedding source image bytes so the RAM dataset is self-contained...")

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

    embedded = encoded.map(
        embed_batch,
        batched=True,
        batch_size=64,
        load_from_cache_file=False,
        desc="Embedding images",
    )
    _log_dataset_summary("Image embedding output", embedded)
    return embedded


def _replace_saved_split(dataset, destination: Path, force: bool) -> None:
    _log(f"Preparing to save split: destination={destination} rows={len(dataset)} force={force}")
    if destination.exists():
        _log_path_status("Existing destination split", destination)
        if not force:
            raise FileExistsError(
                f"{destination} already exists; use --force to replace it"
            )
        _log(f"Removing existing destination split: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        _log(f"Removing stale temporary split directory: {temporary}")
        shutil.rmtree(temporary)
    try:
        t0 = time.time()
        dataset.save_to_disk(str(temporary))
        elapsed = time.time() - t0
        _log_path_status("Temporary saved split", temporary)
        _log(f"Finished save_to_disk for {destination.name} in {elapsed:.1f}s")
        os.replace(temporary, destination)
        _log_path_status("Final saved split", destination)
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
    selection_cache: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create ImageNet-LT train/validation Arrow splits under ``dst``."""
    _log(
        f"prepare_imagenet_lt start: src={src} dst={dst} splits={list(splits)} "
        f"seed={seed} profile={profile} force={force} dry_run={dry_run}"
    )
    _log_path_status("Initial source root", Path(src))
    _log_path_status("Initial destination root", Path(dst))
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
        _log(f"Using official ImageNet-LT manifest: {official_manifest}")
        manifest_filenames, manifest_labels, class_counts = parse_imagenet_lt_manifest(
            official_manifest
        )
        _log(
            f"Parsed official manifest: rows={len(manifest_filenames)} "
            f"classes={len(class_counts)} head={max(class_counts)} tail={min(class_counts)}"
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
    selection_cache_path = (
        _resolve_selection_cache_path(profile, seed, selection_cache)
        if profile == "official"
        else None
    )
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
    if selection_cache_path is not None:
        metadata["selection_cache"] = str(selection_cache_path)
        _log(f"Resolved selection cache path: {selection_cache_path}")
    if official_manifest is not None:
        metadata["manifest"] = official_manifest
        metadata["manifest_sha256"] = _sha256(official_manifest)
        metadata["selection_order"] = "deterministic-global-shuffle"
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
    _log(f"Requested splits after validation: {requested_splits}")
    if dry_run:
        print(f"DRY RUN: would create {requested_splits} under {dst}/arrow")
        return metadata

    existing = [
        Path(dst) / "arrow" / split
        for split in requested_splits
        if (Path(dst) / "arrow" / split).exists()
    ]
    if existing and not force:
        for path in existing:
            _log_path_status("Existing requested destination split", path)
        raise FileExistsError(
            "ImageNet-LT destination split(s) already exist: "
            + ", ".join(str(path) for path in existing)
            + "; use --force to replace them"
        )

    arrow_root = Path(dst) / "arrow"
    if "train" in requested_splits:
        _log("Starting train split build")
        print("Loading source ImageNet-1K train split...")
        train = _load_source_split(src, "train")
        if "label" not in train.column_names:
            raise ValueError(f"Source train split has no 'label' column: {train.column_names}")
        _log("Materializing train labels for fast selection and cache validation")
        train_labels = [int(label) for label in train["label"]]
        _log(f"Materialized {len(train_labels)} train labels")
        if profile == "official":
            assert manifest_filenames is not None and manifest_labels is not None
            assert selection_cache_path is not None
            source_fingerprint = str(
                getattr(
                    train,
                    "_labelmix_source_fingerprint",
                    getattr(train, "_fingerprint", ""),
                )
            )
            source_label_sha256 = _labels_sha256(train_labels)
            manifest_sha256 = str(metadata["manifest_sha256"])
            _log(
                f"Train source summary before selection: rows={len(train)} "
                f"fingerprint={source_fingerprint} label_sha256={source_label_sha256}"
            )
            cached_selection = _load_selection_cache(
                selection_cache_path,
                profile=profile,
                seed=seed,
                manifest_sha256=manifest_sha256,
                source_rows=len(train),
                source_fingerprint=source_fingerprint,
                source_label_sha256=source_label_sha256,
                expected_counts=class_counts,
                source_labels=train_labels,
            )
            if cached_selection is None:
                _log("No valid selection cache found; resolving manifest indices from source")
                selected, selection_strategy = resolve_manifest_indices(
                    train,
                    manifest_filenames,
                    manifest_labels,
                    seed=seed,
                )
                selected_source_labels = [int(train_labels[index]) for index in selected]
                _write_selection_cache(
                    selection_cache_path,
                    profile=profile,
                    seed=seed,
                    manifest_sha256=manifest_sha256,
                    source_rows=len(train),
                    source_fingerprint=source_fingerprint,
                    source_label_sha256=source_label_sha256,
                    class_counts=class_counts,
                    selected=selected,
                    selection_strategy=selection_strategy,
                    selected_labels=selected_source_labels,
                )
                metadata["selection_cache_hit"] = False
                print(f"Created permanent ImageNet-LT selection cache: {selection_cache_path}")
            else:
                selected, selection_strategy = cached_selection
                metadata["selection_cache_hit"] = True
                print(f"Using permanent ImageNet-LT selection cache: {selection_cache_path}")
            _log(
                f"Selection resolved: count={len(selected)} strategy={selection_strategy} "
                f"cache_hit={metadata.get('selection_cache_hit')}"
            )
            metadata["selection_identity_strategy"] = selection_strategy
            metadata["selection_order"] = "deterministic-global-shuffle"
            metadata["exact_official_identities"] = (
                selection_strategy == "exact-official-manifest-identities"
            )
            if not metadata["exact_official_identities"]:
                selected_source_labels = [int(train_labels[index]) for index in selected]
                metadata["selection_sha256"] = _selection_sha256(
                    [str(index) for index in selected],
                    selected_source_labels,
                )
                print(
                    "WARNING: source Arrow rows do not retain the original JPEG "
                    "basenames. Using a deterministic seeded subset with the "
                    "official ImageNet-LT per-class counts."
                )
            actual_counts = class_counts
        else:
            class_counts, class_rank_order = assign_rank_counts_to_classes(
                train_labels, rank_counts, seed=seed
            )
            metadata["class_counts"] = class_counts
            metadata["class_rank_order"] = class_rank_order
            selected, actual_counts = select_long_tail_indices(
                train_labels, class_counts, seed=seed
            )
            selected_labels = [train_labels[index] for index in selected]
            metadata["selection_order"] = "seeded-class-selection-and-global-shuffle"
            metadata["selection_sha256"] = _selection_sha256(
                [str(index) for index in selected],
                selected_labels,
            )
        subset = train.select(selected)
        _log_dataset_summary("Selected globally shuffled train subset", subset)
        observed = Counter(int(label) for label in subset["label"])
        if [observed[label] for label in range(1000)] != actual_counts:
            raise RuntimeError("ImageNet-LT class-count verification failed before save")
        _log(
            f"Class-count verification before save passed: classes={len(actual_counts)} "
            f"samples={sum(actual_counts)}"
        )
        subset = _ensure_embedded_images(subset)
        print(f"Saving {len(subset)} globally shuffled long-tail train samples...")
        _replace_saved_split(subset, arrow_root / "train", force=force)

    if "validation" in requested_splits:
        _log("Starting validation split build")
        print("Loading unchanged ImageNet-1K validation split...")
        validation = _load_source_split(src, "validation")
        metadata["validation_samples"] = len(validation)
        _log_dataset_summary("Validation source split", validation)
        validation = _ensure_embedded_images(validation)
        print(f"Saving {len(validation)} validation samples to RAM...")
        _replace_saved_split(validation, arrow_root / "validation", force=force)

    Path(dst).mkdir(parents=True, exist_ok=True)
    _log(f"Writing ImageNet-LT metadata: {metadata_path}")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _log_path_status("Final ImageNet-LT destination", Path(dst))
    print(f"ImageNet-LT ready at {dst}")
    print(f"Metadata: {metadata_path}")
    return metadata


def verify_imagenet_lt(dst: str) -> dict[str, Any]:
    """Verify saved split sizes and train class counts against metadata."""
    _log(f"verify_imagenet_lt start: dst={dst}")
    metadata_path = Path(dst) / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing ImageNet-LT metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    datasets = _import_datasets()
    train_path = Path(dst) / "arrow" / "train"
    if train_path.is_dir():
        _log_path_status("Verification train split", train_path)
        train = datasets.load_from_disk(str(train_path))
        _log_dataset_summary("Verification loaded train split", train)
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
        _log_path_status("Verification validation split", validation_path)
        validation = datasets.load_from_disk(str(validation_path))
        _log_dataset_summary("Verification loaded validation split", validation)
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
    _log(f"print_imagenet_lt_status start: dst={dst}")
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
        _log_path_status(f"Status split {split}", path)
