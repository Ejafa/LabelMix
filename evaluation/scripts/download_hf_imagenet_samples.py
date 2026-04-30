"""CLI: stream a small ImageNet-1k image pool from Hugging Face.

This intentionally downloads only the requested samples, not the full
ImageNet-1k dataset.  By default it streams animal classes from the validation
split in order with no shuffle buffer, caps samples per class for diversity,
then writes ordinary image files that ``augmentation_showcase.py`` can consume
via ``--source-dir``.

Usage::

    python evaluation/scripts/download_hf_imagenet_samples.py
    python evaluation/figures/augmentation_showcase.py \\
        --source-dir evaluation/data/raw/hf_imagenet1k_animal_samples
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image


_logger = logging.getLogger(__name__)

EVALUATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EVALUATION_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_EVAL_DIR = RAW_DIR / "eval_csv"
RAW_LOGITS_DIR = RAW_DIR / "logits"
WANDB_RAW_DIR = RAW_DIR / "wandb"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DIR / "figures"

DEFAULT_DATASET = "ILSVRC/imagenet-1k"
DEFAULT_SPLIT = "validation"
DEFAULT_OUTPUT_DIR = RAW_DIR / "hf_imagenet1k_animal_samples"
DEFAULT_NUM_IMAGES = 128
DEFAULT_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 0
DEFAULT_CLASS_PRESET = "animals"
DEFAULT_MAX_PER_LABEL = 2
DEFAULT_MAX_SAMPLES_INSPECTED = 20000

IMAGENET_INFO_DIR = EVALUATION_DIR.parent / "timm" / "data" / "_info"
IMAGENET_SYNSETS_PATH = IMAGENET_INFO_DIR / "imagenet_synsets.txt"
IMAGENET_LEMMAS_PATH = IMAGENET_INFO_DIR / "imagenet_synset_to_lemma.txt"


def ensure_dirs() -> None:
    for path in (
        RAW_DIR,
        RAW_EVAL_DIR,
        RAW_LOGITS_DIR,
        WANDB_RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _normalize_dataset(dataset: str) -> str:
    return dataset[len("hfds/") :] if dataset.startswith("hfds/") else dataset


def _resolve_token(token_arg: str | None) -> str | None:
    if token_arg:
        return token_arg
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_DATASETS_TOKEN"):
        token = os.environ.get(env_key)
        if token:
            return token
    return None


def _load_streaming_dataset(
    *,
    dataset: str,
    split: str,
    token: str | None,
    trust_remote_code: bool,
    cache_dir: str | None,
):
    try:
        import datasets
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: `datasets`.\n"
            "Install with: pip install datasets"
        ) from exc

    load_dataset = datasets.load_dataset
    sig = inspect.signature(load_dataset)
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        if "token" in sig.parameters:
            kwargs["token"] = token
        elif "use_auth_token" in sig.parameters:
            kwargs["use_auth_token"] = token

    return load_dataset(dataset, **kwargs)


def _image_from_sample(sample: dict[str, Any], image_key: str) -> Image.Image:
    value = sample.get(image_key)
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(
        f"Sample field {image_key!r} has unsupported type {type(value).__name__}. "
        f"Available keys: {sorted(sample.keys())}"
    )


def _load_imagenet_class_info() -> dict[int, dict[str, str]]:
    """Load ImageNet-1k synset + lemma metadata without importing timm."""
    if not IMAGENET_SYNSETS_PATH.exists() or not IMAGENET_LEMMAS_PATH.exists():
        return {}

    synsets = IMAGENET_SYNSETS_PATH.read_text(encoding="utf-8").splitlines()
    lemmas: dict[str, str] = {}
    for line in IMAGENET_LEMMAS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        synset, lemma = line.split("\t", 1)
        lemmas[synset] = lemma
    return {
        idx: {"synset": synset, "description": lemmas.get(synset, synset)}
        for idx, synset in enumerate(synsets)
    }


def _parse_label_set(
    *,
    class_preset: str,
    labels: list[int],
    label_ranges: list[str],
) -> set[int] | None:
    """Return allowed labels, or ``None`` for all ImageNet classes."""
    allowed: set[int] = set()
    if class_preset == "all":
        pass
    elif class_preset == "animals":
        # In the canonical ImageNet-1k class order, labels 0..397 are animals
        # (fish, birds, reptiles, amphibians, insects, dogs, cats, mammals).
        allowed.update(range(0, 398))
    else:
        raise ValueError(f"unknown class preset: {class_preset}")

    allowed.update(labels)
    for label_range in label_ranges:
        if "-" not in label_range:
            allowed.add(int(label_range))
            continue
        start_str, end_str = label_range.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"invalid --label-range {label_range!r}")
        allowed.update(range(start, end + 1))
    return None if class_preset == "all" and not labels and not label_ranges else allowed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - corrupt or partial image; redownload it.
        return False


def _image_size_ok(image: Image.Image, *, min_side: int) -> bool:
    if min_side <= 0:
        return True
    width, height = image.size
    return min(width, height) >= min_side


def download_hf_imagenet_samples(
    *,
    output_dir: Path,
    dataset: str,
    split: str,
    num_images: int,
    image_key: str,
    label_key: str,
    seed: int,
    shuffle_buffer: int,
    min_side: int,
    class_preset: str,
    labels: list[int],
    label_ranges: list[str],
    max_per_label: int,
    max_samples_inspected: int,
    token: str | None,
    trust_remote_code: bool,
    cache_dir: str | None,
    force: bool,
) -> Path:
    """Stream and save ``num_images`` ImageNet samples."""
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _normalize_dataset(dataset)
    token = _resolve_token(token)
    stream = _load_streaming_dataset(
        dataset=dataset,
        split=split,
        token=token,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    if shuffle_buffer > 1:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    allowed_labels = _parse_label_set(
        class_preset=class_preset,
        labels=labels,
        label_ranges=label_ranges,
    )
    class_info = _load_imagenet_class_info()

    manifest: dict[str, Any] = {
        "source": "Hugging Face ImageNet-1k",
        "dataset": dataset,
        "split": split,
        "num_images": num_images,
        "image_key": image_key,
        "label_key": label_key,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "min_side": min_side,
        "class_preset": class_preset,
        "allowed_labels": sorted(allowed_labels) if allowed_labels is not None else None,
        "explicit_labels": labels,
        "label_ranges": label_ranges,
        "max_per_label": max_per_label,
        "max_samples_inspected": max_samples_inspected,
        "streaming": True,
        "files": [],
    }

    saved = 0
    seen = 0
    skipped_by_label = 0
    skipped_by_label_cap = 0
    skipped_by_size = 0
    label_counts: dict[int, int] = {}
    for sample in stream:
        if saved >= num_images:
            break
        seen += 1
        if max_samples_inspected > 0 and seen > max_samples_inspected:
            break

        raw_label = sample.get(label_key)
        if raw_label is None:
            raise KeyError(f"Sample is missing label field {label_key!r}")
        label = int(raw_label)
        if allowed_labels is not None and label not in allowed_labels:
            skipped_by_label += 1
            continue
        if max_per_label > 0 and label_counts.get(label, 0) >= max_per_label:
            skipped_by_label_cap += 1
            continue

        image = _image_from_sample(sample, image_key)
        if not _image_size_ok(image, min_side=min_side):
            skipped_by_size += 1
            continue

        filename = f"imagenet1k_{saved + 1:04d}.jpg"
        out_path = output_dir / filename
        if force or not out_path.exists() or not _valid_image(out_path):
            image.save(out_path, format="JPEG", quality=95)
            _logger.info("Wrote %s", out_path)
        else:
            _logger.info("Skipping existing %s", out_path)

        width, height = image.size
        label_counts[label] = label_counts.get(label, 0) + 1
        info = class_info.get(label, {})
        manifest["files"].append({
            "filename": filename,
            "label": label,
            "synset": info.get("synset"),
            "description": info.get("description"),
            "width": width,
            "height": height,
            "sha256": _sha256(out_path),
        })
        saved += 1

    manifest["samples_inspected"] = seen
    manifest["skipped_by_label"] = skipped_by_label
    manifest["skipped_by_label_cap"] = skipped_by_label_cap
    manifest["skipped_by_size"] = skipped_by_size
    manifest["saved_per_label"] = {
        str(label): count for label, count in sorted(label_counts.items())
    }

    if saved < num_images:
        raise RuntimeError(
            f"Stream ended after saving {saved}/{num_images} images "
            f"({seen} samples inspected)."
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    _logger.info("Wrote %s", manifest_path)
    return output_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to populate. Default: {DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to stream/save. Default: {DEFAULT_NUM_IMAGES}",
    )
    p.add_argument("--image-key", default="image")
    p.add_argument("--label-key", default="label")
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for streaming shuffle. Default: {DEFAULT_SEED}",
    )
    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER,
        help="Streaming shuffle buffer. Set 0/1 to avoid reading extra samples. "
             f"Default: {DEFAULT_SHUFFLE_BUFFER}",
    )
    p.add_argument(
        "--min-side",
        type=int,
        default=256,
        help="Skip images whose shorter side is below this many pixels. Default: 256",
    )
    p.add_argument(
        "--class-preset",
        choices=("animals", "all"),
        default=DEFAULT_CLASS_PRESET,
        help="ImageNet class preset to keep. Default: animals (labels 0..397)",
    )
    p.add_argument(
        "--label",
        action="append",
        type=int,
        default=[],
        help="Additional ImageNet label id to keep. Repeat as needed.",
    )
    p.add_argument(
        "--label-range",
        action="append",
        default=[],
        help="Additional inclusive label range to keep, e.g. 151-268 for dogs.",
    )
    p.add_argument(
        "--max-per-label",
        type=int,
        default=DEFAULT_MAX_PER_LABEL,
        help="Maximum saved images per class label; 0 disables the cap. "
             f"Default: {DEFAULT_MAX_PER_LABEL}",
    )
    p.add_argument(
        "--max-samples-inspected",
        type=int,
        default=DEFAULT_MAX_SAMPLES_INSPECTED,
        help="Stop streaming after inspecting this many samples; 0 disables. "
             f"Default: {DEFAULT_MAX_SAMPLES_INSPECTED}",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir for streaming metadata/cache files.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote code in the dataset builder if required.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite readable existing image files.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    if args.num_images <= 0:
        raise SystemExit("--num-images must be positive")

    try:
        output_dir = download_hf_imagenet_samples(
            output_dir=args.output_dir,
            dataset=args.dataset,
            split=args.split,
            num_images=args.num_images,
            image_key=args.image_key,
            label_key=args.label_key,
            seed=args.seed,
            shuffle_buffer=max(0, args.shuffle_buffer),
            min_side=max(0, args.min_side),
            class_preset=args.class_preset,
            labels=args.label,
            label_ranges=args.label_range,
            max_per_label=max(0, args.max_per_label),
            max_samples_inspected=max(0, args.max_samples_inspected),
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001 - make common HF auth failure readable.
        msg = str(exc)
        if "gated dataset" in msg or "Unauthorized" in msg or "authenticated" in msg:
            raise SystemExit(
                "Could not access the Hugging Face ImageNet-1k dataset. "
                "It is gated, so you need to accept access on Hugging Face "
                "and pass --token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            ) from exc
        raise
    print(f"Downloaded ImageNet sample pool: {output_dir}")
    print(
        "Run showcase with: "
        f"python evaluation/figures/augmentation_showcase.py --source-dir {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
