#!/usr/bin/env python3
"""
Calculate balanced dataset epoch sizes (min/max/M) vs original dataset size.
Works with Hugging Face datasets configured via the repo YAML configs.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from typing import Dict, Tuple

import yaml


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return cfg


def _normalize_dataset(dataset: str) -> str:
    if dataset.startswith("hfds/"):
        return dataset[len("hfds/") :]
    return dataset


def _resolve_data_dir(data_dir: str, cwd: str) -> str:
    if not data_dir:
        return cwd
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.abspath(os.path.join(cwd, data_dir))


def _resolve_token(token_arg: str | None) -> str | None:
    if token_arg:
        return token_arg
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_DATASETS_TOKEN"):
        token = os.environ.get(env_key)
        if token:
            return token
    return None


def _resolve_label_key(ds, target_key: str | None) -> str:
    if target_key and target_key in ds.column_names:
        return target_key
    if "label" in ds.column_names:
        return "label"
    # Try to find a ClassLabel feature
    for name, feat in ds.features.items():
        if hasattr(feat, "num_classes") and hasattr(feat, "names"):
            return name
    raise ValueError(
        f"Could not infer label column. Available columns: {ds.column_names}. "
        "Provide --target-key or set target_key in the config."
    )


def _count_labels(ds, label_key: str, batch_size: int) -> Tuple[Dict[int, int], int]:
    total = len(ds)
    counts: Dict[int, int] = collections.Counter()
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = ds[start:end]
        labels = batch[label_key]
        if not isinstance(labels, (list, tuple)):
            labels = list(labels)
        for v in labels:
            counts[int(v)] += 1
        if start == 0 or end == total or (start // batch_size) % 50 == 0:
            print(f"Counted {end}/{total} samples", flush=True)
    return counts, total


def _fmt_diff(total: int, original: int) -> str:
    delta = total - original
    pct = (delta / original) * 100.0 if original else 0.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta} ({sign}{pct:.2f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare balanced epoch sizes vs original dataset.")
    parser.add_argument(
        "--config",
        default="config/imagenet1k/mnv4_small.yaml",
        help="Path to config file.",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset name (hfds/...).")
    parser.add_argument("--data-dir", default=None, help="Override data_dir from config.")
    parser.add_argument("--train-split", default=None, help="Override train_split from config.")
    parser.add_argument("--target-key", default=None, help="Override target_key from config.")
    parser.add_argument("--M", type=int, default=None, help="Evaluate a specific M value.")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for counting labels.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote code in Hugging Face dataset builder.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. If omitted, uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset = args.dataset or cfg.get("dataset")
    if not dataset:
        raise SystemExit("Missing dataset in config and no --dataset provided.")
    dataset_name = _normalize_dataset(dataset)

    data_dir = _resolve_data_dir(args.data_dir or cfg.get("data_dir", ""), os.getcwd())
    train_split = args.train_split or cfg.get("train_split", "train")
    target_key = args.target_key or cfg.get("target_key")

    try:
        import datasets
    except ImportError as exc:
        raise SystemExit("Missing dependency: `datasets`. Install with: pip install datasets") from exc

    token = _resolve_token(args.token)
    load_kwargs = dict(split=train_split, cache_dir=data_dir, trust_remote_code=args.trust_remote_code)
    if token:
        import inspect
        sig = inspect.signature(datasets.load_dataset)
        if "token" in sig.parameters:
            load_kwargs["token"] = token
        elif "use_auth_token" in sig.parameters:
            load_kwargs["use_auth_token"] = token

    print(f"Loading dataset: {dataset_name} (split={train_split})")
    ds = datasets.load_dataset(dataset_name, **load_kwargs)

    label_key = _resolve_label_key(ds, target_key)
    print(f"Using label key: {label_key}")

    counts, original_size = _count_labels(ds, label_key, args.batch_size)
    if not counts:
        raise SystemExit("No labels counted; check label key.")

    num_classes = len(counts)
    min_count = min(counts.values())
    max_count = max(counts.values())

    total_min = num_classes * min_count
    total_max = num_classes * max_count

    print("\n=== Summary ===")
    print(f"Original size: {original_size}")
    print(f"Num classes:  {num_classes}")
    print(f"Min class:   {min_count}")
    print(f"Max class:   {max_count}")
    print("")
    print(f"balanced_mode=min  -> {total_min}  diff: {_fmt_diff(total_min, original_size)}")
    print(f"balanced_mode=max  -> {total_max}  diff: {_fmt_diff(total_max, original_size)}")
    if args.M is not None:
        total_m = num_classes * args.M
        print(f"balanced_mode={args.M} -> {total_m}  diff: {_fmt_diff(total_m, original_size)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
