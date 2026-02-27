#!/usr/bin/env python3
"""
Download ImageNet-1k from Hugging Face datasets into the cache directory
defined by a config file (or CLI overrides).
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Iterable

import yaml

DEFAULT_DATASET = "hfds/ilsvrc/imagenet-1k"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return cfg


def _normalize_dataset(dataset: str) -> str:
    if not dataset:
        raise ValueError("Dataset name is empty.")
    if dataset.startswith("hfds/"):
        return dataset[len("hfds/") :]
    return dataset


def _resolve_data_dir(data_dir: str, cwd: str) -> str:
    if not data_dir:
        raise ValueError("data_dir is empty.")
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


def _iter_splits(splits: Iterable[str]) -> list[str]:
    cleaned = []
    for split in splits:
        if split is None:
            continue
        split = split.strip()
        if not split:
            continue
        cleaned.append(split)
    if not cleaned:
        cleaned = ["train", "validation"]
    return cleaned


def _load_dataset_for_split(
    dataset_name: str,
    split: str,
    cache_dir: str,
    token: str | None,
    trust_remote_code: bool,
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
    kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if token:
        if "token" in sig.parameters:
            kwargs["token"] = token
        elif "use_auth_token" in sig.parameters:
            kwargs["use_auth_token"] = token

    ds = load_dataset(dataset_name, **kwargs)
    # Force materialization to ensure download/build completes.
    _ = len(ds)
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ImageNet-1k via Hugging Face datasets.")
    parser.add_argument(
        "--config",
        default="config/imagenet1k/mnv4_small.yaml",
        help="Path to config file with dataset and data_dir entries.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Override dataset name (hfds/...). Default: {DEFAULT_DATASET}",
    )
    parser.add_argument("--data-dir", default=None, help="Override data_dir from config.")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Dataset split to download (repeatable). Default: train + validation.",
    )
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
    dataset = args.dataset or cfg.get("dataset") or DEFAULT_DATASET
    data_dir = args.data_dir or cfg.get("data_dir")

    if not data_dir:
        raise SystemExit("Config missing `data_dir` and no --data-dir override provided.")

    dataset_name = _normalize_dataset(dataset)
    resolved_data_dir = _resolve_data_dir(data_dir, os.getcwd())
    os.makedirs(resolved_data_dir, exist_ok=True)

    token = _resolve_token(args.token)
    splits = _iter_splits(args.split or [])

    print(f"Dataset: {dataset_name}")
    print(f"Cache dir: {resolved_data_dir}")
    print(f"Splits: {', '.join(splits)}")

    for split in splits:
        print(f"Downloading split: {split}")
        _load_dataset_for_split(
            dataset_name=dataset_name,
            split=split,
            cache_dir=resolved_data_dir,
            token=token,
            trust_remote_code=args.trust_remote_code,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
