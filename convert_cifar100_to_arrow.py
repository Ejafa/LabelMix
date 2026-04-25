#!/usr/bin/env python3
"""One-time script: convert cifar100 parquet -> Arrow datasets on disk.

This produces the ``arrow/train/`` and ``arrow/test/`` directories that
``reader_hfds.py`` loads instantly via ``datasets.load_from_disk()``.

After running this script, ``copy_data_to_ram.py`` only needs to do a plain
file copy (just like imagenet-1k / places365) -- no parquet conversion at
runtime, so the "Generating train split: ... examples" step disappears from
every launch.

Usage::

    python convert_cifar100_to_arrow.py                    # default paths
    python convert_cifar100_to_arrow.py --src /path/to/cifar100
    python convert_cifar100_to_arrow.py --splits train     # only convert train
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import datasets

# The HF cifar100 layout on this cluster is:
#   <DEFAULT_SRC>/cifar100/<parquet files>   (inner cifar100/ holds parquet)
# so the "parquet root" we feed to datasets.load_dataset is the inner dir.
DEFAULT_SRC = (
    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/cifar100"
)
DEFAULT_SPLITS = ["train", "test"]


def _resolve_parquet_root(src: str) -> str:
    """Locate the directory that actually contains the parquet files.

    Handles both layouts:
      - <src>/cifar100/*.parquet   (current layout on this cluster)
      - <src>/data/*.parquet       (places365-style layout)
      - <src>/*.parquet            (flat layout)
    """
    candidates = [
        os.path.join(src, "cifar100"),
        os.path.join(src, "data"),
        src,
    ]
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        for entry in os.listdir(cand):
            if entry.endswith(".parquet"):
                return cand
    raise FileNotFoundError(
        f"Could not find any *.parquet files under {src} "
        f"(checked: {candidates})"
    )


def convert(src: str, splits: list[str], force: bool = False) -> None:
    arrow_root = os.path.join(src, "arrow")
    os.makedirs(arrow_root, exist_ok=True)

    parquet_root = _resolve_parquet_root(src)
    print(f"Parquet source: {parquet_root}")
    print(f"Arrow output:   {arrow_root}")

    for split in splits:
        split_dir = os.path.join(arrow_root, split)

        if os.path.isdir(split_dir) and not force:
            print(f"SKIP {split_dir} already exists (use --force to overwrite)")
            continue

        if os.path.isdir(split_dir) and force:
            print(f"Removing existing {split_dir}...")
            shutil.rmtree(split_dir)

        print(f"\nConverting split '{split}'...")

        # Load from the parquet source directory.
        print(f"   Loading from parquet...", end=" ", flush=True)
        t0 = time.time()
        ds = datasets.load_dataset(parquet_root, split=split)
        elapsed = time.time() - t0
        print(f"done ({len(ds)} rows, {elapsed:.1f}s)")

        # Save as Arrow dataset (this is what load_from_disk picks up).
        print(f"   Saving to {split_dir}...", end=" ", flush=True)
        t0 = time.time()
        ds.save_to_disk(split_dir)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        del ds  # free memory

    # Show result
    print(f"\nArrow datasets saved under {arrow_root}/")
    for split in splits:
        split_dir = os.path.join(arrow_root, split)
        if os.path.isdir(split_dir):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(split_dir)
                for f in fns
            )
            print(f"   {split}: {size / 1e6:.1f} MB")

    print(
        "\nYou can now use copy_data_to_ram.py (with the updated cifar100 "
        "entry) to copy the arrow/ directory to /dev/shm for RAM-speed I/O."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cifar100 parquet to Arrow datasets on disk",
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help=f"Path to the cifar100 dataset directory (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"Splits to convert (default: {DEFAULT_SPLITS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Arrow directories",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        parser.error(f"--src {args.src} does not exist or is not a directory")

    convert(args.src, args.splits, force=args.force)


if __name__ == "__main__":
    main()
