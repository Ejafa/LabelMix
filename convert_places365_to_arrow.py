#!/usr/bin/env python3
"""One-time script: convert places365 parquet → Arrow datasets on disk.

This produces the ``arrow/train/`` and ``arrow/validation/`` directories that
``reader_hfds.py`` loads instantly via ``datasets.load_from_disk()``.

After running this script, ``copy_data_to_ram.py`` only needs to do a plain
file copy (just like imagenet-1k) — no parquet conversion at runtime.

Usage::

    python convert_places365_to_arrow.py                    # default paths
    python convert_places365_to_arrow.py --src /path/to/places365
    python convert_places365_to_arrow.py --splits train     # only convert train
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import datasets


DEFAULT_SRC = (
    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/places365"
)
DEFAULT_SPLITS = ["train", "validation"]


def convert(src: str, splits: list[str], force: bool = False) -> None:
    arrow_root = os.path.join(src, "arrow")
    os.makedirs(arrow_root, exist_ok=True)

    # Point at the data/ subdirectory so that HF only sees the parquet files
    # and doesn't get confused by the arrow/ output directory we create here.
    data_dir = os.path.join(src, "data")

    for split in splits:
        split_dir = os.path.join(arrow_root, split)

        if os.path.isdir(split_dir) and not force:
            print(f"⏭️  {split_dir} already exists — skipping (use --force to overwrite)")
            continue

        if os.path.isdir(split_dir) and force:
            print(f"🗑️  Removing existing {split_dir}...")
            shutil.rmtree(split_dir)

        print(f"\n🔄 Converting split '{split}'...")

        # Load from the parquet source directory (data/ subdir).
        print(f"   Loading from parquet...", end=" ", flush=True)
        t0 = time.time()
        ds = datasets.load_dataset(data_dir, split=split)
        elapsed = time.time() - t0
        print(f"done ({len(ds)} rows, {elapsed:.1f}s)")

        # Save as Arrow dataset
        print(f"   Saving to {split_dir}...", end=" ", flush=True)
        t0 = time.time()
        ds.save_to_disk(split_dir)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        del ds  # free memory

    # Show result
    print(f"\n✅ Arrow datasets saved under {arrow_root}/")
    for split in splits:
        split_dir = os.path.join(arrow_root, split)
        if os.path.isdir(split_dir):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(split_dir)
                for f in fns
            )
            print(f"   {split}: {size / 1e9:.1f} GB")

    print(
        "\nYou can now use copy_data_to_ram.py to copy the arrow/ directory to "
        "/dev/shm for RAM-speed I/O."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert places365 parquet to Arrow datasets on disk",
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help=f"Path to the places365 dataset directory (default: {DEFAULT_SRC})",
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

    # Sanity check: parquet source must exist
    data_dir = os.path.join(args.src, "data")
    if not os.path.isdir(data_dir):
        parser.error(
            f"Expected parquet data directory at {data_dir} — "
            f"is --src correct?"
        )

    convert(args.src, args.splits, force=args.force)


if __name__ == "__main__":
    main()
