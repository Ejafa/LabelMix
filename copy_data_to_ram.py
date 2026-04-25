#!/usr/bin/env python3
"""Copy HuggingFace datasets to /dev/shm (RAM-backed tmpfs) for fast I/O.

This eliminates filesystem I/O bottlenecks by serving data directly from RAM.
Run once before starting the daemon. The script prints the data-dir path to
use in generate_jobs.py or as --data-dir.

For places365, run ``convert_places365_to_arrow.py`` first to pre-convert the
parquet source into Arrow datasets on disk. Then this script simply copies the
pre-built Arrow directories to /dev/shm — just like imagenet-1k.

For cifar100, run ``convert_cifar100_to_arrow.py`` first for the same reason:
it turns the parquet source into ``arrow/train/`` and ``arrow/test/`` that
``reader_hfds.py`` loads instantly via ``load_from_disk()``, so the
"Generating train split" step disappears from every launch.

Usage::

    python copy_data_to_ram.py                          # copy imagenet-1k (default)
    python copy_data_to_ram.py --dataset places365      # copy places365
    python copy_data_to_ram.py --dataset cifar100       # copy cifar100
    python copy_data_to_ram.py --dry-run                # show what would be copied
    python copy_data_to_ram.py --verify                 # verify existing copy
    python copy_data_to_ram.py --cleanup                # remove the RAM copy

Supported datasets and their default paths:

    imagenet-1k:
      src: /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k
      dst: /dev/shm/imagenet-1k

    places365:
      src: /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/places365/arrow
      dst: /dev/shm/places365/arrow

    cifar100:
      src: /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/cifar100/arrow
      dst: /dev/shm/cifar100/arrow
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset registry — add new datasets here
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, str]] = {
    "imagenet-1k": {
        "src": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k",
        "dst": "/dev/shm/imagenet-1k",
    },
    "places365": {
        # Pre-converted Arrow datasets (run convert_places365_to_arrow.py first).
        # Source is the arrow/ subdirectory; destination mirrors the structure so
        # that reader_hfds.py finds <data_dir>/arrow/<split>/ at runtime.
        "src": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/places365/arrow",
        "dst": "/dev/shm/places365/arrow",
    },
    "cifar100": {
        # Pre-converted Arrow datasets (run convert_cifar100_to_arrow.py first).
        # Source is the arrow/ subdirectory; destination mirrors the structure
        # so that reader_hfds.py finds <data_dir>/arrow/<split>/ at runtime and
        # uses the instant load_from_disk() fast path instead of regenerating
        # the train/test split from parquet on every launch.
        "src": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/cifar100/arrow",
        "dst": "/dev/shm/cifar100/arrow",
    },
}

DEFAULT_DATASET = "imagenet-1k"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB read chunks for progress reporting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def shm_capacity() -> tuple[int, int, int]:
    """Return (total, used, free) bytes for /dev/shm."""
    return shutil.disk_usage("/dev/shm")


def walk_files(root: str) -> list[tuple[str, int]]:
    """Walk *root* and return [(relative_path, size_bytes), ...]."""
    result = []
    root_path = Path(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            result.append((rel, os.path.getsize(full)))
    return sorted(result)


def quick_hash(filepath: str, sample_bytes: int = 4096) -> str:
    """Fast integrity check: hash first + last sample_bytes + file size."""
    size = os.path.getsize(filepath)
    h = hashlib.md5()
    h.update(str(size).encode())
    with open(filepath, "rb") as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(-sample_bytes, 2)
            h.update(f.read(sample_bytes))
    return h.hexdigest()


def print_bar(current: int, total: int, width: int = 50, extra: str = ""):
    """Print a simple progress bar to stderr."""
    frac = current / total if total else 1
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    sys.stderr.write(f"\r  [{bar}] {pct:5.1f}%  {fmt_size(current):>10s} / {fmt_size(total):>10s}  {extra}")
    sys.stderr.flush()


def copy_with_progress(src: str, dst: str, total_bytes: int, copied_so_far: int) -> int:
    """Copy a single file with progress tracking. Returns bytes copied."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = os.path.getsize(src)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            fout.write(chunk)
            copied_so_far += len(chunk)
            print_bar(copied_so_far, total_bytes, extra=os.path.basename(src))
    return copied_so_far


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_copy(src: str, dst: str, dry_run: bool = False, force: bool = False) -> bool:
    """Copy dataset from src to dst (/dev/shm). Returns True if a fresh copy was made."""
    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    print()

    # Gather files
    files = walk_files(src)
    total_bytes = sum(s for _, s in files)
    total_count = len(files)
    print(f"Files:       {total_count}")
    print(f"Total size:  {fmt_size(total_bytes)}")
    print()

    # Check /dev/shm capacity
    shm_total, shm_used, shm_free = shm_capacity()
    print(f"/dev/shm:    total={fmt_size(shm_total)}  used={fmt_size(shm_used)}  free={fmt_size(shm_free)}")

    if total_bytes > shm_free:
        print(f"\n❌ ERROR: Not enough space in /dev/shm!")
        print(f"   Need {fmt_size(total_bytes)} but only {fmt_size(shm_free)} free.")
        print(f"   Free up {fmt_size(total_bytes - shm_free)} or reduce data.")
        sys.exit(1)

    headroom = shm_free - total_bytes
    print(f"Headroom:    {fmt_size(headroom)} will remain free after copy")
    print()

    if dry_run:
        print("🔍 DRY RUN — nothing copied.")
        print(f"\nWould copy {total_count} files ({fmt_size(total_bytes)}) to {dst}")
        return False

    # Check for existing copy
    if os.path.exists(dst) and not force:
        existing = walk_files(dst)
        if existing:
            existing_size = sum(s for _, s in existing)
            print(f"⚠️  Destination already exists with {len(existing)} files ({fmt_size(existing_size)})")
            # Check if it's a complete copy
            if len(existing) == total_count:
                src_sizes = {rel: sz for rel, sz in files}
                dst_sizes = {rel: sz for rel, sz in existing}
                if src_sizes == dst_sizes:
                    print("✅ Existing copy appears complete and matching!")
                    print(f"\n{'='*60}")
                    print(f"DATA DIR (use in generate_jobs.py or --data-dir):")
                    print(f"  {dst}")
                    print(f"{'='*60}")
                    return False
            print("   Existing copy is incomplete/mismatched. Use --force to overwrite.")
            print("   Or run with --cleanup first.")
            sys.exit(1)

    # Copy
    print(f"🚀 Copying {total_count} files to RAM...")
    t0 = time.time()
    copied_bytes = 0
    copied_count = 0

    for rel, size in files:
        src_file = os.path.join(src, rel)
        dst_file = os.path.join(dst, rel)
        copied_bytes = copy_with_progress(src_file, dst_file, total_bytes, copied_bytes)
        copied_count += 1

    elapsed = time.time() - t0
    throughput = copied_bytes / elapsed if elapsed > 0 else 0
    sys.stderr.write("\n")
    print()
    print(f"✅ Done! Copied {copied_count} files ({fmt_size(copied_bytes)}) in {elapsed:.1f}s")
    print(f"   Throughput: {fmt_size(throughput)}/s")

    # Verify file count
    dst_files = walk_files(dst)
    if len(dst_files) != total_count:
        print(f"⚠️  WARNING: expected {total_count} files but found {len(dst_files)} in destination!")
    else:
        print(f"   File count verified: {len(dst_files)} ✓")

    print(f"\n{'='*60}")
    print(f"DATA DIR (use in generate_jobs.py or --data-dir):")
    print(f"  {dst}")
    print(f"{'='*60}")

    return True  # signal success for post-copy steps


def cmd_verify(src: str, dst: str) -> None:
    """Verify the RAM copy matches the source."""
    print(f"Verifying {dst} against {src}...")
    print()

    src_files = walk_files(src)
    dst_files = walk_files(dst)

    src_map = {rel: sz for rel, sz in src_files}
    dst_map = {rel: sz for rel, sz in dst_files}

    errors = []

    # Check all source files exist in destination
    for rel, src_sz in src_map.items():
        if rel not in dst_map:
            errors.append(f"  MISSING: {rel}")
        elif dst_map[rel] != src_sz:
            errors.append(f"  SIZE MISMATCH: {rel} (src={fmt_size(src_sz)}, dst={fmt_size(dst_map[rel])})")

    # Check for extra files in destination
    for rel in dst_map:
        if rel not in src_map:
            errors.append(f"  EXTRA: {rel}")

    if errors:
        print(f"❌ Found {len(errors)} issue(s):")
        for e in errors:
            print(e)
        sys.exit(1)

    # Spot-check integrity with hashes (first 10 + last 10 + random 10)
    print(f"File count: {len(src_files)} ✓")
    print(f"Size match: all files ✓")

    import random
    sample_indices = set()
    sample_indices.update(range(min(10, len(src_files))))
    sample_indices.update(range(max(0, len(src_files) - 10), len(src_files)))
    remaining = [i for i in range(len(src_files)) if i not in sample_indices]
    sample_indices.update(random.sample(remaining, min(10, len(remaining))))

    print(f"Hash-checking {len(sample_indices)} sample files...")
    hash_errors = []
    for i in sorted(sample_indices):
        rel, _ = src_files[i]
        src_hash = quick_hash(os.path.join(src, rel))
        dst_hash = quick_hash(os.path.join(dst, rel))
        if src_hash != dst_hash:
            hash_errors.append(rel)

    if hash_errors:
        print(f"❌ Hash mismatch on {len(hash_errors)} file(s):")
        for f in hash_errors:
            print(f"  {f}")
        sys.exit(1)

    print(f"Hash check:  {len(sample_indices)} samples ✓")
    print()
    print("✅ Verification passed!")
    print(f"\n{'='*60}")
    print(f"DATA DIR (use in generate_jobs.py or --data-dir):")
    print(f"  {dst}")
    print(f"{'='*60}")


def cmd_cleanup(dst: str) -> None:
    """Remove the RAM copy from /dev/shm."""
    if not os.path.exists(dst):
        print(f"Nothing to clean: {dst} does not exist.")
        return

    files = walk_files(dst)
    total_bytes = sum(s for _, s in files)
    print(f"Removing {len(files)} files ({fmt_size(total_bytes)}) from {dst}...")

    shutil.rmtree(dst)
    print(f"✅ Cleaned up. Freed ~{fmt_size(total_bytes)} in /dev/shm.")


def cmd_status(src: str, dst: str) -> None:
    """Show current status of RAM copy and /dev/shm."""
    shm_total, shm_used, shm_free = shm_capacity()
    print(f"/dev/shm:    total={fmt_size(shm_total)}  used={fmt_size(shm_used)}  free={fmt_size(shm_free)}")
    print()

    src_files = walk_files(src)
    src_total = sum(s for _, s in src_files)
    print(f"Source ({src}):")
    print(f"  Files: {len(src_files)}, Size: {fmt_size(src_total)}")

    if os.path.exists(dst):
        dst_files = walk_files(dst)
        dst_total = sum(s for _, s in dst_files)
        print(f"\nRAM copy ({dst}):")
        print(f"  Files: {len(dst_files)}, Size: {fmt_size(dst_total)}")
        if len(dst_files) == len(src_files):
            print(f"  Status: ✅ Complete ({len(dst_files)}/{len(src_files)} files)")
        else:
            print(f"  Status: ⚠️  Incomplete ({len(dst_files)}/{len(src_files)} files)")
    else:
        print(f"\nRAM copy ({dst}): not found")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy dataset to /dev/shm for RAM-speed I/O",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python copy_data_to_ram.py                         # copy imagenet-1k to /dev/shm\n"
            "  python copy_data_to_ram.py --dataset places365     # copy places365 to /dev/shm\n"
            "  python copy_data_to_ram.py --dataset cifar100      # copy cifar100 to /dev/shm\n"
            "  python copy_data_to_ram.py --dry-run               # preview without copying\n"
            "  python copy_data_to_ram.py --verify                # verify existing copy\n"
            "  python copy_data_to_ram.py --cleanup               # remove RAM copy\n"
            "  python copy_data_to_ram.py --status                # show current state\n"
            "\n"
            "Available datasets: " + ", ".join(DATASETS.keys()) + "\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=list(DATASETS.keys()),
        help=f"Dataset to copy (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--src",
        default=None,
        help="Override source data directory",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help="Override destination in RAM",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without doing it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing RAM copy against source",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the RAM copy to free /dev/shm",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current copy status and /dev/shm usage",
    )
    args = parser.parse_args()

    # Resolve src/dst: explicit overrides take priority, then dataset registry
    dataset_defaults = DATASETS[args.dataset]
    src = args.src or dataset_defaults["src"]
    dst = args.dst or dataset_defaults["dst"]

    if args.cleanup:
        cmd_cleanup(dst)
    elif args.verify:
        cmd_verify(src, dst)
    elif args.status:
        cmd_status(src, dst)
    else:
        cmd_copy(src, dst, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
