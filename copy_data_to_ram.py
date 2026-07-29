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

For coco, the script copies the detectron2-style layout
(``annotations/``, ``train2017/``, ``val2017/``) into ``/dev/shm/coco``. After
copying you typically set ``export DETECTRON2_DATASETS=/dev/shm`` so detectron2
picks up the RAM copy automatically (it looks for ``$DETECTRON2_DATASETS/coco``
at runtime). Use ``--splits`` to copy only a subset, e.g. ``--splits val`` for
eval-only jobs (~1.6 GB) instead of the full ~20 GB.

Usage::

    python copy_data_to_ram.py                          # copy imagenet-1k (default)
    python copy_data_to_ram.py --dataset places365      # copy places365
    python copy_data_to_ram.py --dataset cifar100       # copy cifar100
    python copy_data_to_ram.py --dataset imagenet-lt    # build long-tail ImageNet in RAM
    python copy_data_to_ram.py --dataset coco           # copy coco (full: ~20 GB)
    python copy_data_to_ram.py --dataset coco --splits val  # coco val2017 + anns only
    python copy_data_to_ram.py --dry-run                # show what would be copied
    python copy_data_to_ram.py --verify                 # verify existing copy
    python copy_data_to_ram.py --cleanup                # remove the RAM copy

Supported datasets and their default paths:

    imagenet-1k:
      src: /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k
      dst: /dev/shm/imagenet-1k

    imagenet-lt:
      src: ImageNet-1K Hugging Face cache (same default as imagenet-1k)
      dst: /dev/shm/imagenet-lt
      adapter: bundled official Pareto-alpha=6 ImageNet-LT manifest subset

    places365:
      src: /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/places365/arrow
      dst: /dev/shm/places365/arrow

    cifar100:
      src: /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/cifar100/arrow
      dst: /dev/shm/cifar100/arrow

    coco:
      src: <repo>/detectron2_vitdet/datasets/coco
      dst: /dev/shm/coco
      splits: annotations, train, val   (select with --splits, default: all)
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

# Each entry may optionally declare a ``splits`` mapping:
#   split-alias -> list of subdirectory names (relative to ``src``).
# When present, ``--splits`` can restrict which subdirs are copied; otherwise
# the whole ``src`` tree is copied (imagenet-1k / places365 / cifar100 behaviour).

DATASETS: dict[str, dict] = {
    "imagenet-1k": {
        "src": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k",
        "dst": "/dev/shm/imagenet-1k",
    },
    "imagenet-lt": {
        "src": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k",
        "dst": "/dev/shm/imagenet-lt",
        "adapter": "imagenet-lt",
        "splits": {
            "train": ["train"],
            "val": ["validation"],
            "validation": ["validation"],
            "full": ["train", "validation"],
        },
        "post_copy_hint": (
            "ImageNet-LT is stored as Arrow datasets under "
            "<DATA DIR>/arrow. Set IMAGENET_LT_DATA_DIR to the same DATA DIR "
            "when generating jobs, then use --dataset imagenet-lt in "
            "experiments/generate_jobs.py."
        ),
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
    "coco": {
        # Detectron2-style COCO layout:
        #   <src>/annotations/  ~ 795 MB (instances_{train,val}2017.json, etc.)
        #   <src>/train2017/    ~ 18 GB  (118k jpgs)
        #   <src>/val2017/      ~ 777 MB (5k jpgs)
        # After copying, set DETECTRON2_DATASETS=/dev/shm so detectron2 finds
        # /dev/shm/coco at runtime (it reads $DETECTRON2_DATASETS/coco).
        "src": "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/detectron2_vitdet/datasets/coco",
        "dst": "/dev/shm/coco",
        "splits": {
            # alias -> list of subdirectories (relative to src) to include
            "annotations": ["annotations"],
            "train":       ["train2017"],
            "val":         ["val2017"],
            # convenience presets
            "eval":        ["annotations", "val2017"],        # ~1.6 GB, enough for inference + AP
            "full":        ["annotations", "train2017", "val2017"],
        },
        "post_copy_hint": (
            "NOTE: detectron2 locates COCO via the DETECTRON2_DATASETS env var.\n"
            "      The LabelMix entry points (train_net.py, eval_all.py,\n"
            "      generate_vitdet_jobs.py) auto-detect /dev/shm/coco, so no\n"
            "      manual export is required for those scripts.\n"
            "      For plain detectron2 tools, or to make the choice explicit\n"
            "      in your shell, run:\n"
            "          export DETECTRON2_DATASETS=/dev/shm\n"
            "      (detectron2 will then resolve $DETECTRON2_DATASETS/coco → /dev/shm/coco)"
        ),
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


def resolve_split_subdirs(
    dataset_cfg: dict,
    splits_arg: str | None,
) -> list[str] | None:
    """Translate a --splits CLI value into a list of subdirectories under ``src``.

    Returns ``None`` when no split filtering is needed (copy the whole source).
    Returns a list of relative subdirectory names otherwise (order preserved,
    duplicates removed).
    """
    available = dataset_cfg.get("splits")
    if splits_arg is None or splits_arg.strip().lower() in ("", "all", "full"):
        # No filtering requested. If the dataset defines a "full" preset, use
        # it to stay explicit about what we copy; otherwise fall back to None
        # (= whole source tree).
        if available and "full" in available:
            return list(available["full"])
        return None

    if not available:
        raise SystemExit(
            f"--splits is not supported for this dataset "
            f"(no split registry). Remove --splits or pick a different dataset."
        )

    requested: list[str] = []
    seen: set[str] = set()
    for raw in splits_arg.split(","):
        alias = raw.strip()
        if not alias:
            continue
        if alias not in available:
            valid = ", ".join(sorted(available.keys()))
            raise SystemExit(
                f"Unknown split alias {alias!r}. Valid aliases: {valid}"
            )
        for sub in available[alias]:
            if sub not in seen:
                seen.add(sub)
                requested.append(sub)
    return requested


def walk_files(
    root: str,
    subdirs: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Walk *root* and return ``[(relative_path, size_bytes), ...]``.

    If *subdirs* is given, only those top-level subdirectories of *root* are
    traversed. Paths in the returned tuples remain relative to *root*, so the
    destination layout mirrors the source layout.
    """
    result = []
    if subdirs is None:
        walk_roots = [root]
    else:
        walk_roots = []
        for sub in subdirs:
            full_sub = os.path.join(root, sub)
            if not os.path.exists(full_sub):
                print(f"⚠️  Split subdir not found, skipping: {full_sub}")
                continue
            walk_roots.append(full_sub)

    for walk_root in walk_roots:
        if os.path.isfile(walk_root):
            # Support top-level files listed as a "split" entry if ever needed.
            rel = os.path.relpath(walk_root, root)
            result.append((rel, os.path.getsize(walk_root)))
            continue
        for dirpath, _dirnames, filenames in os.walk(walk_root):
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

def cmd_copy(
    src: str,
    dst: str,
    dry_run: bool = False,
    force: bool = False,
    subdirs: list[str] | None = None,
    post_copy_hint: str | None = None,
) -> bool:
    """Copy dataset from src to dst (/dev/shm). Returns True if a fresh copy was made."""
    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    if subdirs is not None:
        print(f"Splits:      {', '.join(subdirs)}")
    print()

    # Gather files
    files = walk_files(src, subdirs=subdirs)
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

    # Check for existing copy (scoped to the same subdirs we intend to copy)
    if os.path.exists(dst) and not force:
        existing = walk_files(dst, subdirs=subdirs)
        if existing:
            existing_size = sum(s for _, s in existing)
            scope = f" (splits={','.join(subdirs)})" if subdirs else ""
            print(f"⚠️  Destination already exists with {len(existing)} files ({fmt_size(existing_size)}){scope}")
            # Check if it's a complete copy
            if len(existing) == total_count:
                src_sizes = {rel: sz for rel, sz in files}
                dst_sizes = {rel: sz for rel, sz in existing}
                if src_sizes == dst_sizes:
                    print("✅ Existing copy appears complete and matching!")
                    _print_data_dir_banner(dst, post_copy_hint)
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
    dst_files = walk_files(dst, subdirs=subdirs)
    if len(dst_files) != total_count:
        print(f"⚠️  WARNING: expected {total_count} files but found {len(dst_files)} in destination!")
    else:
        print(f"   File count verified: {len(dst_files)} ✓")

    _print_data_dir_banner(dst, post_copy_hint)
    return True  # signal success for post-copy steps


def _print_data_dir_banner(dst: str, post_copy_hint: str | None = None) -> None:
    """Print the final DATA DIR banner (and any per-dataset hint)."""
    print(f"\n{'='*60}")
    print(f"DATA DIR (use in generate_jobs.py or --data-dir):")
    print(f"  {dst}")
    print(f"{'='*60}")
    if post_copy_hint:
        print(post_copy_hint)


def cmd_verify(
    src: str,
    dst: str,
    subdirs: list[str] | None = None,
    post_copy_hint: str | None = None,
) -> None:
    """Verify the RAM copy matches the source."""
    print(f"Verifying {dst} against {src}...")
    if subdirs is not None:
        print(f"Splits:    {', '.join(subdirs)}")
    print()

    src_files = walk_files(src, subdirs=subdirs)
    dst_files = walk_files(dst, subdirs=subdirs)

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
    _print_data_dir_banner(dst, post_copy_hint)


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


def cmd_status(
    src: str,
    dst: str,
    subdirs: list[str] | None = None,
) -> None:
    """Show current status of RAM copy and /dev/shm."""
    shm_total, shm_used, shm_free = shm_capacity()
    print(f"/dev/shm:    total={fmt_size(shm_total)}  used={fmt_size(shm_used)}  free={fmt_size(shm_free)}")
    print()

    src_files = walk_files(src, subdirs=subdirs)
    src_total = sum(s for _, s in src_files)
    scope = f" (splits={','.join(subdirs)})" if subdirs else ""
    print(f"Source ({src}){scope}:")
    print(f"  Files: {len(src_files)}, Size: {fmt_size(src_total)}")

    if os.path.exists(dst):
        dst_files = walk_files(dst, subdirs=subdirs)
        dst_total = sum(s for _, s in dst_files)
        print(f"\nRAM copy ({dst}){scope}:")
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
            "  python copy_data_to_ram.py                              # copy imagenet-1k to /dev/shm\n"
            "  python copy_data_to_ram.py --dataset places365          # copy places365 to /dev/shm\n"
            "  python copy_data_to_ram.py --dataset cifar100           # copy cifar100 to /dev/shm\n"
            "  python copy_data_to_ram.py --dataset imagenet-lt        # build ImageNet-LT in RAM\n"
            "  python copy_data_to_ram.py --dataset coco               # copy full coco (~20 GB)\n"
            "  python copy_data_to_ram.py --dataset coco --splits val  # only annotations+val2017 (~1.6 GB)\n"
            "  python copy_data_to_ram.py --dataset coco --splits annotations,val\n"
            "  python copy_data_to_ram.py --dry-run                    # preview without copying\n"
            "  python copy_data_to_ram.py --verify                     # verify existing copy\n"
            "  python copy_data_to_ram.py --cleanup                    # remove RAM copy\n"
            "  python copy_data_to_ram.py --status                     # show current state\n"
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
        "--splits",
        default=None,
        help=(
            "Comma-separated split aliases to copy (dataset-specific). "
            "For coco: annotations, train, val, eval (=annotations+val), full (default)."
        ),
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
    parser.add_argument(
        "--lt-profile",
        choices=("official", "synthetic-exponential"),
        default="official",
        help=(
            "ImageNet-LT profile. 'official' uses the bundled published manifest; "
            "the synthetic exponential profile is non-canonical (default: official)."
        ),
    )
    parser.add_argument(
        "--lt-train-list",
        default=None,
        help=(
            "Optional override for the bundled official ImageNet_LT_train.txt "
            "manifest. It can also be supplied via IMAGENET_LT_TRAIN_LIST."
        ),
    )
    parser.add_argument(
        "--lt-seed",
        type=int,
        default=42,
        help="ImageNet-LT subset seed (default: 42)",
    )
    parser.add_argument(
        "--lt-max-samples",
        type=int,
        default=1280,
        help="ImageNet-LT head-class sample count (default: 1280)",
    )
    parser.add_argument(
        "--lt-min-samples",
        type=int,
        default=5,
        help="ImageNet-LT tail-class sample count (default: 5)",
    )
    args = parser.parse_args()

    # Resolve src/dst: explicit overrides take priority, then dataset registry
    dataset_defaults = DATASETS[args.dataset]
    src = args.src or dataset_defaults["src"]
    dst = args.dst or dataset_defaults["dst"]
    post_copy_hint = dataset_defaults.get("post_copy_hint")
    subdirs = resolve_split_subdirs(dataset_defaults, args.splits)

    if dataset_defaults.get("adapter") == "imagenet-lt":
        from imagenet_lt_adapter import (
            prepare_imagenet_lt,
            print_imagenet_lt_status,
            verify_imagenet_lt,
        )

        selected_splits = subdirs or ["train", "validation"]
        if args.cleanup:
            cmd_cleanup(dst)
        elif args.verify:
            verify_imagenet_lt(dst)
            _print_data_dir_banner(dst, post_copy_hint)
        elif args.status:
            print_imagenet_lt_status(dst)
        else:
            prepare_imagenet_lt(
                src=src,
                dst=dst,
                splits=selected_splits,
                seed=args.lt_seed,
                max_samples=args.lt_max_samples,
                min_samples=args.lt_min_samples,
                profile=args.lt_profile,
                train_list=args.lt_train_list,
                force=args.force,
                dry_run=args.dry_run,
            )
            _print_data_dir_banner(dst, post_copy_hint)
        return

    if args.cleanup:
        # Cleanup always nukes the whole destination — RAM is precious and
        # partial splits cohabiting in /dev/shm/coco get in each other's way.
        cmd_cleanup(dst)
    elif args.verify:
        cmd_verify(src, dst, subdirs=subdirs, post_copy_hint=post_copy_hint)
    elif args.status:
        cmd_status(src, dst, subdirs=subdirs)
    else:
        cmd_copy(
            src,
            dst,
            dry_run=args.dry_run,
            force=args.force,
            subdirs=subdirs,
            post_copy_hint=post_copy_hint,
        )


if __name__ == "__main__":
    main()
