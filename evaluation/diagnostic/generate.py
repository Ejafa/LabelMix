"""Generate the composed diagnostic evaluation dataset.

For each ``k`` in the requested set, this stage:

    1. Picks ``k`` distinct source images from the held-out pool.
    2. Runs the training-equivalent LabelMix composition (alpha, k,
       max aspect) via :func:`compose_labelmix_with_mask`.
    3. Persists:

        - ``images/sample_<id>.png``  -- composed RGB canvas (uint8)
        - ``masks/sample_<id>.png``   -- patch mask as uint8 L-mode PNG
                                         (pixel value = slot index)
        - one line in ``manifest.jsonl`` with:
            * ``sample_id``, ``k``
            * ``source_pool_indices``, ``source_shard_rows``, ``source_labels``
            * ``patch_to_class``         (slot index -> source label)
            * ``patch_area_ratios``      (slot index -> fraction of canvas)
            * ``merged_class_area_ratios`` (source_label -> summed fraction)
            * ``image_path``, ``patch_mask_path``
            * ``symmetry``, ``alpha``, ``sampling_max_aspect``

All randomness is driven by a single base seed so every model consumes the
same composed dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from evaluation.common import DIAGNOSTIC_DIR, ensure_dirs, setup_logging

from .compose import ComposedSample, compose_labelmix_with_mask
from .source_pool import (
    DEFAULT_HFDS_CACHE_DIR,
    SourcePoolEntry,
    load_source_pool,
    pool_to_preprocessed,
)


_logger = logging.getLogger("evaluation.diagnostic.generate")


# ---------------------------------------------------------------------------
# Defaults (mirroring the user's spec for this diagnostic run).
# ---------------------------------------------------------------------------
DEFAULT_ALPHA: float = 0.5
DEFAULT_MAX_ASPECT: float = 15.0
DEFAULT_K_VALUES: Tuple[int, ...] = (3, 4, 5, 6)
DEFAULT_SAMPLES_PER_K: int = 500
DEFAULT_IMG_SIZE: int = 256
DEFAULT_CROP_PCT: float = 0.95
DEFAULT_SPLIT: str = "validation"
DEFAULT_SEED: int = 20260501


@dataclass
class GenerateConfig:
    out_dir: str
    split: str = DEFAULT_SPLIT
    cache_dir: str = DEFAULT_HFDS_CACHE_DIR
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES
    samples_per_k: int = DEFAULT_SAMPLES_PER_K
    alpha: float = DEFAULT_ALPHA
    sampling_max_aspect: float = DEFAULT_MAX_ASPECT
    img_size: int = DEFAULT_IMG_SIZE
    crop_pct: float = DEFAULT_CROP_PCT
    seed: int = DEFAULT_SEED
    use_symmetries: bool = True
    pool_size: int = 0  # 0 => auto (sum of requested k uses + spare)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)


def _pick_source_indices(
    rng: np.random.Generator,
    pool_size: int,
    k: int,
) -> np.ndarray:
    """Pick ``k`` distinct indices into the source pool."""
    return rng.choice(pool_size, size=k, replace=False)


def _save_rgb_png(tensor_01: torch.Tensor, path: str) -> None:
    """Save a (3, H, W) float tensor in ``[0, 1]`` as an RGB PNG."""
    arr = (tensor_01.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", optimize=False)


def _save_mask_png(mask: torch.Tensor, path: str) -> None:
    """Save the patch mask as an 8-bit PNG (pixel value = slot index).

    Asserts ``k <= 255`` so the slot indices fit into ``L`` mode.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H, W), got shape {tuple(mask.shape)}")
    arr = mask.to(torch.int64).cpu().numpy()
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(
            f"patch mask values out of uint8 range: [{arr.min()}, {arr.max()}]"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path, format="PNG", optimize=True)


def _merged_class_areas(
    source_labels: Sequence[int],
    patch_area_ratios: torch.Tensor,
) -> Dict[str, float]:
    """Sum per-slot areas by source label (JSON-safe: str keys)."""
    out: Dict[str, float] = {}
    for slot_i, label in enumerate(source_labels):
        area = float(patch_area_ratios[slot_i].item())
        key = str(int(label))
        out[key] = out.get(key, 0.0) + area
    return out


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate(cfg: GenerateConfig, append: bool = False) -> str:
    """Materialize the composed diagnostic dataset at ``cfg.out_dir``.

    Returns the path of the written ``manifest.jsonl`` file.

    Args:
        cfg:      generation config (k_values, samples_per_k, alpha, ...).
        append:   If True, existing ``manifest.jsonl`` / ``manifest.json``
                  rows are preserved and new rows for ``cfg.k_values`` are
                  appended. Existing images / masks are not overwritten
                  (filenames are namespaced by k, so no collisions).
    """
    ensure_dirs()
    os.makedirs(cfg.out_dir, exist_ok=True)
    images_dir = os.path.join(cfg.out_dir, "images")
    masks_dir = os.path.join(cfg.out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # How many source images to load: default to k_max * samples_per_k so
    # every composed sample can get a fresh unique tuple.  We cap the pool at
    # the split size (handled inside ``load_source_pool``).  Larger pools
    # are fine but load slower; smaller pools cause more source reuse.
    k_max = max(cfg.k_values)
    implied = int(k_max * cfg.samples_per_k)
    pool_size = cfg.pool_size or min(implied, 50_000)
    _logger.info(
        "Loading %d source samples from ImageNet-1k split=%s (seed=%d).",
        pool_size, cfg.split, cfg.seed,
    )
    pool: List[SourcePoolEntry] = load_source_pool(
        num_samples=pool_size,
        split=cfg.split,
        cache_dir=cfg.cache_dir,
        seed=cfg.seed,
        shuffle=True,
    )

    _logger.info("Preprocessing %d source tensors at %dx%d.",
                 len(pool), cfg.img_size, cfg.img_size)
    source_tensors = pool_to_preprocessed(
        pool, size=cfg.img_size, crop_pct=cfg.crop_pct,
    )

    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    dataset_meta_path = os.path.join(cfg.out_dir, "manifest.json")

    existing_meta: Dict[str, object] = {}
    if append and os.path.isfile(dataset_meta_path):
        with open(dataset_meta_path, "r") as f:
            existing_meta = json.load(f)
        # Guard: the invariants that must match to reuse the same dataset.
        for key, must_match in (
            ("alpha", float(cfg.alpha)),
            ("sampling_max_aspect", float(cfg.sampling_max_aspect)),
            ("seed", int(cfg.seed)),
            ("split", cfg.split),
            ("img_size", int(cfg.img_size)),
            ("crop_pct", float(cfg.crop_pct)),
            ("use_symmetries", bool(cfg.use_symmetries)),
        ):
            have = existing_meta.get(key)
            if have is not None and have != must_match:
                raise ValueError(
                    f"--append mismatch on '{key}': existing manifest has "
                    f"{have!r}, new config has {must_match!r}. Refusing to "
                    "mix incompatible diagnostic samples."
                )

    # Compute the union of k_values for the dataset-level manifest.
    if append and existing_meta.get("k_values"):
        all_k_values = sorted(set(int(k) for k in existing_meta["k_values"])
                              | set(int(k) for k in cfg.k_values))
    else:
        all_k_values = sorted(set(int(k) for k in cfg.k_values))

    dataset_meta = {
        "version": 1,
        "alpha": float(cfg.alpha),
        "sampling_max_aspect": float(cfg.sampling_max_aspect),
        "k_values": list(all_k_values),
        "samples_per_k": int(cfg.samples_per_k),
        "seed": int(cfg.seed),
        "split": cfg.split,
        "cache_dir": cfg.cache_dir,
        "img_size": int(cfg.img_size),
        "crop_pct": float(cfg.crop_pct),
        "pool_size": int(len(pool)),
        "use_symmetries": bool(cfg.use_symmetries),
        "image_format": "PNG (uint8, RGB, normalized-to-[0,1] at compose time)",
        "mask_format": "PNG (L, uint8; pixel value = slot index in [0, k-1])",
    }
    with open(dataset_meta_path, "w") as f:
        json.dump(dataset_meta, f, indent=2, sort_keys=False)
        f.write("\n")
    _logger.info("Wrote dataset-level manifest to %s", dataset_meta_path)

    # Deterministic per-(k, sample_idx) RNG: one numpy.Generator seeded by
    # (base_seed, k, sample_idx) chooses the source tuple, and torch's
    # global RNG is seeded alongside so ``compose_labelmix_with_mask``'s
    # Dirichlet / D4 choices are reproducible.
    existing_ids: set = set()
    if append and os.path.isfile(manifest_path):
        with open(manifest_path, "r") as _prev:
            for _line in _prev:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    existing_ids.add(str(json.loads(_line)["sample_id"]))
                except Exception:  # noqa: BLE001
                    continue
        _logger.info("Append mode: %d existing samples already in manifest.",
                     len(existing_ids))

    n_written = 0
    open_mode = "a" if append else "w"
    with open(manifest_path, open_mode) as mf:
        for k in cfg.k_values:
            for s in range(cfg.samples_per_k):
                sample_id = f"k{k:02d}_{s:05d}"
                if sample_id in existing_ids:
                    continue
                sub_seed = (int(cfg.seed) * 1_000_003 + int(k) * 10_007 + s) & 0x7FFFFFFF
                rng = np.random.default_rng(sub_seed)
                _seed_everything(sub_seed)

                src_idx = _pick_source_indices(rng, len(pool), k)
                imgs = [source_tensors[int(i)] for i in src_idx]
                labels = [int(pool[int(i)].label) for i in src_idx]

                composed: ComposedSample = compose_labelmix_with_mask(
                    imgs,
                    alpha=cfg.alpha,
                    k=k,
                    sampling_max_aspect=cfg.sampling_max_aspect,
                    use_symmetries=cfg.use_symmetries,
                )

                img_path = os.path.join(images_dir, f"{sample_id}.png")
                mask_path = os.path.join(masks_dir, f"{sample_id}.png")
                _save_rgb_png(composed.image, img_path)
                _save_mask_png(composed.patch_mask, mask_path)

                patch_to_class = {
                    str(slot_i): int(labels[slot_i]) for slot_i in range(k)
                }
                row = {
                    "sample_id": sample_id,
                    "k": int(k),
                    "source_pool_indices": [int(i) for i in src_idx.tolist()],
                    "source_shard_rows": [
                        {
                            "shard": os.path.basename(pool[int(i)].shard_path),
                            "row": int(pool[int(i)].row_index),
                        }
                        for i in src_idx
                    ],
                    "source_labels": labels,
                    "patch_to_class": patch_to_class,
                    "patch_area_ratios": [
                        float(x) for x in composed.patch_area_ratios.tolist()
                    ],
                    "merged_class_area_ratios": _merged_class_areas(
                        labels, composed.patch_area_ratios,
                    ),
                    "slot_boxes": [
                        [int(v) for v in composed.slot_boxes[slot_i].tolist()]
                        for slot_i in range(k)
                    ],
                    "symmetry": int(composed.symmetry),
                    "alpha": float(cfg.alpha),
                    "sampling_max_aspect": float(cfg.sampling_max_aspect),
                    "image_path": os.path.relpath(img_path, cfg.out_dir),
                    "patch_mask_path": os.path.relpath(mask_path, cfg.out_dir),
                }
                mf.write(json.dumps(row) + "\n")
                n_written += 1

                if n_written % 200 == 0:
                    _logger.info("Generated %d composed samples...", n_written)

    _logger.info(
        "Done. Wrote %d composed samples across k=%s to %s.",
        n_written, list(cfg.k_values), cfg.out_dir,
    )
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--out-dir", type=str,
        default=str(DIAGNOSTIC_DIR / "composed"),
        help="Directory to write the composed dataset into.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        choices=("validation", "test"),
                        help="Held-out ImageNet-1k split to draw sources from.")
    parser.add_argument("--cache-dir", default=DEFAULT_HFDS_CACHE_DIR,
                        help="Root of the HF Arrow cache for ILSVRC/imagenet-1k.")
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=list(DEFAULT_K_VALUES),
                        help="k values to generate (default: 3 4 5 6).")
    parser.add_argument("--samples-per-k", type=int,
                        default=DEFAULT_SAMPLES_PER_K,
                        help="Number of composed samples per k (default: 500).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="LabelMix Dirichlet alpha (default: 0.5).")
    parser.add_argument("--sampling-max-aspect", type=float,
                        default=DEFAULT_MAX_ASPECT,
                        help="Max aspect ratio per patch (default: 15).")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--crop-pct", type=float, default=DEFAULT_CROP_PCT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Fixed seed so every model evaluates on the same data.")
    parser.add_argument("--no-symmetries", action="store_true",
                        help="Disable D4 symmetry (keeps raw squarify layout).")
    parser.add_argument("--pool-size", type=int, default=0,
                        help="Override the source-pool size (0 = auto).")
    parser.add_argument("--append", action="store_true",
                        help="Append to an existing manifest in --out-dir "
                             "instead of overwriting it. Useful for extending "
                             "a previously generated dataset with extra k values "
                             "(e.g. adding k=1,2 to an existing k=3..6 run).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING")
    cfg = GenerateConfig(
        out_dir=args.out_dir,
        split=args.split,
        cache_dir=args.cache_dir,
        k_values=tuple(int(k) for k in args.k_values),
        samples_per_k=int(args.samples_per_k),
        alpha=float(args.alpha),
        sampling_max_aspect=float(args.sampling_max_aspect),
        img_size=int(args.img_size),
        crop_pct=float(args.crop_pct),
        seed=int(args.seed),
        use_symmetries=not args.no_symmetries,
        pool_size=int(args.pool_size),
    )
    generate(cfg, append=bool(args.append))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
