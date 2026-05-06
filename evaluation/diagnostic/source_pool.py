"""Load a pool of ``(image, label)`` samples from ImageNet-1k held-out splits.

Reads directly from the on-disk HuggingFace ``datasets`` Arrow cache used by
the rest of the repo (see ``copy_data_to_ram.py``) so we don't need the full
``datasets`` library's decoding pipeline at import time.

Default cache layout::

    <root>/ilsvrc___imagenet-1k/default/0.0.0/<fingerprint>/
        imagenet-1k-validation-00000-of-NNN.arrow
        imagenet-1k-test-00000-of-NNN.arrow
        ...

For diagnostic evaluation we read from the ``validation`` (or ``test``)
split so the composed evaluation samples never overlap with training data.
"""
from __future__ import annotations

import glob
import io
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


_logger = logging.getLogger("evaluation.diagnostic.source_pool")


# Root of the HuggingFace Arrow cache for ILSVRC ImageNet-1k on our cluster.
# Mirrors ``augmentation_showcase.DEFAULT_HFDS_CACHE_DIR`` and
# ``copy_data_to_ram.DATASETS['imagenet-1k']``.
DEFAULT_HFDS_CACHE_DIR: str = (
    "/apdcephfs_fsgm/share_303853033/ethangeng/"
    "konstantin-garbers/data/imagenet-1k"
)


@dataclass
class SourcePoolEntry:
    """One held-out source image with its ImageNet-1k label.

    Attributes:
        pool_index: Position within the pool (stable, seed-independent).
        shard_path: Absolute path of the Arrow shard the sample came from.
        row_index: Row within that shard.
        label:     Integer class index (0..num_classes-1) or ``-1`` when
                   the shard carries no ``label`` column.
        image:     HWC uint8 numpy array at the original HF-Arrow resolution.
    """

    pool_index: int
    shard_path: str
    row_index: int
    label: int
    image: np.ndarray


def _iter_arrow_batches(shard_path: str):
    """Yield ``pyarrow.RecordBatch`` from an Arrow shard in file or stream form."""
    import pyarrow.ipc as ipc  # lazy import so non-diagnostic code paths don't pay

    try:
        reader = ipc.open_file(shard_path)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)
    except (OSError, ValueError):
        stream = ipc.open_stream(shard_path)
        for batch in stream:
            yield batch


def _decode_image_entry(entry) -> Optional[np.ndarray]:
    """Decode one HF ``Image`` feature cell into an HWC uint8 RGB array."""
    if isinstance(entry, dict):
        raw = entry.get("bytes")
        if not raw:
            path = entry.get("path")
            if not path:
                return None
            with open(path, "rb") as f:
                raw = f.read()
        pil = Image.open(io.BytesIO(raw))
    elif isinstance(entry, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(entry))
    else:
        arr = np.asarray(entry, dtype=np.uint8)
        pil = Image.fromarray(arr)
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def load_source_pool(
    *,
    num_samples: int,
    split: str = "validation",
    cache_dir: str = DEFAULT_HFDS_CACHE_DIR,
    dataset_subdir: str = "ilsvrc___imagenet-1k",
    seed: int = 0,
    shuffle: bool = True,
) -> List[SourcePoolEntry]:
    """Read ``num_samples`` held-out ImageNet-1k samples from the Arrow cache.

    Samples are drawn deterministically: we enumerate rows shard-by-shard in
    sorted shard order, assign each ``(shard, row)`` a stable ``pool_index``,
    and optionally shuffle ``pool_index`` with :class:`numpy.random.Generator`
    seeded by ``seed``.  The first ``num_samples`` entries of that shuffled
    order are materialized (image bytes + label) and returned.

    Labels are read from the ``label`` column when present; the validation
    and test splits of ILSVRC's HF export carry integer class indices.
    """
    pattern = os.path.join(
        cache_dir, dataset_subdir, "*", "*", "*",
        f"*-{split}-*.arrow",
    )
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No Arrow shard found matching {pattern!r}. "
            f"Check cache_dir and split."
        )

    # The HF Arrow cache sometimes contains multiple fingerprint directories
    # for the same dataset revision. Collapse to one shard per basename so we
    # don't double-count rows.
    _by_base: dict = {}
    for s in shards:
        _by_base.setdefault(os.path.basename(s), s)
    shards = sorted(_by_base.values())

    # Two-pass scheme: (1) index (shard, row) tuples without decoding image
    # bytes, (2) shuffle, truncate, decode only the selected rows.
    index: List[tuple] = []
    has_label_cache: dict = {}
    import pyarrow.ipc as ipc  # noqa: F401  - already imported lazily above

    for shard in shards:
        n_rows = 0
        has_label = False
        for batch in _iter_arrow_batches(shard):
            if n_rows == 0:
                has_label = "label" in batch.schema.names
            n_rows += batch.num_rows
        has_label_cache[shard] = has_label
        for row in range(n_rows):
            index.append((shard, row))

    if len(index) < num_samples:
        raise RuntimeError(
            f"Only {len(index)} rows available in split={split!r}; "
            f"need {num_samples}."
        )

    order = np.arange(len(index), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(order)
    chosen = order[:num_samples]

    # Group selections per shard so we stream each shard at most once.
    # ``per_shard[shard]`` is a list of ``(row, pool_idx)`` tuples -- we
    # keep ``row`` as the key so the ``dict(...)`` below produces a
    # ``{row -> pool_idx}`` mapping (the decoder loop below depends on it).
    per_shard: dict = {}
    for pool_idx, flat_idx in enumerate(chosen.tolist()):
        shard, row = index[flat_idx]
        per_shard.setdefault(shard, []).append((row, pool_idx))

    out: List[Optional[SourcePoolEntry]] = [None] * num_samples
    for shard in sorted(per_shard.keys()):
        wanted = dict(per_shard[shard])  # row -> pool_idx
        wanted_set = set(wanted.keys())
        cursor = 0
        remaining = dict(wanted)  # row -> pool_idx
        for batch in _iter_arrow_batches(shard):
            if not remaining:
                break
            image_col = batch.column("image")
            label_col = batch.column("label") if has_label_cache[shard] else None
            batch_rows = batch.num_rows
            # Which rows within this batch do we want?
            hits = [
                r for r in range(batch_rows)
                if (cursor + r) in wanted_set and (cursor + r) in remaining
            ]
            for r in hits:
                global_row = cursor + r
                pool_idx = remaining.pop(global_row)
                entry = image_col[r].as_py()
                arr = _decode_image_entry(entry)
                if arr is None:
                    raise RuntimeError(
                        f"Failed to decode image at {shard}:{global_row}"
                    )
                label = -1
                if label_col is not None:
                    lv = label_col[r].as_py()
                    if lv is not None:
                        label = int(lv)
                out[pool_idx] = SourcePoolEntry(
                    pool_index=pool_idx,
                    shard_path=shard,
                    row_index=global_row,
                    label=label,
                    image=arr,
                )
            cursor += batch_rows

    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise RuntimeError(
            f"Failed to read {len(missing)} sample(s) from Arrow cache "
            f"(first missing pool indices: {missing[:5]})."
        )

    _logger.info(
        "Loaded %d source samples from split=%s (across %d shards).",
        num_samples, split, len(per_shard),
    )
    return [e for e in out if e is not None]


def entry_to_tensor(
    entry: SourcePoolEntry,
    *,
    size: int,
    crop_pct: float = 0.95,
) -> torch.Tensor:
    """Apply validation-style resize + center-crop to a pool entry.

    Matches :func:`evaluation.plots.augmentation_showcase._resize_centercrop`
    so the diagnostic pipeline preprocesses identically to the rest of the
    repo's clean / validation path (no training-time augmentations).
    """
    import torch.nn.functional as F  # local import to keep module import light

    t = torch.from_numpy(entry.image).permute(2, 0, 1).contiguous().float() / 255.0
    C, H, W = t.shape
    resize_to = int(round(size / float(crop_pct)))
    if H < W:
        new_h = resize_to
        new_w = int(round(W * (resize_to / H)))
    else:
        new_w = resize_to
        new_h = int(round(H * (resize_to / W)))
    out = F.interpolate(
        t.unsqueeze(0), size=(new_h, new_w),
        mode="bilinear", align_corners=False, antialias=True,
    ).squeeze(0)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return out[:, top:top + size, left:left + size].contiguous()


def pool_to_preprocessed(
    pool: Sequence[SourcePoolEntry],
    *,
    size: int,
    crop_pct: float = 0.95,
) -> List[torch.Tensor]:
    """Convenience: preprocess every pool entry at ``size x size``."""
    return [entry_to_tensor(e, size=size, crop_pct=crop_pct) for e in pool]
