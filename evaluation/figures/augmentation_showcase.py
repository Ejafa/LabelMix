#!/usr/bin/env python3
"""Render the augmentation-showcase *images* for the LabelMix paper.

This script emits raw per-image PNGs only — the final figures (captions,
column labels, etc.) are composed in LaTeX later.  Four image sets are
produced, all reading from the *same* pool of source images so that
visual differences between augmentations are attributable to the
augmentation itself, not to a different image crop:

    1. ``aug_comparison_{clean,vit_wee}/`` — one PNG per
       (augmentation, panel) tile, for augmentations in
       ``{none, mixup, cutmix, mosaic, labelmix}``.  ``clean`` applies a
       resize + center-crop only; ``vit_wee`` additionally applies the
       ViT-wee ImageNet-1k training augmentations on top
       (see ``experiments/labelmix_imagenet1k/configs/vit-wee.yaml``).

    2. ``labelmix_randomness/`` — N PNGs of LabelMix with ``alpha=0.5``,
       ``k=6`` and ``sampling_max_aspect=20``, one per seed; no
       single-image augmentation (color jitter, RandAugment,
       random-erasing, shear, ...) is applied.

    3. ``labelmix_k_sweep/`` — one PNG per ``k`` in ``[2, 10]``, all
       with ``alpha=0.5`` and ``sampling_max_aspect=20``.

    4. ``labelmix_alpha_sweep/`` — one PNG per ``alpha`` in
       ``[0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0, 5.0]``, all with ``k=6`` and
       ``sampling_max_aspect=20``.

Every PNG is written at two resolutions (256x256 and 1024x1024) and a
render is skipped if the output PNG already exists.  The random seed is
fixed so the same image set is used across all augmentations; this also
guarantees that re-rendering (after manually deleting a cached PNG)
yields the exact same picture.

Each image subdirectory contains a ``metadata.json`` describing the
scene so a downstream LLM can propose figure/file captions without
rediscovering the generator's semantics.

Source images
-------------
ImageNet-1k is not always cached locally.  The renderer tries, in order:

    1. ``--source-dir <dir>`` — any directory of ``.jpg/.jpeg/.png``
       files; images are picked in sorted-filename order (deterministic).
    2. ``--hfds-cache-dir`` — a HuggingFace ``datasets`` Arrow cache
       (default: the shared ILSVRC ImageNet-1k cache).
    3. ``--use-timm-dataset`` — pull the first ``N`` training samples
       via ``timm.data.create_dataset`` with the same ``hfds/...``
       spec used in training.  Requires the dataset cache to be
       populated.

If no source yields enough images, the renderer prints a clear error.

Output
------
``evaluation/data/processed/figures/augmentation_showcase/``
    ├── aug_comparison_clean/
    │   ├── metadata.json
    │   ├── none__panel1_256.png  / _1024.png
    │   ├── mixup__panel1_256.png / _1024.png
    │   ├── ...
    ├── aug_comparison_vit_wee/        (same layout as above)
    ├── labelmix_randomness/
    │   ├── metadata.json
    │   ├── panel1_256.png / _1024.png
    │   └── ...
    ├── labelmix_k_sweep/
        ├── metadata.json
        ├── k2_256.png / _1024.png
        └── ...
    └── labelmix_alpha_sweep/
        ├── metadata.json
        ├── alpha0_05_256.png / _1024.png
        └── ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Make the repo root importable when running this file as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from timm.data import Mixup, create_transform  # noqa: E402
from timm.data.balanced_dataset import (  # noqa: E402
    _apply_box_symmetry,
    _apply_box_symmetry_rect,
    _boxes_are_valid_and_tile,
    _fallback_stripes_boxes,
    _layout_to_pixel_boxes,
)
from timm.data.labelmix_layout import squarify_core  # noqa: E402
from timm.data.mosaic_dataset import MosaicDataset  # noqa: E402

from evaluation.common.paths import FIGURES_DIR, ensure_dirs  # noqa: E402

_logger = logging.getLogger("augmentation_showcase")


# ---------------------------------------------------------------------------
# Constants mirroring the ViT-wee ImageNet-1k training config.
# ---------------------------------------------------------------------------

# ``experiments/labelmix_imagenet1k/configs/vit-wee.yaml``:
#   img_size: 256, interpolation (default): bicubic, train_interpolation: random
#   scale: [0.08, 1.0], ratio: [0.75, 1.333], hflip: 0.5, vflip: 0.0
#   color_jitter: null, aa: rand-m6-inc1-mstd1.0-n3
#   reprob: 0.2, remode: pixel, recount: 1, crop_pct: 0.95
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

VIT_WEE_IMG_SIZE: int = 256
VIT_WEE_AUG = dict(
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    hflip=0.5,
    vflip=0.0,
    color_jitter=None,
    auto_augment="rand-m6-inc1-mstd1.0-n3",
    interpolation="random",
    re_prob=0.2,
    re_mode="pixel",
    re_count=1,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
)

# LabelMix defaults used by the paper's main experiments.
LABELMIX_ALPHA: float = 0.5
LABELMIX_K: int = 6
LABELMIX_MAX_ASPECT: float = 20.0
LABELMIX_ALPHA_SWEEP: Tuple[float, ...] = (0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0, 5.0)

# Mixup/CutMix defaults mirroring vit-wee.yaml.
MIXUP_ALPHA: float = 0.8
CUTMIX_ALPHA: float = 1.0

# How many source images we need at the most to render every figure.
# - k-sweep needs up to k=10
# - randomness uses 8 panels * K=6 but each panel draws from a disjoint
#   slice so we oversample; same for aug_comparison Mosaic (4 per panel).
# 128 is plenty and cheap to load.
NUM_SOURCE_IMAGES: int = 128

# Default on-disk HuggingFace ``datasets`` cache for ILSVRC ImageNet-1k.
# Laid out as ``<root>/ilsvrc___imagenet-1k/default/0.0.0/<fingerprint>/``
# with ``imagenet-1k-{train,validation,test}-*.arrow`` shard files.
DEFAULT_HFDS_CACHE_DIR: str = (
    "/apdcephfs_fsgm/share_303853033/ethangeng/"
    "konstantin-garbers/data/imagenet-1k"
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_images_from_dir(
    source_dir: str,
    num: int,
    input_size: int,
) -> List[torch.Tensor]:
    """Load the first ``num`` images from ``source_dir``, sorted by filename,
    as CHW float tensors in [0, 1] at native resolution (no crop/resize).
    Returned images will be resized/cropped later by the per-augmentation
    transform.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"source dir does not exist: {source_dir}")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(
        f for f in os.listdir(source_dir)
        if f.lower().endswith(exts)
    )
    if len(files) < num:
        raise RuntimeError(
            f"Need {num} source images, only found {len(files)} in {source_dir}. "
            f"Point --source-dir at a directory with at least {num} images."
        )
    imgs: List[torch.Tensor] = []
    for fname in files[:num]:
        path = os.path.join(source_dir, fname)
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        imgs.append(t)
    _logger.info("Loaded %d source images from %s", len(imgs), source_dir)
    return imgs


def _load_images_from_hfds_arrow(
    cache_dir: str,
    num: int,
    split: str = "train",
    dataset_subdir: str = "ilsvrc___imagenet-1k",
) -> List[torch.Tensor]:
    """Load ``num`` images directly from an on-disk HuggingFace Arrow cache.

    Reads ``<cache_dir>/<dataset_subdir>/*/*/*/<ds>-<split>-*.arrow`` with
    pyarrow's IPC reader (no dependency on the ``datasets`` Image feature
    decoding semantics, and no need to build the full HF cache index).
    Each row's ``image`` column is a struct ``{'bytes': <PNG/JPEG>,
    'path': <str>}`` — we decode ``bytes`` through PIL.

    The default cache layout at ``/apdcephfs_fsgm/.../data/imagenet-1k``
    is::

        ilsvrc___imagenet-1k/default/0.0.0/<fingerprint>/
            imagenet-1k-train-00000-of-00267.arrow
            imagenet-1k-train-00001-of-00267.arrow
            ...
    """
    import glob
    import io

    import pyarrow.ipc as ipc  # type: ignore

    pattern = os.path.join(
        cache_dir, dataset_subdir, "*", "*", "*",
        f"*-{split}-*.arrow",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Arrow shard found matching {pattern!r}. "
            f"Check --hfds-cache-dir and --hfds-split."
        )

    imgs: List[torch.Tensor] = []
    for shard_path in matches:
        if len(imgs) >= num:
            break
        _logger.info("Reading from %s (have %d/%d)", shard_path, len(imgs), num)
        # HF writes Arrow IPC *file* format (with footer), but older
        # shards may use *stream* format; try both.
        try:
            reader = ipc.open_file(shard_path)
            num_batches = reader.num_record_batches
            batches = (reader.get_batch(i) for i in range(num_batches))
        except (OSError, ValueError):
            stream = ipc.open_stream(shard_path)
            batches = iter(stream)
        for batch in batches:
            image_col = batch.column("image")
            for i in range(batch.num_rows):
                if len(imgs) >= num:
                    break
                entry = image_col[i].as_py()
                # entry is {'bytes': <raw>, 'path': <str>} for HF
                # ``Image`` feature with decode=False.
                if isinstance(entry, dict):
                    raw = entry.get("bytes")
                    if not raw:
                        path = entry.get("path")
                        if not path:
                            continue
                        with open(path, "rb") as f:
                            raw = f.read()
                    pil = Image.open(io.BytesIO(raw))
                elif isinstance(entry, (bytes, bytearray)):
                    pil = Image.open(io.BytesIO(entry))
                else:
                    # Already-decoded array (unlikely here).
                    arr = np.asarray(entry, dtype=np.uint8)
                    pil = Image.fromarray(arr)
                arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
                imgs.append(
                    torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
                )
            if len(imgs) >= num:
                break

    if len(imgs) < num:
        raise RuntimeError(
            f"Only {len(imgs)} samples available across {len(matches)} shards "
            f"for split={split!r}; need {num}. Try a smaller --num-source-images."
        )
    _logger.info(
        "Loaded %d source images from Arrow cache (%s/%s).",
        len(imgs), dataset_subdir, split,
    )
    return imgs


def _load_images_from_timm(
    dataset_spec: str,
    data_dir: Optional[str],
    split: str,
    input_key: str,
    num: int,
) -> List[torch.Tensor]:
    """Pull ``num`` training samples from ``timm.data.create_dataset``.

    Falls back to the on-disk reader in ``timm.data.readers`` so this
    path only works if the dataset was pre-copied (see
    ``copy_data_to_ram.py``).
    """
    from timm.data import create_dataset  # local import to keep top clean

    ds = create_dataset(
        dataset_spec,
        root=data_dir,
        split=split,
        is_training=True,
        input_key=input_key,
        target_key=None,
    )
    imgs: List[torch.Tensor] = []
    for i, (img, _tgt) in enumerate(ds):
        if i >= num:
            break
        # The timm dataset returns a PIL image when ``transform=None``.
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"timm dataset returned {type(img)} for sample {i}; "
                "expected PIL.Image (set transform=None)."
            )
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0)
    if len(imgs) < num:
        raise RuntimeError(
            f"timm dataset {dataset_spec!r} yielded only {len(imgs)}/{num} samples"
        )
    _logger.info("Loaded %d source images via timm(%s)", len(imgs), dataset_spec)
    return imgs


# ---------------------------------------------------------------------------
# Clean (no-aug) normalization to a fixed canvas.
# ---------------------------------------------------------------------------


def _resize_centercrop(img: torch.Tensor, size: int, crop_pct: float = 0.95) -> torch.Tensor:
    """Resize the shorter side to ``size/crop_pct`` and center-crop to
    ``size x size``.  Matches the validation-side pipeline of timm.
    """
    C, H, W = img.shape
    resize_to = int(round(size / float(crop_pct)))
    if H < W:
        new_h = resize_to
        new_w = int(round(W * (resize_to / H)))
    else:
        new_w = resize_to
        new_h = int(round(H * (resize_to / W)))
    out = F.interpolate(
        img.unsqueeze(0), size=(new_h, new_w),
        mode="bilinear", align_corners=False, antialias=True,
    ).squeeze(0)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return out[:, top:top + size, left:left + size].contiguous()


def _normalize(
    img: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    m = torch.tensor(list(mean), dtype=img.dtype).view(-1, 1, 1)
    s = torch.tensor(list(std), dtype=img.dtype).view(-1, 1, 1)
    return (img - m) / s


def _denormalize(
    img: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    m = torch.tensor(list(mean), dtype=img.dtype).view(-1, 1, 1)
    s = torch.tensor(list(std), dtype=img.dtype).view(-1, 1, 1)
    return (img * s + m).clamp(0.0, 1.0)


def _build_clean_transform(size: int, crop_pct: float = 0.95) -> Callable[[Image.Image], torch.Tensor]:
    """A deterministic transform: resize + center crop + normalize."""
    def _apply(pil: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        t = _resize_centercrop(t, size=size, crop_pct=crop_pct)
        return _normalize(t)
    return _apply


def _build_vit_wee_transform(size: int = VIT_WEE_IMG_SIZE) -> Callable[[Image.Image], torch.Tensor]:
    """The vit-wee ImageNet-1k training transform, returning a normalized CHW tensor."""
    return create_transform(
        input_size=(3, size, size),
        is_training=True,
        no_aug=False,
        scale=VIT_WEE_AUG["scale"],
        ratio=VIT_WEE_AUG["ratio"],
        hflip=VIT_WEE_AUG["hflip"],
        vflip=VIT_WEE_AUG["vflip"],
        color_jitter=VIT_WEE_AUG["color_jitter"],
        auto_augment=VIT_WEE_AUG["auto_augment"],
        interpolation=VIT_WEE_AUG["interpolation"],
        mean=VIT_WEE_AUG["mean"],
        std=VIT_WEE_AUG["std"],
        re_prob=VIT_WEE_AUG["re_prob"],
        re_mode=VIT_WEE_AUG["re_mode"],
        re_count=VIT_WEE_AUG["re_count"],
        re_num_splits=0,
        use_prefetcher=False,
        separate=False,
    )


def _apply_transform(
    imgs: List[torch.Tensor],
    transform: Callable[[Image.Image], torch.Tensor],
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for t in imgs:
        # Reconstruct a PIL image at the tensor's native size.
        arr = (t.permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        pil = Image.fromarray(arr)
        out.append(transform(pil))
    return out


# ---------------------------------------------------------------------------
# LabelMix compose (distilled from ``BalancedBucketDataset._mix_group_labelmix``
# and ``_layout_is_valid``).  Reimplemented in-place so the figure generator
# doesn't need to instantiate a full BalancedBucketDataset (which requires
# an actual HF arrow dataset on disk).
# ---------------------------------------------------------------------------


def _sample_labelmix_layout(
    *,
    alpha: float,
    k: int,
    H: int,
    W: int,
    sampling_min_side_px: int,
    sampling_max_aspect: float,
    max_attempts: int,
) -> torch.Tensor:
    """Rejection-sample a Dirichlet(alpha) layout and return the *pixel*
    boxes directly (ASC order), so the caller can paste without ever
    re-running :func:`squarify_core` (whose internal RNG would otherwise
    produce a different rendered layout from the one that was validated).

    Mirrors ``BalancedBucketDataset._layout_is_valid`` and
    ``_sample_dirichlet_layout``.  If after ``max_attempts`` the strict
    constraints still fail, the aspect constraint is progressively
    relaxed (aspect -> 2x, 4x, inf) before finally falling back to
    uniform weights.  This is needed because for large ``k`` combined
    with small ``alpha`` (Dirichlet(α=0.5, k=10) is extremely skewed),
    ``sampling_max_aspect=20`` is near-unsatisfiable and the original
    ``torch.full(k, 1/k)`` fallback would silently hide rendering bugs.

    Returns ``(K, 4)`` int64 pixel boxes in ASC-weight order, ready for
    :func:`_apply_box_symmetry`.
    """
    def _attempt_with(aspect_cap: float) -> Optional[torch.Tensor]:
        dirichlet = torch.distributions.Dirichlet(torch.full((k,), alpha))
        for _ in range(max(1, max_attempts)):
            w = dirichlet.sample().to(dtype=torch.float32)
            w_desc, _ = torch.sort(w, descending=True)
            base_layout_desc = squarify_core(w_desc, canvas_size=1.0)
            base_layout_asc = torch.flip(base_layout_desc, dims=[0])
            boxes = _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)
            if not _boxes_are_valid_and_tile(boxes, H=H, W=W):
                continue
            x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            widths = (x1 - x0).to(dtype=torch.float32)
            heights = (y1 - y0).to(dtype=torch.float32)
            min_side = torch.minimum(widths, heights)
            max_side = torch.maximum(widths, heights)
            if sampling_min_side_px > 0 and torch.any(min_side < float(sampling_min_side_px)):
                continue
            if aspect_cap > 0.0:
                aspect = max_side / torch.clamp(min_side, min=1.0)
                if torch.any(aspect > float(aspect_cap)):
                    continue
            return boxes
        return None

    if alpha <= 0.0:
        w_desc = torch.full((k,), 1.0 / k, dtype=torch.float32)
        base_layout_desc = squarify_core(w_desc, canvas_size=1.0)
        base_layout_asc = torch.flip(base_layout_desc, dims=[0])
        return _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)

    # Strict constraint first.
    for cap in (sampling_max_aspect, sampling_max_aspect * 2.0, sampling_max_aspect * 4.0, 0.0):
        boxes = _attempt_with(cap)
        if boxes is not None:
            if cap != sampling_max_aspect:
                _logger.info(
                    "LabelMix rejection sampling relaxed aspect<=%.1f for k=%d alpha=%.2f.",
                    cap if cap > 0 else float('inf'), k, alpha,
                )
            return boxes

    _logger.warning(
        "LabelMix rejection sampling exhausted even the relaxed budget for k=%d; "
        "falling back to uniform layout.", k,
    )
    w_desc = torch.full((k,), 1.0 / k, dtype=torch.float32)
    base_layout_desc = squarify_core(w_desc, canvas_size=1.0)
    base_layout_asc = torch.flip(base_layout_desc, dims=[0])
    return _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)


def _compose_labelmix(
    imgs: Sequence[torch.Tensor],
    *,
    alpha: float,
    k: int,
    sampling_max_aspect: float,
    sampling_min_side_px: int = 6,
    max_attempts: int = 200,
    shift: int = 0,
    use_symmetries: bool = True,
) -> torch.Tensor:
    """Produce a single LabelMix composite from ``k`` source images.

    Reimplements the core of
    ``BalancedBucketDataset._mix_group_labelmix`` for a single output
    (``shift=0``), without the K-way circular batch construction or the
    D4 symmetry caching layer.  For the paper figure we always render
    the ``shift=0`` tile and pick D4 symmetries per slot iff
    ``use_symmetries=True`` (this matches what the training loop does).
    """
    if len(imgs) != k:
        raise ValueError(f"_compose_labelmix expects exactly k={k} images, got {len(imgs)}")

    batch = torch.stack(list(imgs), dim=0).contiguous()
    _, C, H, W = batch.shape
    is_square = H == W

    # 1) Sample a validated pixel layout in one shot (no second squarify
    #    call, whose internal randomness would produce a different layout
    #    from the one that passed validation and potentially leave gaps).
    base_boxes = _sample_labelmix_layout(
        alpha=alpha, k=k, H=H, W=W,
        sampling_min_side_px=sampling_min_side_px,
        sampling_max_aspect=sampling_max_aspect,
        max_attempts=max_attempts,
    )
    if not _boxes_are_valid_and_tile(base_boxes, H=H, W=W):
        # Defensive fallback: guaranteed-tiling vertical stripes with
        # uniform widths.  Never expected to trigger but keeps the
        # canvas from ever rendering as grey.
        base_boxes = _fallback_stripes_boxes(
            torch.full((k,), 1.0 / k, dtype=torch.float32), H=H, W=W,
        )

    # 2) Pick ONE D4 symmetry for this output and transform every slot box
    #    with it (this matches ``_mix_group_labelmix``: per output ``s`` a
    #    single ``sym_choices[s]`` is applied to the *whole* canvas, so the
    #    transformed boxes still tile the canvas).  Picking a different
    #    symmetry per slot — as an earlier version of this helper did —
    #    breaks tiling and leaves uncovered pixels that render as grey
    #    squares in the output.
    if use_symmetries:
        if is_square:
            sym = int(torch.randint(0, 8, (1,)).item())
        else:
            allowed = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
            sym = int(allowed[torch.randint(0, allowed.numel(), (1,))].item())
    else:
        sym = 0

    if is_square:
        boxes_sym = _apply_box_symmetry(base_boxes, sym=sym, S=W)
    else:
        boxes_sym = _apply_box_symmetry_rect(base_boxes, sym=sym, H=H, W=W)

    # 3) Paste each slot's image into the output canvas at its sym'd box.
    out = batch.new_zeros((C, H, W))
    for slot_i in range(k):
        src_idx = (slot_i + shift) % k
        x0 = int(boxes_sym[slot_i, 0].item())
        y0 = int(boxes_sym[slot_i, 1].item())
        x1 = int(boxes_sym[slot_i, 2].item())
        y1 = int(boxes_sym[slot_i, 3].item())
        th = max(1, y1 - y0)
        tw = max(1, x1 - x0)
        src = batch[src_idx].unsqueeze(0)
        patch = F.interpolate(src, size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)
        out[:, y0:y1, x0:x1] = patch
    return out


# ---------------------------------------------------------------------------
# Mixup / CutMix compose (single-output wrapper around timm.data.Mixup).
# ---------------------------------------------------------------------------


def _compose_mixup(
    imgs: Sequence[torch.Tensor],
    *,
    alpha: float,
    num_classes: int = 1000,
) -> torch.Tensor:
    """Classic Mixup: ``out = lam * a + (1 - lam) * b``."""
    if len(imgs) < 2:
        raise ValueError("Mixup needs at least 2 images")
    batch = torch.stack([imgs[0], imgs[1]], dim=0).contiguous()
    targets = torch.tensor([0, 1], dtype=torch.int64)
    mixup = Mixup(
        mixup_alpha=alpha, cutmix_alpha=0.0,
        prob=1.0, switch_prob=0.0, mode="batch",
        label_smoothing=0.0, num_classes=num_classes,
    )
    mixed, _ = mixup(batch.clone(), targets)
    return mixed[0]


def _compose_cutmix(
    imgs: Sequence[torch.Tensor],
    *,
    alpha: float,
    num_classes: int = 1000,
) -> torch.Tensor:
    """Classic CutMix: paste a random bbox of b into a."""
    if len(imgs) < 2:
        raise ValueError("CutMix needs at least 2 images")
    batch = torch.stack([imgs[0], imgs[1]], dim=0).contiguous()
    targets = torch.tensor([0, 1], dtype=torch.int64)
    mixup = Mixup(
        mixup_alpha=0.0, cutmix_alpha=alpha,
        prob=1.0, switch_prob=1.0, mode="batch",
        label_smoothing=0.0, num_classes=num_classes,
    )
    mixed, _ = mixup(batch.clone(), targets)
    return mixed[0]


# ---------------------------------------------------------------------------
# Mosaic compose (wraps MosaicDataset, which needs an IterableDataset base).
# ---------------------------------------------------------------------------


class _ListIterableDataset(torch.utils.data.IterableDataset):
    """Minimal IterableDataset returning (img, label) for a fixed list."""

    def __init__(self, imgs: Sequence[torch.Tensor]) -> None:
        super().__init__()
        self.imgs = list(imgs)

    def __iter__(self):
        for i, img in enumerate(self.imgs):
            yield img, i

    def __len__(self) -> int:
        return len(self.imgs)


def _compose_mosaic(
    imgs: Sequence[torch.Tensor],
    *,
    output_size: Tuple[int, int],
    center_ratio: Tuple[float, float] = (0.8, 1.2),
    post_scale: Optional[Tuple[float, float]] = (0.8, 1.2),
    fill_value: float = 114.0 / 255.0,
    seed: int = 0,
) -> torch.Tensor:
    """Run a single Mosaic composition on the first 4 images.

    The underlying ``MosaicDataset`` expects an IterableDataset and mixes
    in images lazily as it iterates; for a deterministic figure we feed
    it exactly 4 normalized tensors and take the first yielded sample.
    """
    if len(imgs) < 4:
        raise ValueError("Mosaic needs at least 4 images")
    base = _ListIterableDataset(list(imgs[:4]))
    mosaic = MosaicDataset(
        base_dataset=base,
        output_size=output_size,
        prob=1.0,
        center_ratio_range=center_ratio,
        post_scale_range=post_scale,
        close_epochs=0,
        total_epochs=None,
        fill_value=fill_value,
        seed=seed,
    )
    for img, _tgt in mosaic:
        return img
    raise RuntimeError("MosaicDataset yielded no samples")


# ---------------------------------------------------------------------------
# Per-image PNG writers.
# ---------------------------------------------------------------------------


def _tensor_to_uint8_hwc(
    t: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """CHW float (possibly normalized) -> HWC uint8 numpy.

    Heuristic: if the tensor is already in ``[0, 1]`` we treat it as
    already-denormalized (e.g. a straight-through "none" sample pulled
    from the clean transform); otherwise we assume it is normalized with
    ImageNet mean/std and denormalize before quantizing.
    """
    assert t.ndim == 3, f"expected CHW, got {tuple(t.shape)}"
    if float(t.min()) >= -0.05 and float(t.max()) <= 1.05 and float(t.max()) > 0.6:
        img = t.clamp(0.0, 1.0)
    else:
        img = _denormalize(t, mean=mean, std=std)
    return (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _resize_uint8(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 array to (size, size) using PIL bicubic.

    We always render compositions at ``cfg.img_size`` (256) and then
    upsample to the requested display resolution.  This keeps the
    augmentation pipeline at one canonical size and makes the 256 / 1024
    versions pixel-exact siblings.
    """
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    im = Image.fromarray(arr)
    im = im.resize((size, size), resample=Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def _save_png(arr_hwc_uint8: np.ndarray, path: str, *, overwrite: bool = False) -> None:
    """Write an HWC uint8 array to ``path`` (parent dir created on demand).

    Idempotent by default: does nothing if ``path`` already exists.
    """
    if os.path.exists(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(arr_hwc_uint8).save(path, format="PNG", optimize=False)


def _save_tensor_at_sizes(
    t: torch.Tensor,
    dir_path: str,
    stem: str,
    tile_sizes: Sequence[int],
    *,
    overwrite: bool = False,
) -> List[str]:
    """Save ``t`` (CHW float) as ``dir_path/<stem>_<size>.png`` for each size.

    Returns the list of paths that were written or already existed.
    """
    base = _tensor_to_uint8_hwc(t)
    written: List[str] = []
    for tile_px in tile_sizes:
        out_path = os.path.join(dir_path, f"{stem}_{tile_px}.png")
        if overwrite or not os.path.exists(out_path):
            _save_png(_resize_uint8(base, tile_px), out_path, overwrite=overwrite)
            _logger.info("Wrote %s", out_path)
        written.append(out_path)
    return written


def _write_metadata(dir_path: str, meta: Dict) -> None:
    """Write a concise ``metadata.json`` suitable for prompting an LLM.

    The schema is intentionally minimal and stable:

        {
          "group":        str,          # e.g. "aug_comparison_clean"
          "purpose":      str,          # one-line human/LLM hint
          "source":       str,          # e.g. "ImageNet-1k train"
          "img_size":     int,          # compose resolution
          "tile_sizes":   [int, ...],   # per-file output sizes
          "params":       { ... },      # numeric aug params
          "files":        [             # one entry per (stem) tile
            {
              "stem":    str,
              "aug":     str,           # optional
              "panel":   int|None,
              "k":       int|None,
              "notes":   str,
            },
            ...
          ]
        }
    """
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
        f.write("\n")
    _logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Per-figure builders.
# ---------------------------------------------------------------------------


@dataclass
class ShowcaseConfig:
    source_images: List[torch.Tensor]
    img_size: int = VIT_WEE_IMG_SIZE
    seed: int = 42
    out_dir: str = str(FIGURES_DIR / "augmentation_showcase")
    tile_sizes: Tuple[int, ...] = (256, 1024)
    source_name: str = "ImageNet-1k"

    def dir_for(self, group: str) -> str:
        """Return the per-figure subdirectory for ``group``."""
        return os.path.join(self.out_dir, group)


# Canonical augmentation ordering used by the comparison figure.
_AUG_ROW_ORDER = ("none", "mixup", "cutmix", "mosaic", "labelmix")
# Fixed per-aug salt so seeds are reproducible across Python runs
# (``hash(str)`` is randomized by PYTHONHASHSEED by default).
_AUG_SEED_SALT = {
    "none":     0,
    "mixup":    101,
    "cutmix":   202,
    "mosaic":   303,
    "labelmix": 404,
}

# Short per-aug notes used inside metadata.json; kept concise so a
# downstream LLM can cite them verbatim when producing figure captions.
_AUG_NOTES = {
    "none":     "Unmodified source image (no augmentation).",
    "mixup":    f"Classic Mixup: out = lam*a + (1-lam)*b with alpha={MIXUP_ALPHA}.",
    "cutmix":   f"Classic CutMix: paste a random bbox of b into a with alpha={CUTMIX_ALPHA}.",
    "mosaic":   "4-tile Mosaic augmentation (square output).",
    "labelmix": (
        f"LabelMix with alpha={LABELMIX_ALPHA}, k={LABELMIX_K}, "
        f"sampling_max_aspect={LABELMIX_MAX_ASPECT}."
    ),
}


def _panel_for_aug(
    aug: str,
    *,
    pre_imgs: List[torch.Tensor],
    img_size: int,
    panel_seed: int,
    num_panels: int,
) -> List[torch.Tensor]:
    """Return a list of ``num_panels`` composite tensors for the given aug.

    ``pre_imgs`` is already transform'd (size-normalized + optionally
    augmented).  Each panel consumes a fresh, disjoint chunk of
    ``pre_imgs`` so the visible content across augmentations is as
    different as the aug itself (rather than being the same source
    image over and over).
    """
    out: List[torch.Tensor] = []
    # Budget per panel, per aug:
    #   none     = 1, mixup = 2, cutmix = 2, mosaic = 4, labelmix = K
    per_panel = {
        "none": 1,
        "mixup": 2,
        "cutmix": 2,
        "mosaic": 4,
        "labelmix": LABELMIX_K,
    }[aug]
    salt = _AUG_SEED_SALT[aug]
    for p in range(num_panels):
        # Seed per (aug, panel) so randomness of each augmentation is
        # reproducible and independent of row order.
        panel_rng_seed = panel_seed + p * 1000 + salt
        torch.manual_seed(panel_rng_seed)
        np.random.seed(panel_rng_seed)

        start = p * per_panel
        chunk = pre_imgs[start:start + per_panel]
        if len(chunk) < per_panel:
            # Wrap around; stays deterministic because pre_imgs is fixed.
            chunk = (pre_imgs + pre_imgs)[start:start + per_panel]

        if aug == "none":
            out.append(chunk[0])
        elif aug == "mixup":
            out.append(_compose_mixup(chunk, alpha=MIXUP_ALPHA))
        elif aug == "cutmix":
            out.append(_compose_cutmix(chunk, alpha=CUTMIX_ALPHA))
        elif aug == "mosaic":
            out.append(_compose_mosaic(
                chunk, output_size=(img_size, img_size),
                seed=panel_seed + p,
            ))
        elif aug == "labelmix":
            out.append(_compose_labelmix(
                chunk,
                alpha=LABELMIX_ALPHA, k=LABELMIX_K,
                sampling_max_aspect=LABELMIX_MAX_ASPECT,
            ))
        else:
            raise ValueError(aug)
    return out


def render_aug_comparison(
    cfg: ShowcaseConfig,
    *,
    flavor: str,
    num_panels: int = 4,
) -> None:
    """Set 1: one PNG per (augmentation, panel) for the given flavor.

    ``flavor`` is either ``"clean"`` (resize + center-crop only) or
    ``"vit_wee"`` (additionally applies the ViT-wee ImageNet-1k
    training-time single-image augmentations).  Output directory:
    ``<out>/aug_comparison_<flavor>/``.
    """
    group = f"aug_comparison_{flavor}"
    dir_path = cfg.dir_for(group)
    os.makedirs(dir_path, exist_ok=True)

    # Build the per-image transform.
    if flavor == "clean":
        transform = _build_clean_transform(cfg.img_size)
        flavor_desc = "resize-and-center-crop only; no single-image augmentations"
    elif flavor == "vit_wee":
        transform = _build_vit_wee_transform(cfg.img_size)
        flavor_desc = (
            "ViT-wee ImageNet-1k training-time single-image augmentations: "
            "RandomResizedCrop(scale=(0.08, 1.0), ratio=(0.75, 1.333)), "
            "hflip=0.5, rand-m6-inc1-mstd1.0-n3, reprob=0.2"
        )
    else:
        raise ValueError(f"unknown flavor: {flavor}")

    # Skip compositing entirely if every PNG already exists.
    all_paths_exist = True
    for aug in _AUG_ROW_ORDER:
        for p in range(num_panels):
            stem = f"{aug}__panel{p+1}"
            for tp in cfg.tile_sizes:
                if not os.path.exists(os.path.join(dir_path, f"{stem}_{tp}.png")):
                    all_paths_exist = False
                    break
            if not all_paths_exist:
                break
        if not all_paths_exist:
            break

    files_meta: List[Dict] = []
    if not all_paths_exist:
        # NOTE: re-seed BEFORE applying transforms so the vit-wee RRC / aa
        # crops are reproducible per (flavor, seed).
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        pre_imgs = _apply_transform(cfg.source_images, transform)

        for aug in _AUG_ROW_ORDER:
            panels = _panel_for_aug(
                aug, pre_imgs=pre_imgs,
                img_size=cfg.img_size,
                panel_seed=cfg.seed,
                num_panels=num_panels,
            )
            for p, tensor in enumerate(panels):
                stem = f"{aug}__panel{p+1}"
                _save_tensor_at_sizes(tensor, dir_path, stem, cfg.tile_sizes)
                files_meta.append({
                    "stem": stem,
                    "aug": aug,
                    "panel": p + 1,
                    "notes": _AUG_NOTES[aug],
                })
    else:
        # We still want metadata.json refreshed even if PNGs were cached.
        _logger.info("%s: all PNGs already present, refreshing metadata only.", group)
        for aug in _AUG_ROW_ORDER:
            for p in range(num_panels):
                files_meta.append({
                    "stem": f"{aug}__panel{p+1}",
                    "aug": aug,
                    "panel": p + 1,
                    "notes": _AUG_NOTES[aug],
                })

    _write_metadata(dir_path, {
        "group": group,
        "purpose": (
            "Side-by-side comparison of data augmentations on shared source "
            "images; used to illustrate what each method does in input space."
        ),
        "flavor": flavor,
        "flavor_description": flavor_desc,
        "source": f"{cfg.source_name} (shared source images across augmentations)",
        "img_size": cfg.img_size,
        "tile_sizes": list(cfg.tile_sizes),
        "num_panels_per_aug": num_panels,
        "augmentations": list(_AUG_ROW_ORDER),
        "params": {
            "mixup_alpha": MIXUP_ALPHA,
            "cutmix_alpha": CUTMIX_ALPHA,
            "mosaic_tiles": 4,
            "labelmix_alpha": LABELMIX_ALPHA,
            "labelmix_k": LABELMIX_K,
            "labelmix_max_aspect": LABELMIX_MAX_ASPECT,
        },
        "files": files_meta,
    })


def render_labelmix_randomness(
    cfg: ShowcaseConfig,
    *,
    num_panels: int = 8,
) -> None:
    """Set 2: independent LabelMix draws to illustrate layout randomness.

    Output directory: ``<out>/labelmix_randomness/``, one PNG per panel
    per resolution.  Single-image augmentations are *not* applied.
    """
    group = "labelmix_randomness"
    dir_path = cfg.dir_for(group)
    os.makedirs(dir_path, exist_ok=True)

    transform = _build_clean_transform(cfg.img_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pre_imgs = _apply_transform(cfg.source_images, transform)
    shared_chunk = pre_imgs[:LABELMIX_K]
    if len(shared_chunk) < LABELMIX_K:
        shared_chunk = (pre_imgs * (LABELMIX_K // len(pre_imgs) + 1))[:LABELMIX_K]

    metadata_path = os.path.join(dir_path, "metadata.json")
    overwrite_stale_cache = False
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                old_meta = json.load(f)
            old_params = old_meta.get("params", {}) if isinstance(old_meta, dict) else {}
            overwrite_stale_cache = not bool(old_params.get("shared_source_images_across_panels"))
        except (OSError, json.JSONDecodeError):
            overwrite_stale_cache = True

    files_meta: List[Dict] = []
    for p in range(num_panels):
        stem = f"panel{p+1:02d}"
        existing = [
            os.path.exists(os.path.join(dir_path, f"{stem}_{tp}.png"))
            for tp in cfg.tile_sizes
        ]
        if overwrite_stale_cache or not all(existing):
            # Deterministic, independent seed per panel.
            torch.manual_seed(cfg.seed + p * 7919)
            np.random.seed(cfg.seed + p * 7919)
            composite = _compose_labelmix(
                shared_chunk,
                alpha=LABELMIX_ALPHA, k=LABELMIX_K,
                sampling_max_aspect=LABELMIX_MAX_ASPECT,
            )
            _save_tensor_at_sizes(
                composite,
                dir_path,
                stem,
                cfg.tile_sizes,
                overwrite=overwrite_stale_cache,
            )
        files_meta.append({
            "stem": stem,
            "aug": "labelmix",
            "panel": p + 1,
            "notes": (
                "Independent LabelMix draw using the same source images as "
                "every other panel; differs only in seed/layout."
            ),
        })

    _write_metadata(dir_path, {
        "group": group,
        "purpose": (
            "Illustrates the randomness of LabelMix layouts: every PNG uses "
            "the same source images and the same (alpha, k, aspect-cap) "
            "hyperparameters; only the random seed changes."
        ),
        "source": f"{cfg.source_name}, clean (resize + center-crop) transform only",
        "img_size": cfg.img_size,
        "tile_sizes": list(cfg.tile_sizes),
        "num_panels": num_panels,
        "params": {
            "labelmix_alpha": LABELMIX_ALPHA,
            "labelmix_k": LABELMIX_K,
            "labelmix_max_aspect": LABELMIX_MAX_ASPECT,
            "single_image_aug": False,
            "shared_source_images_across_panels": True,
        },
        "files": files_meta,
    })


def render_labelmix_k_sweep(
    cfg: ShowcaseConfig,
    *,
    k_values: Sequence[int] = tuple(range(2, 11)),
) -> None:
    """Set 3: one LabelMix PNG per ``k`` value, same alpha/aspect cap.

    Output directory: ``<out>/labelmix_k_sweep/``.
    """
    group = "labelmix_k_sweep"
    dir_path = cfg.dir_for(group)
    os.makedirs(dir_path, exist_ok=True)

    transform = _build_clean_transform(cfg.img_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pre_imgs = _apply_transform(cfg.source_images, transform)

    files_meta: List[Dict] = []
    for i, k in enumerate(k_values):
        stem = f"k{k:02d}"
        existing = [
            os.path.exists(os.path.join(dir_path, f"{stem}_{tp}.png"))
            for tp in cfg.tile_sizes
        ]
        if not all(existing):
            # Re-seed per k so the layout isn't identical across panels.
            torch.manual_seed(cfg.seed + i * 13)
            np.random.seed(cfg.seed + i * 13)
            chunk = pre_imgs[:k]
            if len(chunk) < k:
                chunk = (pre_imgs * (k // len(pre_imgs) + 1))[:k]
            composite = _compose_labelmix(
                chunk, alpha=LABELMIX_ALPHA, k=k,
                sampling_max_aspect=LABELMIX_MAX_ASPECT,
            )
            _save_tensor_at_sizes(composite, dir_path, stem, cfg.tile_sizes)
        files_meta.append({
            "stem": stem,
            "aug": "labelmix",
            "k": int(k),
            "notes": f"LabelMix composite with k={k} source images.",
        })

    _write_metadata(dir_path, {
        "group": group,
        "purpose": (
            "Shows how LabelMix layouts scale with the number of mixed "
            "images k; all other hyperparameters are held fixed."
        ),
        "source": f"{cfg.source_name}, clean (resize + center-crop) transform only",
        "img_size": cfg.img_size,
        "tile_sizes": list(cfg.tile_sizes),
        "k_values": list(int(k) for k in k_values),
        "params": {
            "labelmix_alpha": LABELMIX_ALPHA,
            "labelmix_max_aspect": LABELMIX_MAX_ASPECT,
            "single_image_aug": False,
        },
        "files": files_meta,
    })


def _alpha_stem(alpha: float) -> str:
    """Stable filename stem for an alpha value."""
    return f"alpha{alpha:g}".replace(".", "_").replace("-", "m")


def render_labelmix_alpha_sweep(
    cfg: ShowcaseConfig,
    *,
    alpha_values: Sequence[float] = LABELMIX_ALPHA_SWEEP,
) -> None:
    """Set 4: one LabelMix PNG per ``alpha`` value, same k/aspect cap.

    Output directory: ``<out>/labelmix_alpha_sweep/``.  Every panel uses
    the same source images and same RNG seed base so visible differences
    are attributable to the Dirichlet concentration alpha.
    """
    group = "labelmix_alpha_sweep"
    dir_path = cfg.dir_for(group)
    os.makedirs(dir_path, exist_ok=True)

    transform = _build_clean_transform(cfg.img_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pre_imgs = _apply_transform(cfg.source_images, transform)
    shared_chunk = pre_imgs[:LABELMIX_K]
    if len(shared_chunk) < LABELMIX_K:
        shared_chunk = (pre_imgs * (LABELMIX_K // len(pre_imgs) + 1))[:LABELMIX_K]

    files_meta: List[Dict] = []
    for alpha in alpha_values:
        alpha = float(alpha)
        stem = _alpha_stem(alpha)
        existing = [
            os.path.exists(os.path.join(dir_path, f"{stem}_{tp}.png"))
            for tp in cfg.tile_sizes
        ]
        if not all(existing):
            # Same seed base for every alpha: alpha changes the Dirichlet
            # distribution, not the chosen source images or high-level seed.
            torch.manual_seed(cfg.seed + 104729)
            np.random.seed(cfg.seed + 104729)
            composite = _compose_labelmix(
                shared_chunk,
                alpha=alpha, k=LABELMIX_K,
                sampling_max_aspect=LABELMIX_MAX_ASPECT,
            )
            _save_tensor_at_sizes(composite, dir_path, stem, cfg.tile_sizes)
        files_meta.append({
            "stem": stem,
            "aug": "labelmix",
            "alpha": alpha,
            "notes": (
                f"LabelMix composite with alpha={alpha:g}; source images, "
                f"k={LABELMIX_K}, aspect cap, and seed base are fixed."
            ),
        })

    _write_metadata(dir_path, {
        "group": group,
        "purpose": (
            "Shows how LabelMix layouts change as the Dirichlet concentration "
            "alpha changes; all panels use the same source images and fixed k."
        ),
        "source": f"{cfg.source_name}, clean (resize + center-crop) transform only",
        "img_size": cfg.img_size,
        "tile_sizes": list(cfg.tile_sizes),
        "alpha_values": [float(a) for a in alpha_values],
        "params": {
            "labelmix_k": LABELMIX_K,
            "labelmix_max_aspect": LABELMIX_MAX_ASPECT,
            "single_image_aug": False,
            "shared_source_images_across_panels": True,
            "fixed_seed_base_across_alpha": True,
        },
        "files": files_meta,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help="Directory of at least %d source images (.jpg/.png). "
             "If set, used instead of the HF Arrow cache." % NUM_SOURCE_IMAGES,
    )
    parser.add_argument(
        "--source-name", type=str, default=None,
        help="Human-readable source name for metadata.json. If omitted, "
             "derived from --source-dir when possible.",
    )
    parser.add_argument(
        "--hfds-cache-dir", type=str, default=DEFAULT_HFDS_CACHE_DIR,
        help="Root of the HuggingFace ``datasets`` Arrow cache for "
             "ILSVRC ImageNet-1k (layout: "
             "<root>/ilsvrc___imagenet-1k/default/0.0.0/<fingerprint>/). "
             "Used as the default source of images.",
    )
    parser.add_argument(
        "--hfds-split", type=str, default="train",
        choices=("train", "validation", "test"),
        help="Split to read from --hfds-cache-dir.",
    )
    parser.add_argument(
        "--use-timm-dataset", type=str, default=None,
        metavar="DATASET_SPEC",
        help="If set, pull source images via timm's create_dataset with this "
             "spec (e.g. 'hfds/ILSVRC/imagenet-1k').  Requires the dataset "
             "cache to be populated on disk.",
    )
    parser.add_argument(
        "--timm-data-dir", type=str, default=None,
        help="Data-dir arg for --use-timm-dataset (e.g. '/dev/shm/imagenet-1k').",
    )
    parser.add_argument(
        "--timm-split", type=str, default="train",
        help="Split to read from when --use-timm-dataset is given.",
    )
    parser.add_argument(
        "--timm-input-key", type=str, default="image",
        help="Image field name within the timm dataset.",
    )
    parser.add_argument(
        "--num-source-images", type=int, default=NUM_SOURCE_IMAGES,
        help="Number of source images to load from --source-dir / --use-timm-dataset.",
    )
    parser.add_argument(
        "--img-size", type=int, default=VIT_WEE_IMG_SIZE,
        help="Canonical compose resolution (all augs run at this size; "
             "upsampling to 1024 happens at render time).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed. Used as the fixed seed for the image-pool order "
             "and for per-panel RNG streams.",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(FIGURES_DIR / "augmentation_showcase"),
        help="Where to write PNG figures (directory will be created).",
    )
    parser.add_argument(
        "--num-comparison-panels", type=int, default=4,
        help="Panels per row in aug_comparison (figure 1).",
    )
    parser.add_argument(
        "--num-randomness-panels", type=int, default=8,
        help="Panels in labelmix_randomness (figure 2).",
    )
    parser.add_argument(
        "--k-sweep", type=int, nargs="+", default=list(range(2, 11)),
        help="k values to render in labelmix_k_sweep (figure 3).",
    )
    parser.add_argument(
        "--alpha-sweep", type=float, nargs="+", default=list(LABELMIX_ALPHA_SWEEP),
        help="alpha values to render in labelmix_alpha_sweep (figure 4). "
             "Rendered by default.",
    )
    parser.add_argument(
        "--tile-sizes", type=int, nargs="+", default=[256, 1024],
        help="Per-tile output resolutions; one PNG per size per figure.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable INFO logging.",
    )
    return parser.parse_args()


def _resolve_source_images(args: argparse.Namespace) -> List[torch.Tensor]:
    if args.source_dir:
        return _load_images_from_dir(
            args.source_dir,
            num=args.num_source_images,
            input_size=args.img_size,
        )
    if args.use_timm_dataset:
        return _load_images_from_timm(
            dataset_spec=args.use_timm_dataset,
            data_dir=args.timm_data_dir,
            split=args.timm_split,
            input_key=args.timm_input_key,
            num=args.num_source_images,
        )
    if args.hfds_cache_dir and os.path.isdir(args.hfds_cache_dir):
        return _load_images_from_hfds_arrow(
            cache_dir=args.hfds_cache_dir,
            num=args.num_source_images,
            split=args.hfds_split,
        )
    raise SystemExit(
        "No source of images available. Either:\n"
        "  - point --source-dir at a directory of images, or\n"
        "  - point --hfds-cache-dir at a HuggingFace datasets Arrow cache "
        f"(default: {DEFAULT_HFDS_CACHE_DIR}), or\n"
        "  - pass --use-timm-dataset hfds/ILSVRC/imagenet-1k "
        "--timm-data-dir /dev/shm/imagenet-1k."
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ensure_dirs()
    os.makedirs(args.out_dir, exist_ok=True)

    images = _resolve_source_images(args)

    source_name = args.source_name
    if source_name is None:
        source_name = "ImageNet-1k"
        if args.source_dir:
            source_dir_name = os.path.basename(os.path.normpath(args.source_dir)).lower()
            if "picsum" in source_dir_name:
                source_name = "Lorem Picsum"
            elif "commons" in source_dir_name:
                source_name = "Wikimedia Commons object photos"
            elif "imagenet" in source_dir_name and "animal" in source_dir_name:
                source_name = "Hugging Face ImageNet-1k animal samples"
            elif "imagenet" in source_dir_name:
                source_name = "Hugging Face ImageNet-1k samples"

    cfg = ShowcaseConfig(
        source_images=images,
        img_size=args.img_size,
        seed=args.seed,
        out_dir=args.out_dir,
        tile_sizes=tuple(args.tile_sizes),
        source_name=source_name,
    )

    # Figure 1 (two flavors).
    render_aug_comparison(cfg, flavor="clean", num_panels=args.num_comparison_panels)
    render_aug_comparison(cfg, flavor="vit_wee", num_panels=args.num_comparison_panels)
    # Figure 2.
    render_labelmix_randomness(cfg, num_panels=args.num_randomness_panels)
    # Figure 3.
    render_labelmix_k_sweep(cfg, k_values=tuple(args.k_sweep))
    # Figure 4.
    render_labelmix_alpha_sweep(cfg, alpha_values=tuple(args.alpha_sweep))


if __name__ == "__main__":
    main()
