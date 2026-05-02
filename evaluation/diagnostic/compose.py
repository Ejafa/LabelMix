"""LabelMix composition + patch-mask extraction for diagnostic evaluation.

The training-time LabelMix pipeline lives in
``timm.data.balanced_dataset.BalancedBucketDataset._mix_group_labelmix``.
For diagnostic evaluation we only need a single composed sample (no K-way
circular batch, no target mixing), so we reuse the distilled single-output
helpers in :mod:`evaluation.figures.augmentation_showcase` and wrap them so
that we additionally return:

    * ``slot_boxes``     -- (k, 4) int64 pixel boxes (x0, y0, x1, y1) in
                            source-rank order ("slot i" -> source i),
                            already transformed by the D4 symmetry applied
                            to the canvas.
    * ``patch_mask``     -- (H, W) int16 map where each pixel stores the
                            source/slot index that was pasted there.  This
                            is what the downstream analysis needs to build
                            per-patch-class segmentation evaluations.
    * ``patch_area_ratios`` -- (k,) float32 fraction of total canvas area
                            covered by each slot's box.

The composition logic (layout sampling, D4 symmetry choice, paste order) is
identical to the training path: we call the exact same helpers from the
showcase module so there is no logic drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F

from evaluation.figures.augmentation_showcase import (
    _apply_box_symmetry,
    _apply_box_symmetry_rect,
    _boxes_are_valid_and_tile,
    _fallback_stripes_boxes,
    _sample_labelmix_layout,
)


@dataclass
class ComposedSample:
    """Output of :func:`compose_labelmix_with_mask`.

    Attributes:
        image:              (3, H, W) float32 tensor in ``[0, 1]``.
        patch_mask:         (H, W) int16 map; pixel value = slot index in
                            ``0..k-1``.
        slot_boxes:         (k, 4) int64 pixel boxes ``(x0, y0, x1, y1)``
                            in slot-index order (post D4 symmetry).
        patch_area_ratios:  (k,) float32 fraction of canvas covered.
        symmetry:           int in ``{0..7}``, the D4 transform index.
    """

    image: torch.Tensor
    patch_mask: torch.Tensor
    slot_boxes: torch.Tensor
    patch_area_ratios: torch.Tensor
    symmetry: int


def compose_labelmix_with_mask(
    imgs: Sequence[torch.Tensor],
    *,
    alpha: float,
    k: int,
    sampling_max_aspect: float,
    sampling_min_side_px: int = 6,
    max_attempts: int = 200,
    shift: int = 0,
    use_symmetries: bool = True,
) -> ComposedSample:
    """Compose ``k`` preprocessed source tensors into one LabelMix canvas.

    This mirrors :func:`evaluation.figures.augmentation_showcase._compose_labelmix`
    (which in turn mirrors the training path), and *additionally* materializes
    the per-pixel patch mask + per-slot boxes/areas needed for diagnostic
    analysis.  All RNG is driven by the ambient ``torch``/``numpy`` state so
    callers can fix a seed outside and get deterministic layouts.
    """
    if len(imgs) != k:
        raise ValueError(
            f"compose_labelmix_with_mask expects exactly k={k} images, got {len(imgs)}"
        )

    batch = torch.stack(list(imgs), dim=0).contiguous()
    _, C, H, W = batch.shape
    is_square = H == W

    # 1) Sample a validated pixel layout (ASC-weight order), mirroring
    #    balanced_dataset._sample_dirichlet_layout.
    base_boxes = _sample_labelmix_layout(
        alpha=alpha, k=k, H=H, W=W,
        sampling_min_side_px=sampling_min_side_px,
        sampling_max_aspect=sampling_max_aspect,
        max_attempts=max_attempts,
    )
    if not _boxes_are_valid_and_tile(base_boxes, H=H, W=W):
        base_boxes = _fallback_stripes_boxes(
            torch.full((k,), 1.0 / k, dtype=torch.float32), H=H, W=W,
        )

    # 2) Pick one D4 symmetry for the whole canvas (matches training).
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

    # 3) Paste each slot into the canvas at its symmetry-transformed box, and
    #    simultaneously paint the slot index into the patch mask.
    out = batch.new_zeros((C, H, W))
    patch_mask = torch.full((H, W), -1, dtype=torch.int16)
    slot_boxes_out = torch.zeros((k, 4), dtype=torch.int64)
    areas = torch.zeros((k,), dtype=torch.float32)
    canvas_area = float(H * W)

    for slot_i in range(k):
        src_idx = (slot_i + shift) % k
        x0 = int(boxes_sym[slot_i, 0].item())
        y0 = int(boxes_sym[slot_i, 1].item())
        x1 = int(boxes_sym[slot_i, 2].item())
        y1 = int(boxes_sym[slot_i, 3].item())
        th = max(1, y1 - y0)
        tw = max(1, x1 - x0)
        src = batch[src_idx].unsqueeze(0)
        patch = F.interpolate(
            src, size=(th, tw), mode="bilinear", align_corners=False,
        ).squeeze(0)
        out[:, y0:y1, x0:x1] = patch
        patch_mask[y0:y1, x0:x1] = int(src_idx)
        slot_boxes_out[slot_i] = torch.tensor(
            [x0, y0, x1, y1], dtype=torch.int64,
        )
        areas[slot_i] = float((y1 - y0) * (x1 - x0)) / max(canvas_area, 1.0)

    # Any pixel still unset (shouldn't happen if _boxes_are_valid_and_tile
    # holds) gets assigned to its nearest slot so downstream code never
    # sees sentinel -1.
    if bool((patch_mask < 0).any().item()):
        patch_mask[patch_mask < 0] = 0

    return ComposedSample(
        image=out.clamp(0.0, 1.0),
        patch_mask=patch_mask,
        slot_boxes=slot_boxes_out,
        patch_area_ratios=areas,
        symmetry=sym,
    )
