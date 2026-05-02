"""Post-hoc visualization of diagnostic Gradient x Input attributions.

Consumes the artifacts already produced by
:mod:`evaluation.diagnostic.attribution` (``input.png``, per-class
``class_<c>.pt`` with a ``heatmap`` tensor, and the dataset ``manifest.jsonl``
with ``patch_mask_path`` + ``patch_to_class`` + ``slot_boxes``) and writes, per
present class, two additional PNGs:

    - ``class_<c>_overlay.png``   : heatmap blended over the preprocessed input
    - ``class_<c>_boxed.png``     : overlay + red rectangle(s) outlining the
                                    source patch(es) belonging to class ``c``

No new attribution is computed here; this module is pure post-processing and
can be re-run cheaply with different colormap/alpha choices.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from evaluation.common import DIAGNOSTIC_DIR, setup_logging

_logger = logging.getLogger("evaluation.diagnostic.visualize")


# ---------------------------------------------------------------------------
# Manifest access
# ---------------------------------------------------------------------------


@dataclass
class _ManifestRow:
    sample_id: str
    image_path: str        # absolute path to composed source image (source coords)
    patch_mask_path: str   # absolute path to source-coord patch mask PNG (int16-like)
    patch_to_class: Dict[int, int]
    slot_boxes: List[List[int]]
    symmetry: int
    merged_class_area_ratios: Dict[str, float]


def _load_manifest_rows(manifest_path: str) -> Dict[str, _ManifestRow]:
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    by_id: Dict[str, _ManifestRow] = {}
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img_path = r["image_path"]
            mask_path = r["patch_mask_path"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(manifest_dir, img_path)
            if not os.path.isabs(mask_path):
                mask_path = os.path.join(manifest_dir, mask_path)
            by_id[str(r["sample_id"])] = _ManifestRow(
                sample_id=str(r["sample_id"]),
                image_path=img_path,
                patch_mask_path=mask_path,
                patch_to_class={int(k): int(v) for k, v in r["patch_to_class"].items()},
                slot_boxes=[[int(x) for x in box] for box in r["slot_boxes"]],
                symmetry=int(r.get("symmetry", 0)),
                merged_class_area_ratios={
                    str(k): float(v) for k, v in r["merged_class_area_ratios"].items()
                },
            )
    return by_id


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


def _cmap_lookup(name: str):
    try:
        import matplotlib.cm as mpl_cm
        return mpl_cm.get_cmap(name)
    except Exception:  # pragma: no cover
        return None


def _resize_heatmap(heatmap: torch.Tensor, target_hw: Tuple[int, int]) -> np.ndarray:
    h = heatmap.detach().float()
    if h.ndim != 2:
        raise ValueError(f"heatmap must be (H, W); got {tuple(h.shape)}")
    h = h.unsqueeze(0).unsqueeze(0)
    h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)
    return h.squeeze(0).squeeze(0).numpy()


def _normalize_heatmap(h: np.ndarray, percentile: float) -> np.ndarray:
    h = np.maximum(h, 0.0)
    if h.size == 0 or not np.isfinite(h).any():
        return np.zeros_like(h, dtype=np.float32)
    p = float(percentile)
    upper = float(np.percentile(h, p)) if 0.0 < p < 100.0 else float(h.max())
    if not np.isfinite(upper) or upper <= 0.0:
        upper = float(h.max()) if h.max() > 0 else 1.0
    return np.clip(h / upper, 0.0, 1.0).astype(np.float32)


def _heatmap_rgba(h_norm: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = _cmap_lookup(cmap_name)
    if cmap is not None:
        return (cmap(h_norm) * 255.0 + 0.5).astype(np.uint8)  # (H, W, 4)
    g = (h_norm * 255.0 + 0.5).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    a = g
    return np.concatenate([rgb, a[..., None]], axis=-1)


def _alpha_blend(base_rgb: np.ndarray, heat_rgba: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a (H, W, 3) uint8 image with a (H, W, 4) uint8 heatmap.

    The heatmap's own alpha channel (set from its intensity) modulates the
    blend so near-zero attribution leaves the pixel almost untouched.
    """
    base = base_rgb.astype(np.float32) / 255.0
    heat_rgb = heat_rgba[..., :3].astype(np.float32) / 255.0
    heat_a = heat_rgba[..., 3:4].astype(np.float32) / 255.0
    eff_a = np.clip(heat_a * float(alpha), 0.0, 1.0)
    out = base * (1.0 - eff_a) + heat_rgb * eff_a
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Source <-> input-space coordinate transform
# ---------------------------------------------------------------------------


@dataclass
class _CropXform:
    """Maps a (H_src, W_src) source-space image into the model's input-space.

    Replicates :func:`_build_val_preprocess` from ``diagnostic.infer``:
    resize the shorter side to ``round(min(h, w) / crop_pct)`` keeping aspect,
    then center-crop to ``(h, w)``. For square 256-in, 256-out, crop_pct=0.95
    this becomes a 269x269 resize + 256x256 center crop (6-7 px symmetric
    shave on each side).
    """

    input_h: int
    input_w: int
    crop_pct: float

    def apply_image(self, pil: Image.Image) -> Image.Image:
        w_src, h_src = pil.size
        new_h, new_w = self._resize_dims(h_src, w_src)
        pil_r = pil.resize((new_w, new_h), Image.BICUBIC)
        top = (new_h - self.input_h) // 2
        left = (new_w - self.input_w) // 2
        return pil_r.crop((left, top, left + self.input_w, top + self.input_h))

    def apply_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        """Nearest-neighbour variant that preserves the integer slot ids."""
        h_src, w_src = mask_u8.shape
        new_h, new_w = self._resize_dims(h_src, w_src)
        im = Image.fromarray(mask_u8, mode="L").resize((new_w, new_h), Image.NEAREST)
        arr = np.array(im, dtype=np.uint8, copy=True)
        top = (new_h - self.input_h) // 2
        left = (new_w - self.input_w) // 2
        return arr[top:top + self.input_h, left:left + self.input_w]

    def _resize_dims(self, h_src: int, w_src: int) -> Tuple[int, int]:
        short_target = int(round(min(self.input_h, self.input_w) / float(self.crop_pct)))
        if h_src < w_src:
            new_h = short_target
            new_w = int(round(w_src * (short_target / h_src)))
        else:
            new_w = short_target
            new_h = int(round(h_src * (short_target / w_src)))
        return new_h, new_w


# ---------------------------------------------------------------------------
# Box extraction from patch mask (robust to any symmetry transform)
# ---------------------------------------------------------------------------


def _class_boxes_from_mask(
    mask_input_space: np.ndarray,
    patch_to_class: Mapping[int, int],
    target_class: int,
) -> List[Tuple[int, int, int, int]]:
    """For the given target class, return list of (x0, y0, x1, y1) rectangles
    covering each contiguous source patch mapped to that class. We take per
    slot_id the tight axis-aligned bbox of its pixels in input-space."""
    boxes: List[Tuple[int, int, int, int]] = []
    for slot_id, cls in patch_to_class.items():
        if int(cls) != int(target_class):
            continue
        ys, xs = np.where(mask_input_space == int(slot_id))
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        boxes.append((x0, y0, x1, y1))
    return boxes


def _draw_boxes(
    img: Image.Image,
    boxes: Sequence[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1) in boxes:
        # PIL ImageDraw.rectangle needs inclusive lower-right.
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class VisualizeConfig:
    attributions_dir: str
    manifest_path: str
    cmap: str = "inferno"
    percentile: float = 99.0
    alpha: float = 0.85
    box_color: Tuple[int, int, int] = (255, 0, 0)
    box_width: int = 3
    overwrite: bool = False
    # Visualization base. When True we use the preprocessed ``input.png`` that
    # sits next to the heatmap (pixel-perfect alignment with the heatmap).
    # When False we use the original composed source image at source-resolution
    # and transform heatmap + mask back into source coords.
    use_input_png: bool = True


def _find_model_dirs(attributions_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(attributions_dir)):
        p = os.path.join(attributions_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "index.json")):
            out.append(p)
    return out


def _infer_xform_from_logits(
    attributions_dir: str,
    model_slug: str,
) -> Optional[_CropXform]:
    """Look up the per-model ``logits.pt`` data_config in a sibling directory.

    Expected layout::
        evaluation/data/raw/diagnostic/
            attributions/<model_slug>/...
            logits/<model_slug>/logits.pt

    Returns None when ``logits.pt`` is missing; in that case the caller should
    default to (256, 256, 0.95).
    """
    root = os.path.dirname(os.path.abspath(attributions_dir))
    logits_path = os.path.join(root, "logits", model_slug, "logits.pt")
    if not os.path.isfile(logits_path):
        return None
    blob = torch.load(logits_path, map_location="cpu", weights_only=False)
    dc = blob.get("data_config") or {}
    isize = list(dc.get("input_size") or [3, 256, 256])
    return _CropXform(
        input_h=int(isize[1]),
        input_w=int(isize[2]),
        crop_pct=float(dc.get("crop_pct") or 0.95),
    )


def visualize_model(model_dir: str, rows: Mapping[str, _ManifestRow], cfg: VisualizeConfig) -> None:
    index_path = os.path.join(model_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    model_slug = os.path.basename(model_dir.rstrip(os.sep))
    xform = _infer_xform_from_logits(cfg.attributions_dir, model_slug) \
        or _CropXform(input_h=256, input_w=256, crop_pct=0.95)

    samples: Dict[str, Dict] = index.get("samples") or {}
    n_done = 0
    n_overlays = 0
    for sample_id, entry in samples.items():
        row = rows.get(sample_id)
        if row is None:
            _logger.warning("[%s] sample %s missing from manifest -- skipping.",
                            model_slug, sample_id)
            continue

        sample_out_dir = os.path.join(model_dir, sample_id)
        meta_path = os.path.join(sample_out_dir, "_sample.json")

        # --- Build the base image ------------------------------------------
        if cfg.use_input_png:
            input_png_path = os.path.join(sample_out_dir, "input.png")
            if not os.path.isfile(input_png_path):
                _logger.warning("  missing input.png for %s/%s", model_slug, sample_id)
                continue
            base_pil = Image.open(input_png_path).convert("RGB")
        else:
            base_pil = Image.open(row.image_path).convert("RGB")

        base_np = np.array(base_pil, dtype=np.uint8)
        target_hw = base_np.shape[:2]

        # --- Build the patch mask in the same space as base ----------------
        mask_src = np.array(Image.open(row.patch_mask_path).convert("L"), dtype=np.uint8)
        if cfg.use_input_png:
            mask_space = xform.apply_mask(mask_src)
            # Sanity: mask must have same spatial size as the base image.
            if mask_space.shape != target_hw:
                # If the input.png wasn't produced by the exact same xform
                # (shouldn't happen), resize to match.
                mask_space = np.array(
                    Image.fromarray(mask_space, mode="L").resize(
                        (target_hw[1], target_hw[0]), Image.NEAREST),
                    dtype=np.uint8,
                )
        else:
            mask_space = mask_src  # already source-sized

        # --- Per-class overlays --------------------------------------------
        overlay_paths: Dict[str, str] = {}
        boxed_paths: Dict[str, str] = {}
        for cls_str, tensor_rel in (entry.get("tensor_paths") or {}).items():
            cls_id = int(cls_str)
            tensor_path = os.path.join(cfg.attributions_dir, tensor_rel)
            if not os.path.isfile(tensor_path):
                _logger.warning("  missing tensor %s", tensor_path)
                continue

            overlay_path = os.path.join(sample_out_dir, f"class_{cls_id}_overlay.png")
            boxed_path = os.path.join(sample_out_dir, f"class_{cls_id}_boxed.png")
            overlay_paths[cls_str] = os.path.relpath(overlay_path, cfg.attributions_dir)
            boxed_paths[cls_str] = os.path.relpath(boxed_path, cfg.attributions_dir)

            if (not cfg.overwrite
                    and os.path.exists(overlay_path)
                    and os.path.exists(boxed_path)):
                continue

            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            heatmap = payload["heatmap"]  # (H, W)
            if cfg.use_input_png:
                h_resized = _resize_heatmap(heatmap, target_hw)
            else:
                # Map heatmap (input-space) back into source-space: inverse of
                # center-crop+resize. Since the crop is symmetric we just
                # resize with padding first. Simpler and close-enough
                # approximation: scale-up by inverse crop and then resize.
                H_src, W_src = target_hw
                h_input_space = _resize_heatmap(heatmap, (xform.input_h, xform.input_w))
                pad_t = (int(round(xform.input_h / xform.crop_pct)) - xform.input_h) // 2
                pad_l = (int(round(xform.input_w / xform.crop_pct)) - xform.input_w) // 2
                padded_h = xform.input_h + 2 * pad_t
                padded_w = xform.input_w + 2 * pad_l
                padded = np.zeros((padded_h, padded_w), dtype=np.float32)
                padded[pad_t:pad_t + xform.input_h, pad_l:pad_l + xform.input_w] = h_input_space
                h_resized = np.array(
                    Image.fromarray(padded).resize((W_src, H_src), Image.BILINEAR),
                    dtype=np.float32,
                )
            h_norm = _normalize_heatmap(h_resized, cfg.percentile)
            rgba = _heatmap_rgba(h_norm, cfg.cmap)
            # Use the heatmap's own intensity as its alpha so low-attr
            # pixels don't wash out the base image.
            rgba[..., 3] = (h_norm * 255.0 + 0.5).astype(np.uint8)

            overlay_np = _alpha_blend(base_np, rgba, alpha=cfg.alpha)
            Image.fromarray(overlay_np, mode="RGB").save(overlay_path)

            boxes = _class_boxes_from_mask(mask_space, row.patch_to_class, cls_id)
            boxed_pil = _draw_boxes(
                Image.fromarray(overlay_np, mode="RGB"),
                boxes=boxes,
                color=cfg.box_color,
                width=cfg.box_width,
            )
            boxed_pil.save(boxed_path)
            n_overlays += 1

        # --- Update per-sample meta + index --------------------------------
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            meta = {}
        meta["overlay_pngs"] = overlay_paths
        meta["boxed_pngs"] = boxed_paths
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        entry["overlay_pngs"] = overlay_paths
        entry["boxed_pngs"] = boxed_paths
        n_done += 1

    index["overlay_cmap"] = cfg.cmap
    index["overlay_percentile"] = float(cfg.percentile)
    index["overlay_alpha"] = float(cfg.alpha)
    index["samples"] = samples
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    _logger.info("[%s] wrote %d overlay+boxed pairs across %d samples.",
                 model_slug, n_overlays, n_done)


def run(cfg: VisualizeConfig) -> None:
    rows = _load_manifest_rows(cfg.manifest_path)
    model_dirs = _find_model_dirs(cfg.attributions_dir)
    if not model_dirs:
        raise SystemExit(f"No model directories with index.json under {cfg.attributions_dir}")
    _logger.info("Visualizing %d model(s) under %s", len(model_dirs), cfg.attributions_dir)
    for md in model_dirs:
        visualize_model(md, rows, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--attributions-dir", type=str,
        default=str(DIAGNOSTIC_DIR / "attributions"),
        help="Root dir produced by diag_gradient_x_input.",
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to manifest.jsonl produced by diagnostic.generate.",
    )
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--alpha", type=float, default=0.85,
                        help="Max blend weight for heatmap over base image.")
    parser.add_argument("--box-color", default="255,0,0",
                        help="Comma-separated R,G,B triplet for outline colour.")
    parser.add_argument("--box-width", type=int, default=3)
    parser.add_argument("--source-image", action="store_true",
                        help="Overlay on the original composed source image "
                             "instead of the preprocessed input.png.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING")
    try:
        r, g, b = (int(x) for x in args.box_color.split(","))
        box_color = (r, g, b)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--box-color must be R,G,B (got {args.box_color!r}): {exc}")
    cfg = VisualizeConfig(
        attributions_dir=args.attributions_dir,
        manifest_path=args.manifest,
        cmap=str(args.cmap),
        percentile=float(args.percentile),
        alpha=float(args.alpha),
        box_color=box_color,
        box_width=int(args.box_width),
        overwrite=bool(args.overwrite),
        use_input_png=not bool(args.source_image),
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
