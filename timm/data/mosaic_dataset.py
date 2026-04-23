"""Mosaic augmentation dataset wrapper.

YOLOv5-faithful Mosaic for image classification.  Four source images are
resized so their **longer side** equals the mosaic tile length
``S = max(S_h, S_w)`` (aspect preserving — same effective behavior as
YOLOv5's ``load_image`` + letterbox), then pasted at that native size into
a ``2S x 2S`` canvas with a gray fill (``114/255`` by default, same as
YOLOv5), using YOLOv5's exact ``(x1a..x2a, x1b..x2b)`` clipping formulas
around a randomly sampled mosaic center ``(xc, yc)``.  A uniform scale is
then sampled and the final ``S x S`` crop is produced by a single fused
affine (``M = T · R · C``) — equivalent to YOLOv5's
``random_perspective(..., border=(-S/2, -S/2))`` call restricted to
rotation=shear=perspective=translate=0 and a scale knob.  The resulting
sample carries four soft labels whose weights equal the area fraction that
each source tile's **pasted rectangle** (clipped to whatever fit on the
2S x 2S canvas) contributes to the final crop under the affine.

Hyperparameters (all tunable from train.py):

* ``prob`` — probability of applying Mosaic per sample (default 1.0).
* ``center_ratio_range`` — mosaic center ``(xc, yc)`` is sampled on each
  axis as ``S * uniform(lo, hi)`` (our parameterization).  This maps
  exactly onto YOLOv5's ``uniform(-b, 2S + b)`` with border
  ``b = S * (hi - 1)`` when ``lo + hi == 2`` (symmetric range around
  ``S``).  Wider ranges => more unbalanced quadrants.
* ``post_scale_range`` — optional uniform scale ``s ~ U(lo, hi)`` applied
  as part of the final affine.  The affine first recenters the ``2S x
  2S`` canvas on ``(xc, yc)`` (``C``), then scales around that center
  by ``s`` (``R`` with rotation=0), then translates back so the ``S x
  S`` output is produced directly (``T``).  This mirrors YOLOv5's
  ``random_perspective`` with ``degrees=translate=shear=perspective=0``
  and ``scale = s - 1``.  If ``None`` (or ``(1.0, 1.0)``) the affine is
  a pure crop (no resample).
* ``close_epochs`` — disable Mosaic during the last N epochs of training
  ("close-mosaic" schedule from YOLOv8).
* ``total_epochs`` — total training epochs (used in conjunction with
  ``close_epochs`` to decide when to disable).

Output format mirrors LabelMix: ``(image, (labels[4], weights[4]))`` so the
existing :class:`LabelMixSoftTargetCrossEntropy` loss can be reused.  When
Mosaic is skipped (probabilistically or during close-mosaic), the wrapped
single-sample label is returned in the same soft-target format with
``labels=[y, y, y, y]`` and ``weights=[1, 0, 0, 0]`` so the downstream
collate / loss path stays on a single tensor-shape code path.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


class MosaicDataset(IterableDataset):
    """Wraps a single-sample iterable dataset to produce Mosaic outputs."""

    def __init__(
        self,
        base_dataset: IterableDataset,
        output_size: Tuple[int, int],
        prob: float = 1.0,
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        post_scale_range: Optional[Tuple[float, float]] = None,
        close_epochs: int = 0,
        total_epochs: Optional[int] = None,
        fill_value: float = 114.0 / 255.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_dataset, IterableDataset):
            raise TypeError(
                "MosaicDataset requires an IterableDataset base "
                "(e.g. BalancedBucketDataset with labelmix=False)."
            )
        if len(output_size) != 2:
            raise ValueError("output_size must be (H, W)")
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        if len(center_ratio_range) != 2 or center_ratio_range[0] > center_ratio_range[1]:
            raise ValueError("center_ratio_range must be (lo, hi) with lo <= hi")
        if post_scale_range is not None:
            if len(post_scale_range) != 2 or post_scale_range[0] <= 0 or post_scale_range[0] > post_scale_range[1]:
                raise ValueError("post_scale_range must be (lo, hi) with 0 < lo <= hi")
        if close_epochs < 0:
            raise ValueError("close_epochs must be >= 0")

        self.base_dataset = base_dataset
        self.out_h, self.out_w = int(output_size[0]), int(output_size[1])
        self.prob = float(prob)
        self.center_lo, self.center_hi = float(center_ratio_range[0]), float(center_ratio_range[1])
        if post_scale_range is None or tuple(post_scale_range) == (1.0, 1.0):
            self.post_scale_range: Optional[Tuple[float, float]] = None
        else:
            self.post_scale_range = (float(post_scale_range[0]), float(post_scale_range[1]))
        self.close_epochs = int(close_epochs)
        self.total_epochs = int(total_epochs) if total_epochs is not None else None
        self.fill_value = float(fill_value)
        # User-supplied seed salt.  When not None, it is mixed into the
        # per-worker, per-epoch RNG so that consecutive runs launched with
        # the same `--seed` reproduce identical Mosaic crops / centers /
        # post-scales, independent of the DataLoader worker_info.seed path.
        self.seed: Optional[int] = int(seed) if seed is not None else None

        self._epoch = 0

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def _mosaic_enabled_this_epoch(self) -> bool:
        """Return False during the final ``close_epochs`` epochs."""
        if self.close_epochs <= 0 or self.total_epochs is None:
            return True
        cutoff = max(0, self.total_epochs - self.close_epochs)
        return self._epoch < cutoff

    def __len__(self) -> int:  # type: ignore[override]
        if hasattr(self.base_dataset, "__len__"):
            return len(self.base_dataset)  # type: ignore[arg-type]
        raise TypeError("Base dataset has no __len__")

    # ------------------------------------------------------------------
    # Core Mosaic composition
    # ------------------------------------------------------------------

    @staticmethod
    def _as_chw_float(img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 2:
            img = img.unsqueeze(0).expand(3, -1, -1)
        if img.ndim != 3:
            raise ValueError(f"Image tensor must be CHW, got shape {tuple(img.shape)}")
        return img.float() if img.dtype != torch.float32 else img

    @staticmethod
    def _resize_longer_side(img: torch.Tensor, target: int) -> torch.Tensor:
        """YOLOv5-style ``load_image`` resize: scale so the **longer side**
        equals ``target``, preserving aspect ratio.

        This is what YOLOv5's ``load_image`` effectively does (ratio
        ``r = img_size / max(h0, w0)``), and is the Option (b') semantics
        we agreed on: per-tile areas in the ``2S x 2S`` canvas become
        deterministic (each tile occupies at most an ``S x S`` region,
        with smaller dimension leaving gray padding within its quadrant
        just like YOLO).
        """
        C, H, W = img.shape
        longest = max(H, W)
        if longest == target:
            return img
        r = float(target) / float(longest)
        new_h = max(1, int(round(H * r)))
        new_w = max(1, int(round(W * r)))
        return F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

    def _compose_mosaic(
        self,
        imgs: List[torch.Tensor],
        labels: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compose a mosaic and return (image, labels_tensor, weights_tensor).

        YOLOv5-faithful pipeline:

        1. Each of the 4 sources is resized so its longer side equals
           ``S = max(S_h, S_w)`` (aspect preserving).
        2. A ``2S x 2S`` canvas filled with ``fill_value`` (gray in
           ``[0, 1]``-normalized space; the YOLOv5 equivalent is
           ``114/255``) is created.
        3. A mosaic center ``(xc, yc)`` is sampled using our
           ``center_ratio_range`` parameterization: ``xc = S_w * rx``,
           ``yc = S_h * ry`` with ``rx, ry ~ U(center_lo, center_hi)``.
        4. Each source is pasted onto the canvas using YOLOv5's exact
           ``(x1a..x2a, x1b..x2b)`` clipping formulas:

              TL: x1a,y1a,x2a,y2a = max(xc-w, 0), max(yc-h, 0), xc, yc
                  x1b,y1b,x2b,y2b = w-(x2a-x1a), h-(y2a-y1a), w, h
              TR: x1a,y1a,x2a,y2a = xc, max(yc-h, 0), min(xc+w, 2S), yc
                  ...  (see code below)

           The **actually-pasted rectangle** for tile i, i.e.
           ``[x1a, x2a) x [y1a, y2a)`` in canvas coords, is recorded for
           later weight computation.  Tiles that are smaller than ``S``
           on one axis leave gray padding inside their quadrant (same as
           YOLOv5); tiles that would extend past the canvas are clipped
           at the border.
        5. A uniform scale ``s ~ U(lo, hi)`` is sampled (``s = 1`` if
           ``post_scale_range is None``).  The final affine, mirroring
           YOLOv5's ``random_perspective`` with
           ``degrees=shear=perspective=translate=0`` and
           ``border=(-S_h/2, -S_w/2)``, is

              M = T · R · C

           with
              C = [[1, 0, -W_c/2], [0, 1, -H_c/2], [0, 0, 1]]  # recenter canvas
              R = [[s, 0, 0],      [0, s, 0],      [0, 0, 1]]  # scale
              T = [[1, 0, out_w/2],[0, 1, out_h/2],[0, 0, 1]]  # recenter output

           This matches YOLOv5 literally: ``C[0,2] = -im.shape[1]/2``
           (here ``im.shape[1] = W_c = 2*S_w``) and
           ``T[0,2] = uniform(0.5-translate, 0.5+translate) * width``
           with ``width = out_w`` and ``translate=0`` ⇒ ``T[0,2] = out_w/2``.
           **Crucially, the sampled mosaic center ``(xc, yc)`` does
           *not* enter the affine** — it enters only through **where
           each source tile was pasted** on the canvas (step 4).  The
           warp always re-centers the canvas center onto the output
           center, so ``scale = 1`` produces the canonical mosaic
           (``S x S`` crop of the canvas center), and tiles pasted
           off-center (large ``|xc - S_w|``) simply contribute less
           area to the crop.  The output size is ``S_h x S_w``
           directly; the crop and the warp happen in a single
           ``grid_sample`` call.
        6. Per-tile area weights are computed by forward-mapping each
           pasted rectangle through ``M`` and intersecting with
           ``[0, S_w) x [0, S_h)``.  Because ``M`` is a uniform scale
           plus translation, the mapped rectangles remain axis-aligned
           and the intersection is closed-form.
        """
        if len(imgs) != 4 or len(labels) != 4:
            raise ValueError("Mosaic requires exactly 4 images/labels")

        C = imgs[0].shape[0]
        S_h, S_w = self.out_h, self.out_w
        # YOLOv5 uses a single square ``s = img_size`` for both canvas
        # axes and the final crop.  For (typically square) classification
        # outputs we pick the longer side as the YOLO-equivalent ``s``;
        # the final ``S_h x S_w`` crop is produced directly by the fused
        # affine in step 5 regardless of whether S_h == S_w.
        S = max(S_h, S_w)
        canvas_h, canvas_w = 2 * S, 2 * S

        # 1) Resize each source longer-side-to-S (YOLO load_image).
        tiles: List[torch.Tensor] = []
        for img in imgs:
            img = self._as_chw_float(img)
            if img.shape[0] != C:
                if img.shape[0] == 1 and C == 3:
                    img = img.expand(3, -1, -1)
                else:
                    raise ValueError("Inconsistent image channels in mosaic group")
            tiles.append(self._resize_longer_side(img, S))

        # 2) Allocate the 2S x 2S canvas filled with `fill_value`.  The
        #    YOLOv5 equivalent is gray 114 in [0, 255]; in our normalized
        #    float pipeline this corresponds to 114/255 ≈ 0.447, but we
        #    keep the user-supplied `fill_value` default (0.0) because
        #    downstream normalization isn't controlled here.
        canvas = torch.full(
            (C, canvas_h, canvas_w), self.fill_value,
            dtype=tiles[0].dtype,
        )

        # 3) Sample mosaic center on the 2S x 2S canvas using our
        #    center_ratio_range parameterization.  YOLOv5 samples
        #    ``xc = int(uniform(-b, 2s + b))`` with
        #    ``mosaic_border = -s // 2`` hard-coded, i.e.
        #    ``xc ~ U(s/2, 3s/2)``.  Our default ``center_ratio_range
        #    = (0.5, 1.5)`` reproduces this exactly since we sample
        #    ``xc = int(S * uniform(0.5, 1.5)) ∈ [S/2, 3S/2]``.  Wider
        #    ranges ⇒ more unbalanced quadrants; narrower ⇒ closer to
        #    a balanced 25/25/25/25 split.
        rx = random.uniform(self.center_lo, self.center_hi)
        ry = random.uniform(self.center_lo, self.center_hi)
        xc = int(round(S * rx))
        yc = int(round(S * ry))

        # 4) Paste each tile at native size with YOLOv5's clipping.
        #    Record the actual canvas-space rectangle each tile occupies
        #    (for weight computation in step 6).
        pasted_rects: List[Tuple[int, int, int, int]] = []
        for i, tile in enumerate(tiles):
            h, w = tile.shape[-2], tile.shape[-1]
            if i == 0:  # TL
                x1a = max(xc - w, 0);          y1a = max(yc - h, 0)
                x2a = xc;                       y2a = yc
                x1b = w - (x2a - x1a);          y1b = h - (y2a - y1a)
                x2b = w;                        y2b = h
            elif i == 1:  # TR
                x1a = xc;                       y1a = max(yc - h, 0)
                x2a = min(xc + w, canvas_w);    y2a = yc
                x1b = 0;                        y1b = h - (y2a - y1a)
                x2b = min(w, x2a - x1a);        y2b = h
            elif i == 2:  # BL
                x1a = max(xc - w, 0);           y1a = yc
                x2a = xc;                       y2a = min(canvas_h, yc + h)
                x1b = w - (x2a - x1a);          y1b = 0
                x2b = w;                        y2b = min(y2a - y1a, h)
            else:  # BR
                x1a = xc;                       y1a = yc
                x2a = min(xc + w, canvas_w);    y2a = min(canvas_h, yc + h)
                x1b = 0;                        y1b = 0
                x2b = min(w, x2a - x1a);        y2b = min(y2a - y1a, h)

            if x2a > x1a and y2a > y1a and x2b > x1b and y2b > y1b:
                canvas[:, y1a:y2a, x1a:x2a] = tile[:, y1b:y2b, x1b:x2b]
                pasted_rects.append((x1a, y1a, x2a, y2a))
            else:
                # Tile contributed no pixels (center at canvas edge).
                pasted_rects.append((0, 0, 0, 0))

        # 5) Sample the post-mosaic scale and do the fused affine + crop.
        if self.post_scale_range is not None:
            s = random.uniform(self.post_scale_range[0], self.post_scale_range[1])
        else:
            s = 1.0

        # Forward pixel-space mapping (canvas -> output), matching YOLOv5
        # random_perspective with degrees=shear=perspective=translate=0
        # and border=(-S_h/2, -S_w/2):
        #   (x, y) -> ( s * (x - W_c/2) + out_w/2, s * (y - H_c/2) + out_h/2 )
        # i.e. recenter canvas on its middle, scale, then place canvas
        # middle at output middle.  The sampled (xc, yc) does NOT enter
        # here — it only controlled tile placement in step 4.
        out = self._warp_and_crop(canvas, scale=s, out_h=S_h, out_w=S_w,
                                   fill_value=self.fill_value)

        # 6) Area weights from forward-mapped pasted rectangles ∩ [0,S).
        Wc_half = canvas_w / 2.0
        Hc_half = canvas_h / 2.0
        outw_half = S_w / 2.0
        outh_half = S_h / 2.0
        weights = torch.zeros(4, dtype=torch.float32)
        for i, (px1, py1, px2, py2) in enumerate(pasted_rects):
            if px2 <= px1 or py2 <= py1:
                continue
            # forward mapping of the pasted rectangle (axis-aligned):
            tx0 = s * (px1 - Wc_half) + outw_half
            tx1 = s * (px2 - Wc_half) + outw_half
            ty0 = s * (py1 - Hc_half) + outh_half
            ty1 = s * (py2 - Hc_half) + outh_half
            ix0 = max(tx0, 0.0);          ix1 = min(tx1, float(S_w))
            iy0 = max(ty0, 0.0);          iy1 = min(ty1, float(S_h))
            w_ = max(0.0, ix1 - ix0)
            h_ = max(0.0, iy1 - iy0)
            weights[i] = float(w_ * h_)

        total = float(weights.sum().item())
        if total > 0:
            weights = weights / total
        else:
            # Degenerate: the crop fell entirely outside any pasted tile
            # (possible only with very small s + extreme center).  Fall
            # back to uniform so downstream losses don't NaN.
            weights = torch.full((4,), 0.25, dtype=torch.float32)

        labels_t = torch.tensor(labels, dtype=torch.long)
        return out, labels_t, weights

    @staticmethod
    def _warp_and_crop(
        canvas: torch.Tensor,
        scale: float,
        out_h: int,
        out_w: int,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """YOLOv5 ``random_perspective``-style fused affine + crop.

        Forward pixel-space map (canvas → output), with
        (H_c, W_c) = canvas.shape[-2:]:
            (x, y) -> ( scale * (x - W_c/2) + out_w/2,
                        scale * (y - H_c/2) + out_h/2 )
        Inverse map (output → canvas):
            x_src = (x_dst - out_w/2) / scale + W_c/2
            y_src = (y_dst - out_h/2) / scale + H_c/2

        ``F.affine_grid`` + ``F.grid_sample`` work in normalized
        ``[-1, 1]`` coords with ``align_corners=False``; for a length-
        ``L`` axis the center of pixel ``p`` is at
        ``n = (2*p + 1)/L - 1``.  Converting dst pixel → dst norm and
        src pixel → src norm gives the linear relation
            x_n_src = a * x_n_dst + b
        with
            a = out_w / (W_c * scale)
            b = (W_c - out_w/scale) / W_c - 1 + (W_c + 1)/W_c - 1
              = 0      (sanity check: scale=1, out_w=W_c ⇒ a=1, b=-1+1=0)
        We compute it from first principles below (numerically
        transparent; the algebraic simplification above assumes
        out_w/scale could equal W_c which isn't generally true).
        """
        if scale <= 0:
            raise ValueError("scale must be > 0")
        C, H_c, W_c = canvas.shape
        device = canvas.device
        dtype = canvas.dtype

        inv_scale = 1.0 / float(scale)
        # dst norm  n_d  ->  dst pixel  p_d = (n_d + 1) * out / 2 - 0.5
        # dst pixel ->  src pixel  via inverse affine (see forward map):
        #   p_s = (p_d - out/2) * inv_scale + Wc/2
        # src pixel ->  src norm  n_s = (2*p_s + 1)/Wc - 1
        # Composing p_s as a function of n_d:
        #   p_s(n_d) = ( (n_d + 1) * out/2 - 0.5 - out/2 ) * inv_scale + Wc/2
        #           = ( n_d * out/2 - 0.5 ) * inv_scale + Wc/2
        # Then
        #   n_s = ( 2 * p_s + 1 )/Wc - 1
        #       = ( 2 * ( n_d*out/2 - 0.5 )*inv_scale + Wc + 1 )/Wc - 1
        #       = ( n_d * out * inv_scale - inv_scale + Wc + 1 )/Wc - 1
        #       = (out * inv_scale / Wc) * n_d + (Wc + 1 - inv_scale)/Wc - 1
        #       = a * n_d + b
        # with a = out/(Wc * scale), b = 1/Wc - inv_scale/Wc
        #                             = (1 - inv_scale)/Wc.
        a_x = float(out_w) / (float(W_c) * float(scale))
        b_x = (1.0 - inv_scale) / float(W_c)
        a_y = float(out_h) / (float(H_c) * float(scale))
        b_y = (1.0 - inv_scale) / float(H_c)

        theta = torch.tensor(
            [[a_x, 0.0, b_x], [0.0, a_y, b_y]],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, C, out_h, out_w), align_corners=False)
        # Use YOLOv5's gray-114 ``borderValue`` behavior.  PyTorch's
        # ``grid_sample`` only supports ``zeros``, ``border``, or
        # ``reflection`` constant-padding modes, so we emulate a
        # constant gray fill by subtracting ``fill_value`` from the
        # canvas, sampling with ``zeros`` padding (so any output pixel
        # whose inverse-mapped source coord falls outside ``[-1, 1]``
        # contributes 0 = ``fill_value - fill_value``), and adding
        # ``fill_value`` back.  End result: both the in-canvas gray
        # padding (set in step 2) and the outside-canvas padding read
        # back as the same gray — exactly YOLOv5's ``borderValue``.
        shifted = canvas.unsqueeze(0).float() - float(fill_value)
        out = F.grid_sample(
            shifted, grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        ).squeeze(0) + float(fill_value)
        return out.to(dtype)

    # ------------------------------------------------------------------
    # Helpers for single-sample passthrough
    # ------------------------------------------------------------------

    def _passthrough_target(self, label: int):
        """Return a LabelMix-style ``(labels[4], weights[4])`` target for a
        single-source sample (Mosaic skipped by probability or close-mosaic).

        The single hard label ``y`` is broadcast to ``labels=[y, y, y, y]``
        with ``weights=[1, 0, 0, 0]`` so the ``LabelMixSoftTargetCrossEntropy``
        loss reduces exactly to ``-log p(y)`` — i.e. ordinary CE on a single
        label — and the downstream collate / loss path stays on a single
        tensor-shape code path across iterations.
        """
        labels_t = torch.tensor([int(label), int(label), int(label), int(label)], dtype=torch.long)
        weights_t = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return labels_t, weights_t

    def _ensure_output_size(self, img: torch.Tensor) -> torch.Tensor:
        img = self._as_chw_float(img)
        if img.shape[-2] != self.out_h or img.shape[-1] != self.out_w:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.out_h, self.out_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return img

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        it = iter(self.base_dataset)
        mosaic_on = self._mosaic_enabled_this_epoch()

        # Per-worker RNG reseed for independence across workers/epochs.
        # We mix four independent sources so the stream of random crops /
        # centers / post-scales is:
        #   * deterministic given (user_seed, epoch, worker_id),
        #   * different across workers in the same epoch,
        #   * different across epochs for the same worker,
        #   * independent of any randomness already consumed upstream.
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        base = torch.initial_seed() if self.seed is None else int(self.seed)
        salt = 0 if self.seed is None else 2654435761  # deterministic tag
        seed = (base + salt + 7919 * self._epoch + 104729 * worker_id) % (2 ** 32 - 1)
        random.seed(seed)

        buf_imgs: List[torch.Tensor] = []
        buf_lbls: List[int] = []

        for item in it:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                raise ValueError(
                    "MosaicDataset expects the base dataset to yield (image, label) pairs."
                )
            img, lbl = item[0], int(item[1])

            if not mosaic_on or random.random() >= self.prob:
                img = self._ensure_output_size(img)
                yield img, self._passthrough_target(lbl)
                continue

            buf_imgs.append(self._as_chw_float(img))
            buf_lbls.append(lbl)

            if len(buf_imgs) == 4:
                out_img, out_lbls, out_w = self._compose_mosaic(buf_imgs, buf_lbls)
                # LabelMix-style soft-target semantics: the mosaic's
                # supervision is the area-weighted 4-way soft target,
                # consumed by ``LabelMixSoftTargetCrossEntropy``.  This
                # matches how YOLOv5's classification path accepts soft
                # targets via ``smartCrossEntropyLoss`` and uses all of
                # the pixel evidence in the crop, not just the anchor
                # tile's label.
                yield out_img, (out_lbls, out_w)
                buf_imgs.clear()
                buf_lbls.clear()

        # Flush leftover samples as single-sample passthroughs so no data
        # is silently dropped.
        for img, lbl in zip(buf_imgs, buf_lbls):
            img = self._ensure_output_size(img)
            yield img, self._passthrough_target(lbl)
