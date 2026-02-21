import io
import os
import math
import pickle
import random
from bisect import bisect_left
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .labelmix_layout import squarify_core


# ---------------------------------------------------------------------
# Optional torchvision import (cached once)
# ---------------------------------------------------------------------
try:
    import torchvision.transforms.functional as TVF  # type: ignore
except Exception:
    TVF = None


# =====================================================================
# Alpha Scheduler (curriculum on Dirichlet alpha)
# =====================================================================


class AlphaScheduler:
    """
    Alpha schedule evaluated once per cycle row.

    Modes:
      schedule: fixed | linear | cosine
      step_mode: epoch | total
      reverse: if True, run max->min instead of min->max
    """

    def __init__(
        self,
        alpha_min: float = 0.2,
        alpha_max: float = 2.0,
        schedule: str = "fixed",
        reverse: bool = False,
        step_mode: str = "epoch",
        warmup_steps: int = 0,
        num_classes: int = 1000,
        cycles_per_epoch: int = 1,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.schedule = schedule.lower()
        self.reverse = bool(reverse)
        self.step_mode = step_mode.lower()
        self.cycles_per_epoch = max(1, int(cycles_per_epoch))

        # Warmup: convert warmup_steps to warmup_cycles (approx)
        self.warmup_cycles = 0
        if warmup_steps > 0 and batch_size and num_classes > 0:
            self.warmup_cycles = int(math.ceil((warmup_steps * batch_size) / num_classes))

        # Total cycles for "total" mode
        self.total_global_cycles = 1
        if self.schedule != "fixed":
            if total_epochs is not None:
                self.total_global_cycles = int(total_epochs) * self.cycles_per_epoch
            elif total_steps is not None and batch_size and num_classes > 0:
                self.total_global_cycles = int(math.ceil((total_steps * batch_size) / num_classes))

        self.scope_cycles = self.cycles_per_epoch if self.step_mode == "epoch" else self.total_global_cycles

    def get_alpha(self, epoch: int, cycle_idx_in_epoch: int) -> float:
        if self.schedule == "fixed":
            return self.alpha_min

        global_idx = epoch * self.cycles_per_epoch + cycle_idx_in_epoch

        # Warmup: hold at one end
        if global_idx < self.warmup_cycles:
            return self.alpha_max if self.reverse else self.alpha_min

        # Progress p in [0,1]
        if self.step_mode == "epoch":
            p = cycle_idx_in_epoch / max(1, self.scope_cycles - 1)
        else:
            eff_curr = global_idx - self.warmup_cycles
            eff_tot = max(1, self.total_global_cycles - self.warmup_cycles - 1)
            p = eff_curr / eff_tot

        p = float(max(0.0, min(1.0, p)))
        if self.schedule == "cosine":
            p = 0.5 * (1.0 - math.cos(math.pi * p))

        span = self.alpha_max - self.alpha_min
        return (self.alpha_max - p * span) if self.reverse else (self.alpha_min + p * span)


# =====================================================================
# Layout -> Integer Boxes (robust tiling helpers)
# =====================================================================


def _cluster_sorted(values: List[float], eps: float) -> List[float]:
    """Cluster sorted floats so near-equal boundaries become identical."""
    if not values:
        return []
    values = sorted(values)
    groups: List[List[float]] = [[values[0]]]
    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= eps:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(sum(g) / len(g)) for g in groups]


def _enforce_strictly_increasing(coords: List[int], size_px: int) -> List[int]:
    """
    Enforce:
      coords[0] == 0, coords[-1] == size_px
      and strictly increasing interior coordinates.

    NOTE: This implies you should not use K larger than the canvas resolution in pixels
    in a way that forces too many unique boundaries.
    """
    if not coords:
        return coords
    coords[0] = 0
    coords[-1] = size_px

    # Forward pass: ensure >= prev+1
    for i in range(1, len(coords) - 1):
        coords[i] = max(coords[i], coords[i - 1] + 1)

    # Backward pass: ensure <= next-1
    for i in range(len(coords) - 2, 0, -1):
        coords[i] = min(coords[i], coords[i + 1] - 1)

    coords[0] = 0
    coords[-1] = size_px
    for i in range(1, len(coords) - 1):
        coords[i] = max(0, min(size_px, coords[i]))
    return coords


def _nearest_index(sorted_vals: Sequence[float], v: float) -> int:
    """Nearest index in a sorted list."""
    i = bisect_left(sorted_vals, v)
    if i <= 0:
        return 0
    if i >= len(sorted_vals):
        return len(sorted_vals) - 1
    return i if abs(sorted_vals[i] - v) < abs(v - sorted_vals[i - 1]) else (i - 1)


def _layout_to_pixel_boxes(
    layout_xywh: torch.Tensor,
    H: int,
    W: int,
    canvas_size: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Convert (K,4) float layout (x,y,w,h) into integer pixel boxes (K,4) [x0,y0,x1,y1],
    using globally-consistent boundary snapping to reduce gaps/overlaps.
    """
    K = int(layout_xywh.shape[0])
    S = float(canvas_size)

    xywh = layout_xywh.detach().cpu().tolist()

    x_edges = [0.0, S]
    y_edges = [0.0, S]
    for i in range(K):
        xi, yi, wi, hi = xywh[i]
        x_edges.extend([float(xi), float(xi + wi)])
        y_edges.extend([float(yi), float(yi + hi)])

    x_edges = [min(S, max(0.0, v)) for v in x_edges]
    y_edges = [min(S, max(0.0, v)) for v in y_edges]

    x_reps = _cluster_sorted(x_edges, eps=eps)
    y_reps = _cluster_sorted(y_edges, eps=eps)

    x_reps[0] = 0.0
    x_reps[-1] = S
    y_reps[0] = 0.0
    y_reps[-1] = S

    x_coords = [int(round((v / S) * W)) for v in x_reps]
    y_coords = [int(round((v / S) * H)) for v in y_reps]
    x_coords = _enforce_strictly_increasing(x_coords, size_px=W)
    y_coords = _enforce_strictly_increasing(y_coords, size_px=H)

    boxes = torch.zeros((K, 4), dtype=torch.int64)
    for i in range(K):
        xi, yi, wi, hi = xywh[i]
        x0f = min(S, max(0.0, float(xi)))
        x1f = min(S, max(0.0, float(xi + wi)))
        y0f = min(S, max(0.0, float(yi)))
        y1f = min(S, max(0.0, float(yi + hi)))

        ix0 = _nearest_index(x_reps, x0f)
        ix1 = _nearest_index(x_reps, x1f)
        iy0 = _nearest_index(y_reps, y0f)
        iy1 = _nearest_index(y_reps, y1f)

        boxes[i, 0] = x_coords[min(ix0, ix1)]
        boxes[i, 1] = y_coords[min(iy0, iy1)]
        boxes[i, 2] = x_coords[max(ix0, ix1)]
        boxes[i, 3] = y_coords[max(iy0, iy1)]

    return boxes


def _fallback_stripes_boxes(weights: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Fallback layout: vertical stripes across full height using weights (K,).
    Produces guaranteed tiling boxes (K,4) [x0,y0,x1,y1].
    """
    K = int(weights.numel())
    w = weights.detach().float().cpu()
    s = float(w.sum())
    if s <= 0:
        w = torch.full((K,), 1.0 / K)
    else:
        w = w / s

    cum = torch.cumsum(w, dim=0)
    xs = [0] + [int(round(float(c) * W)) for c in cum[:-1]] + [W]
    xs = _enforce_strictly_increasing(xs, size_px=W)

    boxes = torch.zeros((K, 4), dtype=torch.int64)
    for i in range(K):
        boxes[i, 0] = xs[i]
        boxes[i, 1] = 0
        boxes[i, 2] = xs[i + 1]
        boxes[i, 3] = H
    return boxes


def _boxes_are_valid_and_tile(boxes: torch.Tensor, H: int, W: int) -> bool:
    if boxes.numel() == 0:
        return False
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if torch.any(x1 <= x0) or torch.any(y1 <= y0):
        return False
    areas = (x1 - x0) * (y1 - y0)
    return int(areas.sum().item()) == int(H * W)


def _apply_box_symmetry(boxes: torch.Tensor, sym: int, S: int) -> torch.Tensor:
    """
    Apply D4 symmetry to integer pixel boxes on a square canvas size S (S == H == W).
    boxes: (K,4) int64 [x0,y0,x1,y1]
    Returns: (K,4) int64 transformed boxes.
    """
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]

    if sym == 0:   # Identity
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    elif sym == 1:  # Rot90 CCW: (x,y)->(y, S-x)
        nx0, ny0 = y0, S - x1
        nx1, ny1 = y1, S - x0
    elif sym == 2:  # Rot180
        nx0, ny0 = S - x1, S - y1
        nx1, ny1 = S - x0, S - y0
    elif sym == 3:  # Rot270 CCW
        nx0, ny0 = S - y1, x0
        nx1, ny1 = S - y0, x1
    elif sym == 4:  # FlipH
        nx0, ny0 = S - x1, y0
        nx1, ny1 = S - x0, y1
    elif sym == 5:  # FlipV
        nx0, ny0 = x0, S - y1
        nx1, ny1 = x1, S - y0
    elif sym == 6:  # Transpose
        nx0, ny0 = y0, x0
        nx1, ny1 = y1, x1
    else:  # 7 AntiTranspose
        nx0, ny0 = S - y1, S - x1
        nx1, ny1 = S - y0, S - x0

    return torch.stack([nx0, ny0, nx1, ny1], dim=1).to(dtype=torch.int64)


def _apply_box_symmetry_rect(boxes: torch.Tensor, sym: int, H: int, W: int) -> torch.Tensor:
    """
    Rect-safe symmetries for non-square inputs. Supports only:
      0 identity, 2 rot180, 4 flipH, 5 flipV
    """
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    if sym == 0:
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    elif sym == 2:
        nx0, ny0 = W - x1, H - y1
        nx1, ny1 = W - x0, H - y0
    elif sym == 4:
        nx0, ny0 = W - x1, y0
        nx1, ny1 = W - x0, y1
    elif sym == 5:
        nx0, ny0 = x0, H - y1
        nx1, ny1 = x1, H - y0
    else:
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1

    return torch.stack([nx0, ny0, nx1, ny1], dim=1).to(dtype=torch.int64)


# =====================================================================
# Dataset
# =====================================================================


class BalancedBucketDataset(IterableDataset):
    """
    Iterable dataset that yields class-balanced samples.

    If labelmix=True, yields:
      (mixed_image, (labels[K], weights[K]))
    where labels/weights are ASCENDING by weight and aligned with slot order.

    HuggingFace optimization:
      - groups are loaded via base_dataset[indices] when possible (batch fetch).

    LabelMix semantics:
      - One Dirichlet draw per CYCLE ROW -> weights_desc -> base_layout_desc cached per cycle row
      - Base boxes and symmetry-transformed boxes are cached once per row/shape
      - For each GROUP of K images within the row:
          * Use cached base boxes to compute slot sizes once (ASC)
          * Precompute resize caches once (second orientation only if needed)
          * Pick K random symmetries -> produce K outputs (one per shift)
          * Emit per-shift output views directly
      - Rolling buffer + random pop; flush with shuffle
    """

    def __init__(
        self,
        base_dataset,
        transform=None,
        mode: str = "max",
        buffer_size: int = 4096,
        cache_path: str = "class_buckets.pkl",
        cache_small_classes_threshold: int = 100,
        input_key: str = "image",
        target_key: str = "label",
        labelmix: bool = False,
        labelmix_kwargs: Optional[Dict[str, Any]] = None,
        dist_rank_override: Optional[int] = None,
        dist_world_size_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.mode = str(mode)
        self.buffer_size = int(buffer_size)
        self.cache_threshold = int(cache_small_classes_threshold)
        self.input_key = input_key
        self.target_key = target_key
        self._epoch = 0

        self.labelmix = bool(labelmix)
        self.lm_config = labelmix_kwargs or {}
        self.mix_k = int(self.lm_config.get("mix_k", 4))
        if self.mix_k < 2:
            raise ValueError("mix_k must be >= 2")

        self.debug_sym = bool(self.lm_config.get("debug_sym", False))

        self.sampling_enabled = bool(self.lm_config.get("sampling", False))
        self.sampling_min_side_px = int(self.lm_config.get("sampling_min_side_px", 6))
        self.sampling_max_aspect = float(self.lm_config.get("sampling_max_aspect", 10.0))
        self.sampling_bins = int(self.lm_config.get("sampling_bins", 16))
        self.sampling_pool_size = int(self.lm_config.get("sampling_pool_size", 128))
        self.sampling_low_watermark = int(self.lm_config.get("sampling_low_watermark", 32))
        self.sampling_max_attempts = int(self.lm_config.get("sampling_max_attempts", 200))
        self._sampling_pools: Optional[List[deque]] = None
        self._sampling_hw: Optional[Tuple[int, int]] = None

        # Precompute indices for cyclic permutation:
        # circulant_idx[shift, slot] = (slot + shift) % K
        K = self.mix_k
        self._shift_idx = torch.arange(K, dtype=torch.int64)
        self._circulant_idx = (self._shift_idx[:, None] + self._shift_idx[None, :]) % K

        # Identify HF-style dataset
        self.is_hf = hasattr(base_dataset, "column_names") or hasattr(base_dataset, "features")

        # Detect sample list for file-based datasets
        self._raw_samples = self._get_samples_list(base_dataset)

        # Build / load buckets
        self.buckets = self._load_or_build_buckets(cache_path)

        self.classes = torch.tensor(sorted(list(self.buckets.keys())), dtype=torch.int64)
        self.num_classes = int(self.classes.numel())
        if self.num_classes == 0:
            raise ValueError("No classes found in buckets.")

        self.bucket_arrays = [torch.tensor(self.buckets[int(c)], dtype=torch.int64) for c in self.classes]
        self.bucket_lengths = torch.tensor([len(b) for b in self.bucket_arrays], dtype=torch.int64)
        self.bucket_len_map = {
            int(c): int(l) for c, l in zip(self.classes, self.bucket_lengths)
        }

        # cycles per epoch
        min_len = int(torch.min(self.bucket_lengths).item())
        max_len = int(torch.max(self.bucket_lengths).item())
        if self.mode == "min":
            self.M = min_len
        elif self.mode == "max":
            self.M = max_len
        else:
            self.M = int(self.mode)
        self.M = max(1, self.M)

        self.total_images_global = int(self.num_classes * self.M)

        is_primary = True
        if dist.is_available() and dist.is_initialized():
            is_primary = dist.get_rank() == 0
        if is_primary:
            print(
                f"[BalancedBucketDataset] mode={self.mode} min_len={min_len} max_len={max_len} "
                f"M={self.M} total_images={self.total_images_global}"
            )

        if self.labelmix:
            self.alpha_scheduler = AlphaScheduler(
                alpha_min=float(self.lm_config.get("alpha_min", 0.2)),
                alpha_max=float(self.lm_config.get("alpha_max", 2.0)),
                schedule=str(self.lm_config.get("schedule", "fixed")),
                reverse=bool(self.lm_config.get("reverse", False)),
                step_mode=str(self.lm_config.get("step_mode", "epoch")),
                warmup_steps=int(self.lm_config.get("warmup_steps", 0)),
                num_classes=self.num_classes,
                cycles_per_epoch=self.M,
                total_epochs=self.lm_config.get("total_epochs"),
                total_steps=self.lm_config.get("total_steps"),
                batch_size=self.lm_config.get("batch_size"),
            )
            if self.sampling_enabled:
                bins = max(1, self.sampling_bins)
                self._sampling_pools = [deque() for _ in range(bins)]

        self.local_byte_cache: Dict[int, Tuple[Any, int]] = {}
        self.dist_rank_override = dist_rank_override
        self.dist_world_size_override = dist_world_size_override

    def _sampling_bin_idx(self, alpha: float) -> int:
        bins = max(1, self.sampling_bins)
        if bins == 1 or self.alpha_scheduler.alpha_max <= self.alpha_scheduler.alpha_min:
            return 0
        t = (alpha - self.alpha_scheduler.alpha_min) / (self.alpha_scheduler.alpha_max - self.alpha_scheduler.alpha_min)
        t = max(0.0, min(1.0, float(t)))
        return int(t * (bins - 1))

    def _layout_is_valid(self, weights_desc: torch.Tensor, H: int, W: int) -> bool:
        base_layout_desc = squarify_core(weights_desc, canvas_size=1.0)
        base_layout_asc = torch.flip(base_layout_desc, dims=[0])
        boxes = _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)
        if not _boxes_are_valid_and_tile(boxes, H=H, W=W):
            return False

        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        widths = (x1 - x0).to(dtype=torch.float32)
        heights = (y1 - y0).to(dtype=torch.float32)
        min_side = torch.minimum(widths, heights)
        max_side = torch.maximum(widths, heights)

        if self.sampling_min_side_px > 0 and torch.any(min_side < float(self.sampling_min_side_px)):
            return False
        if self.sampling_max_aspect > 0.0:
            aspect = max_side / torch.clamp(min_side, min=1.0)
            if torch.any(aspect > float(self.sampling_max_aspect)):
                return False
        return True

    def _sample_dirichlet_layout(self, alpha: float, H: int, W: int) -> torch.Tensor:
        K = self.mix_k
        if alpha <= 0.0:
            return torch.full((K,), 1.0 / K, dtype=torch.float32)
        dirichlet = torch.distributions.Dirichlet(torch.full((K,), alpha))
        for _ in range(max(1, self.sampling_max_attempts)):
            w = dirichlet.sample().to(dtype=torch.float32)
            weights_desc, _ = torch.sort(w, descending=True)
            if self._layout_is_valid(weights_desc, H=H, W=W):
                return w

        # Fallback to uniform to avoid stalling
        return torch.full((K,), 1.0 / K, dtype=torch.float32)

    def _get_sampling_weights(self, alpha: float, H: int, W: int) -> torch.Tensor:
        if self._sampling_pools is None:
            return self._sample_dirichlet_layout(alpha, H=H, W=W)
        if self._sampling_hw != (H, W):
            for pool in self._sampling_pools:
                pool.clear()
            self._sampling_hw = (H, W)

        bin_idx = self._sampling_bin_idx(alpha)
        pool = self._sampling_pools[bin_idx]

        if len(pool) <= self.sampling_low_watermark:
            for _ in range(max(1, self.sampling_pool_size - len(pool))):
                pool.append(self._sample_dirichlet_layout(alpha, H=H, W=W))

        if pool:
            return pool.popleft()

        return self._sample_dirichlet_layout(alpha, H=H, W=W)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return self.total_images_global // dist.get_world_size()
        return self.total_images_global

    # -------------------------
    # Dataset plumbing
    # -------------------------

    def _get_samples_list(self, base_dataset):
        if hasattr(base_dataset, "samples"):
            return base_dataset.samples
        if hasattr(base_dataset, "parser") and hasattr(base_dataset.parser, "samples"):
            return base_dataset.parser.samples
        reader = getattr(base_dataset, "reader", None)
        if reader is not None:
            if hasattr(reader, "samples"):
                return reader.samples
            if hasattr(reader, "parser") and hasattr(reader.parser, "samples"):
                return reader.parser.samples
        return None

    def _load_or_build_buckets(self, path: str) -> Dict[int, List[int]]:
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                    if isinstance(obj, dict) and len(obj) > 0:
                        return obj
            except Exception:
                pass

        buckets: Dict[int, List[int]] = defaultdict(list)

        # 1) File-based datasets
        if self._raw_samples:
            targets = [int(s[1]) for s in self._raw_samples]
            for idx, t in enumerate(targets):
                buckets[int(t)].append(int(idx))
            if path:
                try:
                    with open(path, "wb") as f:
                        pickle.dump(dict(buckets), f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            return buckets

        # 2) Torchvision-style datasets with .targets
        if hasattr(self.base_dataset, "targets"):
            targets = [int(t) for t in self.base_dataset.targets]
            for idx, t in enumerate(targets):
                buckets[int(t)].append(int(idx))
            if path:
                try:
                    with open(path, "wb") as f:
                        pickle.dump(dict(buckets), f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            return buckets

        # 3) Hugging Face datasets wrapped by timm ImageDataset / ReaderHfds
        reader = getattr(self.base_dataset, "reader", None)
        hf_dataset = getattr(reader, "dataset", None) if reader is not None else None
        if hf_dataset is not None and (hasattr(hf_dataset, "column_names") or hasattr(hf_dataset, "features")):
            target_key = self.target_key
            if hasattr(hf_dataset, "column_names") and target_key not in hf_dataset.column_names:
                reader_label_key = getattr(reader, "label_key", None)
                if reader_label_key and reader_label_key in hf_dataset.column_names:
                    target_key = reader_label_key
            if hasattr(hf_dataset, "column_names") and target_key not in hf_dataset.column_names:
                raise ValueError(
                    f"Could not read target column '{target_key}' from HF dataset. "
                    f"Available columns: {hf_dataset.column_names}"
                )
            try:
                targets = hf_dataset[target_key]
            except Exception as e:
                raise ValueError(f"Could not read target column '{target_key}' from HF dataset.") from e
            for idx, t in enumerate(targets):
                buckets[int(t)].append(int(idx))
            if path:
                try:
                    with open(path, "wb") as f:
                        pickle.dump(dict(buckets), f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            return buckets

        # 4) Hugging Face datasets (random access)
        if self.is_hf:
            if not hasattr(self.base_dataset, "__len__") or not hasattr(self.base_dataset, "__getitem__"):
                raise ValueError("Hugging Face streaming datasets are not supported (need random access).")
            try:
                targets = self.base_dataset[self.target_key]
            except Exception as e:
                raise ValueError(f"Could not read target column '{self.target_key}' from HF dataset.") from e
            for idx, t in enumerate(targets):
                buckets[int(t)].append(int(idx))
            if path:
                try:
                    with open(path, "wb") as f:
                        pickle.dump(dict(buckets), f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            return buckets

        raise ValueError(
            "Could not infer targets. Provide a dataset with .samples, .parser.samples, .targets, "
            "or a Hugging Face dataset with a target column."
        )

    def _load_item(self, idx: int) -> Tuple[Any, int]:
        if idx in self.local_byte_cache:
            return self.local_byte_cache[idx]

        if self._raw_samples:
            path, target = self._raw_samples[idx]
            with open(path, "rb") as f:
                img = f.read()
            target = int(target)
        else:
            item = self.base_dataset[idx]
            if isinstance(item, dict):
                img = item[self.input_key]
                target = int(item[self.target_key])
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                img, target = item[0], int(item[1])
            else:
                raise ValueError(f"Unknown sample format at index {idx}: {type(item)}")

        # Cache only bytes; avoid holding PIL objects for HF
        if self.bucket_len_map.get(int(target), 0) <= self.cache_threshold and isinstance(img, (bytes, bytearray)):
            self.local_byte_cache[idx] = (img, target)

        return img, target

    def _load_items_batch(self, indices: List[int]) -> List[Tuple[Any, int]]:
        """
        Batch load items.
        - HF datasets: base_dataset[indices] returns dict-of-lists (fast path).
        - Others: fallback to per-item loads.
        """
        if not indices:
            return []

        if self._raw_samples:
            return [self._load_item(int(i)) for i in indices]

        if self.is_hf:
            try:
                batch = self.base_dataset[indices]
                if isinstance(batch, dict) and self.input_key in batch and self.target_key in batch:
                    imgs = batch[self.input_key]
                    tgts = batch[self.target_key]
                    return [(imgs[i], int(tgts[i])) for i in range(len(tgts))]
                if isinstance(batch, list) and batch and isinstance(batch[0], dict):
                    return [(b[self.input_key], int(b[self.target_key])) for b in batch]
            except Exception:
                pass

        return [self._load_item(int(i)) for i in indices]

    # -------------------------
    # Image conversion (no numpy)
    # -------------------------

    @staticmethod
    def _pil_to_chw_uint8(img: Image.Image) -> torch.Tensor:
        """
        Convert PIL RGB image -> CHW uint8 tensor without numpy.
        Prefers torchvision if available; otherwise uses tobytes().
        """
        if TVF is not None:
            return TVF.pil_to_tensor(img)

        img = img.convert("RGB")
        w, h = img.size
        data = img.tobytes()
        mv = memoryview(data)
        try:
            t = torch.frombuffer(mv, dtype=torch.uint8)
        except Exception:
            t = torch.tensor(bytearray(mv), dtype=torch.uint8)
        t = t.view(h, w, 3).permute(2, 0, 1).contiguous()
        return t

    def _process_image(self, img_obj: Any) -> torch.Tensor:
        """
        Returns CHW float32 in [0,1].
        """
        try:
            if isinstance(img_obj, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img_obj)).convert("RGB")
            elif isinstance(img_obj, Image.Image):
                img = img_obj.convert("RGB")
            else:
                img = Image.fromarray(img_obj).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(img, torch.Tensor):
            t = img
        else:
            t = self._pil_to_chw_uint8(img)

        # Ensure float for interpolate + model input
        if t.dtype not in (torch.float16, torch.float32, torch.float64):
            t = t.float().div(255.0)
        else:
            t = t.float()

        return t

    # -------------------------
    # Schedule generation
    # -------------------------

    def _build_schedule(self, rng: torch.Generator, worker_cycles: torch.Tensor) -> torch.Tensor:
        """
        Build schedule rows for this worker: (num_rows, num_classes).
        Each row has one index per class, then shuffled within-row.
        """
        num_rows = int(worker_cycles.numel())
        schedule = torch.empty((num_rows, self.num_classes), dtype=torch.int64)

        for ci in range(self.num_classes):
            indices = self.bucket_arrays[ci]
            n = int(indices.numel())
            if n <= 0:
                raise ValueError("Empty class bucket encountered.")

            if self.mode == "min":
                schedule[:, ci] = indices[worker_cycles % n]
            else:
                r = torch.randint(n, (num_rows,), generator=rng, dtype=torch.int64)
                schedule[:, ci] = indices[r]

        noise = torch.rand(schedule.shape, generator=rng)
        order = torch.argsort(noise, dim=1)
        return schedule.gather(1, order)

    # -------------------------
    # LabelMix core (your mixing semantics)
    # -------------------------

    def _mix_group_labelmix(
        self,
        imgs: List[torch.Tensor],
        targets: List[int],
        base_boxes: torch.Tensor,
        sym_boxes_cache: Dict[int, torch.Tensor],
        weights_desc: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Group mixing:
          1) Receive precomputed base boxes (ASC slot order)
          2) Precompute resize caches once from base slot sizes:
               resized0[slot] = resize(batch) to (hi,wi)
               resized1[slot] = resize(batch) to (wi,hi) only when swap symmetries are used
          3) Pick K random symmetries (one per output shift)
             and fetch precomputed symmetry boxes
          4) Render K outputs
          5) Output labels/weights in ASC slot order
        """
        K = self.mix_k

        with torch.inference_mode():
            batch = torch.stack(imgs, dim=0).contiguous()
            _, C, H, W = batch.shape
            is_square = H == W

            # Choose symmetries for each shift
            if is_square:
                sym_choices = torch.randint(0, 8, (K,), dtype=torch.int64)
            else:
                allowed = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
                sym_choices = allowed[torch.randint(0, allowed.numel(), (K,))]

            # Output weights are ASC; move to same device
            weights_asc = torch.flip(weights_desc, dims=[0]).float().contiguous()
            weights_asc = weights_asc.to(device=batch.device)

            need_swap_cache = False
            if is_square:
                need_swap_cache = bool(
                    torch.any(
                        (sym_choices == 1)
                        | (sym_choices == 3)
                        | (sym_choices == 6)
                        | (sym_choices == 7)
                    ).item()
                )

            # Precompute resize caches once from base slot sizes
            resized0: List[torch.Tensor] = []
            resized1: Optional[List[torch.Tensor]] = [] if need_swap_cache else None
            for slot_i in range(K):
                x0 = int(base_boxes[slot_i, 0].item())
                y0 = int(base_boxes[slot_i, 1].item())
                x1 = int(base_boxes[slot_i, 2].item())
                y1 = int(base_boxes[slot_i, 3].item())
                hi = max(1, y1 - y0)
                wi = max(1, x1 - x0)

                resized0.append(F.interpolate(batch, size=(hi, wi), mode="bilinear", align_corners=False))
                if resized1 is not None:
                    resized1.append(F.interpolate(batch, size=(wi, hi), mode="bilinear", align_corners=False))

            shifts = self._shift_idx.to(batch.device, non_blocking=True) if batch.device.type != "cpu" else self._shift_idx
            circulant = (
                self._circulant_idx.to(batch.device, non_blocking=True)
                if batch.device.type != "cpu"
                else self._circulant_idx
            )

            tgt = torch.as_tensor(targets, dtype=torch.int64, device=batch.device)
            labels_mat = tgt[circulant]

            out_batch = batch.new_zeros((K, C, H, W))

            for sym_t in torch.unique(sym_choices):
                sym = int(sym_t.item())
                mask = (sym_choices == sym)
                shift_subset = shifts[mask]

                boxes_sym = sym_boxes_cache[sym]

                # Symmetries that swap width/height for slots (square only)
                swap = is_square and (sym in (1, 3, 6, 7))

                for slot_i in range(K):
                    x0 = int(boxes_sym[slot_i, 0].item())
                    y0 = int(boxes_sym[slot_i, 1].item())
                    x1 = int(boxes_sym[slot_i, 2].item())
                    y1 = int(boxes_sym[slot_i, 3].item())

                    cache = resized1[slot_i] if (swap and resized1 is not None) else resized0[slot_i]
                    src_idx = (slot_i + shift_subset) % K
                    patches = cache.index_select(0, src_idx)

                    th = max(1, y1 - y0)
                    tw = max(1, x1 - x0)
                    ph, pw = patches.shape[-2], patches.shape[-1]
                    if ph != th or pw != tw:
                        patches = F.interpolate(patches, size=(th, tw), mode="bilinear", align_corners=False)

                    out_batch[shift_subset, :, y0:y1, x0:x1] = patches

            out: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = []
            for s in range(K):
                if self.debug_sym:
                    out.append((out_batch[s], (labels_mat[s], weights_asc, int(sym_choices[s].item()))))
                else:
                    out.append((out_batch[s], (labels_mat[s], weights_asc)))
            return out

    def _iter_labelmix(self, schedule: torch.Tensor, worker_cycles: torch.Tensor):
        """
        LabelMix iteration:
          - one Dirichlet draw per cycle row
          - base layout built once per cycle row
          - base boxes and symmetry boxes built once per row/shape
          - for each group of K indices inside the row:
              produce K outputs with K random symmetries
          - push to rolling buffer and yield randomly
        """
        buffer: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = []
        K = self.mix_k
        emitted = 0
        last_exc: Optional[Exception] = None

        for row_i in range(schedule.shape[0]):
            cycle_idx_in_epoch = int(worker_cycles[row_i].item())
            alpha = float(self.alpha_scheduler.get_alpha(self._epoch, cycle_idx_in_epoch))

            weights: Optional[torch.Tensor] = None
            weights_desc: Optional[torch.Tensor] = None
            base_layout_desc: Optional[torch.Tensor] = None
            base_layout_asc: Optional[torch.Tensor] = None
            weights_asc: Optional[torch.Tensor] = None
            base_boxes_by_hw: Dict[Tuple[int, int], torch.Tensor] = {}
            sym_boxes_by_hw: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}

            # Within the row, group into chunks of K.
            # If there is a remainder, pad one tail group from this row and
            # keep only `remainder` mixed outputs so row output count stays exact.
            row_indices = schedule[row_i]
            row_n = int(row_indices.numel())
            usable = (row_n // K) * K
            remainder = row_n - usable

            for g0 in range(0, usable, K):
                group_indices = row_indices[g0 : g0 + K].tolist()
                try:
                    items = self._load_items_batch(group_indices)

                    imgs: List[torch.Tensor] = []
                    tgts: List[int] = []
                    for img_obj, tgt in items:
                        imgs.append(self._process_image(img_obj))
                        tgts.append(int(tgt))

                    if weights_desc is None or base_layout_desc is None:
                        shape = imgs[0].shape
                        H, W = int(shape[-2]), int(shape[-1])
                        if self.sampling_enabled:
                            weights = self._get_sampling_weights(alpha, H=H, W=W)
                        else:
                            if alpha > 0.0:
                                dirichlet = torch.distributions.Dirichlet(torch.full((K,), alpha))
                                weights = dirichlet.sample().to(dtype=torch.float32)
                            else:
                                weights = torch.full((K,), 1.0 / K, dtype=torch.float32)
                        weights_desc, _ = torch.sort(weights, descending=True)
                        base_layout_desc = squarify_core(weights_desc, canvas_size=1.0)
                        base_layout_asc = torch.flip(base_layout_desc, dims=[0])
                        weights_asc = torch.flip(weights_desc, dims=[0]).float().contiguous()

                    H, W = int(imgs[0].shape[-2]), int(imgs[0].shape[-1])
                    hw = (H, W)
                    if hw not in base_boxes_by_hw:
                        if base_layout_asc is None or weights_asc is None:
                            raise RuntimeError("Internal error: layout caches not initialized.")

                        base_boxes = _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)
                        if not _boxes_are_valid_and_tile(base_boxes, H=H, W=W):
                            base_boxes = _fallback_stripes_boxes(weights_asc, H=H, W=W).to(base_boxes.device)
                        base_boxes_by_hw[hw] = base_boxes

                        if H == W:
                            sym_boxes_by_hw[hw] = {
                                s: _apply_box_symmetry(base_boxes, sym=s, S=W) for s in range(8)
                            }
                        else:
                            sym_boxes_by_hw[hw] = {
                                s: _apply_box_symmetry_rect(base_boxes, sym=s, H=H, W=W)
                                for s in (0, 2, 4, 5)
                            }

                    mixed_items = self._mix_group_labelmix(
                        imgs=imgs,
                        targets=tgts,
                        base_boxes=base_boxes_by_hw[hw],
                        sym_boxes_cache=sym_boxes_by_hw[hw],
                        weights_desc=weights_desc,
                    )
                    buffer.extend(mixed_items)
                except Exception as exc:
                    last_exc = exc
                    continue

                while len(buffer) >= self.buffer_size:
                    i = random.randint(0, len(buffer) - 1)
                    buffer[i], buffer[-1] = buffer[-1], buffer[i]
                    emitted += 1
                    yield buffer.pop()

            if remainder > 0:
                # Tail indices that would otherwise be dropped.
                tail = row_indices[usable:]
                need = K - remainder

                # Fill from earlier indices in the same row (without replacement).
                pool = row_indices[:usable]
                if int(pool.numel()) < need:
                    # Defensive fallback (should not happen when num_classes >= mix_k).
                    perm = torch.randperm(row_n, dtype=torch.int64)
                    fill = row_indices[perm[:need]]
                else:
                    perm = torch.randperm(int(pool.numel()), dtype=torch.int64)
                    fill = pool[perm[:need]]

                group_indices = torch.cat([tail, fill], dim=0).tolist()
                try:
                    items = self._load_items_batch(group_indices)

                    imgs: List[torch.Tensor] = []
                    tgts: List[int] = []
                    for img_obj, tgt in items:
                        imgs.append(self._process_image(img_obj))
                        tgts.append(int(tgt))

                    if weights_desc is None or base_layout_desc is None:
                        shape = imgs[0].shape
                        H, W = int(shape[-2]), int(shape[-1])
                        if self.sampling_enabled:
                            weights = self._get_sampling_weights(alpha, H=H, W=W)
                        else:
                            if alpha > 0.0:
                                dirichlet = torch.distributions.Dirichlet(torch.full((K,), alpha))
                                weights = dirichlet.sample().to(dtype=torch.float32)
                            else:
                                weights = torch.full((K,), 1.0 / K, dtype=torch.float32)
                        weights_desc, _ = torch.sort(weights, descending=True)
                        base_layout_desc = squarify_core(weights_desc, canvas_size=1.0)
                        base_layout_asc = torch.flip(base_layout_desc, dims=[0])
                        weights_asc = torch.flip(weights_desc, dims=[0]).float().contiguous()

                    H, W = int(imgs[0].shape[-2]), int(imgs[0].shape[-1])
                    hw = (H, W)
                    if hw not in base_boxes_by_hw:
                        if base_layout_asc is None or weights_asc is None:
                            raise RuntimeError("Internal error: layout caches not initialized.")

                        base_boxes = _layout_to_pixel_boxes(base_layout_asc, H=H, W=W, canvas_size=1.0, eps=1e-7)
                        if not _boxes_are_valid_and_tile(base_boxes, H=H, W=W):
                            base_boxes = _fallback_stripes_boxes(weights_asc, H=H, W=W).to(base_boxes.device)
                        base_boxes_by_hw[hw] = base_boxes

                        if H == W:
                            sym_boxes_by_hw[hw] = {
                                s: _apply_box_symmetry(base_boxes, sym=s, S=W) for s in range(8)
                            }
                        else:
                            sym_boxes_by_hw[hw] = {
                                s: _apply_box_symmetry_rect(base_boxes, sym=s, H=H, W=W)
                                for s in (0, 2, 4, 5)
                            }

                    mixed_items = self._mix_group_labelmix(
                        imgs=imgs,
                        targets=tgts,
                        base_boxes=base_boxes_by_hw[hw],
                        sym_boxes_cache=sym_boxes_by_hw[hw],
                        weights_desc=weights_desc,
                    )
                    buffer.extend(mixed_items[:remainder])
                except Exception as exc:
                    last_exc = exc

                while len(buffer) >= self.buffer_size:
                    i = random.randint(0, len(buffer) - 1)
                    buffer[i], buffer[-1] = buffer[-1], buffer[i]
                    emitted += 1
                    yield buffer.pop()

        random.shuffle(buffer)
        for item in buffer:
            emitted += 1
            yield item

        if emitted == 0:
            if last_exc is not None:
                raise RuntimeError("LabelMix produced no samples; last group error attached.") from last_exc
            raise RuntimeError("LabelMix produced no samples; check mix_k and dataset class count.")

    # -------------------------
    # Iterator
    # -------------------------

    def __iter__(self):
        # Distributed
        if self.dist_world_size_override is not None:
            world_size = int(self.dist_world_size_override)
            rank = int(self.dist_rank_override or 0)
        elif dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        # DataLoader workers
        info = get_worker_info()
        num_workers = info.num_workers if info else 1
        worker_id = info.id if info else 0

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        if self.labelmix and self.num_classes < self.mix_k:
            raise ValueError(f"LabelMix requires num_classes >= mix_k (got {self.num_classes} < {self.mix_k})")

        worker_cycles = torch.arange(global_worker_id, self.M, total_workers, dtype=torch.int64)
        if worker_cycles.numel() == 0:
            return iter(())

        # Independence across workers/ranks (not strict determinism)
        seed = (torch.initial_seed() + 1009 * self._epoch + 9176 * global_worker_id) % (2**32 - 1)
        random.seed(seed)
        torch.manual_seed(seed)
        rng = torch.Generator()
        rng.manual_seed(seed)

        schedule = self._build_schedule(rng=rng, worker_cycles=worker_cycles)

        if self.labelmix:
            yield from self._iter_labelmix(schedule=schedule, worker_cycles=worker_cycles)
        else:
            # Non-labelmix path: batch-fetch in chunks if possible (HF speedup)
            buffer: List[Tuple[torch.Tensor, int]] = []
            flat = schedule.flatten()

            chunk_size = max(64, min(1024, self.buffer_size))
            for i0 in range(0, int(flat.numel()), chunk_size):
                chunk = flat[i0 : i0 + chunk_size].tolist()
                items = self._load_items_batch(chunk)

                for img_obj, tgt in items:
                    try:
                        buffer.append((self._process_image(img_obj), int(tgt)))
                    except Exception:
                        continue

                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []

            random.shuffle(buffer)
            for item in buffer:
                yield item
