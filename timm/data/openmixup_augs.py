"""OpenMixup-style data augmentations (self-contained port).

This module bundles the mixup-style augmentations described in
``data_augs.xml`` (taken from `OpenMixup
<https://github.com/Westlake-AI/openmixup>`_) into a single, dependency-light
file that can be imported from both the training script
(:mod:`train.py`) and the figure-rendering script
(:mod:`evaluation.plots.augmentation_showcase`).

The methods can be split into two groups:

* **Image-only (model-free)** – These can be applied as drop-in replacements
  for ``timm.data.Mixup`` during training. They take only ``(img, gt_label)``
  and produce mixed images plus soft labels:

      ``fmix``, ``gridmix``, ``resizemix``, ``smoothmix``, ``saliencymix``.

* **Model-aware** – These additionally need feature maps, attention maps,
  saliency maps, or loss gradients from a trained model. They are still
  ported here for completeness so the showcase can visualize them with
  synthetic stand-in features:

      ``alignmix``, ``attentivemix``, ``snapmix``, ``transmix``,
      ``mixpro``, ``smmix``, ``tla``, ``tokenmix``, ``guidedmix``,
      ``puzzlemix``.

.. note::
   The OpenMixup repository also defines ``mixup``, ``cutmix`` and
   ``augmix`` augmentations, but the LabelMix codebase already implements
   those (``timm.data.Mixup`` / ``FastCollateMixup`` for mixup+cutmix,
   ``timm.data.auto_augment.AugMixAugment`` for augmix). We deliberately
   do not duplicate those here – use ``--mixup`` / ``--cutmix`` /
   ``--aa augmix-...`` instead.

All functions return ``(img_mixed, target_info)`` where ``target_info`` is
either a tuple ``(y_a, y_b, lam)`` or a richer tuple if the original method
required it. A small helper :func:`mix_to_soft_target` converts the various
forms back to a single soft-label tensor of shape ``(N, num_classes)``.

The implementations follow the OpenMixup originals as faithfully as
possible, but with two simplifications:

* Distributed shuffling (``dist_mode=True``) is intentionally **not**
  supported – we always use within-rank batch permutations. This matches
  the pattern used by the rest of the LabelMix codebase, where each rank
  shuffles its own batch.
* Dependencies on the ``openmixup`` Python package are replaced by inline
  definitions or PIL-based equivalents.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "OPENMIXUP_AUG_NAMES",
    "OPENMIXUP_CLI_AUG_NAMES",
    "OPENMIXUP_IMAGE_ONLY_AUGS",
    "OPENMIXUP_MODEL_AWARE_AUGS",
    "OPENMIXUP_AUG_INFO",
    "OpenMixupAug",
    "apply_openmixup_aug",
    "openmixup_to_soft_target",
    # individual functions (mixup/cutmix/augmix are intentionally NOT
    # exposed: timm already implements them; see the module docstring).
    "fmix",
    "gridmix",
    "resizemix",
    "smoothmix",
    "saliencymix",
    "alignmix",
    "attentivemix",
    "snapmix",
    "transmix",
    "mixpro",
    "smmix",
    "tla",
    "tokenmix",
    "guidedmix",
    "puzzlemix",
]


# ---------------------------------------------------------------------------
# Helpers used by multiple augmentations.
# ---------------------------------------------------------------------------


def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        if len(x) == 2:
            return tuple(x)
        if len(x) == 1:
            return (x[0], x[0])
    return (x, x)


def _no_repeat_shuffle_idx(batch_size: int, ignore_failure: bool = False, device=None):
    """Generate a no-repeat shuffle index within a single rank.

    Mirrors ``openmixup.data._no_repeat_shuffle_idx`` so two consecutive
    indices are never identical (except as a deterministic fallback when
    repeats would be impossible to avoid).
    """
    device = device if device is not None else torch.device("cpu")
    idx = torch.randperm(batch_size, device=device)
    base = torch.arange(batch_size, device=device)
    for _ in range(10):
        if not bool((base == idx).any().item()):
            return idx
        idx = torch.randperm(batch_size, device=device)
    if ignore_failure:
        return idx
    shift = np.random.randint(1, max(2, batch_size - 1))
    return torch.tensor(
        [(i + shift) % batch_size for i in range(batch_size)],
        device=device, dtype=torch.long,
    )


def _one_hot(target: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    """Convert int labels (or already-soft labels) to a smoothed one-hot tensor."""
    if target.ndim == 2 and target.shape[1] == num_classes:
        return target.float()
    if smoothing < 0.0 or smoothing >= 1.0:
        smoothing = 0.0
    on = 1.0 - smoothing
    off = smoothing / max(1, num_classes)
    out = target.new_full((target.shape[0], num_classes), off, dtype=torch.float32)
    out.scatter_(1, target.long().view(-1, 1), on)
    return out


def _rand_bbox(size: Tuple[int, int, int, int], lam: float, return_mask: bool = False, device=None):
    """Standard CutMix bbox sampler."""
    W, H = size[2], size[3]
    cut_rat = float(np.sqrt(1.0 - max(0.0, min(1.0, lam))))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
    bby1 = int(np.clip(cy - cut_h // 2, 0, H))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_h // 2, 0, H))
    if not return_mask:
        return bbx1, bby1, bbx2, bby2
    mask = torch.zeros((1, 1, W, H), device=device or torch.device("cpu"))
    mask[:, :, bbx1:bbx2, bby1:bby2] = 1.0
    mask = mask.expand(size[0], 1, W, H)
    return bbx1, bby1, bbx2, bby2, mask


# ---------------------------------------------------------------------------
# Image-only augmentations.
# ---------------------------------------------------------------------------


# Note: ``mixup`` and ``cutmix`` are not implemented here – use
# :class:`timm.data.Mixup` / :class:`timm.data.FastCollateMixup` instead.


# ---- FMix ---------------------------------------------------------------


def _fftfreqnd(h, w=None, z=None):
    fz = fx = 0
    fy = np.fft.fftfreq(h)
    if w is not None:
        fy = np.expand_dims(fy, -1)
        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]
    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]
    return np.sqrt(fx * fx + fy * fy + fz * fz)


def _fmix_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    scale = np.ones(1) / (np.maximum(freqs, np.array([1.0 / max(w, h, z)])) ** decay_power)
    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)
    scale = np.expand_dims(scale, -1)[None, :]
    return scale * param


def _fmix_low_freq_image(decay, shape, ch=1):
    freqs = _fftfreqnd(*shape)
    spectrum = _fmix_spectrum(freqs, decay, ch, *shape)
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))
    if len(shape) == 1:
        mask = mask[:1, : shape[0]]
    elif len(shape) == 2:
        mask = mask[:1, : shape[0], : shape[1]]
    elif len(shape) == 3:
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]
    mask = mask - mask.min()
    denom = mask.max() if mask.max() > 0 else 1.0
    return mask / denom


def _fmix_binarise_mask(mask, lam, in_shape, max_soft=0.0):
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = (
        math.ceil(lam * mask.size)
        if random.random() > 0.5
        else math.floor(lam * mask.size)
    )
    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)
    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft
    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    if num_high > num_low:
        mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))
    return mask.reshape((1, *in_shape))


def _fmix_sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    if isinstance(shape, int):
        shape = (shape,)
    try:
        from scipy.stats import beta as _beta
        lam = float(_beta.rvs(alpha + 1, alpha) if reformulate else _beta.rvs(alpha, alpha))
    except Exception:
        lam = float(np.random.beta(alpha, alpha))
    mask = _fmix_low_freq_image(decay_power, shape)
    mask = _fmix_binarise_mask(mask, lam, shape, max_soft)
    return lam, mask


@torch.no_grad()
def fmix(img, gt_label, alpha=1.0, lam=None, decay_power=3, size=None,
         max_soft=0.0, reformulate=False, return_mask=False, **kwargs):
    """FMix (Harris et al., 2020) – Fourier-domain mixup mask."""
    if size is None:
        size = (int(img.shape[-2]), int(img.shape[-1]))
    lam_, mask = _fmix_sample_mask(alpha, decay_power, size, max_soft, reformulate)
    mask_t = torch.from_numpy(mask).to(device=img.device).type_as(img)
    if lam is None:
        lam = lam_
    else:
        if lam_ < lam:
            mask_t = 1 - mask_t
            lam = 1 - lam_
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]
    out = mask_t * img + (1 - mask_t) * img_
    if return_mask:
        N, _, H, W = out.shape
        out = (out, mask_t.expand(N, 1, H, W))
    return out, (y_a, y_b, float(lam))


# ---- GridMix ------------------------------------------------------------


def _gridmix_grid_mask(lam, size, cut_area_ratio, cut_aspect_ratio, n_holes,
                       hole_aspect_ratio, device):
    W, H = size[2], size[3]
    cut_area = int(H * W * cut_area_ratio)
    cut_w = int(np.sqrt(cut_area / max(1e-6, cut_aspect_ratio)))
    cut_h = int(cut_w * cut_aspect_ratio)
    cx = np.random.random()
    cy = np.random.random()
    xc1 = max(0, int((W - cut_w) * cx))
    yc1 = max(0, int((H - cut_h) * cy))
    xc2 = min(W, xc1 + cut_w)
    yc2 = min(H, yc1 + cut_h)
    width, height = max(2, xc2 - xc1), max(2, yc2 - yc1)
    n_holes = max(1, min(n_holes, width // 2))
    patch_width = max(1, math.ceil(width / n_holes))
    patch_height = max(1, int(patch_width * hole_aspect_ratio))
    ny = max(1, math.ceil(height / patch_height))
    ratio = float(np.sqrt(max(0.0, 1 - lam)))
    hole_width = int(patch_width * ratio)
    hole_height = int(patch_height * ratio)
    hole_width = min(max(hole_width, 1), patch_width - 1) if patch_width > 1 else 1
    hole_height = min(max(hole_height, 1), patch_height - 1) if patch_height > 1 else 1
    mask = torch.zeros((1, 1, W, H), device=device)
    for i in range(n_holes + 1):
        for j in range(ny + 1):
            x1 = min(patch_width * i, width)
            y1 = min(patch_height * j, height)
            x2 = min(x1 + hole_width, width)
            y2 = min(y1 + hole_height, height)
            mask[0, 0, yc1 + y1: yc1 + y2, xc1 + x1: xc1 + x2] = 1.0
    return mask


@torch.no_grad()
def gridmix(img, gt_label, alpha=1.0, lam=None, n_holes=20, hole_aspect_ratio=1.0,
            cut_area_ratio=1.0, cut_aspect_ratio=1.0, return_mask=False, **kwargs):
    """GridMix (Baek et al., 2021)."""
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    n_holes = _to_2tuple(n_holes)
    hole_aspect_ratio = _to_2tuple(hole_aspect_ratio)
    cut_area_ratio = _to_2tuple(cut_area_ratio)
    cut_aspect_ratio = _to_2tuple(cut_aspect_ratio)
    n_holes_v = random.randint(int(n_holes[0]), int(n_holes[1]))
    har = float(np.random.uniform(hole_aspect_ratio[0], hole_aspect_ratio[1]))
    car = float(np.random.uniform(cut_area_ratio[0], cut_area_ratio[1]))
    cas = float(np.random.uniform(cut_aspect_ratio[0], cut_aspect_ratio[1]))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]
    mask = _gridmix_grid_mask(lam, img.size(), car, cas, n_holes_v, har, img.device)
    out = img * (1 - mask) + img_ * mask
    lam = 1.0 - float(mask[0, 0, ...].sum().item() / (img.shape[-1] * img.shape[-2]))
    if return_mask:
        N, _, H, W = out.size()
        out = (out, mask.expand(N, 1, H, W))
    return out, (y_a, y_b, float(lam))


# ---- ResizeMix ----------------------------------------------------------


@torch.no_grad()
def resizemix(img, gt_label, scope=(0.1, 0.8), alpha=1.0, lam=None,
              use_alpha=False, interpolate_mode="nearest", return_mask=False,
              **kwargs):
    """ResizeMix (Qin et al., 2020)."""
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_resize = img[rand_idx].clone()
    shuffled_gt = gt_label[rand_idx]
    _, _, h, w = img.size()
    if lam is None:
        if use_alpha:
            tao = float(np.random.beta(alpha, alpha))
            if tao < scope[0] or tao > scope[1]:
                tao = float(np.random.uniform(scope[0], scope[1]))
        else:
            tao = float(np.random.uniform(scope[0], scope[1]))
    else:
        tao = float(min(max(lam, scope[0]), scope[1]))
    bbx1, bby1, bbx2, bby2 = _rand_bbox(img.size(), tao)
    img_resize = F.interpolate(
        img_resize, (bby2 - bby1, bbx2 - bbx1), mode=interpolate_mode,
    )
    img = img.clone()
    img[:, :, bby1:bby2, bbx1:bbx2] = img_resize
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(w * h))
    if return_mask:
        mask = torch.zeros((img.size(0), 1, h, w), device=img.device)
        mask[:, :, bby1:bby2, bbx1:bbx2] = 1.0
        img = (img, mask)
    return img, (gt_label, shuffled_gt, float(lam))


# ---- SmoothMix ----------------------------------------------------------


def _gaussian_kernel(kernel_size, rand_w, rand_h, sigma, device):
    s = kernel_size * 2
    x_cord = torch.arange(s, device=device)
    x_grid = x_cord.repeat(s).view(s, s)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    xy_grid = torch.roll(xy_grid, int(rand_w), 0)
    xy_grid = torch.roll(xy_grid, int(rand_h), 1)
    crop = s // 4
    xy_grid = xy_grid[crop: s - crop, crop: s - crop]
    mean = (s - 1) / 2.0
    var = float(sigma) ** 2
    g = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * var))
    return g.view(kernel_size, kernel_size)


@torch.no_grad()
def smoothmix(img, gt_label, alpha=1.0, lam=None, return_mask=False, **kwargs):
    """SmoothMix (Lee et al., 2020)."""
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]
    b, _, h, w = img.size()
    rand_w = int(torch.randint(0, w, (1,)).item() - w / 2)
    rand_h = int(torch.randint(0, h, (1,)).item() - h / 2)
    sigma = float(((torch.rand(1).item() / 4) + 0.25) * h)
    kernel = _gaussian_kernel(h, rand_h, rand_w, sigma, device=img.device)
    out = img * (1 - kernel) + img_ * kernel
    lam_eff = float(torch.sum(kernel).item() / (h * w))
    if return_mask:
        out = (out, kernel.expand(b, 1, h, w))
    return out, (y_a, y_b, lam_eff)


# ---- SaliencyMix --------------------------------------------------------


def _saliency_bbox(img: torch.Tensor, lam: float):
    try:
        from cv2.saliency import StaticSaliencyFineGrained_create
    except Exception:
        StaticSaliencyFineGrained_create = None
    size = img.size()
    W, H = size[1], size[2]
    cut_rat = float(np.sqrt(1.0 - max(0.0, min(1.0, lam))))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    if StaticSaliencyFineGrained_create is None:
        # Fall back to a centered-but-jittered bbox so the visual still
        # mimics SaliencyMix when opencv-contrib is unavailable.
        x = int(W * 0.5 + np.random.uniform(-0.1, 0.1) * W)
        y = int(H * 0.5 + np.random.uniform(-0.1, 0.1) * H)
    else:
        temp = (
            img.detach().to(torch.float32).clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
        )
        saliency = StaticSaliencyFineGrained_create()
        ok, smap = saliency.computeSaliency(temp)
        if not ok:
            x = int(W * 0.5)
            y = int(H * 0.5)
        else:
            smap = (smap * 255).astype(np.uint8)
            mx = np.unravel_index(np.argmax(smap, axis=None), smap.shape)
            x, y = int(mx[0]), int(mx[1])
    bbx1 = int(np.clip(x - cut_w // 2, 0, W))
    bby1 = int(np.clip(y - cut_h // 2, 0, H))
    bbx2 = int(np.clip(x + cut_w // 2, 0, W))
    bby2 = int(np.clip(y + cut_h // 2, 0, H))
    return bbx1, bby1, bbx2, bby2


@torch.no_grad()
def saliencymix(img, gt_label, alpha=1.0, lam=None, return_mask=False, **kwargs):
    """SaliencyMix (Uddin et al., 2021)."""
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]
    b, _, h, w = img.size()
    bbx1, bby1, bbx2, bby2 = _saliency_bbox(img[rand_idx[0]], lam)
    img = img.clone()
    img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(w * h))
    if return_mask:
        mask = torch.zeros((1, 1, h, w), device=img.device)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 1
        mask = mask.expand(b, 1, h, w)
        img = (img, mask)
    return img, (y_a, y_b, float(lam))


# ---------------------------------------------------------------------------
# Model-aware augmentations.
# ---------------------------------------------------------------------------


@torch.no_grad()
def attentivemix(img, gt_label, alpha=1.0, lam=None, features=None,
                 grid_scale=32, top_k=6, return_mask=False, **kwargs):
    """AttentiveMix (Walawalkar et al., 2020).

    ``features`` must be a ``(N, C, h, w)`` tensor – e.g. the last feature
    map of the model. For showcase rendering you can pass random features.
    """
    if features is None:
        raise ValueError("attentivemix requires `features=(N,C,h,w)` feature maps")
    bs, _, att_size, _ = features.size()
    att_grid = att_size ** 2
    if att_size * grid_scale != img.size(2):
        grid_scale = img.size(2) / att_size
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    if top_k is None:
        top_k = min(max(1, int(att_grid * lam)), att_grid)

    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]

    feats = features.mean(1)
    _, att_idx = feats.view(bs, att_grid).topk(top_k)
    att_idx = torch.cat([
        (att_idx // att_size).unsqueeze(1),
        (att_idx % att_size).unsqueeze(1),
    ], dim=1)
    mask = torch.zeros(bs, 1, att_size, att_size, device=img.device)
    for i in range(bs):
        mask[i, 0, att_idx[i, 0, :], att_idx[i, 1, :]] = 1.0
    mask = F.interpolate(mask, scale_factor=float(grid_scale), mode="nearest")
    lam_eff = float(mask[0, 0, ...].mean().item())
    out = mask * img + (1 - mask) * img_
    if return_mask:
        out = (out, mask)
    return out, (y_a, y_b, lam_eff)


# ---- AlignMix (Sinkhorn over feature spaces) ----------------------------


class _SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=100):
        super().__init__()
        self.eps = float(eps)
        self.max_iter = int(max_iter)

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        return torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

    def _M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def forward(self, x, y):
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        bs = x.shape[0] if x.dim() == 3 else 1
        mu = torch.full((bs, x_points), 1.0 / x_points, dtype=torch.float32, device=x.device)
        nu = torch.full((bs, y_points), 1.0 / y_points, dtype=torch.float32, device=x.device)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        for _ in range(self.max_iter):
            u_prev = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self._M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(
                self._M(C, u, v).transpose(-2, -1), dim=-1)) + v
            if (u - u_prev).abs().sum(-1).mean().item() < 1e-1:
                break
        return torch.exp(self._M(C, u, v))


@torch.no_grad()
def alignmix(img, gt_label, alpha=1.0, lam=None, eps=0.1, max_iter=100, **kwargs):
    """AlignMix (Venkataramanan et al., 2022).

    Note: AlignMix mixes *feature maps*, not raw images. ``img`` is
    interpreted as a feature tensor of shape ``(N, C, h, w)``.
    """
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]
    B, C, H, W = img.shape
    f1 = img.view(B, C, -1)
    f2 = img_.view(B, C, -1)
    P = _SinkhornDistance(eps=eps, max_iter=max_iter)(
        f1.permute(0, 2, 1), f2.permute(0, 2, 1),
    ).detach() * (H * W)
    if random.randint(0, 1) == 0:
        f_aligned = torch.matmul(f2, P.permute(0, 2, 1)).view(B, C, H, W)
        out = img * lam + f_aligned * (1 - lam)
    else:
        f_aligned = torch.matmul(f1, P).view(B, C, H, W)
        out = f_aligned * lam + img_ * (1 - lam)
    return out, (y_a, y_b, float(lam))


# ---- SnapMix ------------------------------------------------------------


@torch.no_grad()
def snapmix(img, gt_label, alpha=1.0, lam=None, features=None, **kwargs):
    """SnapMix (Huang et al., 2021).

    ``features`` is the per-image saliency / CAM tensor of shape
    ``(N, h, w)``.
    """
    if features is None:
        raise ValueError("snapmix requires saliency `features=(N,h,w)`")
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
        lam_ = float(np.random.beta(alpha, alpha))
    else:
        lam_ = lam

    b, _, h, w = img.size()
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    feats = features
    feats_ = feats[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]

    def _bbox(size, lam_v):
        W, H = size[2], size[3]
        cr = float(np.sqrt(max(0.0, 1.0 - lam_v)))
        cw, ch = int(W * cr), int(H * cr)
        cx, cy = (np.random.randint(W), np.random.randint(H)) if W > 0 and H > 0 else (0, 0)
        return (
            int(np.clip(cx - cw // 2, 0, W)), int(np.clip(cy - ch // 2, 0, H)),
            int(np.clip(cx + cw // 2, 0, W)), int(np.clip(cy + ch // 2, 0, H)),
        )

    bbx1, bby1, bbx2, bby2 = _bbox(img.size(), lam)
    bbx1_, bby1_, bbx2_, bby2_ = _bbox(img.size(), lam_)
    area = (bby2 - bby1) * (bbx2 - bbx1)
    area_ = (bby2_ - bby1_) * (bbx2_ - bbx1_)

    img = img.clone()
    if area_ > 0 and area > 0:
        ncont = img[rand_idx, :, bbx1_:bbx2_, bby1_:bby2_].clone()
        ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1),
                              mode="bilinear", align_corners=True)
        img[:, :, bbx1:bbx2, bby1:bby2] = ncont
        lam_a = 1.0 - feats[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (feats.sum(2).sum(1) + 1e-8)
        lam_b = feats_[:, bbx1_:bbx2_, bby1_:bby2_].sum(2).sum(1) / (feats_.sum(2).sum(1) + 1e-8)
        lam_a[torch.isnan(lam_a)] = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(h * w))
        lam_b[torch.isnan(lam_b)] = (bbx2 - bbx1) * (bby2 - bby1) / float(h * w)
    else:
        lam_a = torch.ones(b, device=img.device)
        lam_b = torch.zeros(b, device=img.device)
    return img, (y_a, y_b, lam_a, lam_b)


# ---- TransMix-style label-only adjustments (visualization fallback) -----


@torch.no_grad()
def transmix(img, gt_label, alpha=1.0, lam=None, attn=None, mask=None,
             patch_shape=None, return_mask=False, ratio=0.5, **kwargs):
    """TransMix (Chen et al., 2022)."""
    if lam is None and mask is None:
        lam0 = float(np.random.beta(alpha, alpha))
    else:
        lam0 = float(lam) if lam is not None else 0.5
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    b, _, h, w = img.size()
    y_a = gt_label
    y_b = gt_label[rand_idx]
    if mask is None:
        bbx1, bby1, bbx2, bby2, mask = _rand_bbox(img.size(), lam0, return_mask=True, device=img.device)
        img = img.clone()
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam0 = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(h * w))
    else:
        img = (1 - mask) * img + mask * img_

    lam1 = lam0
    if attn is not None and patch_shape is not None:
        mask_ = nn.Upsample(size=patch_shape)(mask).view(b, -1).int()
        attn_ = torch.mean(attn[:, :, 0, 1:], dim=1)
        w1 = torch.sum(mask_ * attn_, dim=1)
        w2 = torch.sum((1 - mask_) * attn_, dim=1)
        lam1 = (w2 / (w1 + w2)).mean().item()
    lam_eff = float(lam0) * ratio + float(lam1) * (1 - ratio)
    if return_mask:
        img = (img, mask)
    return img, (y_a, y_b, lam_eff)


# ---- MixPro (MaskMix + progressive attention) ---------------------------


@torch.no_grad()
def mixpro(img, gt_label, attn=None, alpha=1.0, lam=None, mask_patch_size=64,
           model_patch_size=16, return_mask=False, **kwargs):
    """MixPro (Zhao et al., 2023) – MaskMix variant for ViTs."""
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    b, _, h, w = img.size()
    mask_num = math.ceil(h / mask_patch_size)
    scale_ = mask_patch_size // model_patch_size
    scale = mask_patch_size

    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    y_a = gt_label
    y_b = gt_label[rand_idx]

    token_count = mask_num ** 2
    mask_count = int(np.ceil(token_count * lam))
    masks = []
    for _ in range(b):
        m = np.zeros((token_count,), dtype=np.int32)
        idx = np.random.permutation(token_count)[:mask_count]
        m[idx] = 1
        masks.append(m.reshape((mask_num, mask_num)))
    masks_full = np.array([m.repeat(scale, axis=0).repeat(scale, axis=1) for m in masks])
    mask = torch.from_numpy(masks_full).to(device=img.device, dtype=img.dtype)
    mask = mask.unsqueeze(1).repeat(1, img.shape[1], 1, 1)
    mask = mask[:, :, :h, :w]
    out = img * mask + img[rand_idx] * (1 - mask)
    lam_eff = float(mask_count / max(1, token_count))
    if return_mask:
        out = (out, mask[:, :1])
    return out, (y_a, y_b, lam_eff)


# ---- SMMix (token-level swap based on attention) ------------------------


@torch.no_grad()
def smmix(img, gt_label, attn=None, lam=None, side=14, min_side_ratio=0.25,
          max_side_ratio=0.75, return_mask=False, **kwargs):
    """SMMix (Chen et al., 2023). Requires attention map."""
    if attn is None:
        # Fall back to a uniform attention map so the showcase still renders.
        b = img.size(0)
        attn = torch.ones(b, 1, side * side + 1, side * side + 1, device=img.device)

    b, _, h, w = img.size()
    y_a = gt_label
    y_b = gt_label.flip(0)
    min_side = max(1, int(side * min_side_ratio))
    max_side = max(min_side + 1, int(side * max_side_ratio))
    rect_size = (random.randint(min_side, max_side - 1),) * 2
    if lam is None:
        lam = (side ** 2 - rect_size[0] * rect_size[1]) / float(side ** 2)
    patch_size = h // side
    inputs = F.unfold(img, patch_size, stride=patch_size).transpose(1, 2)
    a = torch.mean(attn[:, :, 0, 1:], dim=1).reshape(-1, side, side).unsqueeze(1)
    rect_attn = F.unfold(a, rect_size, stride=1).sum(dim=1)
    min_idx = torch.argmin(rect_attn, dim=1)
    max_idx = torch.argmax(rect_attn, dim=1)

    def _expand(idx, rect, total=(side, side)):
        total_idx = torch.arange(total[0] * total[1], device=img.device).reshape(1, 1, total[0], total[1]).float()
        unfolded = F.unfold(total_idx, rect, stride=1).transpose(1, 2).long()
        return unfolded.index_select(dim=1, index=idx).squeeze(0)

    min_region = _expand(min_idx, rect_size)
    max_region = _expand(max_idx.flip(0), rect_size)

    def _batch_idx(x_shape, idx):
        B, N, _ = x_shape
        offset = torch.arange(B, device=img.device).view(B, 1) * N
        return (idx + offset).reshape(-1)

    min_region_global = _batch_idx(inputs.shape, min_region)
    max_region_global = _batch_idx(inputs.shape, max_region)
    inputs_flat = inputs.reshape(b * inputs.shape[1], inputs.shape[2])
    flipped = inputs.flip(0).reshape(b * inputs.shape[1], inputs.shape[2])
    inputs_flat[min_region_global] = flipped[max_region_global]
    inputs = inputs_flat.reshape(b, inputs.shape[1], inputs.shape[2])
    out = F.fold(inputs.transpose(1, 2), h, patch_size, stride=patch_size)
    if return_mask:
        src = torch.zeros((b, 1, side, side), device=img.device).reshape(-1)
        src[min_region_global] = 1
        src = src.reshape(b, 1, side, side)
        src = F.interpolate(src, scale_factor=patch_size, mode="nearest")
        out = (out, (1 - src, src))
    return out, (y_a, y_b, float(lam))


# ---- TLA (Token Labeling Align) -----------------------------------------


@torch.no_grad()
def tla(img, gt_label, alpha=1.0, lam=None, patch_size=16, return_mask=False,
        **kwargs):
    """TLA (Wang et al., 2023)."""
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    b, c, h, w = img.size()
    y_a = gt_label
    y_b = gt_label[rand_idx]

    img = img.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    img = img.permute(0, 1, 2, 4, 3, 5).contiguous()
    ratio = float(np.sqrt(max(0.0, 1.0 - lam)))
    img_h, img_w = h // patch_size, w // patch_size
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    cy = np.random.randint(0, max(1, img_h))
    cx = np.random.randint(0, max(1, img_w))
    yl = int(np.clip(cy - cut_h // 2, 0, img_h))
    yu = int(np.clip(cy + cut_h // 2, 0, img_h))
    xl = int(np.clip(cx - cut_w // 2, 0, img_w))
    xu = int(np.clip(cx + cut_w // 2, 0, img_w))
    img[:, :, yl:yu, xl:xu, :, :] = img[rand_idx][:, :, yl:yu, xl:xu, :, :]
    img = img.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, -1, h, w)
    bbox_area = (yu - yl) * (xu - xl)
    lam = 1.0 - bbox_area / float(img_h * img_w)
    mask = torch.zeros((b, 1, img_h, img_w), device=img.device)
    mask[:, :, yl:yu, xl:xu] = 1
    mask = F.interpolate(mask, scale_factor=patch_size, mode="nearest")
    img = (1 - mask) * img + mask * img_
    if return_mask:
        img = (img, mask)
    return img, (y_a, y_b, float(lam))


# ---- TokenMix -----------------------------------------------------------


@torch.no_grad()
def tokenmix(img, gt_label, attn=None, alpha=1.0, lam=None, mask_type="block",
             minimum_tokens=14, return_mask=False, **kwargs):
    """TokenMix (Liu et al., 2022)."""
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    b, _, h, w = img.size()
    y_a = gt_label
    y_b = gt_label[rand_idx]

    width, height = 14, 14
    mask_ratio = 1.0 - lam
    num_masking = min(width * height, int(mask_ratio * width * height) + minimum_tokens)
    if mask_type == "random":
        mask_arr = np.zeros((width * height,), dtype=np.int32)
        idx = np.random.permutation(width * height)[:num_masking]
        mask_arr[idx] = 1
        mask_arr = mask_arr.reshape(width, height)
    else:  # block
        mask_arr = np.zeros((width, height), dtype=np.int32)
        mask_count = 0
        log_aspect = (math.log(0.3), math.log(1 / 0.3))
        while mask_count < num_masking:
            max_patches = num_masking - mask_count
            delta = 0
            for _ in range(10):
                target_area = random.uniform(1, max_patches)
                aspect = math.exp(random.uniform(*log_aspect))
                hh = int(round(math.sqrt(target_area * aspect)))
                ww = int(round(math.sqrt(target_area / aspect)))
                if 0 < ww < width and 0 < hh < height:
                    top = random.randint(0, height - hh)
                    left = random.randint(0, width - ww)
                    block = mask_arr[top: top + hh, left: left + ww]
                    n_masked = int(block.sum())
                    if 0 < hh * ww - n_masked <= max_patches:
                        for i in range(top, top + hh):
                            for j in range(left, left + ww):
                                if mask_arr[i, j] == 0:
                                    mask_arr[i, j] = 1
                                    delta += 1
                        if delta > 0:
                            break
            if delta == 0:
                break
            mask_count += delta
    mask = torch.from_numpy(mask_arr).float().unsqueeze(0).unsqueeze(0).to(img.device)
    mask_full = F.interpolate(mask, size=(h, w), mode="nearest")
    out = (1 - mask_full) * img + mask_full * img_
    if return_mask:
        out = (out, mask_full.expand(b, 1, h, w))
    return out, (y_a, y_b, float(lam))


# ---- GuidedMix ----------------------------------------------------------


@torch.no_grad()
def guidedmix(img, gt_label, alpha=1.0, lam=None, features=None,
              size=(7, 7), sigma=(3.0, 3.0), return_mask=False, **kwargs):
    """GuidedMix (Kang et al., 2023). ``features`` is per-image saliency."""
    if features is None:
        raise ValueError("guidedmix requires saliency `features=(N,1,H,W)`")
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]

    import torchvision.transforms.functional as TF
    feats = TF.gaussian_blur(features, list(size), list(sigma))
    feats = feats / (feats.sum(dim=[-1, -2], keepdim=True) + 1e-8)
    feats_ = feats[rand_idx]
    norm_feats = feats / (feats + feats_ + 1e-8)
    lam_v = norm_feats.mean(dim=[-1, -2]).squeeze(-1)
    mask = torch.cat([norm_feats] * img.shape[1], dim=1)
    out = mask * img + (1 - mask) * img_
    if return_mask:
        out = (out, mask[:, :1])
    return out, (y_a, y_b, float(lam_v.mean().item()))


# ---- PuzzleMix (simplified, gradient-free fallback) ---------------------


@torch.no_grad()
def puzzlemix(img, gt_label, alpha=0.5, lam=None, features=None, block_num=4,
              **kwargs):
    """PuzzleMix (Kim et al., 2020) – simplified, model-aware version.

    The full PuzzleMix needs ``pyGCO`` (graphcut) and pixel-level loss
    gradients which we do not have at the showcase site. This stand-in
    implementation produces a saliency-weighted block-level mask which is
    visually similar to the original output without those dependencies.
    If proper saliency `features=(N,h,w)` are provided, they guide the
    mask; otherwise random noise is used.
    """
    if lam is None:
        lam = float(np.random.beta(alpha, alpha))
    rand_idx = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True, device=img.device)
    img_ = img[rand_idx]
    y_a = gt_label
    y_b = gt_label[rand_idx]
    b, _, h, w = img.shape
    if isinstance(block_num, (tuple, list)):
        block_num = int(2 ** np.random.randint(int(block_num[0]), int(block_num[1])))
    block_num = int(block_num)
    if features is None:
        sal = torch.rand(b, block_num, block_num, device=img.device)
    else:
        sal = F.adaptive_avg_pool2d(features.unsqueeze(1), (block_num, block_num)).squeeze(1)
    sal_ = sal[rand_idx]
    diff = sal - sal_
    flat = diff.view(b, -1)
    k = max(1, int(round(block_num * block_num * (1 - lam))))
    _, top_idx = flat.topk(k, dim=1)
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(1, top_idx, 1.0)
    mask = mask_flat.view(b, 1, block_num, block_num)
    mask = F.interpolate(mask, size=(h, w), mode="nearest")
    out = mask * img + (1 - mask) * img_
    lam_eff = float(mask.view(b, -1).mean().item())
    return out, (y_a, y_b, lam_eff)


# ---------------------------------------------------------------------------
# Unified dispatcher.
# ---------------------------------------------------------------------------


# (name, requires_features?, requires_attention?, description)
# Note: ``mixup``, ``cutmix`` and ``augmix`` are intentionally NOT in this
# table because the LabelMix codebase already provides them via
# ``timm.data.Mixup`` / ``timm.data.auto_augment``. See module docstring.
OPENMIXUP_AUG_INFO: Dict[str, Dict[str, Any]] = {
    "fmix":        {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Fourier-domain low-frequency binary mask."},
    "gridmix":     {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Grid-of-holes mixup."},
    "resizemix":   {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Resize b and paste it into a."},
    "smoothmix":   {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Gaussian-kernel smooth blend."},
    "saliencymix": {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Saliency-guided CutMix bbox (uses cv2 saliency, no model)."},
    "alignmix":    {"image_only": False, "needs_features": True,  "needs_attn": False,
                    "desc": "Optimal-transport alignment in feature space."},
    "attentivemix":{"image_only": False, "needs_features": True,  "needs_attn": False,
                    "desc": "Top-k attentive grid CutMix."},
    "snapmix":     {"image_only": False, "needs_features": True,  "needs_attn": False,
                    "desc": "Saliency-proportional CutMix with CAM."},
    "transmix":    {"image_only": False, "needs_features": False, "needs_attn": True,
                    "desc": "CutMix + ViT attention-corrected labels."},
    "mixpro":      {"image_only": False, "needs_features": False, "needs_attn": True,
                    "desc": "MaskMix + progressive attention labels."},
    "smmix":       {"image_only": False, "needs_features": False, "needs_attn": True,
                    "desc": "Self-motivated min/max attention swap."},
    "tla":         {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Token-Label Align CutMix on ViT patches."},
    "tokenmix":    {"image_only": True,  "needs_features": False, "needs_attn": False,
                    "desc": "Block / random token-level mix for ViTs."},
    "guidedmix":   {"image_only": False, "needs_features": True,  "needs_attn": False,
                    "desc": "Saliency-guided soft mask mix."},
    "puzzlemix":   {"image_only": False, "needs_features": True,  "needs_attn": False,
                    "desc": "Saliency-aware block puzzle (simplified)."},
}

OPENMIXUP_AUG_NAMES: Tuple[str, ...] = tuple(OPENMIXUP_AUG_INFO.keys())
OPENMIXUP_IMAGE_ONLY_AUGS: Tuple[str, ...] = tuple(
    n for n, info in OPENMIXUP_AUG_INFO.items() if info["image_only"]
)
OPENMIXUP_MODEL_AWARE_AUGS: Tuple[str, ...] = tuple(
    n for n, info in OPENMIXUP_AUG_INFO.items() if not info["image_only"]
)

# In this codebase every augmentation that has a CLI flag is in
# OPENMIXUP_AUG_NAMES (mixup/cutmix/augmix are *not* present, since they
# are handled by their own canonical CLI flags). This alias is kept for
# backwards compatibility with imports added in a previous refactor.
OPENMIXUP_CLI_AUG_NAMES: Tuple[str, ...] = OPENMIXUP_AUG_NAMES

_AUG_FUNCS = {
    "fmix": fmix,
    "gridmix": gridmix,
    "resizemix": resizemix,
    "smoothmix": smoothmix,
    "saliencymix": saliencymix,
    "alignmix": alignmix,
    "attentivemix": attentivemix,
    "snapmix": snapmix,
    "transmix": transmix,
    "mixpro": mixpro,
    "smmix": smmix,
    "tla": tla,
    "tokenmix": tokenmix,
    "guidedmix": guidedmix,
    "puzzlemix": puzzlemix,
}


def apply_openmixup_aug(name: str, img: torch.Tensor, target: torch.Tensor,
                        **kwargs) -> Tuple[torch.Tensor, Tuple]:
    """Run the named OpenMixup augmentation and return ``(img, info)``.

    ``info`` follows the per-aug convention. Use :func:`openmixup_to_soft_target`
    to convert it to a soft label tensor of shape ``(N, num_classes)``.
    """
    name = name.lower()
    if name not in _AUG_FUNCS:
        raise KeyError(f"Unknown OpenMixup aug '{name}'. Choose from {OPENMIXUP_AUG_NAMES}.")
    return _AUG_FUNCS[name](img, target, **kwargs)


def openmixup_to_soft_target(target_info: Tuple, num_classes: int,
                             smoothing: float = 0.0) -> torch.Tensor:
    """Convert a per-aug ``target_info`` tuple into a soft-label tensor."""
    if isinstance(target_info, tuple) and len(target_info) >= 3:
        y_a = target_info[0]
        y_b = target_info[1]
        y_a = _one_hot(y_a, num_classes, smoothing)
        y_b = _one_hot(y_b, num_classes, smoothing)
        if len(target_info) == 3:
            lam = float(target_info[2])
            return lam * y_a + (1.0 - lam) * y_b
        if len(target_info) == 4:
            # SnapMix-style: per-image lam_a and lam_b (or lam, lam_).
            third = target_info[2]
            fourth = target_info[3]
            if torch.is_tensor(third) and third.ndim >= 1:
                la = third.float().view(-1, 1)
                lb = fourth.float().view(-1, 1)
                return la * y_a + lb * y_b
            lam = float(third)
            return lam * y_a + (1.0 - lam) * y_b
    raise ValueError(f"Unsupported target_info: {target_info!r}")


class OpenMixupAug:
    """Callable wrapper that applies one configured OpenMixup augmentation.

    Mirrors the interface of :class:`timm.data.Mixup` so it can be slotted
    into the training loop with minimal changes:

        ``input, target = openmixup_aug(input, target)``

    where ``target`` becomes a soft-label tensor of shape ``(N, num_classes)``.

    Methods that need feature/attention maps will raise unless the caller
    sets ``self.features`` / ``self.attn`` before each call (or passes
    ``features=`` / ``attn=`` via ``**call_kwargs``).
    """

    def __init__(self, name: str, num_classes: int, *, alpha: float = 1.0,
                 prob: float = 1.0, label_smoothing: float = 0.0,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 **aug_kwargs: Any) -> None:
        if name not in _AUG_FUNCS:
            raise KeyError(f"Unknown OpenMixup aug '{name}'.")
        self.name = name
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.prob = float(prob)
        self.label_smoothing = float(label_smoothing)
        self.mean = tuple(mean) if mean is not None else None
        self.std = tuple(std) if std is not None else None
        self.aug_kwargs = dict(aug_kwargs)
        self.mixup_enabled = True
        self.features: Optional[torch.Tensor] = None
        self.attn: Optional[torch.Tensor] = None

    @property
    def info(self) -> Dict[str, Any]:
        return OPENMIXUP_AUG_INFO[self.name]

    def __call__(self, img: torch.Tensor, target: torch.Tensor, **call_kwargs):
        if not self.mixup_enabled or np.random.rand() >= self.prob:
            return img, _one_hot(target, self.num_classes, self.label_smoothing)
        kwargs = dict(self.aug_kwargs)
        kwargs.setdefault("alpha", self.alpha)
        if self.info["needs_features"] and "features" not in call_kwargs:
            if self.features is None:
                raise RuntimeError(
                    f"{self.name} needs `features`; set OpenMixupAug.features "
                    "before calling, or pass it as keyword argument."
                )
            call_kwargs["features"] = self.features
        if self.info["needs_attn"] and "attn" not in call_kwargs:
            if self.attn is None:
                raise RuntimeError(
                    f"{self.name} needs `attn`; set OpenMixupAug.attn before "
                    "calling, or pass it as keyword argument."
                )
            call_kwargs["attn"] = self.attn
        kwargs.update(call_kwargs)
        out, info = _AUG_FUNCS[self.name](img, target, **kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        soft_target = openmixup_to_soft_target(info, self.num_classes, self.label_smoothing)
        return out, soft_target
