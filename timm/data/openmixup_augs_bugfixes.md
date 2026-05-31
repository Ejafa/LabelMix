# OpenMixup Augmentation Port — Bugfix Log

This file records the correctness fixes applied to
[`timm/data/openmixup_augs.py`](openmixup_augs.py) after auditing it against
the reference OpenMixup implementations in
[`augs.xml`](../../augs.xml).

All fixes are inlined in `openmixup_augs.py`. This document explains *what*
was wrong, *why* it was wrong, and *what* the fix does. Each fix is marked
with a `# Bugfix:` comment in the source for traceability.

---

## 1. `resizemix` — wrong bbox side ratio

**Before**

```python
bbx1, bby1, bbx2, bby2 = _rand_bbox(img.size(), tao)
```

`_rand_bbox` is the CutMix sampler that interprets `tao` as `lam` and uses
`sqrt(1 - tao)` as the side ratio. ResizeMix, however, samples `tao`
*directly* as the side ratio of the pasted patch (the OpenMixup reference
defines `rand_bbox_tao(size, tao)` with `cut_w = int(W * tao)`).

The previous port therefore shrank every pasted patch to `sqrt(1 - tao)`
of the image — for the documented `scope=(0.1, 0.8)`, the actual side
ratio range was `[~0.45, ~0.95]` instead of `[0.1, 0.8]`, and the produced
`lam` was systematically biased.

**After**

We replicate `rand_bbox_tao` inline:

```python
cut_w = int(W * tao)
cut_h = int(H * tao)
```

so the pasted patch is exactly `tao × tao` of the image, matching the
OpenMixup reference.

---

## 2. `snapmix` — missing same-label lambda swap

**Before**

The port computed `lam_a` (1 minus the saliency mass under the pasted
patch in image *a*) and `lam_b` (saliency mass under the source patch in
image *b*) but skipped the same-label collapse step from the SnapMix
reference:

```python
tmp = lam_a.clone()
lam_a[same_label] += lam_b[same_label]
lam_b[same_label] += tmp[same_label]
```

When the source and target images of a sample carry the same class label,
the soft target `lam_a · y_a + lam_b · y_b` collapses to `(lam_a + lam_b) · y`,
which generally does **not** sum to 1 and therefore violates the
probability-mass invariant of soft labels.

**After**

We compute `same_label` (handling both integer and one-hot inputs) and
apply the swap exactly as in the reference:

```python
tmp = lam_a.clone()
if bool(same_label.any()):
    lam_a[same_label] = lam_a[same_label] + lam_b[same_label]
    lam_b[same_label] = tmp[same_label] + lam_b[same_label]
```

The fallback for NaN entries was also corrected so it sums to 1
(`lam_b ← 1 - lam_fallback` instead of the `area / (h * w)` placeholder).

---

## 3. `transmix` — per-image lambda flattened to a scalar

**Before**

```python
lam1 = (w2 / (w1 + w2)).mean().item()
lam_eff = float(lam0) * ratio + float(lam1) * (1 - ratio)
```

TransMix's whole point is that the attention map yields a *per-image*
mixing ratio that corrects the area-based `lam0`. Calling `.mean()`
collapses the per-image vector to a single scalar, which discards the
attention-correction signal for everything except the batch average and
defeats the augmentation's purpose.

**After**

```python
lam1 = w2 / (w1 + w2 + 1e-8)              # (B,)
lam_eff = float(lam0) * ratio + lam1 * (1 - ratio)   # (B,) tensor
```

`openmixup_to_soft_target` now recognises a 1-D `lam` in the
`(y_a, y_b, lam)` triple and broadcasts it into `(B, 1)` before forming
the soft label.

---

## 4. `mixpro` — attention-corrected label branch deleted

**Before**

The port produced a single batch-wide scalar `lam_eff = mask_count / token_count`
and never consumed `attn`, so MixPro was indistinguishable from a
naïve area-based MaskMix.

**After**

We follow the reference `mask_mix` flow:

* generate per-image token masks at `mask_num × mask_num`,
* upsample to `model_patch_size × model_patch_size` (used for the
  attention-correction step) and to image resolution (used to mix the
  pixels);
* when `attn` is provided, derive the per-image
  `lam_ = w1 / (w1 + w2)` from the attention map and surface it as the
  third entry of `target_info` so the soft label becomes
  `lam_ · y_a + (1 - lam_) · y_b`;
* when `attn` is absent (e.g. inside the showcase script) we fall back
  to the area-based scalar so the function still has a sensible
  rendering-only behaviour.

---

## 5. `tokenmix` — attention-weighted soft label deleted

**Before**

The port returned `(y_a, y_b, float(lam))` even when an attention map was
supplied. The OpenMixup reference, however, returns
`(y_a, y_b, score_a, score_b)` where

```python
score_a = (mask * attn).reshape(B, -1).sum(1)
score_b = ((1 - mask) * attn[rand_idx]).reshape(B, -1).sum(1)
```

are the per-image attention-weighted scores used to form the soft label.

**After**

* When `attn` is provided we now return `(y_a, y_b, score_a, score_b)`.
  `openmixup_to_soft_target` already handles the 4-tuple form by
  broadcasting `score_a` / `score_b` into `(B, 1)`.
* When `attn` is absent we keep the area-based scalar lambda fallback
  used by the showcase.

---

## 6. `guidedmix` — per-image lambda averaged out

**Before**

```python
lam_v = norm_feats.mean(dim=[-1, -2]).squeeze(-1)
return out, (y_a, y_b, float(lam_v.mean().item()))
```

The reference produces a per-image `lam` and `lam_ = 1 - lam` and returns
`(y_a, y_b, lam, lam_)`. Averaging across the batch defeats the
saliency-guided weighting that GuidedMix is built around.

Additionally, the previous port used `torch.cat([norm_feats] * C, dim=1)`,
which only worked when `C == 3`; for greyscale or 4-channel inputs it
would silently fail.

**After**

```python
lam_a = norm_feats.mean(dim=[-1, -2]).reshape(-1)   # (B,)
lam_b = 1.0 - lam_a                                 # (B,)
mask = norm_feats.expand(-1, img.shape[1], -1, -1)  # any C
return out, (y_a, y_b, lam_a, lam_b)
```

---

## 7. `openmixup_to_soft_target` — accept per-image lambda in 3-tuples

The TransMix and MixPro fixes above introduce 3-tuples whose third
element is a 1-D `(B,)` tensor instead of a scalar. The helper now
broadcasts that vector into a `(B, 1)` weight before mixing, mirroring
the existing 4-tuple path used by SnapMix / GuidedMix / TokenMix.

---

## 8. `puzzlemix` — left as a documented approximation

The full PuzzleMix algorithm depends on `pyGCO` (graph-cut) and
pixel-level loss gradients which are not available in this codebase. The
existing port already documents this in its docstring and produces a
saliency-weighted block mask which is visually close to the reference
output. No correctness fix is possible without adding those
dependencies; we therefore keep the simplified block-puzzle stand-in.

---

## Verification checklist

* [x] `resizemix` paste-side ratio matches `tao` (not `sqrt(1 - tao)`).
* [x] `snapmix` same-label lambda swap restored; NaN fallback sums to 1.
* [x] `transmix` returns per-image `lam` when an attention map is given.
* [x] `mixpro` attention-corrected lambda surfaced through `target_info`.
* [x] `tokenmix` returns `(y_a, y_b, score_a, score_b)` when `attn` is set.
* [x] `guidedmix` returns per-image `(lam_a, lam_b)`; mask works for any C.
* [x] `openmixup_to_soft_target` handles 1-D lambda in 3-tuples.
* [x] No new lints in `openmixup_augs.py`.
