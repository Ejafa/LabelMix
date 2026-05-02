"""Gradient x Input attribution for the diagnostic pipeline.

Given a trained model + a composed diagnostic sample, compute per-class
Gradient-x-Input saliency maps::

    x = x.clone().detach()
    x.requires_grad_(True)

    logits = model(x)
    target_logit = logits[0, target_class]

    model.zero_grad(set_to_none=True)
    target_logit.backward()

    attr = x * x.grad
    heatmap = attr.abs().sum(dim=1)     # spatial map

We persist:

    - ``attr``     -- raw (1, 3, H, W) signed attribution   (float32, CPU)
    - ``heatmap``  -- (H, W) absolute-sum spatial map       (float32, CPU)
    - ``logits``   -- (C,) logits for the sample            (float32, CPU)
    - the normalized input tensor ``x``                     (float32, CPU)

Per sample, one file per *present* class:
``<out_dir>/<model_name>/<sample_id>/class_<class_id>.pt``.

This module does **no** statistical analysis; it only emits artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from evaluation.common import DIAGNOSTIC_DIR, ensure_dirs, setup_logging
from evaluation.offline_eval.io import load_args_yaml, load_mapping
from evaluation.offline_eval.model_builder import build_model

from .infer import ManifestRow, _build_val_preprocess, _load_manifest


_logger = logging.getLogger("evaluation.diagnostic.attribution")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "model"


# ---------------------------------------------------------------------------
# Core Gradient x Input.  Implemented verbatim to the spec in the user's
# message: autograd on the input, no smoothing, no baseline subtraction.
# ---------------------------------------------------------------------------


def gradient_x_input(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> Dict[str, torch.Tensor]:
    """Compute Gradient-x-Input for a single image / single class.

    ``x`` is expected to be a ``(1, C, H, W)`` tensor already on the model's
    device, *not* wrapped in ``torch.inference_mode``.  We keep ``model`` in
    ``eval()`` throughout (dropout / BN are frozen) but re-enable autograd so
    the input's ``.grad`` is populated.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"x must be (1, C, H, W); got {tuple(x.shape)}")

    was_training = model.training
    model.eval()

    x = x.clone().detach()
    x.requires_grad_(True)

    # Make sure autograd is enabled (defensive: the caller may have wrapped
    # the attribution loop in ``torch.inference_mode`` by mistake).
    with torch.enable_grad():
        logits = model(x)
        target_logit = logits[0, int(target_class)]

        model.zero_grad(set_to_none=True)
        target_logit.backward()

    if x.grad is None:
        raise RuntimeError(
            "x.grad is None after backward(); the model must be differentiable "
            "end-to-end in fp32."
        )

    attr = (x * x.grad).detach()
    heatmap = attr.abs().sum(dim=1).squeeze(0).detach()  # (H, W)
    logits_out = logits.detach()[0].float().cpu()

    if was_training:
        model.train()

    return {
        "attr": attr.float().cpu(),
        "heatmap": heatmap.float().cpu(),
        "logits": logits_out,
        "target_class": torch.tensor(int(target_class), dtype=torch.int64),
    }


# ---------------------------------------------------------------------------
# Batch driver: run attribution for selected samples + present classes.
# ---------------------------------------------------------------------------


@dataclass
class AttributionConfig:
    manifest_path: str
    out_dir: str
    checkpoint_name: str = "model_best.pth.tar"
    use_ema: bool = True
    device: str = "cuda"
    max_samples_per_k: Optional[int] = None  # None => all samples in manifest
    sample_ids: Optional[Sequence[str]] = None  # explicit subset (overrides max_samples_per_k)
    save_input: bool = False  # also persist the normalized input tensor in the .pt
    overwrite: bool = False
    selection_seed: int = 0  # random seed for per-k sample selection
    # Visualization
    save_png: bool = True          # write input.png + class_<c>_heatmap.png
    heatmap_cmap: str = "inferno"  # matplotlib colormap name
    heatmap_percentile: float = 99.0  # robust upper bound for normalization


def _select_sample_ids(
    rows: Sequence[ManifestRow],
    cfg: AttributionConfig,
) -> List[ManifestRow]:
    if cfg.sample_ids:
        wanted = set(cfg.sample_ids)
        return [r for r in rows if r.sample_id in wanted]
    if cfg.max_samples_per_k is None:
        return list(rows)
    # Random-sample up to ``max_samples_per_k`` per k-bucket with a fixed seed
    # so every model sees the same subset and it stays reproducible.
    per_k: Dict[int, List[ManifestRow]] = {}
    for r in rows:
        per_k.setdefault(r.k, []).append(r)
    rng = random.Random(int(cfg.selection_seed))
    out: List[ManifestRow] = []
    for k in sorted(per_k.keys()):
        bucket = per_k[k]
        if len(bucket) <= cfg.max_samples_per_k:
            picked = list(bucket)
        else:
            picked = rng.sample(bucket, cfg.max_samples_per_k)
        # Preserve deterministic ordering by sample_id within a bucket.
        picked.sort(key=lambda r: r.sample_id)
        out.extend(picked)
    return out


# ---------------------------------------------------------------------------
# PNG rendering helpers
# ---------------------------------------------------------------------------


def _denormalize_input_to_png(
    x: torch.Tensor,
    data_config: Mapping,
) -> Image.Image:
    """Undo timm-style mean/std normalization and return a PIL RGB image.

    ``x`` is the exact tensor fed to the model, shape ``(1, 3, H, W)`` or
    ``(3, H, W)``.
    """
    if x.ndim == 4:
        x = x[0]
    mean = torch.tensor(list(data_config["mean"]), dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(list(data_config["std"]), dtype=torch.float32).view(-1, 1, 1)
    img = x.detach().float().cpu() * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _heatmap_to_png(
    heatmap: torch.Tensor,
    cmap_name: str,
    percentile: float,
) -> Image.Image:
    """Render a (H, W) non-negative attribution map to an RGB PIL image.

    Normalization uses a robust upper bound (the requested percentile) so a
    handful of extreme pixels don't wash everything out.
    """
    h = heatmap.detach().float().cpu().numpy()
    h = np.maximum(h, 0.0)
    if h.size == 0:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    p = float(percentile)
    upper = float(np.percentile(h, p)) if 0.0 < p < 100.0 else float(h.max())
    if not np.isfinite(upper) or upper <= 0.0:
        upper = float(h.max()) if h.max() > 0 else 1.0
    h_norm = np.clip(h / upper, 0.0, 1.0)
    try:
        import matplotlib.cm as mpl_cm
        cmap = mpl_cm.get_cmap(cmap_name)
        rgba = cmap(h_norm)  # (H, W, 4), float in [0, 1]
        rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    except Exception:  # pragma: no cover -- fallback when matplotlib unavailable
        rgb = np.stack([(h_norm * 255.0 + 0.5).astype(np.uint8)] * 3, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def run_attribution_for_run(
    name: str,
    run_dir: str,
    rows: Sequence[ManifestRow],
    cfg: AttributionConfig,
    *,
    manifest_dir: str,
) -> str:
    from timm.data import resolve_data_config

    train_args = load_args_yaml(run_dir)
    checkpoint_path = os.path.join(run_dir, cfg.checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{cfg.checkpoint_name}' not found in {run_dir}"
        )

    device = torch.device(cfg.device)
    model = build_model(train_args, device, checkpoint_path, cfg.use_ema)
    data_config = resolve_data_config(dict(train_args), model=model, verbose=False)
    preprocess = _build_val_preprocess(data_config)

    slug = _slug(name)
    model_out_dir = os.path.join(cfg.out_dir, slug)
    os.makedirs(model_out_dir, exist_ok=True)

    selected = _select_sample_ids(rows, cfg)
    _logger.info(
        "[%s] gradient-x-input on %d sample(s).",
        name, len(selected),
    )

    # Per-model index mapping sample_id -> {artifacts} so we can easily
    # iterate on disk later without re-parsing every ``_sample.json``.
    index_entries: Dict[str, Dict[str, object]] = {}

    for r in selected:
        sample_out_dir = os.path.join(model_out_dir, r.sample_id)
        os.makedirs(sample_out_dir, exist_ok=True)
        meta_path = os.path.join(sample_out_dir, "_sample.json")

        with Image.open(r.image_path) as im:
            x = preprocess(im).unsqueeze(0).to(device)

        # Render + save the (de-normalized) input once per sample.
        input_png_rel: Optional[str] = None
        if cfg.save_png:
            input_png_path = os.path.join(sample_out_dir, "input.png")
            if cfg.overwrite or not os.path.exists(input_png_path):
                _denormalize_input_to_png(x, data_config).save(input_png_path)
            input_png_rel = os.path.relpath(input_png_path, cfg.out_dir)

        per_class_tensor_paths: Dict[str, str] = {}
        per_class_heatmap_pngs: Dict[str, str] = {}
        for target_class in r.present_classes:
            tc = int(target_class)
            tensor_path = os.path.join(sample_out_dir, f"class_{tc}.pt")
            heatmap_png_path = os.path.join(sample_out_dir, f"class_{tc}_heatmap.png")
            per_class_tensor_paths[str(tc)] = os.path.relpath(tensor_path, cfg.out_dir)
            if cfg.save_png:
                per_class_heatmap_pngs[str(tc)] = os.path.relpath(
                    heatmap_png_path, cfg.out_dir,
                )

            tensor_exists = os.path.exists(tensor_path)
            png_needed = cfg.save_png and (
                cfg.overwrite or not os.path.exists(heatmap_png_path)
            )
            if tensor_exists and not cfg.overwrite and not png_needed:
                continue

            result = gradient_x_input(model, x, tc)

            if cfg.overwrite or not tensor_exists:
                payload = {
                    "attr": result["attr"],
                    "heatmap": result["heatmap"],
                    "logits": result["logits"],
                    "target_class": result["target_class"],
                    "sample_id": r.sample_id,
                    "k": int(r.k),
                    "present_classes": [int(c) for c in r.present_classes],
                    "class_area_ratios": {
                        str(k_): float(v) for k_, v in r.class_area_ratios.items()
                    },
                }
                if cfg.save_input:
                    payload["input"] = x.detach().float().cpu()
                torch.save(payload, tensor_path)

            if png_needed:
                _heatmap_to_png(
                    result["heatmap"],
                    cmap_name=cfg.heatmap_cmap,
                    percentile=cfg.heatmap_percentile,
                ).save(heatmap_png_path)

        sample_meta: Dict[str, object] = {
            "sample_id": r.sample_id,
            "k": int(r.k),
            "present_classes": [int(c) for c in r.present_classes],
            "class_area_ratios": {
                str(kk): float(vv) for kk, vv in r.class_area_ratios.items()
            },
            "source_image_path": r.image_path,
            "input_png": input_png_rel,
            "tensor_paths": per_class_tensor_paths,
            "heatmap_pngs": per_class_heatmap_pngs,
            "model_name": name,
            "run_dir": run_dir,
            "checkpoint": checkpoint_path,
        }
        with open(meta_path, "w") as f:
            json.dump(sample_meta, f, indent=2)

        index_entries[r.sample_id] = {
            "k": int(r.k),
            "present_classes": [int(c) for c in r.present_classes],
            "class_area_ratios": sample_meta["class_area_ratios"],
            "input_png": input_png_rel,
            "heatmap_pngs": per_class_heatmap_pngs,
            "tensor_paths": per_class_tensor_paths,
            "source_image_path": r.image_path,
        }

    index_path = os.path.join(model_out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump({
            "model_name": name,
            "run_dir": run_dir,
            "checkpoint": checkpoint_path,
            "use_ema": bool(cfg.use_ema),
            "selection_seed": int(cfg.selection_seed),
            "max_samples_per_k": cfg.max_samples_per_k,
            "heatmap_cmap": cfg.heatmap_cmap,
            "heatmap_percentile": float(cfg.heatmap_percentile),
            "samples": index_entries,
        }, f, indent=2)
    _logger.info("[%s] index.json written (%d samples) -> %s",
                 name, len(index_entries), index_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _logger.info("[%s] wrote attributions to %s", name, model_out_dir)
    return model_out_dir


def run_attribution_for_mapping(
    mapping: Mapping[str, str],
    cfg: AttributionConfig,
) -> Dict[str, str]:
    rows, _dataset_meta = _load_manifest(cfg.manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(cfg.manifest_path))
    ensure_dirs()
    os.makedirs(cfg.out_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for name, run_dir in mapping.items():
        try:
            out[name] = run_attribution_for_run(
                name, run_dir, rows, cfg,
                manifest_dir=manifest_dir,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.exception("[%s] FAILED: %s", name, exc)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.jsonl produced by diagnostic.generate.")
    parser.add_argument("--mapping", required=True,
                        help="YAML/JSON {name: run_dir} mapping of trained models.")
    parser.add_argument(
        "--out-dir", type=str,
        default=str(DIAGNOSTIC_DIR / "attributions"),
        help="Where to dump <model_name>/<sample_id>/class_<c>.pt files.",
    )
    parser.add_argument("--checkpoint-name", default="model_best.pth.tar")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples-per-k", type=int, default=20,
                        help="Cap the number of attribution samples per k, chosen "
                             "uniformly at random within each bucket. Default: 20.")
    parser.add_argument("--selection-seed", type=int, default=0,
                        help="Seed for per-k random sample selection.")
    parser.add_argument("--sample-ids", nargs="*", default=None,
                        help="Explicit list of sample_ids to attribute. Overrides "
                             "--max-samples-per-k.")
    parser.add_argument("--save-input", action="store_true",
                        help="Also persist the preprocessed input tensor in the .pt.")
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=True,
                        help="Write a viewable input.png and per-class heatmap PNGs.")
    parser.add_argument("--heatmap-cmap", default="inferno",
                        help="Matplotlib colormap used to render heatmap PNGs.")
    parser.add_argument("--heatmap-percentile", type=float, default=99.0,
                        help="Robust upper bound for heatmap normalization.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING")
    mapping = load_mapping(args.mapping)
    cfg = AttributionConfig(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        checkpoint_name=args.checkpoint_name,
        use_ema=args.use_ema,
        device=args.device,
        max_samples_per_k=args.max_samples_per_k,
        sample_ids=tuple(args.sample_ids) if args.sample_ids else None,
        save_input=bool(args.save_input),
        overwrite=bool(args.overwrite),
        selection_seed=int(args.selection_seed),
        save_png=bool(args.save_png),
        heatmap_cmap=str(args.heatmap_cmap),
        heatmap_percentile=float(args.heatmap_percentile),
    )
    run_attribution_for_mapping(mapping, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
