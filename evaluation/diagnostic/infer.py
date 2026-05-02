"""Run trained models over the composed diagnostic dataset and dump logits.

Input: ``manifest.jsonl`` produced by :mod:`evaluation.diagnostic.generate`
Output per model: ``<out_dir>/<model_name>/logits.pt`` containing::

    {
        "sample_ids":              list[str]            # length N, order == manifest
        "k":                       torch.int16   (N,)
        "logits":                  torch.float32 (N, C)   # pre-softmax
        "present_classes":         list[list[int]]        # unique source labels per sample
        "class_area_ratios":       list[dict[str, float]] # merged_class_area_ratios per sample
        "model_name":              str
        "run_dir":                 str
        "checkpoint":              str
        "args":                    dict  # verbatim args.yaml
        "data_config":             dict  # timm normalization / crop pct used
        "manifest_dir":            str
        "manifest_sha256":         str   # so we can reject a stale dataset later
    }

Normal validation preprocessing is applied: resize+center-crop+normalize
using the model's own ``data_config`` (mean/std/crop_pct), in eval mode
with ``torch.inference_mode()``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from timm.data import resolve_data_config

from evaluation.common import DIAGNOSTIC_DIR, ensure_dirs, setup_logging
from evaluation.offline_eval.io import load_args_yaml, load_mapping
from evaluation.offline_eval.model_builder import build_model


_logger = logging.getLogger("evaluation.diagnostic.infer")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "model"


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


@dataclass
class ManifestRow:
    sample_id: str
    k: int
    present_classes: List[int]
    class_area_ratios: Dict[str, float]
    image_path: str  # absolute path


def _load_manifest(manifest_path: str) -> Tuple[List[ManifestRow], Dict[str, Any]]:
    """Parse ``manifest.jsonl`` + its sibling ``manifest.json``.

    Returns ``(rows, dataset_meta)``.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    meta_path = os.path.join(manifest_dir, "manifest.json")
    dataset_meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            dataset_meta = json.load(f)

    rows: List[ManifestRow] = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            present = sorted(set(int(x) for x in row["source_labels"]))
            img_path = row["image_path"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(manifest_dir, img_path)
            rows.append(ManifestRow(
                sample_id=str(row["sample_id"]),
                k=int(row["k"]),
                present_classes=present,
                class_area_ratios={
                    str(kk): float(vv)
                    for kk, vv in row["merged_class_area_ratios"].items()
                },
                image_path=img_path,
            ))
    return rows, dataset_meta


def _manifest_sha256(manifest_path: str) -> str:
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Preprocessing (normal validation pipeline)
# ---------------------------------------------------------------------------


def _build_val_preprocess(data_config: Mapping[str, Any]):
    """Return a callable PIL -> (3, H, W) float32 tensor in model space.

    Uses ``resolve_data_config`` values: ``input_size``, ``mean``, ``std``,
    ``crop_pct``.  This mirrors what ``timm.data.create_loader`` does for
    ``is_training=False`` without the prefetcher overhead.
    """
    c, h, w = data_config["input_size"]
    mean = torch.tensor(list(data_config["mean"]), dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(list(data_config["std"]), dtype=torch.float32).view(-1, 1, 1)
    crop_pct = float(data_config.get("crop_pct") or 0.95)
    resize_h = int(round(h / crop_pct))
    resize_w = int(round(w / crop_pct))

    def _apply(pil: Image.Image) -> torch.Tensor:
        pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        _, hh, ww = t.shape
        # Shorter side to resize_to / crop_pct (matching timm's val path).
        short_target = int(round(min(h, w) / crop_pct))
        if hh < ww:
            new_h = short_target
            new_w = int(round(ww * (short_target / hh)))
        else:
            new_w = short_target
            new_h = int(round(hh * (short_target / ww)))
        t = F.interpolate(
            t.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False, antialias=True,
        ).squeeze(0)
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        t = t[:, top:top + h, left:left + w].contiguous()
        return (t - mean) / std

    return _apply


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


@dataclass
class InferConfig:
    manifest_path: str
    out_dir: str
    checkpoint_name: str = "model_best.pth.tar"
    use_ema: bool = True
    batch_size: int = 64
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "bfloat16"


def _autocast_factory(cfg: InferConfig, device: torch.device):
    if not cfg.amp:
        return suppress
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    return partial(torch.autocast, device_type=device.type, dtype=amp_dtype)


def export_logits_for_run(
    name: str,
    run_dir: str,
    rows: Sequence[ManifestRow],
    cfg: InferConfig,
    *,
    manifest_dir: str,
    manifest_sha: str,
) -> str:
    """Run one model on the composed samples and persist its logits.

    Returns the path of the written ``.pt`` file.
    """
    train_args = load_args_yaml(run_dir)
    checkpoint_path = os.path.join(run_dir, cfg.checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{cfg.checkpoint_name}' not found in {run_dir}"
        )

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = build_model(train_args, device, checkpoint_path, cfg.use_ema)
    data_config = resolve_data_config(dict(train_args), model=model, verbose=False)
    preprocess = _build_val_preprocess(data_config)
    amp_autocast = _autocast_factory(cfg, device)

    n = len(rows)
    logits_buf: List[torch.Tensor] = []
    sample_ids: List[str] = []
    ks: List[int] = []
    present_classes: List[List[int]] = []
    class_area_ratios: List[Dict[str, float]] = []

    batch_imgs: List[torch.Tensor] = []
    batch_meta: List[int] = []  # indices into rows

    def _flush() -> None:
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        with torch.inference_mode(), amp_autocast():
            out = model(x)
        logits_buf.append(out.detach().float().cpu())
        batch_imgs.clear()
        batch_meta.clear()

    _logger.info(
        "[%s] running diagnostic inference over %d samples (batch=%d).",
        name, n, cfg.batch_size,
    )
    for i, row in enumerate(rows):
        with Image.open(row.image_path) as im:
            t = preprocess(im)
        batch_imgs.append(t)
        batch_meta.append(i)
        sample_ids.append(row.sample_id)
        ks.append(int(row.k))
        present_classes.append([int(c) for c in row.present_classes])
        class_area_ratios.append({str(k_): float(v) for k_, v in row.class_area_ratios.items()})
        if len(batch_imgs) >= cfg.batch_size:
            _flush()
            if (i + 1) % (cfg.batch_size * 4) == 0:
                _logger.info("  ...%d/%d", i + 1, n)
    _flush()

    logits = torch.cat(logits_buf, dim=0) if logits_buf else torch.zeros((0, 0))

    slug = _slug(name)
    out_subdir = os.path.join(cfg.out_dir, slug)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, "logits.pt")
    torch.save({
        "sample_ids": sample_ids,
        "k": torch.tensor(ks, dtype=torch.int16),
        "logits": logits,
        "present_classes": present_classes,
        "class_area_ratios": class_area_ratios,
        "model_name": name,
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "args": dict(train_args),
        "data_config": {
            "input_size": list(data_config["input_size"]),
            "mean": list(data_config["mean"]),
            "std": list(data_config["std"]),
            "crop_pct": float(data_config.get("crop_pct") or 0.95),
            "interpolation": data_config.get("interpolation"),
        },
        "manifest_dir": manifest_dir,
        "manifest_sha256": manifest_sha,
    }, out_path)
    _logger.info("[%s] wrote %d logits to %s", name, logits.shape[0], out_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_path


def export_logits_for_mapping(
    mapping: Mapping[str, str],
    cfg: InferConfig,
) -> Dict[str, str]:
    """Run every ``(name, run_dir)`` in ``mapping``; returns ``{name: out_path}``."""
    rows, _dataset_meta = _load_manifest(cfg.manifest_path)
    manifest_sha = _manifest_sha256(cfg.manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(cfg.manifest_path))
    ensure_dirs()
    os.makedirs(cfg.out_dir, exist_ok=True)

    out: Dict[str, str] = {}
    for name, run_dir in mapping.items():
        try:
            path = export_logits_for_run(
                name, run_dir, rows, cfg,
                manifest_dir=manifest_dir, manifest_sha=manifest_sha,
            )
            out[name] = path
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
        default=str(DIAGNOSTIC_DIR / "logits"),
        help="Where to dump <model_name>/logits.pt files.",
    )
    parser.add_argument("--checkpoint-name", default="model_best.pth.tar")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING")
    mapping = load_mapping(args.mapping)
    cfg = InferConfig(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        checkpoint_name=args.checkpoint_name,
        use_ema=args.use_ema,
        batch_size=int(args.batch_size),
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    export_logits_for_mapping(mapping, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
