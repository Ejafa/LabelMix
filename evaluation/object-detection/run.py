#!/usr/bin/env python3
"""
Evaluate a timm backbone with an MMDetection config.

Usage (example):
  python evaluation/object-detection.py \
    --configs /path/to/coco_config.py,/path/to/voc_config.py \
    --timm-model mobilenetv4_conv_small \
    --backbone-checkpoints /path/to/ckpt1.pth,/path/to/ckpt2.pth \
    --data-root /path/to/datasets \
    --work-dir output_test/mmdet_eval \
    --results-file output_test/mmdet_eval/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import timm
from timm.data import resolve_data_config
from timm.models import load_checkpoint

_LOG = logging.getLogger("mmdet_timm_eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMDetection eval with timm backbone")
    parser.add_argument("--config", default="", type=str, help="Path to an MMDetection config")
    parser.add_argument(
        "--configs",
        default="",
        type=str,
        help="Comma/space list of MMDetection configs (or @file with one path per line)",
    )
    parser.add_argument("--timm-model", required=True, type=str, help="timm model name")
    parser.add_argument(
        "--out-indices",
        default="",
        type=str,
        help="Comma/space separated out_indices (optional, defaults to timm model defaults)",
    )
    parser.add_argument(
        "--timm-pretrained",
        action="store_true",
        default=False,
        help="Use timm pretrained weights (in addition to optional backbone checkpoint)",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default="",
        type=str,
        help="Path to a timm backbone checkpoint (optional)",
    )
    parser.add_argument(
        "--backbone-checkpoints",
        default="",
        type=str,
        help="Comma/space list of backbone checkpoints (or @file with one path per line)",
    )
    parser.add_argument(
        "--det-checkpoint",
        default="",
        type=str,
        help="Path to a detector checkpoint for evaluation (optional)",
    )
    parser.add_argument(
        "--data-root",
        default="",
        type=str,
        help="Override data_root in the config (optional)",
    )
    parser.add_argument(
        "--batch-size",
        default=0,
        type=int,
        help="Override test dataloader batch size (optional)",
    )
    parser.add_argument(
        "--num-workers",
        default=-1,
        type=int,
        help="Override test dataloader workers (optional)",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--work-dir", default="output_test/mmdet_eval", type=str)
    parser.add_argument(
        "--run",
        default="test",
        choices=("test", "train", "train+test"),
        help="Run type: test, train, or train+test",
    )
    parser.add_argument(
        "--results-file",
        default="output_test/mmdet_eval/results.jsonl",
        type=str,
        help="Write JSONL results to this path",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop immediately on first failure",
    )
    parser.add_argument("--dump-config", default="", type=str, help="Write patched config to path and exit")
    parser.add_argument("--show-config", action="store_true", default=False)
    return parser.parse_args()


def _parse_out_indices(text: str) -> Optional[Tuple[int, ...]]:
    if not text:
        return None
    parts = [p for p in text.replace(",", " ").split(" ") if p.strip()]
    return tuple(int(p) for p in parts)


def _split_list_arg(text: str) -> List[str]:
    if not text:
        return []
    if text.startswith("@"):
        path = text[1:]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"List file not found: {path}")
        items = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                items.append(line)
        return items
    return [p for p in re.split(r"[,\s]+", text.strip()) if p]


def _default_data_root() -> str:
    path = Path("./data")
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path)


def _to_list(x: Iterable[int]) -> List[int]:
    return [int(v) for v in x]


def _get_registry():
    try:
        from mmdet.registry import MODELS

        return MODELS
    except Exception:
        try:
            from mmdet.models.builder import BACKBONES

            return BACKBONES
        except Exception as exc:
            raise RuntimeError("Could not import MMDetection registry") from exc


def _register_timm_backbone():
    registry = _get_registry()
    try:
        from mmengine.model import BaseModule
    except Exception:
        BaseModule = torch.nn.Module

    @registry.register_module()
    class TimmDetBackbone(BaseModule):
        def __init__(
            self,
            model_name: str,
            out_indices: Optional[Sequence[int]] = None,
            checkpoint_path: str = "",
            pretrained: bool = False,
            in_chans: int = 3,
        ):
            if BaseModule is torch.nn.Module:
                super().__init__()
            else:
                super().__init__(init_cfg=None)

            kwargs = dict(model_name=model_name, pretrained=pretrained, features_only=True, in_chans=in_chans)
            if out_indices is not None:
                kwargs["out_indices"] = tuple(out_indices)
            self.model = timm.create_model(**kwargs)
            self.out_channels = self.model.feature_info.channels()
            self.out_indices = tuple(self.model.feature_info.out_indices)
            self.checkpoint_path = checkpoint_path
            if checkpoint_path:
                incompatible = load_checkpoint(self.model, checkpoint_path, strict=False)
                _LOG.info("Loaded backbone checkpoint '%s' (incompatible: %s)", checkpoint_path, incompatible)

        def forward(self, x):
            return self.model(x)

    return TimmDetBackbone


def _resolve_feature_info(model_name: str, out_indices: Optional[Tuple[int, ...]]):
    kwargs = dict(model_name=model_name, pretrained=False, features_only=True)
    if out_indices is not None:
        kwargs["out_indices"] = out_indices
    model = timm.create_model(**kwargs)
    info = model.feature_info
    return info.channels(), info.reduction(), tuple(info.out_indices), model


def _maybe_get(cfg_obj, key: str):
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key)
    return getattr(cfg_obj, key, None)


def _override_data_root(cfg, data_root: str):
    if not data_root:
        return
    if hasattr(cfg, "data_root"):
        cfg.data_root = data_root
    for loader_key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = _maybe_get(cfg, loader_key)
        if not loader:
            continue
        dataset = _maybe_get(loader, "dataset")
        if isinstance(dataset, dict) and "data_root" in dataset:
            dataset["data_root"] = data_root


def _override_test_loader(cfg, batch_size: int, num_workers: int):
    loader = _maybe_get(cfg, "test_dataloader")
    if not loader:
        return
    if batch_size and "batch_size" in loader:
        loader["batch_size"] = batch_size
    if num_workers >= 0 and "num_workers" in loader:
        loader["num_workers"] = num_workers


def _patch_config(
    cfg,
    args: argparse.Namespace,
    in_channels: List[int],
    out_indices: Tuple[int, ...],
    mean: Optional[List[float]],
    std: Optional[List[float]],
    backbone_checkpoint: str,
    work_dir: str,
    single_scale: bool,
):
    backbone_cfg = dict(
        type="TimmDetBackbone",
        model_name=args.timm_model,
        out_indices=out_indices,
        checkpoint_path=backbone_checkpoint or "",
        pretrained=bool(args.timm_pretrained),
    )
    cfg.model.backbone = backbone_cfg

    neck = _maybe_get(cfg.model, "neck")
    if single_scale:
        out_channels = None
        num_outs = None
        if isinstance(neck, dict):
            out_channels = neck.get("out_channels")
            num_outs = neck.get("num_outs")
        elif neck is not None:
            out_channels = getattr(neck, "out_channels", None)
            num_outs = getattr(neck, "num_outs", None)
        if out_channels is None:
            out_channels = 256
        if num_outs is None:
            num_outs = 5
        cfg.model.neck = dict(
            type="SimpleFPN",
            in_channels=int(in_channels[0]),
            out_channels=int(out_channels),
            num_outs=int(num_outs),
        )
    else:
        if isinstance(neck, dict):
            if "in_channels" in neck:
                neck["in_channels"] = in_channels
            if "num_outs" in neck and isinstance(neck["num_outs"], int):
                if neck["num_outs"] < len(in_channels):
                    neck["num_outs"] = len(in_channels)
        elif neck is not None:
            if hasattr(neck, "in_channels"):
                neck.in_channels = in_channels
            if hasattr(neck, "num_outs") and neck.num_outs < len(in_channels):
                neck.num_outs = len(in_channels)

    if mean is not None and std is not None:
        data_preproc = _maybe_get(cfg.model, "data_preprocessor")
        if isinstance(data_preproc, dict):
            data_preproc["mean"] = mean
            data_preproc["std"] = std
            data_preproc["bgr_to_rgb"] = True
        elif data_preproc is not None:
            data_preproc.mean = mean
            data_preproc.std = std
            data_preproc.bgr_to_rgb = True

    if args.det_checkpoint:
        cfg.load_from = args.det_checkpoint

    cfg.work_dir = work_dir
    cfg.device = args.device
    _override_data_root(cfg, args.data_root)
    _override_test_loader(cfg, args.batch_size, args.num_workers)


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _collect_configs(args: argparse.Namespace) -> List[str]:
    configs = []
    if args.config:
        configs.append(args.config)
    configs.extend(_split_list_arg(args.configs))
    configs = _unique_keep_order([c for c in configs if c])
    if not configs:
        raise ValueError("Provide --config or --configs")
    return configs


def _collect_checkpoints(args: argparse.Namespace) -> List[str]:
    ckpts = []
    if args.backbone_checkpoint:
        ckpts.append(args.backbone_checkpoint)
    ckpts.extend(_split_list_arg(args.backbone_checkpoints))
    ckpts = _unique_keep_order([c for c in ckpts if c])
    return ckpts


def _safe_stem(text: str) -> str:
    if not text:
        return "none"
    name = Path(text).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "none"


def _dump_config_path(dump_root: str, config_path: str, ckpt_path: str) -> str:
    root = Path(dump_root)
    if root.suffix in (".py", ".yaml", ".yml", ".json"):
        base = root.parent
        stem = root.stem
    else:
        base = root
        stem = "config"
    cfg_stem = _safe_stem(config_path)
    ckpt_stem = _safe_stem(ckpt_path)
    filename = f"{stem}-{cfg_stem}-{ckpt_stem}.py"
    return str(base / filename)


def _jsonify(value):
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if np is not None and isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.data_root:
        args.data_root = _default_data_root()
        if args.data_root:
            _LOG.info("Using default data_root: %s", args.data_root)

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except Exception as exc:
        _LOG.error("MMEngine is required (mmdet >= 3.x). Import failed: %s", exc)
        return 2

    try:
        import mmdet  # noqa: F401
    except Exception as exc:
        _LOG.error("MMDetection is required. Import failed: %s", exc)
        return 2

    _register_timm_backbone()

    try:
        configs = _collect_configs(args)
    except Exception as exc:
        _LOG.error("%s", exc)
        return 2

    try:
        checkpoints = _collect_checkpoints(args)
    except Exception as exc:
        _LOG.error("%s", exc)
        return 2
    if not checkpoints and not args.timm_pretrained:
        _LOG.warning(
            "No backbone checkpoints provided and --timm-pretrained is false; using random init."
        )
        checkpoints = [""]
    elif not checkpoints:
        checkpoints = [""]

    for cfg_path in configs:
        if not os.path.isfile(cfg_path):
            _LOG.error("Config not found: %s", cfg_path)
            return 2

    out_indices = _parse_out_indices(args.out_indices)
    in_channels, reductions, out_indices_resolved, feat_model = _resolve_feature_info(
        args.timm_model, out_indices
    )
    out_indices = out_indices_resolved
    single_scale = len(set(reductions)) == 1
    if single_scale and len(out_indices) > 1:
        _LOG.info("Single-scale backbone detected; using last out_index %s for SimpleFPN.", out_indices[-1])
        out_indices = (out_indices[-1],)
        in_channels, reductions, out_indices_resolved, feat_model = _resolve_feature_info(
            args.timm_model, out_indices
        )
        out_indices = out_indices_resolved

    data_cfg = resolve_data_config({}, model=feat_model)
    mean = [float(v) * 255.0 for v in data_cfg.get("mean", [])] if "mean" in data_cfg else None
    std = [float(v) * 255.0 for v in data_cfg.get("std", [])] if "std" in data_cfg else None

    _LOG.info("timm model: %s", args.timm_model)
    _LOG.info("out_indices: %s", out_indices)
    _LOG.info("in_channels: %s", in_channels)
    _LOG.info("reductions: %s", reductions)
    _LOG.info("single_scale: %s", single_scale)

    for cfg_path in configs:
        cfg_stem = _safe_stem(cfg_path)
        for ckpt in checkpoints:
            ckpt_stem = _safe_stem(ckpt)
            run_work_dir = os.path.join(args.work_dir, cfg_stem, ckpt_stem)
            _LOG.info("Run: config=%s ckpt=%s work_dir=%s", cfg_path, ckpt or "none", run_work_dir)

            cfg = Config.fromfile(cfg_path)
            _patch_config(
                cfg,
                args,
                _to_list(in_channels),
                out_indices,
                mean,
                std,
                ckpt,
                run_work_dir,
                single_scale,
            )

            if args.show_config:
                _LOG.info("Patched config:\n%s", cfg.pretty_text)

            if args.dump_config:
                dump_path = _dump_config_path(args.dump_config, cfg_path, ckpt)
                Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
                cfg.dump(dump_path)
                _LOG.info("Wrote config to %s", dump_path)
                continue

            result = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "config": cfg_path,
                "timm_model": args.timm_model,
                "backbone_checkpoint": ckpt or "",
                "det_checkpoint": args.det_checkpoint or "",
                "work_dir": run_work_dir,
                "out_indices": list(out_indices),
                "in_channels": _to_list(in_channels),
                "reductions": _to_list(reductions),
                "run": args.run,
                "status": "ok",
            }

            try:
                runner = Runner.from_cfg(cfg)
                metrics = None
                if args.run in ("train", "train+test"):
                    runner.train()
                if args.run in ("test", "train+test"):
                    metrics = runner.test()
                if metrics is not None:
                    result["metrics"] = _jsonify(metrics)
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                _LOG.error("Run failed: %s", exc)
                if args.fail_fast:
                    _append_jsonl(args.results_file, result)
                    return 1

            _append_jsonl(args.results_file, result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
