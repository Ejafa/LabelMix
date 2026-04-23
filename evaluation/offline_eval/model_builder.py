"""Rebuild a training-time model from ``args.yaml`` + checkpoint."""
from __future__ import annotations

import logging
from typing import Any, Mapping

import torch
import torch.nn as nn

from timm.models import create_model, load_checkpoint


_logger = logging.getLogger("evaluation.offline_eval.model_builder")


def build_model(
    train_args: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: str,
    use_ema: bool,
) -> nn.Module:
    """Rebuild the training-time model and load its weights for inference.

    Mirrors the subset of :mod:`train` needed to reach a numerically-identical
    forward pass.  ViT/Eva/DeiT families get ``img_size`` threaded through
    ``model_kwargs``; other architectures ignore the kwarg silently.
    """
    in_chans = train_args.get("in_chans")
    if in_chans is None:
        input_size = train_args.get("input_size")
        in_chans = input_size[0] if input_size else 3

    model_kwargs = dict(train_args.get("model_kwargs") or {})
    img_size = train_args.get("img_size")
    if img_size is not None and "img_size" not in model_kwargs:
        model_name = str(train_args.get("model", ""))
        vit_like = any(tag in model_name.lower() for tag in ("vit", "eva", "deit"))
        if vit_like:
            model_kwargs["img_size"] = img_size

    model = create_model(
        train_args["model"],
        pretrained=False,
        in_chans=in_chans,
        num_classes=train_args.get("num_classes"),
        drop_rate=train_args.get("drop", 0.0),
        drop_path_rate=train_args.get("drop_path"),
        drop_block_rate=train_args.get("drop_block"),
        global_pool=train_args.get("gp"),
        bn_momentum=train_args.get("bn_momentum"),
        bn_eps=train_args.get("bn_eps"),
        **model_kwargs,
    )

    _logger.info("Loading checkpoint: %s (use_ema=%s)", checkpoint_path, use_ema)
    load_checkpoint(model, checkpoint_path, use_ema=use_ema)

    model.to(device=device)
    model.eval()
    return model
