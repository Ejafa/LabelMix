"""Rebuild the training-time validation data loader."""
from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn

from timm.data import create_dataset, create_loader, resolve_data_config

from ..config import EvalConfig


def build_val_loader(
    train_args: Mapping[str, Any],
    model: nn.Module,
    cfg: EvalConfig,
    device: torch.device,
) -> Tuple[Any, Mapping[str, Any]]:
    """Return ``(val_loader, data_config)`` matching the training recipe.

    Overrides on :class:`EvalConfig` (``data_dir_override``,
    ``dataset_override``, ``val_split_override``) take precedence over the
    values recorded in ``args.yaml``.
    """
    data_config = resolve_data_config(dict(train_args), model=model, verbose=False)

    dataset_name = cfg.dataset_override or train_args.get("dataset", "")
    data_dir = (
        cfg.data_dir_override
        or train_args.get("data_dir")
        or train_args.get("data")
    )
    val_split = cfg.val_split_override or train_args.get("val_split", "validation")

    input_img_mode = train_args.get("input_img_mode")
    if input_img_mode is None:
        input_img_mode = "RGB" if data_config["input_size"][0] == 3 else "L"

    dataset = create_dataset(
        dataset_name,
        root=data_dir,
        split=val_split,
        is_training=False,
        class_map=train_args.get("class_map", ""),
        download=train_args.get("dataset_download", False),
        batch_size=cfg.batch_size,
        input_img_mode=input_img_mode,
        input_key=train_args.get("input_key"),
        target_key=train_args.get("target_key"),
        num_samples=train_args.get("val_num_samples"),
        trust_remote_code=train_args.get("dataset_trust_remote_code", False),
    )

    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=cfg.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=cfg.workers,
        crop_pct=data_config["crop_pct"],
        pin_memory=train_args.get("pin_mem", True),
        device=device,
        img_dtype=torch.float32,
        distributed=False,
    )
    return loader, data_config
