#!/usr/bin/env python3
"""Inspect LabelMix inputs and weight distributions without training."""
import argparse
import json
import logging
import os
from typing import Any, List, Optional, Tuple

import torch
import yaml
from PIL import Image

from timm import utils
from timm.data import BalancedBucketDataset, create_dataset, resolve_data_config, create_transform
from timm.data.loader import fast_collate

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_logger = logging.getLogger("inspect_labelmix")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LabelMix Inputs")
    parser.add_argument("-c", "--config", default="", type=str, metavar="FILE",
                        help="YAML config file specifying default arguments")
    parser.add_argument("--data-dir", default=None, type=str, help="Dataset root override")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--steps-per-epoch", default=0, type=int,
                        help="Steps per epoch for implicit epoch advancement (0 = use len(dataloader)).")
    parser.add_argument("--sample-every", default=10, type=int)
    parser.add_argument("--sample-count", default=10, type=int)
    parser.add_argument("--output-dir", default="output_test", type=str)
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument("--no-aug", action="store_true", default=False)
    parser.add_argument("--labelmix-sampling", action="store_true", default=False)
    parser.add_argument("--labelmix-sampling-min-side-px", default=8, type=int)
    parser.add_argument("--labelmix-sampling-max-aspect", default=10, type=float)
    parser.add_argument("--labelmix-sampling-bins", default=16, type=int)
    parser.add_argument("--labelmix-sampling-pool-size", default=128, type=int)
    parser.add_argument("--labelmix-sampling-low-watermark", default=32, type=int)
    parser.add_argument("--labelmix-sampling-max-attempts", default=200, type=int)
    parser.add_argument("--labelmix-debug-sym", action="store_true", default=False)
    parser.add_argument("--labelmix-mix-k", default=5, type=int,
                        help="LabelMix K (number of source images per output, >= 1).")
    parser.add_argument("--labelmix-k-min", default=None, type=int,
                        help="LabelMix K minimum for K scheduler (default: --labelmix-mix-k).")
    parser.add_argument("--labelmix-k-max", default=None, type=int,
                        help="LabelMix K maximum for K scheduler (default: --labelmix-mix-k).")
    parser.add_argument("--labelmix-k-schedule", default="linear", type=str,
                        choices=["fixed", "linear", "cosine"],
                        help="LabelMix K schedule over epochs.")
    parser.add_argument("--labelmix-k-reverse", action="store_true", default=False,
                        help="Reverse LabelMix K schedule (max->min).")
    parser.add_argument("--labelmix-k-warmup-epochs", default=0, type=int,
                        help="Warmup epochs for LabelMix K schedule.")
    parser.add_argument("--labelmix-k-total-epochs", default=None, type=int,
                        help="Total epochs for LabelMix K schedule (default: inferred from total run length).")
    parser.add_argument("--labelmix-epoch", default=0, type=int,
                        help="Epoch index to inspect (for K/alpha scheduling).")
    args, _ = parser.parse_known_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        defaults = {a.dest: parser.get_default(a.dest) for a in parser._actions}
        for k, v in cfg.items():
            if getattr(args, k, defaults.get(k)) == defaults.get(k):
                setattr(args, k, v)

    return args


def _ensure_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def _is_ascending(weights: torch.Tensor, eps: float = 1e-7) -> bool:
    if weights.numel() <= 1:
        return True
    return bool(torch.all(weights[1:] + eps >= weights[:-1]))


def _filter_labelmix_samples(
    dataset: BalancedBucketDataset,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    sym_ids: Optional[torch.Tensor],
    sample_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Reject samples whose LabelMix layout violates min-side/aspect constraints."""
    keep: List[int] = []
    rejected = 0
    max_keep = min(int(sample_count), int(labels.shape[0]))

    for i in range(int(labels.shape[0])):
        if len(keep) >= max_keep:
            break
        w = weights[i]
        w_desc, _ = torch.sort(w.detach(), descending=True)
        H = int(imgs[i].shape[-2])
        W = int(imgs[i].shape[-1])
        if dataset._layout_is_valid(w_desc, H=H, W=W):
            keep.append(i)
        else:
            rejected += 1

    if not keep:
        labels_empty = labels[:0]
        weights_empty = weights[:0]
        return (
            labels_empty,
            weights_empty,
            imgs[:0],
            sym_ids[:0] if sym_ids is not None else None,
            rejected,
        )

    labels = labels[keep]
    weights = weights[keep]
    imgs = imgs[keep]
    if sym_ids is not None:
        sym_ids = sym_ids[keep]
    return labels, weights, imgs, sym_ids, rejected


def labelmix_collate(batch):
    if not batch:
        return fast_collate(batch)
    sample = batch[0]
    if isinstance(sample, tuple) and isinstance(sample[1], (tuple, list)) and len(sample[1]) >= 2:
        imgs = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.stack([b[1][0] for b in batch], dim=0)
        weights = torch.stack([b[1][1] for b in batch], dim=0)
        if len(sample[1]) == 3:
            sym_ids = torch.tensor([int(b[1][2]) for b in batch], dtype=torch.int64)
            return imgs, (labels, weights, sym_ids)
        return imgs, (labels, weights)
    return fast_collate(batch)


def _to_uint8_img(t: torch.Tensor, mean, std) -> Image.Image:
    if t.ndim != 3:
        raise ValueError("Expected CHW tensor")
    c, h, w = t.shape
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device).view(c, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device).view(c, 1, 1)
    img = (t * std_t + mean_t).clamp(0, 1)
    img = (img * 255.0).to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def _pil_resample(interp: str) -> int:
    interp = (interp or "").lower()
    if interp == "nearest":
        return Image.NEAREST
    if interp == "bilinear":
        return Image.BILINEAR
    if interp == "bicubic":
        return Image.BICUBIC
    if interp == "lanczos":
        return Image.LANCZOS
    return Image.BICUBIC


class ResizeOnlyTransform:
    def __init__(self, size_hw, interp: str) -> None:
        h, w = size_hw
        self.h = int(h)
        self.w = int(w)
        self.resample = _pil_resample(interp)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img.resize((self.w, self.h), resample=self.resample)
        try:
            return Image.fromarray(img).resize((self.w, self.h), resample=self.resample)
        except Exception:
            return img


def _save_sample_plot(
    path: str,
    img: Image.Image,
    labels: torch.Tensor,
    weights: torch.Tensor,
    label_names: Optional[List[str]] = None,
) -> None:
    if plt is None:
        img.save(path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("LabelMix sample")

    x = list(range(int(weights.numel())))
    axes[1].bar(x, weights.cpu().tolist())
    axes[1].set_xlabel("slot")
    axes[1].set_ylabel("weight")
    axes[1].set_title("weights")
    axes[1].set_xticks(x)
    names = [_label_id_to_name(int(v), label_names) for v in labels.cpu().tolist()]
    axes[1].set_xticklabels(names, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _resolve_label_names(dataset, target_key: str) -> Optional[List[str]]:
    reader = getattr(dataset, "reader", None)
    if reader is not None:
        ds = getattr(reader, "dataset", None)
        if ds is not None:
            info = getattr(ds, "info", None)
            if info and hasattr(info, "features") and target_key in info.features:
                names = getattr(info.features[target_key], "names", None)
                if names:
                    return list(names)
            feats = getattr(ds, "features", None)
            if feats and target_key in feats:
                names = getattr(feats[target_key], "names", None)
                if names:
                    return list(names)
        class_to_idx = getattr(reader, "class_to_idx", None)
        if isinstance(class_to_idx, dict) and class_to_idx:
            max_idx = max(class_to_idx.values())
            names = [""] * (max_idx + 1)
            for name, idx in class_to_idx.items():
                if 0 <= idx <= max_idx:
                    names[idx] = str(name)
            if any(names):
                return names

    if hasattr(dataset, "classes") and getattr(dataset, "classes"):
        return list(getattr(dataset, "classes"))
    if hasattr(dataset, "class_to_idx"):
        class_to_idx = getattr(dataset, "class_to_idx")
        if isinstance(class_to_idx, dict) and class_to_idx:
            max_idx = max(class_to_idx.values())
            names = [""] * (max_idx + 1)
            for name, idx in class_to_idx.items():
                if 0 <= idx <= max_idx:
                    names[idx] = str(name)
            if all(names):
                return names
    feats = getattr(dataset, "features", None)
    if feats and target_key in feats:
        label_feat = feats[target_key]
        names = getattr(label_feat, "names", None)
        if names:
            return list(names)
    return None


def _label_id_to_name(label_id: int, label_names: Optional[List[str]]) -> str:
    if label_names and 0 <= label_id < len(label_names):
        return str(label_names[label_id])
    return str(label_id)


def main() -> None:
    utils.setup_default_logging()
    args = _parse_args()

    _logger.info("no_aug=%s", _ensure_bool(getattr(args, "no_aug", False)))
    _logger.info("args=%s", json.dumps(vars(args), indent=2, default=str))

    if getattr(args, "dataset", None) is None:
        raise ValueError("Config must specify dataset.")

    args.device = torch.device(args.device)

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=getattr(args, "train_split", "train"),
        is_training=True,
        download=getattr(args, "dataset_download", False),
        input_img_mode=getattr(args, "input_img_mode", None),
        input_key=getattr(args, "input_key", None),
        target_key=getattr(args, "target_key", None),
        trust_remote_code=getattr(args, "dataset_trust_remote_code", False),
    )

    data_config = resolve_data_config(vars(args), model=None)

    no_aug = _ensure_bool(getattr(args, "no_aug", False))
    if no_aug:
        _, h, w = data_config["input_size"]
        interp = getattr(args, "train_interpolation", data_config.get("interpolation", "bicubic"))
        train_transform = ResizeOnlyTransform((h, w), interp)
    else:
        train_transform = create_transform(
            input_size=data_config['input_size'],
            is_training=True,
            auto_augment=getattr(args, "aa", None),
            interpolation=getattr(args, "train_interpolation", "random"),
            mean=data_config['mean'],
            std=data_config['std'],
            re_prob=getattr(args, "reprob", 0.0),
            re_mode=getattr(args, "remode", "pixel"),
            re_count=getattr(args, "recount", 1),
            color_jitter=getattr(args, "color_jitter", 0.4),
            scale=getattr(args, "scale", (0.08, 1.0)),
            ratio=getattr(args, "ratio", (3./4., 4./3.)),
            hflip=getattr(args, "hflip", 0.5),
            vflip=getattr(args, "vflip", 0.0),
            grayscale_prob=getattr(args, "grayscale_prob", 0.0),
            gaussian_blur_prob=getattr(args, "gaussian_blur_prob", 0.0),
        )

    balanced_input_key = getattr(args, "balanced_input_key", None)
    if balanced_input_key is None:
        balanced_input_key = args.input_key if args.input_key is not None else "image"
    balanced_target_key = getattr(args, "balanced_target_key", None)
    if balanced_target_key is None:
        balanced_target_key = args.target_key if args.target_key is not None else "label"

    label_names = _resolve_label_names(dataset_train, balanced_target_key)
    if label_names is None and hasattr(dataset_train, "dataset"):
        label_names = _resolve_label_names(getattr(dataset_train, "dataset"), balanced_target_key)
    if label_names is None and hasattr(dataset_train, "ds"):
        label_names = _resolve_label_names(getattr(dataset_train, "ds"), balanced_target_key)

    output_dir = str(getattr(args, "output_dir", "output_test") or "output_test")
    balanced_cache_path = getattr(args, "balanced_cache_path", "")
    if not balanced_cache_path:
        balanced_cache_path = os.path.join(output_dir, "class_buckets.pkl")
    balanced_cache_dir = os.path.dirname(balanced_cache_path)
    if balanced_cache_dir:
        os.makedirs(balanced_cache_dir, exist_ok=True)

    sampling_enabled = _ensure_bool(getattr(args, "labelmix_sampling", False))
    labelmix_mix_k = int(getattr(args, "labelmix_mix_k", 5))
    labelmix_k_min = getattr(args, "labelmix_k_min", None)
    if labelmix_k_min is None:
        labelmix_k_min = labelmix_mix_k
    labelmix_k_max = getattr(args, "labelmix_k_max", None)
    if labelmix_k_max is None:
        labelmix_k_max = labelmix_mix_k

    default_train_epochs = getattr(args, "epochs", None)
    if default_train_epochs is None:
        default_train_epochs = getattr(args, "labelmix_total_epochs", None)
    if default_train_epochs is None:
        default_train_epochs = 1

    explicit_k_total_epochs = (
        getattr(args, "labelmix_k_total_epochs", None)
        if getattr(args, "labelmix_k_total_epochs", None) is not None
        else getattr(args, "labelmix_total_epochs", None)
    )
    labelmix_k_total_epochs = (
        explicit_k_total_epochs if explicit_k_total_epochs is not None else default_train_epochs
    )

    labelmix_kwargs = {
        "mix_k": int(labelmix_mix_k),
        "k_min": int(labelmix_k_min),
        "k_max": int(labelmix_k_max),
        "k_schedule": str(getattr(args, "labelmix_k_schedule", "linear")),
        "k_reverse": _ensure_bool(getattr(args, "labelmix_k_reverse", False)),
        "k_warmup_epochs": int(getattr(args, "labelmix_k_warmup_epochs", 0)),
        "k_total_epochs": labelmix_k_total_epochs,
        "train_epochs": default_train_epochs,
        "alpha_min": float(getattr(args, "labelmix_alpha_min", 0.1)),
        "alpha_max": float(getattr(args, "labelmix_alpha_max", 1.0)),
        "schedule": str(getattr(args, "labelmix_schedule", "linear")),
        "reverse": _ensure_bool(getattr(args, "labelmix_reverse", False)),
        "step_mode": str(getattr(args, "labelmix_step_mode", "total")),
        "warmup_steps": int(getattr(args, "labelmix_warmup_steps", 0)),
        "total_epochs": getattr(args, "labelmix_total_epochs", None),
        "total_steps": getattr(args, "labelmix_total_steps", getattr(args, "num_steps", None)),
        "batch_size": int(getattr(args, "batch_size", 64)),
        "sampling": sampling_enabled,
        "sampling_min_side_px": int(getattr(args, "labelmix_sampling_min_side_px", 6)),
        "sampling_max_aspect": float(getattr(args, "labelmix_sampling_max_aspect", 10.0)),
        "sampling_bins": int(getattr(args, "labelmix_sampling_bins", 16)),
        "sampling_pool_size": int(getattr(args, "labelmix_sampling_pool_size", 128)),
        "sampling_low_watermark": int(getattr(args, "labelmix_sampling_low_watermark", 32)),
        "sampling_max_attempts": int(getattr(args, "labelmix_sampling_max_attempts", 200)),
        "debug_sym": _ensure_bool(getattr(args, "labelmix_debug_sym", False)),
    }

    dataset_train = BalancedBucketDataset(
        base_dataset=dataset_train,
        transform=train_transform,
        mode=getattr(args, "balanced_mode", "max"),
        buffer_size=int(getattr(args, "balanced_buffer", 256)),
        cache_path=balanced_cache_path,
        cache_small_classes_threshold=int(getattr(args, "balanced_cache_threshold", 256)),
        input_key=balanced_input_key,
        target_key=balanced_target_key,
        labelmix=_ensure_bool(getattr(args, "labelmix", True)),
        labelmix_kwargs=labelmix_kwargs,
    )

    num_workers = int(getattr(args, "workers", 0) or 0)

    loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=int(getattr(args, "batch_size", 64)),
        num_workers=num_workers,
        collate_fn=labelmix_collate,
        pin_memory=False,
        drop_last=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    steps = int(args.steps)
    sample_every = int(args.sample_every)
    sample_count = int(args.sample_count)
    steps_per_epoch_arg = int(getattr(args, "steps_per_epoch", 0) or 0)
    try:
        steps_per_epoch = steps_per_epoch_arg if steps_per_epoch_arg > 0 else int(len(loader))
    except TypeError:
        steps_per_epoch = steps
    steps_per_epoch = max(1, steps_per_epoch)

    mean = data_config['mean']
    std = data_config['std']

    base_epoch = int(getattr(args, "labelmix_epoch", 0))
    current_epoch = None
    data_iter = None

    for step in range(1, steps + 1):
        epoch = base_epoch + (step - 1) // steps_per_epoch
        if epoch != current_epoch:
            current_epoch = epoch
            if _ensure_bool(getattr(args, "labelmix", True)) and hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(current_epoch)
            data_iter = iter(loader)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        inputs, targets = batch

        if step % sample_every != 0:
            continue

        if not (isinstance(targets, (tuple, list)) and len(targets) >= 2):
            _logger.warning("Targets are not LabelMix tuple; got %s", type(targets))
            continue

        labels, weights = targets[0], targets[1]
        sym_ids = targets[2] if len(targets) >= 3 else None
        imgs = inputs

        if sampling_enabled and hasattr(dataset_train, "_layout_is_valid"):
            labels, weights, imgs, sym_ids, rejected = _filter_labelmix_samples(
                dataset_train,
                imgs=imgs,
                labels=labels,
                weights=weights,
                sym_ids=sym_ids,
                sample_count=sample_count,
            )
            if labels.numel() == 0:
                _logger.warning(
                    "All samples rejected by aspect ratio constraints at step %d.", step
                )
                continue
            if rejected:
                _logger.info(
                    "Rejected %d samples by aspect ratio constraints at step %d.",
                    rejected,
                    step,
                )
        else:
            labels = labels[:sample_count]
            weights = weights[:sample_count]
            imgs = imgs[:sample_count]
            if sym_ids is not None:
                sym_ids = sym_ids[:sample_count]

        print(f"\nStep {step} (epoch {current_epoch}): showing {labels.shape[0]} samples")
        for i in range(labels.shape[0]):
            lbl = labels[i].detach().cpu()
            w = weights[i].detach().cpu()
            asc = _is_ascending(w)
            name_list = [_label_id_to_name(int(v), label_names) for v in lbl.tolist()]
            dist = {
                "labels": lbl.tolist(),
                "label_names": name_list,
                "weights": [float(x) for x in w.tolist()],
                "ascending": bool(asc),
            }
            if sym_ids is not None:
                dist["sym_id"] = int(sym_ids[i].item())
            print(json.dumps(dist))

            img = _to_uint8_img(imgs[i].detach().cpu(), mean, std)
            out_path = os.path.join(output_dir, f"step_{step:04d}_sample_{i:02d}.png")
            _save_sample_plot(out_path, img, lbl, w, label_names=label_names)


if __name__ == "__main__":
    main()
