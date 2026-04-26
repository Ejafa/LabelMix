"""Convert a timm checkpoint produced by LabelMix training into a state dict
that can be loaded by detectron2's ``ViT`` backbone (ViTDet).

The LabelMix ViT family uses ``no_embed_class=True`` plus ``init_values=1e-5``
LayerScale, register tokens, and ``GAP`` classification head. Detectron2's
``ViT`` backbone has no LayerScale and no register / cls tokens, so we:

* drop the classification head and final ``norm`` (it is absent from the
  detection backbone -- detection adds its own norms via the feature pyramid);
* drop ``cls_token`` and ``reg_token`` (register tokens are not used during
  dense prediction);
* absorb ``ls1.gamma`` into ``attn.proj.{weight,bias}`` and ``ls2.gamma`` into
  ``mlp.fc2.{weight,bias}`` (mathematically equivalent: ``gamma * proj(x)``
  equals ``(gamma * W_proj) @ x + gamma * b_proj`` because LayerScale is a
  channel-wise scalar);
* keep the absolute positional embedding. Detectron2 interpolates it on the fly
  when ``pretrain_img_size`` differs from the detection ``img_size`` (set
  ``pretrain_use_cls_token=False`` because we trained without a CLS token).

Example::

    python convert_timm_to_vitdet.py \\
        --input mixed_loss/vit-wee__in1k__img256__.../model_best.pth.tar \\
        --output converted/vit_wee_in1k.pth \\
        --use-ema
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import torch


# Known LabelMix ViT variants. ``pretrain_img_size`` is the training image
# size; detectron2 uses it to interpolate the position embedding at 1024x1024.
MODEL_META: Dict[str, Dict[str, int]] = {
    "vit_wee_patch16_reg1_gap_256":     {"embed_dim": 256, "depth": 14, "num_heads": 4,  "mlp_ratio": 5.0, "patch_size": 16, "pretrain_img_size": 256, "reg_tokens": 1},
    "vit_little_patch16_reg4_gap_256":  {"embed_dim": 320, "depth": 14, "num_heads": 5,  "mlp_ratio": 5.6, "patch_size": 16, "pretrain_img_size": 256, "reg_tokens": 4},
    "vit_medium_patch16_reg1_gap_256":  {"embed_dim": 512, "depth": 12, "num_heads": 8,  "mlp_ratio": 4.0, "patch_size": 16, "pretrain_img_size": 256, "reg_tokens": 1},
    "vit_betwixt_patch16_reg4_gap_256": {"embed_dim": 640, "depth": 12, "num_heads": 10, "mlp_ratio": 4.0, "patch_size": 16, "pretrain_img_size": 256, "reg_tokens": 4},
}

# Prefixes that get dropped outright (classification head, dist/cls tokens, ...)
_DROP_PREFIXES = (
    "head.",
    "head_drop.",
    "pre_logits.",
    "fc_norm.",
    # register / cls tokens are discarded (ViTDet uses only dense tokens)
    "cls_token",
    "reg_token",
)

# Keys that should be dropped by exact match (top-level final norm is unused by
# ViTDet -- detectron2 adds its own norms in SimpleFeaturePyramid).
_DROP_EXACT = {
    "norm.weight",
    "norm.bias",
    "norm_pre.weight",
    "norm_pre.bias",
}


def _strip_prefix(k: str) -> str:
    """Strip ``module.`` / ``_orig_mod.`` wrappers that DDP + torch.compile add."""
    for p in ("module.", "_orig_mod."):
        while k.startswith(p):
            k = k[len(p):]
    return k


def _should_drop(k: str) -> bool:
    if k in _DROP_EXACT:
        return True
    return any(k.startswith(p) for p in _DROP_PREFIXES)


def _infer_model(args_dict: dict, state_dict: Dict[str, torch.Tensor]) -> str:
    """Try to infer which LabelMix ViT variant produced the checkpoint."""
    if args_dict is not None and "model" in args_dict:
        m = args_dict["model"]
        if m in MODEL_META:
            return m
    # Fallback: infer from embed_dim & depth of the state dict.
    embed_dim = None
    depth = 0
    for k, v in state_dict.items():
        if k.endswith("patch_embed.proj.weight"):
            embed_dim = v.shape[0]
        m = re.match(r"(?:.*\.)?blocks\.(\d+)\.norm1\.weight", k)
        if m:
            depth = max(depth, int(m.group(1)) + 1)
    for name, meta in MODEL_META.items():
        if meta["embed_dim"] == embed_dim and meta["depth"] == depth:
            return name
    raise ValueError(
        f"Unable to infer model variant from checkpoint (embed_dim={embed_dim}, depth={depth}). "
        "Pass --model explicitly."
    )


def _fuse_layer_scale(
    state_dict: Dict[str, torch.Tensor],
    depth: int,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Fold ``ls1.gamma`` into ``attn.proj`` and ``ls2.gamma`` into ``mlp.fc2``."""
    out = OrderedDict(state_dict)
    for i in range(depth):
        for ls_name, proj_w, proj_b in (
            (f"blocks.{i}.ls1.gamma", f"blocks.{i}.attn.proj.weight", f"blocks.{i}.attn.proj.bias"),
            (f"blocks.{i}.ls2.gamma", f"blocks.{i}.mlp.fc2.weight",   f"blocks.{i}.mlp.fc2.bias"),
        ):
            if ls_name not in out:
                continue
            gamma = out.pop(ls_name)
            if proj_w in out:
                # weight shape: (out_dim, in_dim). Scale output rows.
                out[proj_w] = out[proj_w] * gamma.view(-1, 1).to(out[proj_w].dtype)
            if proj_b in out:
                out[proj_b] = out[proj_b] * gamma.to(out[proj_b].dtype)
            if verbose:
                print(f"  fused {ls_name} -> {proj_w}, {proj_b}")
    return out


def _strip_pos_embed_tokens(
    pos_embed: torch.Tensor,
    num_prefix_tokens: int,
    reg_tokens: int,
) -> torch.Tensor:
    """LabelMix uses ``no_embed_class=True`` so pos_embed only contains the
    spatial grid -- nothing to strip. We keep this helper for safety in case a
    future variant re-enables class token pos embeddings."""
    total_prefix = num_prefix_tokens + reg_tokens
    if pos_embed.ndim != 3:
        raise ValueError(f"Unexpected pos_embed shape {tuple(pos_embed.shape)}")
    _, n, _ = pos_embed.shape
    grid = int(round(math.sqrt(n)))
    if grid * grid == n:
        # already pure grid -> nothing to do
        return pos_embed
    if total_prefix and n - total_prefix > 0:
        grid = int(round(math.sqrt(n - total_prefix)))
        if grid * grid == n - total_prefix:
            return pos_embed[:, total_prefix:, :].contiguous()
    raise ValueError(
        f"pos_embed of length {n} cannot be interpreted as a square grid "
        f"(prefix={total_prefix})"
    )


def convert_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_name: str,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Convert *state_dict* into a dict loadable by detectron2's ViT backbone.

    Returns (new_state_dict, backbone_config) where ``backbone_config`` contains
    the kwargs to pass to ``detectron2.modeling.ViT``.
    """
    meta = MODEL_META[model_name]
    depth = meta["depth"]
    reg_tokens = meta["reg_tokens"]

    # 1) unwrap DDP / torch.compile prefixes & drop head / token keys
    clean: Dict[str, torch.Tensor] = OrderedDict()
    dropped = []
    for k, v in state_dict.items():
        k = _strip_prefix(k)
        if _should_drop(k):
            dropped.append(k)
            continue
        clean[k] = v
    if verbose and dropped:
        print(f"Dropped {len(dropped)} head/token keys, e.g. {dropped[:5]}")

    # 2) fuse LayerScale into adjacent projections
    clean = _fuse_layer_scale(clean, depth, verbose=verbose)

    # 3) sanitize pos_embed if a cls-style prefix sneaks in (defensive)
    if "pos_embed" in clean:
        clean["pos_embed"] = _strip_pos_embed_tokens(
            clean["pos_embed"], num_prefix_tokens=0, reg_tokens=reg_tokens
        )

    # 4) Detectron2 wraps weights inside a flat OrderedDict under key "model"
    backbone_kwargs = {
        "img_size": 1024,                            # detection image size
        "patch_size": meta["patch_size"],
        "embed_dim": meta["embed_dim"],
        "depth": meta["depth"],
        "num_heads": meta["num_heads"],
        "mlp_ratio": meta["mlp_ratio"],
        "qkv_bias": True,
        "pretrain_img_size": meta["pretrain_img_size"],
        "pretrain_use_cls_token": False,             # no_embed_class=True upstream
        "use_abs_pos": True,
        "use_rel_pos": True,                         # added on detection side
        "window_size": 14,
        # standard ViTDet global-attention block indexes (every depth/4-th block)
        "window_block_indexes": _default_window_block_indexes(meta["depth"]),
    }
    return clean, backbone_kwargs


def _default_window_block_indexes(depth: int) -> list:
    """Replicate the ViTDet heuristic: global attention at every depth/4-th
    block (the last block in each quarter). All other blocks use window attn."""
    global_period = max(depth // 4, 1)
    global_idxs = set(range(global_period - 1, depth, global_period))
    # Make sure the last block does global attention
    global_idxs.add(depth - 1)
    return sorted(i for i in range(depth) if i not in global_idxs)


def load_checkpoint(path: str) -> dict:
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    # raw state dict
    return {"state_dict": ckpt, "args": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", "-i", required=True, help="timm checkpoint (*.pth.tar)")
    ap.add_argument("--output", "-o", required=True, help="detectron2 .pth output")
    ap.add_argument(
        "--model", default=None,
        help="Variant name (auto-detected from args if omitted). One of: "
             + ", ".join(MODEL_META.keys()),
    )
    ap.add_argument("--use-ema", action="store_true", help="Export the EMA weights if present.")
    ap.add_argument("--quiet", action="store_true")

    # ── Optional: auto-generate a jobdaemon-compatible jobs.yaml entry ────
    ap.add_argument(
        "--emit-job", nargs="?", const="vitdet_jobs.yaml", default=None,
        metavar="YAML",
        help="After writing the converted checkpoint, append a training-job "
             "entry for it to the given YAML (default: ./vitdet_jobs.yaml) "
             "using detectron2_vitdet/generate_vitdet_jobs.py.",
    )
    ap.add_argument(
        "--emit-seeds", nargs="+", type=int, default=[42],
        help="Seeds to include in the auto-generated jobs (default: [42]).",
    )
    ap.add_argument(
        "--emit-lrs", nargs="*", default=[],
        help="Learning rates to sweep in the auto-generated jobs. Empty "
             "means use each config's default LR.",
    )
    ap.add_argument(
        "--emit-gpus", type=int, default=8,
        help="GPUs per job for the auto-generated YAML (default: 8). "
             "jobdaemon.py enforces a single value per file.",
    )
    ap.add_argument(
        "--emit-output-root", default="./output",
        help="Parent directory for train.output_dir in the auto-generated "
             "jobs (default: ./output).",
    )
    args = ap.parse_args()

    ckpt = load_checkpoint(args.input)

    # Select weights source
    if args.use_ema and isinstance(ckpt.get("state_dict_ema"), dict):
        src = ckpt["state_dict_ema"]
        print("Using state_dict_ema")
    else:
        src = ckpt["state_dict"]
        print("Using state_dict")

    model_name = args.model or _infer_model(ckpt.get("args"), src)
    print(f"Detected model: {model_name}")

    new_sd, backbone_cfg = convert_state_dict(src, model_name, verbose=not args.quiet)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            # detectron2's DetectionCheckpointer recognises this format:
            # when the top-level dict has a "model" entry it is used directly.
            "model": new_sd,
            "__author__": "labelmix/detectron2_vitdet/convert_timm_to_vitdet.py",
            "source_model": model_name,
            "backbone_kwargs": backbone_cfg,
            "matching_heuristics": True,
        },
        args.output,
    )
    print(f"\nSaved {len(new_sd)} tensors -> {args.output}")
    print("Use this file as ``train.init_checkpoint`` in the ViTDet config.")
    print("Backbone kwargs to use in the config:")
    for k, v in backbone_cfg.items():
        print(f"  {k}: {v}")

    # ── Optional: auto-emit a jobs.yaml entry ──────────────────────────
    if args.emit_job:
        try:
            from generate_vitdet_jobs import (
                emit_job_for_checkpoint,
                VARIANTS as _VITDET_VARIANTS,
            )
        except ImportError:
            # convert_timm_to_vitdet.py may be invoked from any cwd — ensure
            # we can import the sibling module living next to this script.
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from generate_vitdet_jobs import (
                emit_job_for_checkpoint,
                VARIANTS as _VITDET_VARIANTS,
            )

        # Map the timm model name (e.g. 'vit_wee_patch16_reg1_gap_256')
        # to the short variant tag ('wee') expected by the generator.
        variant_tag = None
        for tag in _VITDET_VARIANTS:
            if f"vit_{tag}_" in model_name:
                variant_tag = tag
                break
        if variant_tag is None:
            print(
                f"\n⚠️  Could not map '{model_name}' to a known ViTDet variant "
                f"({', '.join(_VITDET_VARIANTS)}) — skipping --emit-job."
            )
        else:
            lrs: list = []
            for s in args.emit_lrs:
                if s.lower() in ("none", "default", ""):
                    lrs.append(None)
                else:
                    lrs.append(float(s))
            if not lrs:
                lrs = [None]

            added, total = emit_job_for_checkpoint(
                variant=variant_tag,
                checkpoint_path=os.path.abspath(args.output),
                jobs_yaml_path=args.emit_job,
                seeds=args.emit_seeds,
                learning_rates=lrs,
                output_root=args.emit_output_root,
                gpus_per_job=args.emit_gpus,
            )
            print(
                f"\n📝 Appended {added} job(s) to {args.emit_job} "
                f"(total in file: {total}). "
                f"Submit with:\n"
                f"   python job_scheduler.py --input {args.emit_job} "
                f"--schedule-name vitdet --num-nodes 1"
            )


if __name__ == "__main__":
    main()
