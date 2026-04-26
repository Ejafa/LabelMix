#!/usr/bin/env python3
"""Generate a jobs.yaml for ViTDet Mask R-CNN sweeps (COCO / LVIS).

Produces a YAML in the schema consumed by ``jobdaemon.py submit`` and by the
smart scheduler in ``job_scheduler.py`` (``--num-gpus`` + ``--dist-url``
placeholders are resolved at launch time).

Typical usage::

    # Sweep all converted checkpoints under ./converted/ across 3 seeds:
    python detectron2_vitdet/generate_vitdet_jobs.py \\
        --checkpoints-dir converted \\
        --seeds 42 123 777 \\
        --output vitdet_jobs.yaml

    # Only one variant, one seed, custom LR:
    python detectron2_vitdet/generate_vitdet_jobs.py \\
        --variant wee \\
        --checkpoint converted/vit_wee_in1k.pth \\
        --seeds 42 --learning-rates 1e-4 2e-4 \\
        --output vitdet_wee_sweep.yaml

    # Append a single job entry for a freshly-converted checkpoint
    # (used by convert_timm_to_vitdet.py --emit-job):
    python detectron2_vitdet/generate_vitdet_jobs.py \\
        --append vitdet_jobs.yaml \\
        --variant wee \\
        --checkpoint converted/vit_wee_in1k.pth \\
        --seeds 42

The daemon expects a single ``gpus`` value per file, so this script never
mixes GPU counts across jobs. Use ``--gpus-per-job`` to change it.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
# Maps short CLI tags ("wee", "little", ...) to:
#   - the LazyConfig file under ``configs/COCO/``
#   - a pattern used to auto-match a converted checkpoint to the variant.
#
# Keep this in sync with ``MODEL_META`` in ``convert_timm_to_vitdet.py``.
VARIANTS: Dict[str, Dict[str, str]] = {
    "wee": {
        "config": "configs/COCO/mask_rcnn_vitdet_wee_100ep.py",
        "ckpt_pattern": "vit_wee",
    },
    "little": {
        "config": "configs/COCO/mask_rcnn_vitdet_little_100ep.py",
        "ckpt_pattern": "vit_little",
    },
    "medium": {
        "config": "configs/COCO/mask_rcnn_vitdet_medium_100ep.py",
        "ckpt_pattern": "vit_medium",
    },
    "betwixt": {
        "config": "configs/COCO/mask_rcnn_vitdet_betwixt_100ep.py",
        "ckpt_pattern": "vit_betwixt",
    },
}


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def build_job_name(
    variant: str,
    seed: int,
    lr: Optional[float],
    dataset: str,
    img_size: int,
) -> str:
    lr_tag = f"__lr{lr:g}" if lr is not None else ""
    return (
        f"vitdet-{variant}__{dataset}__img{img_size}"
        f"__seed{seed}{lr_tag}"
    )


def build_job_cmd(
    config_path: str,
    checkpoint: str,
    output_dir: str,
    seed: int,
    lr: Optional[float],
    extra_overrides: List[str],
) -> str:
    """Build the training command for a single ViTDet run.

    ``{gpus}`` and ``{port}`` are left as placeholders and get resolved by
    ``jobdaemon.py`` / ``job_scheduler.py`` at launch time.
    """
    overrides: List[str] = [
        f"train.init_checkpoint={checkpoint}",
        f"train.output_dir={output_dir}",
        f"train.seed={seed}",
    ]
    if lr is not None:
        overrides.append(f"optimizer.lr={lr:g}")
    overrides.extend(extra_overrides)

    cmd = (
        f"python train_net.py "
        f"--config-file {config_path} "
        f"--num-gpus {{gpus}} "
        f"--dist-url tcp://127.0.0.1:{{port}} "
        + " ".join(overrides)
    )
    return cmd


def _resolve_checkpoint_for_variant(
    variant: str,
    checkpoints_dir: Optional[str],
    explicit: Optional[str],
) -> Optional[str]:
    """Find the converted .pth for a variant, either explicitly or by pattern."""
    if explicit:
        return os.path.abspath(explicit)
    if not checkpoints_dir:
        return None
    pattern = VARIANTS[variant]["ckpt_pattern"]
    if not os.path.isdir(checkpoints_dir):
        return None
    candidates = sorted(
        os.path.join(checkpoints_dir, f)
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".pth") and pattern in f
    )
    if not candidates:
        return None
    # Prefer the most recently modified match for reproducibility across re-runs.
    candidates.sort(key=os.path.getmtime, reverse=True)
    return os.path.abspath(candidates[0])


def generate_jobs(
    variants: List[str],
    seeds: List[int],
    learning_rates: List[Optional[float]],
    checkpoints_dir: Optional[str],
    explicit_checkpoint: Optional[str],
    output_root: str,
    dataset: str,
    img_size: int,
    extra_overrides: List[str],
) -> List[Dict[str, Any]]:
    """Cartesian product sweep over (variant × seed × lr)."""
    jobs: List[Dict[str, Any]] = []

    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(
                f"Unknown variant '{variant}'. Known: {', '.join(VARIANTS)}"
            )
        config_path = os.path.join(_THIS_DIR, VARIANTS[variant]["config"])
        if not os.path.exists(config_path):
            raise SystemExit(f"Config not found for variant '{variant}': {config_path}")

        ckpt = _resolve_checkpoint_for_variant(
            variant,
            checkpoints_dir=checkpoints_dir,
            explicit=explicit_checkpoint if len(variants) == 1 else None,
        )
        if ckpt is None:
            print(
                f"  ⚠️  No checkpoint found for variant '{variant}' "
                f"(pattern='{VARIANTS[variant]['ckpt_pattern']}' in "
                f"'{checkpoints_dir}'). Skipping.",
                file=sys.stderr,
            )
            continue
        if not os.path.exists(ckpt):
            print(f"  ⚠️  Checkpoint missing on disk: {ckpt}. Skipping.", file=sys.stderr)
            continue

        for seed in seeds:
            for lr in learning_rates:
                name = build_job_name(variant, seed, lr, dataset, img_size)
                job_output_dir = os.path.join(output_root, name)
                cmd = build_job_cmd(
                    config_path=config_path,
                    checkpoint=ckpt,
                    output_dir=job_output_dir,
                    seed=seed,
                    lr=lr,
                    extra_overrides=extra_overrides,
                )
                jobs.append({"name": name, "cmd": cmd})

    return jobs


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

DEFAULTS_TEMPLATE: Dict[str, Any] = {
    "gpus": 8,
    "max_retries": 3,
    "working_dir": _THIS_DIR,
}


def write_jobs_yaml(
    path: str,
    defaults: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    append: bool = False,
) -> Tuple[int, int]:
    """Write or append to a jobs YAML. Returns (num_added, total_after)."""
    if append and os.path.exists(path):
        with open(path, "r") as f:
            existing = yaml.safe_load(f) or {}
        existing_defaults = existing.get("defaults", {}) or {}
        existing_jobs = existing.get("jobs", []) or []
        # Keep existing defaults as the source of truth when appending
        # (the daemon enforces a single gpus_needed per file, so we do
        # NOT silently overwrite a user-curated defaults block).
        if existing_defaults and existing_defaults.get("gpus") != defaults.get("gpus"):
            raise SystemExit(
                f"Refusing to append: existing defaults.gpus={existing_defaults.get('gpus')} "
                f"but new jobs use gpus={defaults.get('gpus')}. "
                f"jobdaemon.py rejects mixed gpus_needed in one file."
            )
        seen_names = {j.get("name") for j in existing_jobs}
        fresh = [j for j in jobs if j.get("name") not in seen_names]
        combined = existing_jobs + fresh
        out = {"defaults": existing_defaults or defaults, "jobs": combined}
        added = len(fresh)
        total = len(combined)
    else:
        out = {"defaults": defaults, "jobs": jobs}
        added = len(jobs)
        total = len(jobs)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=False)
    return added, total


# ---------------------------------------------------------------------------
# Public API (used by convert_timm_to_vitdet.py --emit-job)
# ---------------------------------------------------------------------------

def emit_job_for_checkpoint(
    variant: str,
    checkpoint_path: str,
    jobs_yaml_path: str,
    seeds: Optional[List[int]] = None,
    learning_rates: Optional[List[Optional[float]]] = None,
    output_root: str = "./output",
    dataset: str = "coco",
    img_size: int = 256,
    gpus_per_job: int = 8,
    extra_overrides: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Convenience entry point: append one variant's jobs to ``jobs_yaml_path``.

    Called by ``convert_timm_to_vitdet.py`` after writing the ``.pth``.
    Returns ``(num_added, total_after)``.
    """
    seeds = list(seeds) if seeds else [42]
    learning_rates = list(learning_rates) if learning_rates else [None]
    extra_overrides = list(extra_overrides) if extra_overrides else []

    jobs = generate_jobs(
        variants=[variant],
        seeds=seeds,
        learning_rates=learning_rates,
        checkpoints_dir=None,
        explicit_checkpoint=checkpoint_path,
        output_root=output_root,
        dataset=dataset,
        img_size=img_size,
        extra_overrides=extra_overrides,
    )
    defaults = dict(DEFAULTS_TEMPLATE, gpus=gpus_per_job)
    return write_jobs_yaml(jobs_yaml_path, defaults, jobs, append=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_lrs(raw: List[str]) -> List[Optional[float]]:
    if not raw:
        return [None]  # single run: use the config-default LR
    out: List[Optional[float]] = []
    for s in raw:
        if s.lower() in ("none", "default", ""):
            out.append(None)
        else:
            out.append(float(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Known variants: " + ", ".join(VARIANTS),
    )
    ap.add_argument(
        "--variant", action="append", default=None,
        choices=sorted(VARIANTS),
        help="ViT variant(s) to sweep. Repeat for multiple (default: all).",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="Path to ONE converted .pth. Only valid when --variant is a "
             "single value.",
    )
    ap.add_argument(
        "--checkpoints-dir", default=None,
        help="Directory containing converted checkpoints (auto-matches by "
             "filename: e.g. 'vit_wee_*.pth' -> wee variant).",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Seeds to sweep (default: [42]).",
    )
    ap.add_argument(
        "--learning-rates", nargs="*", default=[],
        help="LRs to sweep. Empty uses each config's default LR. "
             "Use 'none' / 'default' as a literal placeholder for the "
             "config default within a mixed sweep.",
    )
    ap.add_argument(
        "--output-root", default="./output",
        help="Parent directory for per-job train.output_dir (default: ./output).",
    )
    ap.add_argument(
        "--dataset", default="coco",
        help="Dataset tag baked into job names (default: coco).",
    )
    ap.add_argument(
        "--img-size", type=int, default=256,
        help="Image size tag baked into job names (default: 256). "
             "Informational only — the config controls the actual size.",
    )
    ap.add_argument(
        "--gpus-per-job", type=int, default=8,
        help="GPUs per job; written to 'defaults.gpus' (default: 8). "
             "jobdaemon.py enforces a single value across the whole file.",
    )
    ap.add_argument(
        "--extra-override", action="append", default=[],
        help="Extra LazyConfig override appended to every job command, "
             "e.g. --extra-override dataloader.train.total_batch_size=128 "
             "(repeatable).",
    )
    ap.add_argument(
        "-o", "--output", required=False, default=None,
        help="Write the sweep to this YAML (default: <cwd>/vitdet_jobs.yaml).",
    )
    ap.add_argument(
        "--append", default=None,
        help="Append to an existing YAML instead of overwriting. Mutually "
             "exclusive with --output.",
    )
    args = ap.parse_args()

    if args.output and args.append:
        ap.error("--output and --append are mutually exclusive.")
    if args.checkpoint and (not args.variant or len(args.variant) != 1):
        ap.error("--checkpoint requires exactly one --variant.")
    if not args.checkpoint and not args.checkpoints_dir:
        ap.error("Provide either --checkpoint or --checkpoints-dir.")

    target_variants = args.variant or sorted(VARIANTS)
    lrs = _parse_lrs(args.learning_rates)

    jobs = generate_jobs(
        variants=target_variants,
        seeds=args.seeds,
        learning_rates=lrs,
        checkpoints_dir=args.checkpoints_dir,
        explicit_checkpoint=args.checkpoint,
        output_root=args.output_root,
        dataset=args.dataset,
        img_size=args.img_size,
        extra_overrides=args.extra_override,
    )

    if not jobs:
        print("❌ No jobs were generated (no matching checkpoints?).", file=sys.stderr)
        sys.exit(1)

    defaults = dict(DEFAULTS_TEMPLATE, gpus=args.gpus_per_job)

    target = args.append or args.output or os.path.join(os.getcwd(), "vitdet_jobs.yaml")
    added, total = write_jobs_yaml(
        target, defaults, jobs, append=bool(args.append)
    )

    action = "Appended" if args.append else "Wrote"
    print(f"\n✅ {action} {added} job(s) -> {target} (total in file: {total})")
    print(f"   Variants: {', '.join(target_variants)}")
    print(f"   Seeds:    {args.seeds}")
    print(f"   LRs:      {args.learning_rates or '[config default]'}")
    print(f"   GPUs/job: {args.gpus_per_job}")
    print(f"\n🚀 Next step:")
    print(f"   python job_scheduler.py --input {os.path.basename(target)} "
          f"--schedule-name vitdet --num-nodes 1")
    print(f"   # or directly:")
    print(f"   python jobdaemon.py -s vitdet --node-index 0 submit {os.path.basename(target)}")


if __name__ == "__main__":
    main()
