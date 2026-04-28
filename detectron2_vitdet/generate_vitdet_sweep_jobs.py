#!/usr/bin/env python3
"""Generate a curated jobs.yaml for the vit-wee + vit-betwixt sweep.

Selection criteria (matches the user brief exactly):

* Only ``vit-wee`` and ``vit-betwixt`` variants.
* Only checkpoints whose filename contains at least one of the tokens
  ``pl-loss``, ``soft-ce``, ``mosaic``, or ``baseline``.
* All three pretraining seeds (42, 43, 44).
* Detection training seed is fixed to 42.
* Jobs are ordered so that *all* jobs for pretraining seed 42 are emitted
  first, then all for seed 43, then all for seed 44. Within one seed the
  order is (variant, category) with variants sorted (``betwixt`` < ``wee``)
  and categories sorted alphabetically.

With 2 variants * 4 categories * 3 pretraining seeds this produces 24 jobs.

The built-in ``emit_job_for_checkpoint`` helper in ``generate_vitdet_jobs``
cannot be used here: its job-name template is
``vitdet-{variant}__{dataset}__img{img}__seed{training_seed}``, which does
NOT encode the checkpoint identity — so appending 12 betwixt jobs with the
same training seed would collapse to a single entry via the built-in
name-dedup logic. This script therefore builds jobs directly and writes
the YAML once.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from generate_vitdet_jobs import (  # noqa: E402
    VARIANTS as _VARIANT_REGISTRY,
    _quote_override_value,
)


CONVERTED_DIR = os.path.join(_THIS_DIR, "converted")
OUTPUT_YAML = os.path.join(
    os.path.dirname(_THIS_DIR), "vitdet_sweep_jobs.yaml"
)

# Variants we care about, in the desired within-seed order.
# Left: short tag (matches VARIANTS in generate_vitdet_jobs.py).
# Right: prefix of the converted .pth filenames for that variant.
VARIANTS: List[Tuple[str, str]] = [
    ("betwixt", "betwixt__"),
    ("wee", "wee__"),
]

CATEGORIES: List[str] = ["baseline", "mosaic", "pl-loss", "soft-ce"]

PRETRAIN_SEEDS: List[int] = [42, 43, 44]
TRAINING_SEED: int = 42

GPUS_PER_JOB: int = 4
# Global (across all DDP workers) batch size. The default in
# configs/common/coco_loader_lsj.py is already 256, but we set it explicitly
# per-job so the YAML is self-documenting and immune to config drift.
GLOBAL_BATCH_SIZE: int = 256
OUTPUT_ROOT: str = "./output"
DATASET_TAG: str = "coco"
IMG_SIZE: int = 256

_SEED_RE = re.compile(r"seed=(\d+)")


def _matches_category(filename: str, category: str) -> bool:
    """Case-insensitive, hyphen/underscore-equivalent substring match."""
    norm = filename.lower().replace("_", "-")
    return category.lower().replace("_", "-") in norm


def _checkpoint_seed(filename: str) -> Optional[int]:
    m = _SEED_RE.search(filename)
    return int(m.group(1)) if m else None


def _find_checkpoint(
    variant_prefix: str, category: str, pretrain_seed: int
) -> Optional[str]:
    if not os.path.isdir(CONVERTED_DIR):
        return None
    matches: List[str] = []
    for fn in os.listdir(CONVERTED_DIR):
        if not fn.endswith(".pth"):
            continue
        if not fn.startswith(variant_prefix):
            continue
        if not _matches_category(fn, category):
            continue
        if _checkpoint_seed(fn) != pretrain_seed:
            continue
        matches.append(os.path.join(CONVERTED_DIR, fn))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _run_id_from_filename(filename: str) -> str:
    """Extract the ``runid-<id>`` tag from a converted checkpoint filename."""
    m = re.search(r"runid-([A-Za-z0-9]+)", filename)
    return m.group(1) if m else "unknown"


def _build_job(
    variant: str,
    category: str,
    pretrain_seed: int,
    checkpoint_path: str,
) -> Dict[str, Any]:
    """Build a single job dict with a unique name and resolved command."""
    config_rel = _VARIANT_REGISTRY[variant]["config"]
    config_path = os.path.join(_THIS_DIR, config_rel)

    # Human-readable, collision-proof name: includes category + pretrain seed
    # + the 8-char wandb run id tail, so no two jobs ever share a name.
    run_id = _run_id_from_filename(os.path.basename(checkpoint_path))
    name = (
        f"vitdet-{variant}__{DATASET_TAG}__img{IMG_SIZE}"
        f"__{category}__ptseed{pretrain_seed}__trseed{TRAINING_SEED}"
        f"__{run_id}"
    )
    job_output_dir = os.path.join(OUTPUT_ROOT, name)

    overrides = [
        _quote_override_value("train.init_checkpoint", checkpoint_path),
        _quote_override_value("train.output_dir", job_output_dir),
        f"train.seed={TRAINING_SEED}",
        f"dataloader.train.total_batch_size={GLOBAL_BATCH_SIZE}",
    ]
    cmd = (
        f"python train_net.py "
        f"--config-file {config_path} "
        f"--num-gpus {{gpus}} "
        f"--dist-url tcp://127.0.0.1:{{port}} "
        + " ".join(overrides)
    )
    return {"name": name, "cmd": cmd}


def main() -> int:
    jobs: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str, int]] = []

    # Outer: pretraining seed -> guarantees all seed-42 jobs come first.
    for pretrain_seed in PRETRAIN_SEEDS:
        for variant, variant_prefix in VARIANTS:
            for category in CATEGORIES:
                ckpt = _find_checkpoint(
                    variant_prefix, category, pretrain_seed
                )
                if ckpt is None:
                    missing.append((variant, category, pretrain_seed))
                    continue
                jobs.append(
                    _build_job(variant, category, pretrain_seed, ckpt)
                )

    if missing:
        print(f"⚠️  Missing {len(missing)} checkpoint(s):")
        for v, c, s in missing:
            print(f"    - variant={v} category={c} seed={s}")

    defaults = {
        "gpus": GPUS_PER_JOB,
        "max_retries": 3,
        "working_dir": _THIS_DIR,
    }
    payload = {"defaults": defaults, "jobs": jobs}

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_YAML)) or ".", exist_ok=True)
    with open(OUTPUT_YAML, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Wrote {len(jobs)} job(s) -> {OUTPUT_YAML}")
    print("Order (first 3 and last 3):")
    for j in jobs[:3] + (["..."] if len(jobs) > 6 else []) + jobs[-3:]:
        if isinstance(j, str):
            print(f"   {j}")
        else:
            print(f"   - {j['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
