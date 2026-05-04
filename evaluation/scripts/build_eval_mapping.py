"""Build a ``{run_name: run_dir}`` YAML mapping for W&B runs in the in1k group.

Scans ``evaluation/data/raw/wandb/in1k/*`` for runs whose ``metadata.json`` has
``group == "in1k"``, then resolves each run's on-disk checkpoint directory via
its ``config.yaml`` (``output`` + ``experiment``). Only runs that have the
requested checkpoint file on disk are emitted.

For relative ``output`` paths the resolver tries each ``--extra-root`` in order
(after the primary ``REPO_ROOT``) and keeps the first match that contains the
checkpoint. This lets us stitch together runs from sibling checkouts (e.g.
``ggez/LabelMix`` and an older ``labelmix/`` tree) without moving files.

Usage::

    python -m evaluation.scripts.build_eval_mapping \\
        --output evaluation/data/raw/in1k_mapping.yaml \\
        --checkpoint-name model_best.pth.tar \\
        --extra-root /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(
    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix"
)
DEFAULT_WANDB_DIR = REPO_ROOT / "evaluation/data/raw/wandb/in1k"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation/data/raw/in1k_mapping.yaml"


def _candidate_run_dirs(output: str, experiment: str,
                        extra_roots: list[Path]) -> list[Path]:
    """Yield plausible run directories for an ``(output, experiment)`` pair.

    Absolute ``output`` is returned as-is. Otherwise REPO_ROOT is tried first,
    then every ``--extra-root`` in the order they were given.
    """
    out = output[2:] if output.startswith("./") else output
    if os.path.isabs(out):
        return [Path(out) / experiment]
    roots = [REPO_ROOT, *extra_roots]
    return [root / out / experiment for root in roots]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wandb-dir", default=str(DEFAULT_WANDB_DIR),
                   help="Directory with per-run W&B sync folders.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Where to write the mapping YAML.")
    p.add_argument("--checkpoint-name", default="model_best.pth.tar",
                   help="Checkpoint file expected inside each run dir.")
    p.add_argument("--group", default="in1k",
                   help="Only keep runs whose metadata.json group matches this.")
    p.add_argument("--include-states", nargs="*",
                   default=["finished", "running", "crashed"],
                   help="Run states to include (default: all).")
    p.add_argument("--extra-root", action="append", default=[],
                   help="Additional filesystem root to try when resolving the "
                        "run directory from a relative ``output`` in "
                        "config.yaml. May be given multiple times.")
    args = p.parse_args(argv)

    wandb_dir = Path(args.wandb_dir)
    if not wandb_dir.is_dir():
        print(f"ERROR: {wandb_dir} does not exist.", file=sys.stderr)
        return 2

    extra_roots = [Path(r) for r in args.extra_root]

    mapping: dict[str, str] = {}
    skipped_no_ckpt: list[tuple[str, str, list[str]]] = []
    skipped_wrong_group: list[str] = []
    skipped_state: list[tuple[str, str]] = []

    for entry in sorted(wandb_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "metadata.json"
        cfg_path = entry / "config.yaml"
        if not (meta_path.is_file() and cfg_path.is_file()):
            continue

        meta = json.loads(meta_path.read_text())
        cfg = yaml.safe_load(cfg_path.read_text())

        if meta.get("group") != args.group:
            skipped_wrong_group.append(entry.name)
            continue
        state = meta.get("state")
        if state not in args.include_states:
            skipped_state.append((entry.name, state))
            continue

        candidates = _candidate_run_dirs(cfg.get("output", ""),
                                         cfg.get("experiment", ""),
                                         extra_roots)
        name = meta.get("display_name") or cfg.get("experiment") or entry.name
        hit: Path | None = None
        for cand in candidates:
            if (cand / args.checkpoint_name).is_file():
                hit = cand
                break
        if hit is None:
            skipped_no_ckpt.append((entry.name, name, [str(c) for c in candidates]))
            continue

        # W&B run id kept in the name to guarantee uniqueness across seeds.
        key = f"{name}__{entry.name}"
        mapping[key] = str(hit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(mapping, f, sort_keys=True)

    print(f"Wrote {len(mapping)} entries to {out_path}")
    if skipped_wrong_group:
        print(f"Skipped (wrong group): {len(skipped_wrong_group)}")
    if skipped_state:
        print(f"Skipped (state not in {args.include_states}): {len(skipped_state)}")
        for rid, st in skipped_state:
            print(f"  - {rid} state={st}")
    if skipped_no_ckpt:
        print(f"Skipped (missing {args.checkpoint_name}): {len(skipped_no_ckpt)}")
        for rid, name, rds in skipped_no_ckpt:
            print(f"  - {rid} | {name}")
            for rd in rds:
                print(f"       tried: {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
