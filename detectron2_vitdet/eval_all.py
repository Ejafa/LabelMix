#!/usr/bin/env python3
"""Batch-evaluate LabelMix ViT backbones on COCO via ViTDet Mask R-CNN.

Given a directory of backed-up LabelMix runs (as produced by
``evaluation/scripts/backup_runs.py``), this script

1. walks every subdirectory that contains both ``model_best.pth.tar``
   and ``args.yaml``,
2. reads the backbone variant name from ``args.yaml`` (e.g.
   ``vit_wee_patch16_reg1_gap_256``),
3. converts the timm checkpoint into the ViTDet-compatible layout with
   :mod:`convert_timm_to_vitdet` (result is cached on disk),
4. launches ``train_net.py`` with the matching
   ``configs/COCO/mask_rcnn_vitdet_<variant>_30ep.py`` recipe (or, with
   ``--eval-only``, just evaluates an existing ``model_final.pth``),
5. aggregates the COCO AP numbers from every run's ``metrics.json`` into
   one summary CSV.

Expected backup layout (matches ``backup_runs.py``)::

    <backup_root>/
        <group>/                         # e.g. "in1k"
            <experiment_name>/           # e.g. "vit-little__in1k__..."
                args.yaml
                model_best.pth.tar
                summary.csv
                run_status.yaml

The backup root can also be flat (just ``<run>/args.yaml`` + checkpoint)
- the walker descends ``--max-depth`` levels looking for matching runs.

Typical invocations::

    # Train + evaluate every backed-up run on 8 GPUs.
    python eval_all.py --backup-root ../backup --num-gpus 8

    # Dry-run: list what would be done.
    python eval_all.py --backup-root ../backup --dry-run

    # Only re-evaluate runs that already finished training.
    python eval_all.py --backup-root ../backup --eval-only

    # Restrict to one backbone variant + one seed, on 4 GPUs.
    python eval_all.py --backup-root ../backup \\
        --include-variants wee --include-pattern "seed=42" --num-gpus 4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# Ensure DETECTRON2_DATASETS is set *before* we spawn any train_net.py
# subprocesses, so they inherit it via os.environ automatically.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coco_env import ensure_detectron2_datasets  # noqa: E402

ensure_detectron2_datasets()


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
CONVERT_SCRIPT = THIS_DIR / "convert_timm_to_vitdet.py"
TRAIN_SCRIPT = THIS_DIR / "train_net.py"
CONFIG_DIR = THIS_DIR / "configs" / "COCO"
DEFAULT_CONVERTED_DIR = THIS_DIR / "converted"
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
DEFAULT_SUMMARY_CSV = THIS_DIR / "output" / "vitdet_eval_summary.csv"

#: Map timm model name (as written in args.yaml) to ViTDet variant tag.
#: Only ``wee`` and ``betwixt`` are supported for object detection.
MODEL_TO_VARIANT: Dict[str, str] = {
    "vit_wee_patch16_reg1_gap_256":     "wee",
    "vit_betwixt_patch16_reg4_gap_256": "betwixt",
}

#: COCO metric keys we pull out of ``metrics.json`` for the summary CSV.
SUMMARY_METRIC_KEYS: Tuple[str, ...] = (
    "bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl",
    "segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl",
)

_logger = logging.getLogger("eval_all")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """One backed-up LabelMix run discovered under ``--backup-root``."""

    run_dir: Path                   # folder containing model_best.pth.tar
    experiment: str                 # basename of run_dir
    group: str                      # parent folder name (e.g. "in1k")
    model_name: str                 # timm name, e.g. "vit_wee_patch16_reg1_gap_256"
    variant: str                    # ViTDet tag: wee/little/medium/betwixt
    seed: Optional[int] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"{self.group}__{self.experiment}"


@dataclass
class EvalStats:
    scanned: int = 0
    launched: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_unsupported: int = 0
    skipped_cached: int = 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        _logger.debug("YAML load failed for %s: %s", path, exc)
        return {}


def _iter_run_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    """Yield every directory under ``root`` that looks like a backed-up run.

    A directory qualifies when it contains ``args.yaml`` **and**
    ``model_best.pth.tar``.  We descend at most ``max_depth`` levels to
    handle both flat and ``<group>/<experiment>/`` layouts.
    """
    root = root.resolve()
    if not root.is_dir():
        return

    def _walk(d: Path, depth: int) -> Iterable[Path]:
        if (d / "args.yaml").is_file() and (d / "model_best.pth.tar").is_file():
            yield d
            return
        if depth >= max_depth:
            return
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            yield from _walk(child, depth + 1)

    yield from _walk(root, 0)


def discover_runs(
    backup_root: Path,
    *,
    max_depth: int = 3,
    include_variants: Optional[List[str]] = None,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> List[RunSpec]:
    """Return every supported run under ``backup_root`` matching the filters."""
    runs: List[RunSpec] = []
    include_set = set(include_variants) if include_variants else None

    for run_dir in _iter_run_dirs(backup_root, max_depth=max_depth):
        args = _read_yaml(run_dir / "args.yaml")
        model_name = str(args.get("model") or "").strip()
        if not model_name:
            _logger.warning("skip %s: no 'model' in args.yaml", run_dir)
            continue

        variant = MODEL_TO_VARIANT.get(model_name)
        if variant is None:
            _logger.info("skip %s: unsupported model '%s'", run_dir, model_name)
            continue

        if include_set is not None and variant not in include_set:
            continue

        experiment = run_dir.name
        if include_pattern and include_pattern not in experiment:
            continue
        if exclude_pattern and exclude_pattern in experiment:
            continue

        group_dir = run_dir.parent
        group = group_dir.name if group_dir != backup_root else "runs"

        seed = args.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None

        runs.append(RunSpec(
            run_dir=run_dir,
            experiment=experiment,
            group=group,
            model_name=model_name,
            variant=variant,
            seed=seed,
            extra_args=args,
        ))

    return runs


# ---------------------------------------------------------------------------
# Conversion + execution
# ---------------------------------------------------------------------------

def _run(cmd: List[str], *, dry_run: bool, cwd: Optional[Path] = None) -> int:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    _logger.info("$ %s", pretty)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def convert_checkpoint(
    run: RunSpec,
    converted_dir: Path,
    *,
    use_ema: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> Optional[Path]:
    """Convert ``run``'s timm checkpoint to the ViTDet layout (cached)."""
    converted_dir.mkdir(parents=True, exist_ok=True)
    ema_suffix = "_ema" if use_ema else ""
    out_path = converted_dir / f"{run.tag}{ema_suffix}.pth"

    if out_path.exists() and not force:
        _logger.info("[%s] converted checkpoint cached: %s", run.tag, out_path)
        return out_path

    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        "--input",  str(run.run_dir / "model_best.pth.tar"),
        "--output", str(out_path),
        "--model",  run.model_name,
    ]
    if use_ema:
        cmd.append("--use-ema")

    rc = _run(cmd, dry_run=dry_run)
    if rc != 0:
        _logger.error("[%s] convert_timm_to_vitdet failed (exit=%d)", run.tag, rc)
        return None
    return out_path


def _build_train_cmd(
    run: RunSpec,
    init_ckpt: Path,
    output_dir: Path,
    *,
    num_gpus: int,
    master_port: int,
    eval_only: bool,
    extra_overrides: List[str],
) -> List[str]:
    cfg = CONFIG_DIR / f"mask_rcnn_vitdet_{run.variant}_30ep.py"
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port}",
        str(TRAIN_SCRIPT),
        "--config-file", str(cfg),
        "--num-gpus",    str(num_gpus),
    ]
    if eval_only:
        cmd.append("--eval-only")
    # Hydra's override parser rejects un-quoted ``=`` inside the VALUE of a
    # ``key=value`` override ("mismatched input '=' expecting <EOF>"),
    # which bites us for checkpoint paths like
    # ``…/baseline_seed=42__runid-xyz.pth``.  Wrap values containing ``=``
    # in double quotes so Hydra takes them literally.  Argv-style exec
    # means no outer shell is involved, so the quotes go straight into
    # Hydra's input.
    def _q(key: str, value: str) -> str:
        if "=" in value or " " in value:
            return f'{key}="{value}"'
        return f"{key}={value}"

    cmd.extend([
        _q("train.init_checkpoint", str(init_ckpt)),
        _q("train.output_dir",     str(output_dir)),
    ])
    cmd.extend(extra_overrides)
    return cmd


def run_vitdet(
    run: RunSpec,
    *,
    converted_dir: Path,
    output_root: Path,
    num_gpus: int,
    master_port: int,
    eval_only: bool,
    use_ema: bool,
    force_convert: bool,
    force_retrain: bool,
    extra_overrides: List[str],
    dry_run: bool,
) -> Tuple[bool, Optional[Path]]:
    """Convert + train/evaluate one run. Returns (success, output_dir)."""
    output_dir = output_root / run.tag
    final_ckpt = output_dir / "model_final.pth"

    if eval_only:
        if not final_ckpt.is_file() and not dry_run:
            _logger.warning("[%s] --eval-only but %s not found; skipping",
                            run.tag, final_ckpt)
            return (False, output_dir)
        init_ckpt = final_ckpt
    else:
        if final_ckpt.is_file() and not force_retrain:
            _logger.info("[%s] model_final.pth already exists; will --eval-only "
                         "(use --force-retrain to re-train)", run.tag)
            init_ckpt = final_ckpt
            eval_only = True
        else:
            converted = convert_checkpoint(
                run, converted_dir,
                use_ema=use_ema, force=force_convert, dry_run=dry_run,
            )
            if converted is None:
                return (False, output_dir)
            init_ckpt = converted

    cmd = _build_train_cmd(
        run, init_ckpt, output_dir,
        num_gpus=num_gpus,
        master_port=master_port,
        eval_only=eval_only,
        extra_overrides=extra_overrides,
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    rc = _run(cmd, dry_run=dry_run, cwd=THIS_DIR)
    success = (rc == 0)
    if not success:
        _logger.error("[%s] train_net.py failed (exit=%d)", run.tag, rc)
    return (success, output_dir)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _last_eval_metrics(metrics_path: Path) -> Dict[str, float]:
    """Read the latest eval block from detectron2's ``metrics.json``.

    detectron2 writes one JSON object per line; eval lines contain keys
    like ``bbox/AP``.  We return the most recent line that has any such
    key.
    """
    if not metrics_path.is_file():
        return {}

    last: Dict[str, float] = {}
    try:
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if any(k.startswith(("bbox/", "segm/")) for k in rec):
                    last = rec
    except OSError as exc:
        _logger.warning("cannot read %s: %s", metrics_path, exc)
        return {}

    return last


def write_summary_csv(
    rows: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag", "group", "experiment", "variant", "model_name",
                  "seed", "output_dir"] + list(SUMMARY_METRIC_KEYS)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _logger.info("Wrote summary CSV: %s (%d rows)", csv_path, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Batch ViTDet evaluation for backed-up LabelMix runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backup-root", required=True, type=Path,
                   help="Directory containing backed-up runs (subfolders with "
                        "args.yaml + model_best.pth.tar).")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_DIR, type=Path,
                   help="Where to place per-run ViTDet training outputs.")
    p.add_argument("--converted-dir", default=DEFAULT_CONVERTED_DIR, type=Path,
                   help="Cache dir for converted ViTDet-format checkpoints.")
    p.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, type=Path,
                   help="Aggregate COCO metrics into this CSV.")
    p.add_argument("--num-gpus", type=int, default=8,
                   help="GPUs per run (passed to torchrun and --num-gpus).")
    p.add_argument("--master-port-base", type=int, default=29500,
                   help="Base master_port; incremented per run to avoid clashes.")
    p.add_argument("--max-depth", type=int, default=3,
                   help="Max descent depth when walking --backup-root.")
    p.add_argument("--include-variants", nargs="*",
                   choices=sorted(set(MODEL_TO_VARIANT.values())),
                   help="Only evaluate these ViTDet variants.")
    p.add_argument("--include-pattern", default=None,
                   help="Substring that must appear in the experiment folder name.")
    p.add_argument("--exclude-pattern", default=None,
                   help="Substring that, if present, skips the run.")
    p.add_argument("--use-ema", action="store_true",
                   help="Export the EMA weights when converting the checkpoint.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, evaluate an existing model_final.pth.")
    p.add_argument("--force-convert", action="store_true",
                   help="Re-run the timm->ViTDet conversion even if cached.")
    p.add_argument("--force-retrain", action="store_true",
                   help="Re-train even if model_final.pth already exists.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going when a single run fails.")
    p.add_argument("--dry-run", action="store_true",
                   help="Log the commands without executing them.")
    p.add_argument("--summary-only", action="store_true",
                   help="Do not train/evaluate; only aggregate existing metrics.json.")
    p.add_argument("--extra-override", action="append", default=[],
                   help="Extra LazyConfig override passed verbatim to train_net.py, "
                        "e.g. --extra-override dataloader.train.total_batch_size=32. "
                        "Repeat the flag for multiple overrides.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runs = discover_runs(
        args.backup_root,
        max_depth=args.max_depth,
        include_variants=args.include_variants,
        include_pattern=args.include_pattern,
        exclude_pattern=args.exclude_pattern,
    )
    _logger.info("Discovered %d run(s) under %s", len(runs), args.backup_root)

    stats = EvalStats(scanned=len(runs))
    summary_rows: List[Dict[str, Any]] = []

    for i, run in enumerate(runs):
        _logger.info("=" * 72)
        _logger.info("[%d/%d] %s  (variant=%s, seed=%s)",
                     i + 1, len(runs), run.tag, run.variant, run.seed)

        output_dir = args.output_root / run.tag

        if not args.summary_only:
            success, output_dir = run_vitdet(
                run,
                converted_dir=args.converted_dir,
                output_root=args.output_root,
                num_gpus=args.num_gpus,
                master_port=args.master_port_base + i,
                eval_only=args.eval_only,
                use_ema=args.use_ema,
                force_convert=args.force_convert,
                force_retrain=args.force_retrain,
                extra_overrides=list(args.extra_override),
                dry_run=args.dry_run,
            )
            stats.launched += 1
            if success:
                stats.succeeded += 1
            else:
                stats.failed += 1
                if not args.continue_on_error:
                    _logger.error("Aborting after first failure (use "
                                  "--continue-on-error to keep going).")
                    break

        metrics = _last_eval_metrics(output_dir / "metrics.json")
        row: Dict[str, Any] = {
            "tag": run.tag,
            "group": run.group,
            "experiment": run.experiment,
            "variant": run.variant,
            "model_name": run.model_name,
            "seed": run.seed,
            "output_dir": str(output_dir),
        }
        row.update({k: metrics.get(k) for k in SUMMARY_METRIC_KEYS})
        summary_rows.append(row)

    if summary_rows and not args.dry_run:
        write_summary_csv(summary_rows, args.summary_csv)

    _logger.info(
        "Summary: scanned=%d launched=%d succeeded=%d failed=%d",
        stats.scanned, stats.launched, stats.succeeded, stats.failed,
    )
    return 0 if stats.failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
