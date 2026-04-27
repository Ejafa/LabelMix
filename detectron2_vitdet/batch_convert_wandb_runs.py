#!/usr/bin/env python3
"""Batch-convert finished LabelMix W&B runs into ViTDet-compatible checkpoints.

This script glues together three pieces of the repo that already existed:

* ``evaluation/scripts/backup_runs.py`` -- walks the local W&B materialization
  tree under ``evaluation/data/raw/wandb/<group>/<run_id>/`` and locates the
  corresponding on-disk training run (the place where the timm
  ``model_best.pth.tar`` lives).
* ``detectron2_vitdet/convert_timm_to_vitdet.py`` -- converts a single timm
  ViT checkpoint into the layout expected by detectron2's ViT backbone
  (LayerScale folded into proj/fc2, reg / cls tokens dropped, ...).
* ``detectron2_vitdet/generate_vitdet_jobs.py`` -- (optionally) emits a
  jobdaemon-compatible YAML entry per converted checkpoint.

What it does, step by step
--------------------------
1. Iterate every ``<wandb_root>/in1k/<run_id>/`` folder that has a
   ``metadata.json``.
2. Keep only runs whose ``metadata.state == "finished"``.
3. Keep only runs whose display name (== ``config.experiment``) mentions at
   least one of the user-specified categories. By default the categories are
   ``pl_loss`` / ``soft_ce`` / ``baseline`` / ``mosaic`` (hyphen variants are
   accepted too, because the codebase uses both spellings).
4. Resolve the on-disk run dir via ``config.yaml``'s ``output`` +
   ``experiment`` fields (same heuristic as ``backup_runs.py``) and require
   ``model_best.pth.tar`` to exist there.
5. Skip if the run is already recorded in the conversion manifest *and* the
   output file still exists on disk. Otherwise run
   ``convert_timm_to_vitdet.convert_state_dict`` in-process and save the
   result as ``<output_dir>/<variant>__<experiment>__runid-<id>.pth``.
6. Persist the manifest after every conversion (so a crash doesn't lose
   progress) and also mirror it into a human-readable CSV alongside.
7. Optionally append a ViTDet training-job entry per converted checkpoint to
   a jobs.yaml via ``generate_vitdet_jobs.emit_job_for_checkpoint``.

Typical usage::

    # Default: scan evaluation/data/raw/wandb/in1k, write .pth files into
    # detectron2_vitdet/converted/, update the manifest, EMA weights if present.
    python detectron2_vitdet/batch_convert_wandb_runs.py --use-ema

    # Also append training-job entries to vitdet_jobs.yaml in one go.
    python detectron2_vitdet/batch_convert_wandb_runs.py \
        --use-ema --emit-jobs-yaml vitdet_jobs.yaml --emit-seeds 42 43 44

    # Preview without touching the disk.
    python detectron2_vitdet/batch_convert_wandb_runs.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml

# The conversion logic lives in a sibling module — make sure it is importable
# no matter which cwd the user launches us from.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from convert_timm_to_vitdet import (  # noqa: E402
    MODEL_META,
    convert_state_dict,
    load_checkpoint,
)

_logger = logging.getLogger("batch_convert_wandb_runs")


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

#: Project root = the repo containing this file's parent directory.
_PROJECT_ROOT: Path = _THIS_DIR.parent

#: Default W&B materialization root -- matches evaluation/common/paths.py.
DEFAULT_WANDB_ROOT: Path = _PROJECT_ROOT / "evaluation" / "data" / "raw" / "wandb"

#: We only ever care about the ``in1k`` group for object-detection transfer.
DEFAULT_GROUP: str = "in1k"

#: Where the converted ViTDet-layout .pth files will be written.
DEFAULT_OUTPUT_DIR: Path = _THIS_DIR / "converted"

#: Manifest file name, kept next to the converted .pth files.
MANIFEST_NAME: str = "_conversion_manifest.json"

#: Mirror of the manifest for quick eyeballing from the shell / spreadsheets.
MANIFEST_CSV_NAME: str = "_conversion_manifest.csv"

#: W&B ``state`` values considered safe to convert.
FINISHED_STATES: frozenset[str] = frozenset({"finished"})

#: Category filter — an experiment name must contain at least one of these
#: substrings (case-insensitive, hyphen/underscore equivalent) to be eligible.
#: The tuples group spelling variants; a run matches a category if any of its
#: spellings appear in the experiment name.
DEFAULT_CATEGORY_SPELLINGS: Tuple[Tuple[str, ...], ...] = (
    ("pl_loss", "pl-loss"),
    ("soft_ce", "soft-ce"),
    ("baseline",),
    ("mosaic",),
)


# ---------------------------------------------------------------------------
# Small YAML / JSON helpers (duplicated from backup_runs.py to keep this
# script self-contained — they're 3 lines each and we don't want a cross-
# package import just for that).
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        _logger.debug("YAML load failed for %s: %s", path, exc)
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        _logger.debug("JSON load failed for %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Run discovery (same logic as backup_runs.py but restricted to one group)
# ---------------------------------------------------------------------------

def _iter_run_dirs(wandb_root: Path, group: str) -> Iterable[Path]:
    """Yield ``<wandb_root>/<group>/<run_id>/`` dirs that have metadata.json."""
    group_dir = wandb_root / group
    if not group_dir.is_dir():
        _logger.error("W&B group directory not found: %s", group_dir)
        return
    for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        if (run_dir / "metadata.json").is_file():
            yield run_dir


def _resolve_on_disk_run_dir(
    config: Dict[str, Any], project_root: Path
) -> Optional[Path]:
    """Map (output, experiment) from ``config.yaml`` to a real directory.

    This is the same rule used by ``evaluation/scripts/backup_runs.py``:
    ``output`` may be absolute or repo-relative; the final run dir is always
    ``<output>/<experiment>``. We additionally keep the legacy
    ``in1k-main/<experiment>`` fallback for historical runs.
    """
    experiment = config.get("experiment")
    output = config.get("output")
    if not experiment:
        return None

    candidates: List[Path] = []
    if output:
        out = Path(str(output))
        if out.is_absolute():
            candidates.append(out / str(experiment))
        else:
            candidates.append((project_root / out / str(experiment)).resolve())
            candidates.append((Path.cwd() / out / str(experiment)).resolve())
    candidates.append((project_root / "in1k-main" / str(experiment)).resolve())

    for c in candidates:
        if c.is_dir():
            return c
    return None


# ---------------------------------------------------------------------------
# Category filter
# ---------------------------------------------------------------------------

def _matches_category(experiment: str, categories: Iterable[Tuple[str, ...]]) -> Optional[str]:
    """Return the canonical (first) spelling of the matching category, or None.

    Matching is case-insensitive and treats '-' and '_' as equivalent so that
    runs named ``...pl-loss...`` and ``...pl_loss...`` are both picked up.
    """
    norm_exp = experiment.lower().replace("-", "_")
    for spellings in categories:
        for spelling in spellings:
            if spelling.lower().replace("-", "_") in norm_exp:
                return spellings[0]
    return None


# ---------------------------------------------------------------------------
# Variant / short-tag resolution
# ---------------------------------------------------------------------------

#: timm model name  →  short ViTDet variant tag (used by
#: generate_vitdet_jobs.py). Keep in sync with MODEL_META.
_TIMM_TO_VITDET_TAG: Dict[str, str] = {
    "vit_wee_patch16_reg1_gap_256":     "wee",
    "vit_little_patch16_reg4_gap_256":  "little",
    "vit_medium_patch16_reg1_gap_256":  "medium",
    "vit_betwixt_patch16_reg4_gap_256": "betwixt",
}


def _vitdet_tag_for_model(model_name: str) -> Optional[str]:
    if model_name in _TIMM_TO_VITDET_TAG:
        return _TIMM_TO_VITDET_TAG[model_name]
    # Fallback: substring match on known variant prefixes.
    for timm_name, tag in _TIMM_TO_VITDET_TAG.items():
        if f"vit_{tag}_" in model_name:
            return tag
    return None


# ---------------------------------------------------------------------------
# Conversion manifest
# ---------------------------------------------------------------------------

@dataclass
class ConversionManifest:
    """JSON-backed ledger of every checkpoint we've already converted.

    Keyed by W&B ``run_id``. Presence of a run id whose recorded output file
    still exists on disk is what makes :meth:`already_done` short-circuit,
    so accidentally deleting a ``.pth`` transparently triggers a re-convert
    on the next run.
    """

    path: Path
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_or_empty(cls, output_dir: Path) -> "ConversionManifest":
        p = output_dir / MANIFEST_NAME
        if not p.is_file():
            return cls(path=p)
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("manifest is not a JSON object")
            return cls(path=p, entries=data)
        except (OSError, ValueError) as exc:
            _logger.warning(
                "Could not read manifest %s (%s); starting fresh.", p, exc,
            )
            return cls(path=p)

    def already_done(self, run_id: str) -> bool:
        entry = self.entries.get(run_id)
        if not entry:
            return False
        out = entry.get("output_path")
        return bool(out) and Path(out).is_file()

    def record(self, run_id: str, entry: Dict[str, Any]) -> None:
        self.entries[run_id] = entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file + rename so a ^C mid-write can't truncate it.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.entries, f, indent=2, sort_keys=True, default=str)
        tmp.replace(self.path)
        self._write_csv_mirror()

    def _write_csv_mirror(self) -> None:
        """Write a flat CSV alongside the JSON so humans can skim it quickly."""
        csv_path = self.path.parent / MANIFEST_CSV_NAME
        fieldnames = [
            "run_id", "status", "category", "variant", "model",
            "experiment", "seed", "use_ema",
            "source_checkpoint", "output_path",
            "output_sha256_short", "converted_at",
            "error",
        ]
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for run_id in sorted(self.entries):
                    entry = self.entries[run_id]
                    row = {k: entry.get(k, "") for k in fieldnames}
                    row["run_id"] = run_id
                    writer.writerow(row)
        except OSError as exc:
            _logger.debug("Could not write CSV mirror %s: %s", csv_path, exc)


# ---------------------------------------------------------------------------
# Run metadata record
# ---------------------------------------------------------------------------

_SEED_RE = re.compile(r"seed\s*=\s*(\d+)", re.IGNORECASE)


@dataclass
class RunRecord:
    run_id: str
    experiment: str
    model: str                 # timm model name, e.g. vit_wee_patch16_reg1_gap_256
    variant: str               # short tag, e.g. "wee"
    category: str              # which category (pl_loss / soft_ce / ...) matched
    seed: Optional[int]
    source_checkpoint: Path    # path to model_best.pth.tar on disk

    @property
    def output_filename(self) -> str:
        """Deterministic output file name.

        Example::
            wee__model_ablation__vit-wee__in1k__img256__k6-6_a0.1-0.5_pl-loss__seed=42__runid-abc123.pth

        The run id suffix guarantees uniqueness across seeds / re-runs even
        when two experiments happen to share an identical name.
        """
        safe_exp = re.sub(r"[^A-Za-z0-9._=-]+", "_", self.experiment)
        return f"{self.variant}__{safe_exp}__runid-{self.run_id}.pth"


def _parse_seed(experiment: str) -> Optional[int]:
    m = _SEED_RE.search(experiment)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Discovery pipeline
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryStats:
    scanned: int = 0
    not_finished: int = 0
    category_mismatch: int = 0
    missing_config: int = 0
    missing_ondisk_dir: int = 0
    missing_best_ckpt: int = 0
    unknown_variant: int = 0
    eligible: int = 0


def discover_runs(
    wandb_root: Path,
    *,
    group: str,
    project_root: Path,
    categories: Tuple[Tuple[str, ...], ...],
    include_regex: Optional[re.Pattern] = None,
    exclude_regex: Optional[re.Pattern] = None,
) -> Tuple[List[RunRecord], DiscoveryStats]:
    """Walk ``<wandb_root>/<group>/`` and return eligible run records."""
    records: List[RunRecord] = []
    stats = DiscoveryStats()

    for run_dir in _iter_run_dirs(wandb_root, group):
        stats.scanned += 1
        run_id = run_dir.name

        metadata = _load_json(run_dir / "metadata.json")
        state = str(metadata.get("state") or "").lower()
        if state not in FINISHED_STATES:
            _logger.debug("[%s] skip: state=%r", run_id, state)
            stats.not_finished += 1
            continue

        experiment = str(metadata.get("display_name") or "").strip()
        config = _load_yaml(run_dir / "config.yaml")
        if not experiment:
            experiment = str(config.get("experiment") or "").strip()
        if not experiment:
            _logger.warning("[%s] skip: no experiment name in metadata/config", run_id)
            stats.missing_config += 1
            continue

        category = _matches_category(experiment, categories)
        if category is None:
            _logger.debug("[%s] skip: no category match in %r", run_id, experiment)
            stats.category_mismatch += 1
            continue

        if include_regex is not None and not include_regex.search(experiment):
            _logger.debug("[%s] skip: include-regex miss", run_id)
            stats.category_mismatch += 1
            continue
        if exclude_regex is not None and exclude_regex.search(experiment):
            _logger.debug("[%s] skip: exclude-regex hit", run_id)
            stats.category_mismatch += 1
            continue

        model = str(config.get("model") or "").strip()
        variant = _vitdet_tag_for_model(model) if model else None
        if not model or variant is None:
            _logger.warning(
                "[%s] skip: unknown ViT variant for model=%r (experiment=%r)",
                run_id, model, experiment,
            )
            stats.unknown_variant += 1
            continue

        src_dir = _resolve_on_disk_run_dir(config, project_root)
        if src_dir is None:
            _logger.warning(
                "[%s] skip: on-disk run dir not found for experiment=%r "
                "(output=%r)", run_id, experiment, config.get("output"),
            )
            stats.missing_ondisk_dir += 1
            continue

        src_ckpt = src_dir / "model_best.pth.tar"
        if not src_ckpt.is_file():
            _logger.warning(
                "[%s] skip: %s has no model_best.pth.tar", run_id, src_dir,
            )
            stats.missing_best_ckpt += 1
            continue

        records.append(RunRecord(
            run_id=run_id,
            experiment=experiment,
            model=model,
            variant=variant,
            category=category,
            seed=_parse_seed(experiment),
            source_checkpoint=src_ckpt,
        ))
        stats.eligible += 1

    return records, stats


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def _sha256_short(path: Path, nbytes: int = 8) -> str:
    """Return the first ``nbytes`` hex chars of the file's SHA-256 digest.

    This gives the manifest a compact checksum we can use to verify that a
    cached conversion wasn't silently overwritten between runs.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[: nbytes * 2]


def convert_one(
    record: RunRecord,
    *,
    output_dir: Path,
    use_ema: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    """Convert a single run's checkpoint. Returns a manifest-shaped entry."""
    output_path = output_dir / record.output_filename
    entry: Dict[str, Any] = {
        "experiment": record.experiment,
        "category": record.category,
        "variant": record.variant,
        "model": record.model,
        "seed": record.seed,
        "use_ema": bool(use_ema),
        "source_checkpoint": str(record.source_checkpoint),
        "output_path": str(output_path),
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if dry_run:
        entry["status"] = "dry-run"
        _logger.info(
            "[%s] DRY-RUN would convert %s -> %s (variant=%s, use_ema=%s)",
            record.run_id, record.source_checkpoint, output_path,
            record.variant, use_ema,
        )
        return entry

    try:
        ckpt = load_checkpoint(str(record.source_checkpoint))
        if use_ema and isinstance(ckpt.get("state_dict_ema"), dict):
            src_sd = ckpt["state_dict_ema"]
            _logger.info("[%s] using state_dict_ema", record.run_id)
        else:
            src_sd = ckpt["state_dict"]
            _logger.info("[%s] using state_dict", record.run_id)

        new_sd, backbone_cfg = convert_state_dict(
            src_sd, record.model, verbose=False,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(
            {
                "model": new_sd,
                "__author__": "labelmix/detectron2_vitdet/batch_convert_wandb_runs.py",
                "source_model": record.model,
                "backbone_kwargs": backbone_cfg,
                "matching_heuristics": True,
                # Extra provenance so downstream users can trace every
                # converted .pth back to the exact run / checkpoint.
                "source_run_id": record.run_id,
                "source_experiment": record.experiment,
                "source_checkpoint": str(record.source_checkpoint),
                "use_ema": bool(use_ema),
            },
            tmp_path,
        )
        tmp_path.replace(output_path)

        entry["status"] = "ok"
        entry["num_tensors"] = len(new_sd)
        entry["output_sha256_short"] = _sha256_short(output_path)
        _logger.info(
            "[%s] ✓ converted -> %s (%d tensors, sha=%s)",
            record.run_id, output_path, len(new_sd), entry["output_sha256_short"],
        )
    except Exception as exc:  # noqa: BLE001 — we want to keep going on errors
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        _logger.error("[%s] conversion failed: %s", record.run_id, exc)
        _logger.debug("%s", traceback.format_exc())

    return entry


# ---------------------------------------------------------------------------
# Optional: emit a training jobs.yaml entry per converted checkpoint
# ---------------------------------------------------------------------------

def _maybe_emit_job(
    *,
    record: RunRecord,
    output_path: Path,
    jobs_yaml: Path,
    seeds: List[int],
    output_root: Path,
    gpus_per_job: int,
) -> None:
    try:
        from generate_vitdet_jobs import emit_job_for_checkpoint  # noqa: WPS433
    except ImportError as exc:
        _logger.warning(
            "Could not import generate_vitdet_jobs.emit_job_for_checkpoint "
            "(%s); skipping jobs.yaml emission.", exc,
        )
        return

    added, total = emit_job_for_checkpoint(
        variant=record.variant,
        checkpoint_path=str(output_path),
        jobs_yaml_path=str(jobs_yaml),
        seeds=seeds,
        learning_rates=[None],
        output_root=str(output_root),
        gpus_per_job=gpus_per_job,
    )
    _logger.info(
        "[%s] appended %d job(s) to %s (total: %d)",
        record.run_id, added, jobs_yaml, total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--wandb-root", default=str(DEFAULT_WANDB_ROOT),
        help=f"Root of the W&B sync tree (default: {DEFAULT_WANDB_ROOT}).",
    )
    p.add_argument(
        "--group", default=DEFAULT_GROUP,
        help=f"W&B group subdirectory to scan (default: {DEFAULT_GROUP}).",
    )
    p.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where to write converted .pth files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--project-root", default=str(_PROJECT_ROOT),
        help="Repo root, used to resolve relative 'output' paths from config.yaml "
             f"(default: {_PROJECT_ROOT}).",
    )
    p.add_argument(
        "--categories", nargs="+", default=None,
        help="Override the category filter. Each token is a substring (hyphen/"
             "underscore equivalent) that the experiment name must contain. "
             "Default: pl_loss, soft_ce, baseline, mosaic.",
    )
    p.add_argument(
        "--include-pattern", default=None,
        help="Additional Python regex; experiment name must match.",
    )
    p.add_argument(
        "--exclude-pattern", default=None,
        help="Additional Python regex; experiment name must NOT match.",
    )
    p.add_argument(
        "--use-ema", action="store_true",
        help="Export state_dict_ema when present (recommended).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-convert even if the manifest says the run is already done.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be converted without writing any file.",
    )
    p.add_argument(
        "--emit-jobs-yaml", default=None, metavar="PATH",
        help="After a successful conversion, append a ViTDet training-job "
             "entry per checkpoint to this jobs.yaml "
             "(via generate_vitdet_jobs.emit_job_for_checkpoint).",
    )
    p.add_argument(
        "--emit-seeds", type=int, nargs="+", default=[42],
        help="Seeds to sweep in the emitted jobs (default: [42]).",
    )
    p.add_argument(
        "--emit-output-root", default="./output",
        help="train.output_dir parent for emitted jobs (default: ./output).",
    )
    p.add_argument(
        "--emit-gpus-per-job", type=int, default=8,
        help="GPUs per emitted job (default: 8).",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ---- Weights & Biases (Option A: one run per conversion batch) ------
    # W&B logging is ALWAYS ON for real (non-dry-run) invocations. A broken
    # wandb install or failed init is downgraded to a warning so it never
    # blocks the conversion batch itself. --dry-run still skips W&B.
    p.add_argument(
        "--wandb-project", default="vitdet-conversions",
        help="W&B project to log the batch run into (default: vitdet-conversions).",
    )
    p.add_argument(
        "--wandb-entity", default=None,
        help="W&B entity (team/user). Defaults to whatever WANDB_ENTITY / the "
             "local wandb config resolves to.",
    )
    p.add_argument(
        "--wandb-run-name", default=None,
        help="Optional explicit name for the batch W&B run. Default is a "
             "timestamped auto-name.",
    )
    p.add_argument(
        "--wandb-tags", nargs="+", default=None,
        help="Optional W&B tags to attach to the batch run.",
    )
    return p


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_categories(raw: Optional[List[str]]) -> Tuple[Tuple[str, ...], ...]:
    if not raw:
        return DEFAULT_CATEGORY_SPELLINGS
    # One-token-per-category; we still accept both hyphen/underscore spellings
    # via the normalization in ``_matches_category``.
    return tuple((tok,) for tok in raw)


def _init_wandb_run(args: argparse.Namespace):
    """Lazily import wandb and start the batch run.

    W&B logging is always on for real invocations. Returns the live
    ``wandb.run`` object on success, or ``None`` if we are in ``--dry-run``
    mode or wandb is unavailable / fails to initialise. We deliberately
    swallow import / init errors (with a warning) so that a broken W&B
    setup never blocks an actual conversion batch.
    """
    if args.dry_run:
        _logger.info("W&B logging skipped: running in --dry-run mode.")
        return None

    try:
        import wandb  # noqa: WPS433 — optional dep, lazy import
    except ImportError as exc:
        _logger.warning(
            "wandb is not importable (%s); continuing without W&B logging.",
            exc,
        )
        return None

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            job_type="conversion",
            config={
                "wandb_root": str(Path(args.wandb_root).expanduser().resolve()),
                "group": args.group,
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "categories": list(args.categories) if args.categories else None,
                "include_pattern": args.include_pattern,
                "exclude_pattern": args.exclude_pattern,
                "use_ema": bool(args.use_ema),
                "force": bool(args.force),
                "emit_jobs_yaml": args.emit_jobs_yaml,
                "emit_seeds": list(args.emit_seeds) if args.emit_seeds else None,
            },
            reinit=True,
        )
        _logger.info("W&B batch run initialised: %s", run.url if run else "?")
        return run
    except Exception as exc:  # noqa: BLE001 — never let wandb break conversion
        _logger.warning("wandb.init failed (%s); continuing without W&B.", exc)
        return None


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.log_level)

    wandb_root = Path(args.wandb_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    categories = _parse_categories(args.categories)
    include_re = re.compile(args.include_pattern) if args.include_pattern else None
    exclude_re = re.compile(args.exclude_pattern) if args.exclude_pattern else None

    _logger.info("W&B root      : %s", wandb_root)
    _logger.info("Output dir    : %s", output_dir)
    _logger.info("Group         : %s", args.group)
    _logger.info("Categories    : %s", [c[0] for c in categories])
    _logger.info("Use EMA       : %s", args.use_ema)
    _logger.info("Force         : %s", args.force)
    _logger.info("Dry run       : %s", args.dry_run)

    if not wandb_root.is_dir():
        _logger.error("W&B root does not exist: %s", wandb_root)
        return 2

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest = ConversionManifest.load_or_empty(output_dir)

    wb_run = _init_wandb_run(args)
    t_start = time.time()

    records, dstats = discover_runs(
        wandb_root=wandb_root,
        group=args.group,
        project_root=project_root,
        categories=categories,
        include_regex=include_re,
        exclude_regex=exclude_re,
    )

    _logger.info(
        "Discovery: scanned=%d eligible=%d not_finished=%d category_miss=%d "
        "missing_config=%d missing_dir=%d missing_ckpt=%d unknown_variant=%d",
        dstats.scanned, dstats.eligible, dstats.not_finished,
        dstats.category_mismatch, dstats.missing_config,
        dstats.missing_ondisk_dir, dstats.missing_best_ckpt,
        dstats.unknown_variant,
    )

    converted = 0
    skipped_cached = 0
    failed = 0
    total_tensors = 0
    emitted_jobs_for: List[Tuple[RunRecord, Path]] = []
    # Rows for the per-checkpoint W&B table (populated regardless of whether
    # wb_run is live — it's cheap and lets us also dump it to logs).
    wb_table_rows: List[List[Any]] = []

    try:
        for record in records:
            if not args.force and manifest.already_done(record.run_id):
                cached = manifest.entries[record.run_id]
                _logger.info(
                    "[%s] skip: already converted -> %s (use --force to redo)",
                    record.run_id, cached.get("output_path"),
                )
                skipped_cached += 1
                wb_table_rows.append([
                    record.run_id, record.variant, record.category,
                    record.experiment, record.seed,
                    cached.get("num_tensors", ""),
                    cached.get("output_sha256_short", ""),
                    str(cached.get("output_path", "")),
                    "skipped_cached", "",
                ])
                continue

            entry = convert_one(
                record,
                output_dir=output_dir,
                use_ema=args.use_ema,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                # Don't mutate the manifest on dry-run; we still want to report
                # what would have happened.
                continue

            manifest.record(record.run_id, entry)
            manifest.save()

            if entry["status"] == "ok":
                converted += 1
                total_tensors += int(entry.get("num_tensors", 0) or 0)
                if args.emit_jobs_yaml:
                    emitted_jobs_for.append((record, Path(entry["output_path"])))
            else:
                failed += 1

            wb_table_rows.append([
                record.run_id, record.variant, record.category,
                record.experiment, record.seed,
                entry.get("num_tensors", ""),
                entry.get("output_sha256_short", ""),
                str(entry.get("output_path", "")),
                entry.get("status", ""),
                entry.get("error", ""),
            ])

        # Emit jobs.yaml entries at the end so we write the file exactly once
        # per batch, not once per checkpoint (keeps YAML job order stable).
        if not args.dry_run and args.emit_jobs_yaml and emitted_jobs_for:
            jobs_yaml = Path(args.emit_jobs_yaml).expanduser().resolve()
            for record, out_path in emitted_jobs_for:
                _maybe_emit_job(
                    record=record,
                    output_path=out_path,
                    jobs_yaml=jobs_yaml,
                    seeds=list(args.emit_seeds),
                    output_root=Path(args.emit_output_root),
                    gpus_per_job=args.emit_gpus_per_job,
                )

        duration = time.time() - t_start

        _logger.info("=" * 60)
        _logger.info(
            "Summary: eligible=%d converted=%d skipped_cached=%d failed=%d "
            "(manifest: %s)",
            len(records), converted, skipped_cached, failed,
            manifest.path if not args.dry_run else "(dry-run, not written)",
        )

        # ---- Push batch metrics to W&B (Option A) ------------------------
        if wb_run is not None:
            try:
                import wandb  # noqa: WPS433
                summary = {
                    "discovery/scanned": dstats.scanned,
                    "discovery/eligible": dstats.eligible,
                    "discovery/not_finished": dstats.not_finished,
                    "discovery/category_mismatch": dstats.category_mismatch,
                    "discovery/missing_config": dstats.missing_config,
                    "discovery/missing_dir": dstats.missing_ondisk_dir,
                    "discovery/missing_ckpt": dstats.missing_best_ckpt,
                    "discovery/unknown_variant": dstats.unknown_variant,
                    "convert/converted": converted,
                    "convert/skipped_cached": skipped_cached,
                    "convert/failed": failed,
                    "convert/total_tensors": total_tensors,
                    "convert/duration_seconds": duration,
                }
                wandb.log(summary)
                # Also mirror into run.summary so they show up as top-level
                # columns in the W&B runs table without needing a chart.
                for k, v in summary.items():
                    wb_run.summary[k] = v

                table = wandb.Table(
                    columns=[
                        "run_id", "variant", "category", "experiment",
                        "seed", "num_tensors", "sha256", "output_path",
                        "status", "error",
                    ],
                    data=wb_table_rows,
                )
                wandb.log({"checkpoints": table})
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Failed to push metrics to W&B: %s", exc)

        # Non-zero exit if nothing got done and there were failures, useful
        # for scheduled runs / cron.
        if failed and converted == 0:
            return 1
        return 0
    finally:
        if wb_run is not None:
            try:
                import wandb  # noqa: WPS433
                wandb.finish()
            except Exception as exc:  # noqa: BLE001
                _logger.debug("wandb.finish() failed: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
