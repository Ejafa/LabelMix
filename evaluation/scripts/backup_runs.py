"""CLI: back up the key files of finished W&B runs to a local directory.

The script walks every locally-materialized W&B run under
``evaluation/data/raw/wandb/<group>/<run_id>/``, looks up the on-disk
training-run directory via the ``experiment`` / ``output`` fields in the
downloaded ``config.yaml``, and copies the following files together into a
single per-run backup directory:

* ``args.yaml``
* ``run_status.yaml``
* ``summary.csv``
* ``model_best.pth.tar``

Only runs whose W&B ``metadata.json`` reports ``state == "finished"`` are
copied.  A small manifest at ``<backup_root>/_manifest.json`` keeps track of
previously-backed-up runs so re-running the script is cheap and idempotent —
each run is copied at most once unless ``--force`` is given.

The backup layout mirrors the W&B group/experiment split::

    <backup_root>/
      <group>/                   # e.g. ``in1k``
        <experiment_name>/       # e.g. ``vit-little__in1k__img256__...``
          args.yaml
          run_status.yaml
          summary.csv
          model_best.pth.tar

Typical invocations::

    # Default: back up every finished run into ``../backup`` (relative to
    # the project root, i.e. the repository containing ``evaluation/``).
    # Runs whose ``model_best.pth.tar`` is missing are *not* recorded in
    # the manifest, so the next pass will retry them automatically.
    python -m evaluation.scripts.backup_runs

    # Archive non-checkpoint artifacts (args.yaml, summary.csv, ...) even
    # when the checkpoint file is absent -- useful for salvaging crashed
    # sweeps.
    python -m evaluation.scripts.backup_runs --allow-missing-best-checkpoint

    # Custom destination + force re-copy of everything.
    python -m evaluation.scripts.backup_runs \\
        --backup-root /data/labelmix_backup --force
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ..common import WANDB_RAW_DIR, setup_logging

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Files we try to copy from the on-disk run directory into the backup.
BACKUP_FILES: tuple[str, ...] = (
    "args.yaml",
    "run_status.yaml",
    "summary.csv",
    "model_best.pth.tar",
)

#: Wandb run states that we consider "finished" and therefore backup-worthy.
FINISHED_STATES: frozenset[str] = frozenset({"finished"})

#: Name of the manifest we drop next to the backup root.
MANIFEST_NAME: str = "_manifest.json"

#: Project root = ``evaluation/`` package parent directory.
#: (``scripts`` lives at ``evaluation/scripts/``.)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default backup destination: ``<project_root>/../backup``.
DEFAULT_BACKUP_ROOT: Path = (PROJECT_ROOT.parent / "backup").resolve()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class BackupManifest:
    """Tracks which runs have already been backed up.

    Stored at ``<backup_root>/_manifest.json`` as a mapping from W&B run id
    to a small per-run dict.  Presence of a run id is what makes
    :meth:`already_done` short-circuit on the next invocation.
    """

    path: Path
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_or_empty(cls, backup_root: Path) -> "BackupManifest":
        p = backup_root / MANIFEST_NAME
        if not p.exists():
            return cls(path=p)
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("manifest is not a JSON object")
            return cls(path=p, entries=data)
        except (OSError, ValueError) as exc:
            _logger.warning("Could not read manifest %s (%s); starting fresh.", p, exc)
            return cls(path=p)

    def already_done(self, run_id: str) -> bool:
        return run_id in self.entries

    def record(self, run_id: str, entry: Dict[str, Any]) -> None:
        self.entries[run_id] = entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.entries, f, indent=2, sort_keys=True, default=str)
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file as a dict, returning ``{}`` on any failure."""
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        _logger.debug("YAML load failed for %s: %s", path, exc)
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file as a dict, returning ``{}`` on any failure."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        _logger.debug("JSON load failed for %s: %s", path, exc)
        return {}


def _iter_run_dirs(wandb_root: Path) -> Iterable[Path]:
    """Yield every run directory directly under ``<wandb_root>/<group>/``.

    A "run directory" is any leaf directory containing a ``metadata.json``
    file (that's the artifact the sync pipeline always writes).
    """
    for group_dir in sorted(p for p in wandb_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if (run_dir / "metadata.json").exists():
                yield run_dir


def _candidate_run_dirs(config: Dict[str, Any], project_root: Path) -> List[Path]:
    """Return the plausible on-disk run directories implied by ``config``.

    The timm training script writes ``<output>/<experiment>/`` where both
    values are stored verbatim in the W&B config.  We accept ``output`` as
    either an absolute path or a path relative to the project root.
    """
    experiment = config.get("experiment")
    output = config.get("output")
    if not experiment:
        return []

    candidates: List[Path] = []
    if output:
        out = Path(str(output))
        if out.is_absolute():
            candidates.append(out / str(experiment))
        else:
            # Try project-root-relative first (how timm writes on this repo),
            # then CWD-relative as a secondary fallback.
            candidates.append((project_root / out / str(experiment)).resolve())
            candidates.append((Path.cwd() / out / str(experiment)).resolve())

    # Historical runs lived under ``in1k-main/<experiment>``; keep this as a
    # last-ditch fallback so pre-``output_runs`` runs still get backed up.
    candidates.append((project_root / "in1k-main" / str(experiment)).resolve())

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _resolve_run_dir(config: Dict[str, Any], project_root: Path) -> Optional[Path]:
    """Pick the first existing candidate run directory, if any."""
    for c in _candidate_run_dirs(config, project_root):
        if c.is_dir():
            return c
    return None


def _copy_run_files(
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool,
) -> tuple[List[str], List[str]]:
    """Copy the backup files from ``src_dir`` into ``dst_dir``.

    Returns ``(copied, missing)`` -- two lists of filenames, classifying
    each entry of :data:`BACKUP_FILES` as either successfully copied (or
    would-be-copied in ``dry_run`` mode) or missing at the source.
    """
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    missing: List[str] = []
    for name in BACKUP_FILES:
        src = src_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        if dry_run:
            copied.append(name)
            continue
        dst = dst_dir / name
        # ``copy2`` preserves timestamps which makes ``rsync``-style
        # diffing against a mirror trivial.
        shutil.copy2(src, dst)
        copied.append(name)
    return copied, missing


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

@dataclass
class BackupStats:
    scanned: int = 0
    backed_up: int = 0
    skipped_not_finished: int = 0
    skipped_cached: int = 0
    skipped_no_run_dir: int = 0
    skipped_no_best_checkpoint: int = 0
    skipped_incomplete: int = 0
    failed: int = 0


def backup_runs(
    *,
    wandb_root: Path,
    backup_root: Path,
    project_root: Path,
    force: bool = False,
    dry_run: bool = False,
    require_best_checkpoint: bool = True,
) -> BackupStats:
    """Walk ``wandb_root`` and back up every finished run into ``backup_root``.

    Args:
        wandb_root: Root of the W&B sync tree (``evaluation/data/raw/wandb``).
        backup_root: Destination root; runs land in
            ``<backup_root>/<group>/<experiment>/``.
        project_root: Repository root, used to resolve relative ``output``
            paths stored in the run config.
        force: If True, re-copy runs even if the manifest already lists them.
        dry_run: If True, only log what would happen, don't touch the disk.
        require_best_checkpoint: If True (the default), *refuse* to record a
            backup entry when ``model_best.pth.tar`` is missing at the
            resolved source directory. This prevents the silent-drift
            failure mode where the manifest would otherwise cache a
            half-populated backup and short-circuit all future retries.
            Set to False via ``--allow-missing-best-checkpoint`` if you
            genuinely want to archive non-checkpoint artifacts (e.g. logs
            from a crashed sweep).
    """
    wandb_root = wandb_root.resolve()
    backup_root = backup_root.resolve()
    project_root = project_root.resolve()

    if not wandb_root.is_dir():
        _logger.error("W&B root %s does not exist; nothing to do.", wandb_root)
        return BackupStats()

    _logger.info(
        "Backing up runs: wandb_root=%s -> backup_root=%s (force=%s, dry_run=%s)",
        wandb_root, backup_root, force, dry_run,
    )

    if not dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)

    manifest = BackupManifest.load_or_empty(backup_root)
    stats = BackupStats()

    for run_dir in _iter_run_dirs(wandb_root):
        stats.scanned += 1
        run_id = run_dir.name

        metadata = _load_json(run_dir / "metadata.json")
        state = str(metadata.get("state") or "").lower()
        group = metadata.get("group") or run_dir.parent.name

        if state not in FINISHED_STATES:
            _logger.debug("[%s] skip: state=%r not in %s", run_id, state, sorted(FINISHED_STATES))
            stats.skipped_not_finished += 1
            continue

        if not force and manifest.already_done(run_id):
            _logger.debug("[%s] skip: already in manifest", run_id)
            stats.skipped_cached += 1
            continue

        config = _load_yaml(run_dir / "config.yaml")
        experiment = str(config.get("experiment") or "").strip()
        if not experiment:
            _logger.warning("[%s] skip: no 'experiment' in config.yaml", run_id)
            stats.failed += 1
            continue

        src_dir = _resolve_run_dir(config, project_root)
        if src_dir is None:
            tried = _candidate_run_dirs(config, project_root)
            _logger.warning(
                "[%s] skip: no on-disk run directory found for experiment %r (tried: %s)",
                run_id, experiment, [str(p) for p in tried],
            )
            stats.skipped_no_run_dir += 1
            continue

        if require_best_checkpoint and not (src_dir / "model_best.pth.tar").is_file():
            _logger.warning(
                "[%s] skip: no model_best.pth.tar at %s "
                "(require_best_checkpoint=True). Not recording in manifest so "
                "the next pass can retry. Pass --allow-missing-best-checkpoint "
                "to archive this run without its checkpoint.",
                run_id, src_dir / "model_best.pth.tar",
            )
            stats.skipped_no_best_checkpoint += 1
            continue

        dst_dir = backup_root / str(group) / experiment

        try:
            copied, missing = _copy_run_files(src_dir, dst_dir, dry_run=dry_run)
        except OSError as exc:
            _logger.exception("[%s] copy failed: %s", run_id, exc)
            stats.failed += 1
            continue

        # Treat an incomplete best-ckpt copy as a soft failure under strict
        # mode: log it, bump the stat, and *skip the manifest write* so the
        # next invocation will retry (e.g. once the checkpoint is produced
        # or the right --project-root is used).
        is_incomplete = require_best_checkpoint and (
            "model_best.pth.tar" in missing
        )

        _logger.info(
            "[%s] %s %s  (copied=%s, missing=%s, src=%s%s)",
            run_id,
            "DRY-RUN would back up" if dry_run else (
                "incomplete backup (NOT recorded)" if is_incomplete else "backed up"
            ),
            dst_dir,
            copied, missing, src_dir,
            " [will retry next pass]" if is_incomplete else "",
        )

        if is_incomplete:
            stats.skipped_incomplete += 1
            continue

        if not dry_run:
            manifest.record(run_id, {
                "group": group,
                "experiment": experiment,
                "src_dir": str(src_dir),
                "dst_dir": str(dst_dir),
                "copied": copied,
                "missing": missing,
                "state": state,
            })
            # Persist after every run so a crash doesn't lose progress.
            manifest.save()

        stats.backed_up += 1

    _logger.info(
        "Done: scanned=%d backed_up=%d skipped_not_finished=%d skipped_cached=%d "
        "skipped_no_run_dir=%d skipped_no_best_checkpoint=%d "
        "skipped_incomplete=%d failed=%d",
        stats.scanned, stats.backed_up, stats.skipped_not_finished,
        stats.skipped_cached, stats.skipped_no_run_dir,
        stats.skipped_no_best_checkpoint, stats.skipped_incomplete, stats.failed,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Back up key files of finished W&B runs to a local directory.",
    )
    p.add_argument(
        "--wandb-root", default=str(WANDB_RAW_DIR),
        help=f"Root of the W&B sync tree (default: {WANDB_RAW_DIR}).",
    )
    p.add_argument(
        "--backup-root", default=str(DEFAULT_BACKUP_ROOT),
        help=f"Destination directory (default: {DEFAULT_BACKUP_ROOT}).",
    )
    p.add_argument(
        "--project-root", default=str(PROJECT_ROOT),
        help=f"Project root used to resolve relative 'output' paths in "
             f"config.yaml (default: {PROJECT_ROOT}).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore the manifest and re-copy every matched run.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log what would be copied without touching the destination.",
    )
    ckpt_group = p.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--require-best-checkpoint", dest="require_best_checkpoint",
        action="store_true", default=True,
        help="(default) Only record a backup when model_best.pth.tar exists "
             "at the source. Runs whose checkpoint is missing are logged "
             "and left unrecorded so the next pass can retry.",
    )
    ckpt_group.add_argument(
        "--allow-missing-best-checkpoint", dest="require_best_checkpoint",
        action="store_false",
        help="Escape hatch: archive runs even when model_best.pth.tar is "
             "absent (e.g. to keep summary.csv/args.yaml from a crashed "
             "sweep). The manifest will record which files were missing.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    setup_logging(args.log_level)

    stats = backup_runs(
        wandb_root=Path(args.wandb_root),
        backup_root=Path(args.backup_root),
        project_root=Path(args.project_root),
        force=args.force,
        dry_run=args.dry_run,
        require_best_checkpoint=args.require_best_checkpoint,
    )
    # Non-zero exit if every scanned run failed outright — useful for cron.
    if stats.scanned > 0 and stats.failed == stats.scanned:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
