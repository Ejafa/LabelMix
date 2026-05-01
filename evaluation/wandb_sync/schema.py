"""On-disk layout for a single materialized W&B run.

Each run is stored under ``<root>/<group>/<run_id>/`` with this exact layout:

    <group>/<run_id>/
    ├── metadata.json      -- id, name, state, tags, group, created_at, updated_at
    ├── config.yaml        -- run.config (serialized from nested dict)
    ├── summary.json       -- run.summary (final metric values)
    ├── history.parquet    -- full step-indexed history (preferred)
    ├── history.csv        -- fallback when pyarrow is unavailable
    └── files/             -- downloaded run files (optional, opt-in)

If a run has no group, it is stored under ``<root>/_ungrouped/<run_id>/``.

When a run comes from a non-root W&B project, the group folder is prefixed
with the project name (e.g. ``<root>/<project>_<group>/<run_id>/``) so
artifacts from different projects never collide under a shared ``--root``.
The root project (see ``DEFAULT_PROJECT`` in ``wandb_sync.config``) keeps
the bare ``<group>/`` layout for backwards compatibility.

``run_id`` is the W&B primary key; ``run.name`` is *not* unique across a
project so we never put it into the directory name.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

HISTORY_PARQUET = "history.parquet"
HISTORY_CSV = "history.csv"
CONFIG_FILE = "config.yaml"
SUMMARY_FILE = "summary.json"
METADATA_FILE = "metadata.json"
FILES_SUBDIR = "files"

#: Name of the project-local manifest used by :mod:`cache`.
MANIFEST_FILE = "_manifest.json"

#: Subfolder for runs that do not belong to any W&B group.
UNGROUPED_DIR = "_ungrouped"


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(name: str) -> str:
    """Filesystem-safe slug for human-readable labels (NOT used as a key)."""
    return _SLUG_RE.sub("_", name).strip("_") or "run"


def run_dir(
    root: Path,
    run_id: str,
    group: Optional[str] = None,
    project_prefix: Optional[str] = None,
) -> Path:
    """Canonical directory for a single run, nested under its W&B group.

    When ``project_prefix`` is given (typically only for non-root projects),
    the group folder is renamed to ``<project_prefix>_<group_slug>`` so runs
    from different projects stay isolated even when sharing a ``--root``.
    """
    group_slug = slug(group) if group else UNGROUPED_DIR
    if project_prefix:
        group_slug = f"{slug(project_prefix)}_{group_slug}"
    return Path(root) / group_slug / run_id


def manifest_path(root: Path) -> Path:
    """Canonical path of the project-local manifest."""
    return Path(root) / MANIFEST_FILE
