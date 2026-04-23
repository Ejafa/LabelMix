"""Local manifest for incremental W&B syncs.

The W&B public API does not ship a cross-invocation "only fetch new runs"
cache, but every run exposes a server-side ``updated_at`` timestamp.  We
persist ``{run_id: {updated_at, ...}}`` to a JSON file next to the run
directories and skip runs whose ``updated_at`` has not advanced since the
last successful sync.

The manifest is the single source of truth for "what have we already
downloaded".  It is written atomically (tmp + rename) so an interrupted
sync cannot leave a corrupt manifest behind.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import manifest_path

_logger = logging.getLogger(__name__)

#: Bump when the manifest schema changes incompatibly.
MANIFEST_VERSION = 1


@dataclass
class CacheEntry:
    """One row of the manifest, keyed by ``run_id``."""

    run_id: str
    name: str
    state: str
    updated_at: str          # ISO-8601 UTC timestamp from W&B
    saved_at: str            # ISO-8601 UTC timestamp when we wrote the dir
    path: str                # relative path from manifest dir to run dir
    tags: list = field(default_factory=list)
    group: Optional[str] = None
    include_files: bool = False


@dataclass
class Manifest:
    """Wrapper around ``{run_id: CacheEntry}`` with atomic IO."""

    root: Path
    version: int = MANIFEST_VERSION
    entries: Dict[str, CacheEntry] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @classmethod
    def load_or_empty(cls, root: Path) -> "Manifest":
        """Load ``manifest.json`` if present; otherwise return an empty one."""
        p = manifest_path(root)
        if not p.is_file():
            return cls(root=Path(root))

        try:
            with open(p, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            _logger.warning(
                "Manifest at %s is unreadable (%s); starting fresh.", p, exc,
            )
            return cls(root=Path(root))

        version = int(raw.get("version", 0))
        if version != MANIFEST_VERSION:
            _logger.warning(
                "Manifest version mismatch (found %s, expected %s); "
                "ignoring existing cache and re-downloading everything.",
                version, MANIFEST_VERSION,
            )
            return cls(root=Path(root))

        entries: Dict[str, CacheEntry] = {}
        for run_id, entry in (raw.get("entries") or {}).items():
            try:
                entries[run_id] = CacheEntry(**entry)
            except TypeError as exc:
                _logger.warning(
                    "Dropping malformed manifest entry for %s: %s", run_id, exc,
                )
        return cls(root=Path(root), version=MANIFEST_VERSION, entries=entries)

    def save(self) -> None:
        """Atomically persist the manifest to ``<root>/_manifest.json``."""
        self.root.mkdir(parents=True, exist_ok=True)
        p = manifest_path(self.root)
        payload: Dict[str, Any] = {
            "version": self.version,
            "entries": {k: asdict(v) for k, v in self.entries.items()},
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="._manifest_", suffix=".tmp", dir=str(self.root),
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, p)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------
    def needs_update(
        self,
        run_id: str,
        server_updated_at: str,
        *,
        want_files: bool = False,
    ) -> bool:
        """Return True when a run should be (re-)downloaded.

        A run is considered fresh iff:
          * it exists in the manifest,
          * its server-side ``updated_at`` matches what we last recorded,
          * and if the caller now requests ``want_files``, we have already
            downloaded them before (``include_files=True`` in the entry).
        """
        entry = self.entries.get(run_id)
        if entry is None:
            return True
        if entry.updated_at != server_updated_at:
            return True
        if want_files and not entry.include_files:
            return True
        # Also re-download if the on-disk directory was removed manually.
        if not (self.root / entry.path).exists():
            return True
        return False

    def upsert(self, entry: CacheEntry) -> None:
        self.entries[entry.run_id] = entry
