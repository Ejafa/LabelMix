"""Materialize filtered W&B runs to disk with an incremental cache.

Public entry point: :func:`sync_project`.

Design summary:
    1. Query :class:`WandbClient` with a server-side filter.
    2. Load the local manifest.
    3. For each matching run, decide :meth:`Manifest.needs_update`.
    4. Download ``config``, ``summary``, ``history`` (parquet preferred,
       CSV fallback) and optionally ``run.files()``.  Writes happen into
       ``<run_dir>.tmp`` and are atomically renamed on success so a
       Ctrl-C cannot leave half-written artifacts that look complete.
    5. Persist the manifest once at the end (also atomic).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .cache import CacheEntry, Manifest
from .client import WandbClient, no_proxy_env
from .config import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    DEFAULT_GROUP,
    DEFAULT_INCLUDE_FILES,
    DEFAULT_BYPASS_PROXY,
    DEFAULT_FORCE,
    DEFAULT_LIMIT,
)
from .filters import build_filters
from .schema import (
    CONFIG_FILE,
    FILES_SUBDIR,
    HISTORY_CSV,
    HISTORY_PARQUET,
    METADATA_FILE,
    SUMMARY_FILE,
    run_dir as schema_run_dir,
)

_logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Summary of one :func:`sync_project` invocation."""

    scanned: int = 0
    downloaded: int = 0
    skipped_cached: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _server_updated_at(run: Any) -> str:
    """Best-effort extraction of the server-side last-modified timestamp.

    Prefer ``heartbeatAt`` (updated every ~30s for live runs, frozen once
    the run terminates) and fall back to ``updatedAt`` then ``createdAt``.
    """
    attrs = getattr(run, "_attrs", {}) or {}
    for key in ("heartbeatAt", "updatedAt", "createdAt"):
        v = attrs.get(key)
        if v:
            return str(v)
    return ""


def _write_history(run: Any, out_parquet: Path, out_csv: Path) -> None:
    """Write the full sampled history to parquet (or CSV as fallback)."""
    history_df = run.history(pandas=True, samples=None)
    try:
        history_df.to_parquet(out_parquet, index=False)
    except (ImportError, ValueError) as exc:
        _logger.debug("Parquet write failed (%s); falling back to CSV.", exc)
        history_df.to_csv(out_csv, index=False)


def _atomic_swap(src: Path, dst: Path) -> None:
    """Replace ``dst`` with ``src`` atomically on POSIX."""
    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

# Top-level keys that W&B considers "internal" (runtime, code, system info, ...)
# and that we therefore keep in a separate ``_wandb`` section of the YAML so
# the user-facing hyperparameters remain trivially readable.
_WANDB_INTERNAL_KEY_PREFIX = "_"


def _unwrap_value(v: Any) -> Any:
    """Strip the ``{"value": ..., "desc": ...}`` wrapper W&B uses on the wire.

    W&B stores every config entry as a mapping with at least a ``value`` key
    (and optionally ``desc`` / ``policy``).  The public-API ``Run.config``
    property normally does this for us, but depending on how the ``Run`` was
    instantiated (lazy fetch, cached ``_attrs``, partial load) it can return
    either the already-unwrapped dict or the raw nested form.  This helper
    is idempotent: unwrapped values pass through unchanged.
    """
    if isinstance(v, dict) and "value" in v and set(v.keys()) <= {"value", "desc", "policy"}:
        return v["value"]
    return v


def _unwrap_config_mapping(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Apply :func:`_unwrap_value` to every top-level entry."""
    return {k: _unwrap_value(v) for k, v in raw.items()}


def _extract_config(run: Any) -> Dict[str, Any]:
    """Return the richest possible config dict for ``run``.

    Tries several sources in order and unwraps ``{value, desc}`` wrappers.
    The result always carries two top-level namespaces:

    * every non-internal key verbatim at the top level (hyperparameters), and
    * a single ``_wandb`` key holding W&B's internal config namespace
      (runtime metadata, code-saving settings, etc.) when present.

    The function never raises: on total failure it returns ``{}``.
    """
    # If the Run was returned by a lazy listing, its lightweight fragment does
    # NOT include ``config`` -- forcing a full-data load here guarantees every
    # subsequent accessor sees the populated, unwrapped dicts the SDK builds
    # in ``_load_from_attrs``.  This is idempotent and cheap on non-lazy runs
    # (``load_full_data`` short-circuits when ``_full_data_loaded`` is True).
    if getattr(run, "_lazy", False) and not getattr(run, "_full_data_loaded", False):
        try:
            run.load_full_data()
        except Exception as exc:  # noqa: BLE001
            _logger.debug("[%s] load_full_data failed: %s",
                          getattr(run, "id", "?"), exc)

    sources: List[tuple[str, Dict[str, Any]]] = []

    # 1. ``run.rawconfig`` -- user keys + internal ``_wandb*`` keys, already
    #    unwrapped by the SDK when ``_load_from_attrs`` has been invoked.
    rawconfig = getattr(run, "rawconfig", None)
    if isinstance(rawconfig, dict) and rawconfig:
        sources.append(("rawconfig", rawconfig))

    # 2. ``run.config`` -- user-only view (no internal keys).
    try:
        cfg = run.config
        if hasattr(cfg, "items"):
            cfg_dict = dict(cfg.items())
            if cfg_dict:
                sources.append(("config", cfg_dict))
    except Exception as exc:  # noqa: BLE001
        _logger.debug("[%s] run.config unavailable: %s", getattr(run, "id", "?"), exc)

    # 3. ``run.json_config`` -- authoritative raw JSON straight from the server
    #    (string form, still wrapped).  This is the safety net for runs where
    #    the SDK's convenience properties come back empty.
    json_cfg = getattr(run, "json_config", None)
    if isinstance(json_cfg, str) and json_cfg.strip():
        try:
            parsed = json.loads(json_cfg)
            if isinstance(parsed, dict) and parsed:
                sources.append(("json_config", parsed))
        except (ValueError, TypeError) as exc:
            _logger.debug("[%s] json_config parse failed: %s",
                          getattr(run, "id", "?"), exc)

    # 4. ``run._attrs['config']`` -- last resort.
    attrs_cfg = (getattr(run, "_attrs", {}) or {}).get("config")
    if isinstance(attrs_cfg, str) and attrs_cfg.strip():
        try:
            attrs_cfg = json.loads(attrs_cfg)
        except (ValueError, TypeError):
            attrs_cfg = None
    if isinstance(attrs_cfg, dict) and attrs_cfg:
        sources.append(("_attrs.config", attrs_cfg))

    if not sources:
        _logger.warning("[%s] no config data found in any known source",
                        getattr(run, "id", "?"))
        return {}

    # Prefer the first source that has at least one non-internal key; that way
    # a stray empty ``rawconfig`` on a pre-loaded run won't mask a populated
    # ``json_config``.
    chosen_label, chosen = sources[0]
    for label, candidate in sources:
        non_internal = [k for k in candidate if not str(k).startswith(_WANDB_INTERNAL_KEY_PREFIX)]
        if non_internal:
            chosen_label, chosen = label, candidate
            break

    # Unwrap + split into user-facing keys and the ``_wandb`` namespace.
    unwrapped = _unwrap_config_mapping(chosen)
    user_keys = {k: v for k, v in unwrapped.items()
                 if not str(k).startswith(_WANDB_INTERNAL_KEY_PREFIX)}
    internal_keys = {k: v for k, v in unwrapped.items()
                     if str(k).startswith(_WANDB_INTERNAL_KEY_PREFIX)}

    out: Dict[str, Any] = dict(user_keys)
    if internal_keys:
        # Collapse all internal ``_wandb*`` entries under a single key so the
        # YAML stays readable; the ``_wandb`` sub-dict itself is commonly the
        # only one present and holds the rich metadata (python version,
        # executable, git commit, hostname, cli_version, ...).
        out["_wandb"] = internal_keys

    _logger.debug(
        "[%s] config extracted from %s: %d user keys, %d internal keys",
        getattr(run, "id", "?"), chosen_label,
        len(user_keys), len(internal_keys),
    )
    return out


def _download_one(
    run: Any,
    dest: Path,
    *,
    include_files: bool,
) -> None:
    """Materialize a single run into ``dest`` (directory is overwritten)."""
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=False)

    # 1. Metadata (stable subset of _attrs we actually use downstream).
    attrs = getattr(run, "_attrs", {}) or {}
    metadata = {
        "id": run.id,
        "name": getattr(run, "name", None),
        "display_name": getattr(run, "display_name", None),
        "state": getattr(run, "state", None),
        "tags": list(getattr(run, "tags", []) or []),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "path": "/".join(getattr(run, "path", []) or []),
        "created_at": attrs.get("createdAt"),
        "updated_at": attrs.get("updatedAt"),
        "heartbeat_at": attrs.get("heartbeatAt"),
        "url": getattr(run, "url", None),
    }
    with open(tmp / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    # 2. Config.  Pull from the richest available source (rawconfig / config /
    #    json_config / _attrs) and unwrap the ``{value, desc}`` nesting W&B
    #    uses on the wire.  We keep internal ``_wandb*`` entries under a
    #    single ``_wandb`` key so the YAML still contains them (git commit,
    #    python version, cli_version, ...), which is invaluable for
    #    reproducing a run later on.
    config_dict = _extract_config(run)
    with open(tmp / CONFIG_FILE, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=True, default_flow_style=False)

    # 3. Summary (final metric values).
    summary_dict = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
    with open(tmp / SUMMARY_FILE, "w") as f:
        json.dump(summary_dict, f, indent=2, sort_keys=True, default=str)

    # 4. Full history.
    _write_history(run, tmp / HISTORY_PARQUET, tmp / HISTORY_CSV)

    # 5. Files (checkpoints, configs, stdout, ...), opt-in.
    if include_files:
        files_dir = tmp / FILES_SUBDIR
        files_dir.mkdir(exist_ok=True)
        for f in run.files():
            try:
                f.download(root=str(files_dir), replace=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("[%s] file download failed for %s: %s",
                                run.id, getattr(f, "name", "?"), exc)

    _atomic_swap(tmp, dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sync_project(
    *,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    root: Path,
    tags: Optional[Iterable[str]] = None,
    any_tags: Optional[Iterable[str]] = None,
    group: Optional[str] = DEFAULT_GROUP,
    state: Optional[Iterable[str]] = None,
    name_regex: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    include_files: bool = DEFAULT_INCLUDE_FILES,
    force: bool = DEFAULT_FORCE,
    limit: Optional[int] = DEFAULT_LIMIT,
    bypass_proxy: bool = DEFAULT_BYPASS_PROXY,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> SyncStats:
    """Download matching runs from ``<entity>/<project>`` into ``<root>``.

    The local manifest at ``<root>/_manifest.json`` is consulted first, so
    subsequent invocations only fetch runs whose server-side
    ``heartbeatAt`` / ``updatedAt`` has advanced.  Pass ``force=True`` to
    bypass the cache.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    _logger.info("Initializing W&B client for entity %s, project %s", entity, project)
    client = WandbClient(base_url=base_url, api_key=api_key)
    filters = build_filters(
        tags=tags,
        any_tags=any_tags,
        group=group,
        state=state,
        name_regex=name_regex,
        created_after=created_after,
        created_before=created_before,
    )
    _logger.info(
        "Syncing %s/%s into %s (filters=%s, include_files=%s, force=%s)",
        entity, project, root, filters, include_files, force,
    )

    manifest = Manifest.load_or_empty(root)
    _logger.debug("Loaded manifest with %d cached runs", len(manifest.entries))
    stats = SyncStats()

    proxy_ctx = no_proxy_env() if bypass_proxy else _NullContext()
    with proxy_ctx:
        _logger.info("Querying W&B API for runs matching filters")
        runs = client.runs(path=f"{entity}/{project}", filters=filters)
        # NOTE: do NOT call ``len(list(runs))`` here -- the W&B ``Runs``
        # iterator is paginated and single-pass, so materializing it would
        # (a) block for a long time on large projects and (b) exhaust the
        # iterator before the download loop below ever runs.

        for i, run in enumerate(runs):
            if limit is not None and i >= limit:
                _logger.info("Reached limit of %d runs, stopping sync", limit)
                break

            run_group = getattr(run, "group", None)

            # Client-side guard: the server-side filter is occasionally
            # lenient (e.g. ``group`` stored as a tag on legacy runs).
            # When the caller specified a group, enforce an exact match
            # here so we never materialize runs from a different group.
            if group is not None and run_group != group:
                _logger.debug(
                    "[%s] skipping: group=%r does not match requested %r",
                    run.id, run_group, group,
                )
                continue

            stats.scanned += 1

            server_ts = _server_updated_at(run)
            if not force and not manifest.needs_update(
                run.id, server_ts, want_files=include_files,
            ):
                stats.skipped_cached += 1
                _logger.debug("[%s] cache HIT (updated_at=%s)", run.id, server_ts)
                continue

            _logger.info(
                "[%s] Processing run: %s (group=%s, state=%s)",
                run.id, getattr(run, "display_name", "unnamed"),
                run_group, getattr(run, "state", "unknown"),
            )
            dest = schema_run_dir(root, run.id, group=run_group)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                _download_one(run, dest, include_files=include_files)
            except Exception as exc:  # noqa: BLE001
                _logger.exception("[%s] download failed: %s", run.id, exc)
                stats.failed += 1
                continue

            manifest.upsert(CacheEntry(
                run_id=run.id,
                name=getattr(run, "display_name", None) or run.id,
                state=str(getattr(run, "state", "") or ""),
                updated_at=server_ts,
                saved_at=_now_iso(),
                path=str(dest.relative_to(root)),
                tags=list(getattr(run, "tags", []) or []),
                group=run_group,
                include_files=include_files,
            ))
            stats.downloaded += 1
            # Persist after every run so a crash doesn't lose progress.
            manifest.save()
            _logger.info(
                "[%s] saved (%d/%d) name=%r state=%s",
                run.id, stats.downloaded, stats.scanned,
                getattr(run, "display_name", ""), getattr(run, "state", ""),
            )

    manifest.save()
    _logger.info(
        "Sync complete: scanned=%d downloaded=%d skipped_cached=%d failed=%d",
        stats.scanned, stats.downloaded, stats.skipped_cached, stats.failed,
    )
    return stats


class _NullContext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False
