"""Build server-side W&B filter dicts from CLI flags.

``wandb.Api().runs(path, filters=...)`` accepts MongoDB-style queries.  Doing
the filtering server-side means we never pull a run just to discard it
client-side, which matters once a project has hundreds of runs.

Supported dimensions (all optional, all AND-combined):

* ``tags``        -- list[str], matches runs with *all* given tags.
* ``any_tags``    -- list[str], matches runs with *any* of the given tags.
* ``group``       -- str, exact match on ``run.group``.
* ``state``       -- list[str] subset of
                     {"running", "finished", "failed", "crashed", "preempted"}.
* ``name_regex``  -- str, matches ``run.display_name``.
* ``created_after`` / ``created_before`` -- ISO-8601 strings.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def build_filters(
    *,
    tags: Optional[Iterable[str]] = None,
    any_tags: Optional[Iterable[str]] = None,
    group: Optional[str] = None,
    state: Optional[Iterable[str]] = None,
    name_regex: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a Mongo-style filter dict for :meth:`wandb.Api.runs`.

    Empty input returns an empty dict, meaning "no server-side filter".
    """
    clauses: list[Dict[str, Any]] = []

    if tags:
        # Runs must have *all* of these tags.
        clauses.append({"tags": {"$all": list(tags)}})
    if any_tags:
        clauses.append({"tags": {"$in": list(any_tags)}})
    if group:
        clauses.append({"group": group})
    if state:
        clauses.append({"state": {"$in": list(state)}})
    if name_regex:
        clauses.append({"display_name": {"$regex": name_regex}})
    if created_after:
        clauses.append({"created_at": {"$gte": created_after}})
    if created_before:
        clauses.append({"created_at": {"$lte": created_before}})
    if extra:
        clauses.append(dict(extra))

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
