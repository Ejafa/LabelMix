"""CLI: incrementally download W&B runs to ``evaluation/data/raw/wandb/``.

Typical invocations::

    # Pull every run tagged ``in1k`` from ``my-team/labelmix``:
    python -m evaluation.scripts.sync_wandb \\
        --entity my-team --project labelmix --tags in1k

    # Only finished runs from a specific W&B run group:
    python -m evaluation.scripts.sync_wandb --entity labelmix_gpu_cluster --project labelmix --group openmixup-in1k --state finished

    # Also pull every file (checkpoints / logs / args.yaml), force-refresh:
    python -m evaluation.scripts.sync_wandb --entity my-team --project labelmix \\
        --include-files --force

The second invocation onwards re-uses a local manifest at
``data/raw/wandb/_manifest.json`` and skips runs whose ``heartbeatAt`` /
``updatedAt`` has not advanced, so routine syncs are near-instant.
"""
from __future__ import annotations

import argparse
import os

from ..common import WANDB_RAW_DIR, ensure_dirs, setup_logging
from ..wandb_sync import sync_project, DEFAULT_ENTITY, DEFAULT_PROJECT, DEFAULT_GROUP


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download W&B runs with an incremental cache.")

    # --- Target project -----------------------------------------------------
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
                   help=f"W&B entity/team. Defaults to $WANDB_ENTITY or '{DEFAULT_ENTITY}'.")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
                   help=f"W&B project. Defaults to $WANDB_PROJECT or '{DEFAULT_PROJECT}'.")

    # --- Filters ------------------------------------------------------------
    p.add_argument("--tags", nargs="*", default=None,
                   help="Only runs that have ALL of these tags.")
    p.add_argument("--any-tags", nargs="*", default=None,
                   help="Only runs that have ANY of these tags.")
    p.add_argument("--group", default=DEFAULT_GROUP,
                   help=f"Exact match on W&B run group. Defaults to '{DEFAULT_GROUP}'. "
                        f"Pass --group '' to disable group filtering.")
    p.add_argument("--state", nargs="*", default=None,
                   choices=["running", "finished", "failed", "crashed", "preempted"],
                   help="Restrict to these run states.")
    p.add_argument("--name-regex", default=None,
                   help="Regex applied to run.display_name.")
    p.add_argument("--created-after", default=None,
                   help="ISO-8601 UTC timestamp lower bound on createdAt.")
    p.add_argument("--created-before", default=None,
                   help="ISO-8601 UTC timestamp upper bound on createdAt.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many matched runs (debugging).")

    # --- Output -------------------------------------------------------------
    p.add_argument("--root", default=str(WANDB_RAW_DIR),
                   help=f"Destination directory (default: {WANDB_RAW_DIR}).")
    p.add_argument("--include-files", action="store_true",
                   help="Also download run.files() (checkpoints, configs, logs).")

    # --- Cache control ------------------------------------------------------
    p.add_argument("--force", action="store_true",
                   help="Bypass the local manifest and re-download everything matched.")

    # --- Transport ----------------------------------------------------------
    p.add_argument("--base-url", default=os.environ.get("WANDB_BASE_URL"),
                   help="Override W&B server URL (for self-hosted W&B).")
    p.add_argument("--api-key", default=None,
                   help="Override WANDB_API_KEY (prefer leaving this unset).")
    p.add_argument("--bypass-proxy", action="store_true",
                   help="Clear http(s)_proxy env vars for the sync call. "
                        "Useful when connecting to a LAN-hosted W&B through a "
                        "corporate HTTP proxy that mis-routes local requests.")

    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    ensure_dirs()

    sync_project(
        entity=args.entity,
        project=args.project,
        root=args.root,
        tags=args.tags,
        any_tags=args.any_tags,
        group=args.group or None,
        state=args.state,
        name_regex=args.name_regex,
        created_after=args.created_after,
        created_before=args.created_before,
        limit=args.limit,
        include_files=args.include_files,
        force=args.force,
        bypass_proxy=args.bypass_proxy,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
