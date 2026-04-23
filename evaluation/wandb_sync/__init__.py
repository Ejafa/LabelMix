"""Download W&B run data to ``evaluation/data/raw/wandb/`` with an incremental cache.

Public API::

    from evaluation.wandb_sync import sync_project, SyncStats

CLI::

    python -m evaluation.scripts.sync_wandb \\
        --entity my-team --project labelmix --tags in1k
"""
from .cache import CacheEntry, Manifest
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
from .pull import SyncStats, sync_project

__all__ = [
    "sync_project",
    "SyncStats",
    "Manifest",
    "CacheEntry",
    "build_filters",
    "DEFAULT_ENTITY",
    "DEFAULT_PROJECT",
    "DEFAULT_GROUP",
    "DEFAULT_INCLUDE_FILES",
    "DEFAULT_BYPASS_PROXY",
    "DEFAULT_FORCE",
    "DEFAULT_LIMIT",
]
