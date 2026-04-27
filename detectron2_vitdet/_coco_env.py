"""Auto-configure ``DETECTRON2_DATASETS`` for LabelMix ViTDet entry points.

Detectron2 discovers built-in datasets (including COCO) at *import time* via
the ``DETECTRON2_DATASETS`` environment variable, which must point at the
*parent* directory of a ``coco/`` subdirectory. Forgetting to ``export`` it is
the single most common cause of ``FileNotFoundError`` during dataset
registration in this repo.

This helper centralises the lookup logic so every entry point
(``train_net.py``, ``eval_all.py``, ``generate_vitdet_jobs.py``) behaves
identically:

1. If ``DETECTRON2_DATASETS`` is already set by the caller, keep it.
2. Else pick the first candidate whose ``<root>/coco`` exists, in this order:
     - ``/dev/shm``              (RAM copy from ``copy_data_to_ram.py``)
     - ``<repo>/detectron2_vitdet/datasets``   (on-disk default used by
       ``download_coco.sh``)
3. Export the chosen value into ``os.environ`` *before* detectron2 is
   imported, so child processes spawned later inherit it automatically.

The function is idempotent and safe to call from any entry point. It never
raises when no candidate is found; it only emits a warning, because some
tooling (e.g. ``generate_vitdet_jobs.py``) legitimately runs without needing
COCO to be resolvable locally.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "ensure_detectron2_datasets",
    "resolve_detectron2_datasets_root",
]

_ENV_VAR = "DETECTRON2_DATASETS"

# Module file lives at <repo>/detectron2_vitdet/_coco_env.py
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent


def _candidate_roots() -> List[Path]:
    """Parent directories that might contain a ``coco/`` subdir.

    Order matters: earlier candidates win. ``/dev/shm`` is preferred when a
    RAM copy exists because it makes I/O roughly two orders of magnitude
    faster than the shared filesystem.
    """
    return [
        Path("/dev/shm"),
        _THIS_DIR / "datasets",   # <repo>/detectron2_vitdet/datasets
        _REPO_ROOT / "datasets",  # <repo>/datasets (fallback)
    ]


def resolve_detectron2_datasets_root(
    *,
    require_coco: bool = True,
) -> Optional[Path]:
    """Return the directory that should be used as ``DETECTRON2_DATASETS``.

    Parameters
    ----------
    require_coco:
        When ``True`` (default), a candidate is only accepted if it contains a
        ``coco/`` subdirectory. Set to ``False`` for tools that only need the
        root to exist (e.g. when generating a jobs file from a login node
        where the RAM copy has not been made yet).
    """
    existing = os.environ.get(_ENV_VAR)
    if existing:
        return Path(existing)

    for root in _candidate_roots():
        if not root.is_dir():
            continue
        if require_coco and not (root / "coco").is_dir():
            continue
        return root

    return None


def ensure_detectron2_datasets(
    *,
    require_coco: bool = True,
    verbose: bool = True,
) -> Tuple[Optional[str], str]:
    """Idempotently set ``DETECTRON2_DATASETS`` in ``os.environ``.

    Returns a ``(value, source)`` tuple where ``source`` is one of
    ``"preset"``, ``"auto"`` or ``"unset"`` so callers can log what happened.

    This is safe to call multiple times and from any entry point. It must be
    called *before* ``import detectron2`` triggers dataset registration,
    because detectron2 reads the env var exactly once at import time.
    """
    preset = os.environ.get(_ENV_VAR)
    if preset:
        if verbose:
            _log(f"{_ENV_VAR} already set by caller → {preset}")
        return preset, "preset"

    root = resolve_detectron2_datasets_root(require_coco=require_coco)
    if root is None:
        if verbose:
            searched = ", ".join(str(p) for p in _candidate_roots())
            warnings.warn(
                f"{_ENV_VAR} is not set and no 'coco/' directory was found "
                f"under any of: {searched}. Detectron2 dataset registration "
                f"will fail unless you export {_ENV_VAR} manually or run "
                f"copy_data_to_ram.py --dataset coco.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None, "unset"

    value = str(root)
    os.environ[_ENV_VAR] = value
    if verbose:
        _log(f"{_ENV_VAR} auto-set to {value} (coco → {root / 'coco'})")
    return value, "auto"


def _log(msg: str) -> None:
    # A plain print to stderr is deliberate: we want the message to appear
    # before detectron2's own logger is configured.
    print(f"[coco_env] {msg}", file=sys.stderr)
