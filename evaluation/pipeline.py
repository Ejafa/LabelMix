"""Deprecated flat module — kept as a thin back-compat shim.

The implementation moved to :mod:`evaluation.offline_eval`.  Public names are
re-exported verbatim so legacy callers keep working.  Update imports to::

    from evaluation.offline_eval import evaluate_mapping, evaluate_run
    from evaluation.config       import EvalConfig
    from evaluation.types        import EvalResult
"""
from __future__ import annotations

from warnings import warn as _warn

from .config import DEFAULT_ARGS_COLUMNS, EvalConfig
from .offline_eval import evaluate_mapping, evaluate_run
from .offline_eval.io import load_args_yaml as _load_args_yaml
from .offline_eval.io import write_results_csv as _write_csv
from .types import EvalResult

_warn(
    "`evaluation.pipeline` is deprecated; import from "
    "`evaluation.offline_eval` / `evaluation.config` / `evaluation.types`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_ARGS_COLUMNS",
    "EvalConfig",
    "EvalResult",
    "evaluate_mapping",
    "evaluate_run",
]
