"""Deprecated CLI entry point — use ``python -m evaluation.scripts.run_eval``.

Kept as a thin forwarder so existing shell scripts continue to work.
"""
from __future__ import annotations

from warnings import warn as _warn

from .scripts.run_eval import main

_warn(
    "`python -m evaluation.run_eval` is deprecated; use "
    "`python -m evaluation.scripts.run_eval` instead.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    raise SystemExit(main())
