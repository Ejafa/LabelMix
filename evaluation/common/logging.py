"""Uniform logger setup used by every script in the evaluation package."""
from __future__ import annotations

import logging


_DEFAULT_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger for CLI scripts.

    Idempotent: repeated calls replace the existing handlers so that scripts
    imported from notebooks do not duplicate log lines.
    """
    if isinstance(level, str):
        level = level.upper()
    logging.basicConfig(level=level, format=_DEFAULT_FMT, force=True)
