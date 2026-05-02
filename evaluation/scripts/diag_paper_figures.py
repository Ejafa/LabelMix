"""CLI wrapper for :mod:`evaluation.diagnostic.paper_figures`.

Produces paper-ready figure packages (input + per-model attribution tiles)
with a machine-readable ``figure_spec.json`` per (sample, class), plus an
optional rendered PDF + PNG panel.

Run with::

    python -m evaluation.scripts.diag_paper_figures [OPTIONS]
"""
from __future__ import annotations

from evaluation.diagnostic.paper_figures import main


if __name__ == "__main__":
    main()
