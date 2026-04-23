"""Shared matplotlib styling for all paper plots.

Import and call :func:`apply_paper_style` *once* at the top of every plot
script to get consistent fonts, figure sizes, and color palettes across
every figure shipped in the paper.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt


#: Single-column figure width (inches) — matches standard NeurIPS/ICML templates.
SINGLE_COL_WIDTH: float = 3.3

#: Double-column / two-panel figure width (inches).
DOUBLE_COL_WIDTH: float = 6.8


_PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed TrueType so LaTeX can re-render cleanly
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    """Install the paper rcParams globally.  Idempotent."""
    mpl.rcParams.update(_PAPER_RC)


@contextmanager
def paper_style() -> Iterator[None]:
    """Context manager form of :func:`apply_paper_style` for notebooks."""
    with mpl.rc_context(_PAPER_RC):
        yield


def savefig(fig: plt.Figure, out_path: str) -> None:
    """Save a figure in PDF *and* PNG next to each other.

    Passing ``out_path = ".../foo.pdf"`` will also write ``foo.png``.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
