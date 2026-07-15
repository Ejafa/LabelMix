"""Shared matplotlib styling for all paper plots.

Import and call :func:`apply_paper_style` *once* at the top of every plot
script to get consistent fonts, figure sizes, and color palettes across
every figure shipped in the paper.

Color scheme
------------
All figures use seaborn's standard ``"pastel"`` qualitative palette as
the shared base.  :data:`PALETTE` exposes that palette as a list of hex
strings and :data:`METHOD_COLORS` pins a stable method → hex mapping so
the same training config always prints in the same colour across every
panel of the paper.

``seaborn`` is imported lazily and is *optional* — if it is not
installed we fall back to the hard-coded hex values for the ``pastel``
palette (which are part of seaborn's public API and therefore stable).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


#: NeurIPS 2026 native figure widths (inches).  The paper deliberately does
#: not resize PDFs in LaTeX, so generated canvases must fit these dimensions.
SINGLE_COL_WIDTH: float = 2.625
DOUBLE_COL_WIDTH: float = 5.5

#: Native widths for the repeated panel layouts used by the paper.  The
#: image-row layouts account for LaTeX's 2 pt ``\tabcolsep`` on both sides of
#: every cell.
THREE_PANEL_WIDTH: float = 0.32 * DOUBLE_COL_WIDTH
FOUR_PANEL_WIDTH: float = (DOUBLE_COL_WIDTH - 16.0 / 72.27) / 4.0
FIVE_PANEL_WIDTH: float = (DOUBLE_COL_WIDTH - 20.0 / 72.27) / 5.0

# ---------------------------------------------------------------------------
# Global typography
# ---------------------------------------------------------------------------

# Keep all plot text sizes defined in one place.  Individual plot modules
# should call ``apply_paper_style()`` and avoid local font-size rcParams.
BASE_FONT_SIZE: int = 8
TITLE_FONT_SIZE: int = 8
AXIS_LABEL_FONT_SIZE: int = 8
TICK_LABEL_FONT_SIZE: int = 8
LEGEND_FONT_SIZE: int = 8


# ---------------------------------------------------------------------------
# Seaborn "pastel" palette (stable hex values, reproduced verbatim so the
# module stays importable even when seaborn is not installed).
# ---------------------------------------------------------------------------

_PASTEL_FALLBACK: tuple[str, ...] = (
    "#A1C9F4",  # 0 blue
    "#FFB482",  # 1 orange
    "#8DE5A1",  # 2 green
    "#FF9F9B",  # 3 red
    "#D0BBFF",  # 4 purple
    "#DEBB9B",  # 5 brown
    "#FAB0E4",  # 6 pink
    "#CFCFCF",  # 7 grey
    "#FFFEA3",  # 8 olive/yellow
    "#B9F2F0",  # 9 cyan
)


def _load_palette() -> list[str]:
    """Return seaborn's ``pastel`` palette as hex, falling back if needed."""
    try:
        import seaborn as sns  # type: ignore

        return list(sns.color_palette("pastel").as_hex())
    except Exception:  # pragma: no cover - optional dep
        return list(_PASTEL_FALLBACK)

#: Seaborn ``pastel`` palette as a list of ``#rrggbb`` strings (length 10).
PALETTE: list[str] = _load_palette()

# ---------------------------------------------------------------------------
# Seaborn "deep" palette — the darker sibling of "pastel" with the same hue
# ordering (blue, orange, green, red, purple, brown, pink, grey, olive,
# cyan).  Used by line plots where the pale pastel fills become hard to
# read as thin strokes on a white background.
# ---------------------------------------------------------------------------

_DEEP_FALLBACK: tuple[str, ...] = (
    "#4C72B0",  # 0 blue
    "#DD8452",  # 1 orange
    "#55A467",  # 2 green  (seaborn deep index 2)
    "#C44E52",  # 3 red
    "#8172B3",  # 4 purple
    "#937860",  # 5 brown
    "#DA8BC3",  # 6 pink
    "#8C8C8C",  # 7 grey
    "#CCB974",  # 8 olive/yellow
    "#64B5CD",  # 9 cyan
)

def _load_palette_dark() -> list[str]:
    """Return seaborn's ``deep`` palette as hex, falling back if needed."""
    try:
        import seaborn as sns  # type: ignore

        return list(sns.color_palette("deep").as_hex())
    except Exception:  # pragma: no cover - optional dep
        return list(_DEEP_FALLBACK)

#: Seaborn ``deep`` palette as a list of ``#rrggbb`` strings (length 10).
PALETTE_DARK: list[str] = _load_palette_dark()


# ---------------------------------------------------------------------------
# Canonical method → colour mapping.
#
# Every plot script that draws per-method curves / bars must look up its
# colour here instead of hard-coding a hex string, so a given method
# always reads the same across every figure in the paper.
#
# Indices refer to the seaborn ``pastel`` palette above.  Neutral /
# reference series use explicit greys so they never fight the coloured
# methods for visual attention.  With the pastel palette we keep the
# greys slightly darker than the fills so reference lines stay readable.
# ---------------------------------------------------------------------------

#: Dark grey for the zero/reference baseline (e.g. Cutmix + Mixup refline).
REF_DARK_GREY: str = "#4D4D4D"
#: Mid grey for the secondary reference (e.g. single-image augmentation).
REF_MID_GREY: str = "#9A9A9A"

METHOD_COLORS: dict[str, str] = {
    # Non-LabelMix methods.
    "bare":           PALETTE[7],   # grey
    "noaug":          PALETTE[8],   # olive
    "baseline":       PALETTE[2],   # green
    "mixup":          PALETTE[9],   # cyan
    "cutmix":         PALETTE[5],   # brown
    "mosaic":         PALETTE[4],   # purple
    "fmix":           PALETTE[6],   # pink
    "gridmix":        PALETTE[0],   # blue
    "resizemix":      PALETTE[3],   # red
    "saliencymix":    PALETTE[8],   # olive
    "smoothmix":      PALETTE[9],   # cyan
    "tokenmix":       PALETTE[4],   # purple
    "tla":            PALETTE[5],   # brown
    # LabelMix family — kept clearly distinct from the non-LabelMix block.
    "labelmix-mixed": PALETTE[1],   # orange
    "labelmix-pl":    PALETTE[0],   # blue
    "labelmix-sce":   PALETTE[3],   # red
}

def method_color(method: str, default: str = REF_MID_GREY) -> str:
    """Return the canonical colour for ``method`` (or ``default``)."""
    return METHOD_COLORS.get(method, default)

# ---------------------------------------------------------------------------
# Darker method → colour mapping for line plots.
#
# Line plots on white backgrounds render the pastel fills as washed-out
# strokes that fight for legibility.  ``METHOD_COLORS_DARK`` mirrors
# :data:`METHOD_COLORS` index-for-index but draws from the seaborn
# ``deep`` palette, so every method keeps its canonical hue (blue =
# baseline, orange = labelmix-mixed, …) while lines stay readable.
# Bar plots should keep using :data:`METHOD_COLORS` (pastel).
# ---------------------------------------------------------------------------

METHOD_COLORS_DARK: dict[str, str] = {
    # Non-LabelMix methods.
    "bare":           PALETTE_DARK[7],   # grey
    "noaug":          PALETTE_DARK[8],   # olive
    "baseline":       PALETTE_DARK[2],   # green
    "mixup":          PALETTE_DARK[9],   # cyan
    "cutmix":         PALETTE_DARK[5],   # brown
    "mosaic":         PALETTE_DARK[4],   # purple
    "fmix":           PALETTE_DARK[6],   # pink
    "gridmix":        PALETTE_DARK[0],   # blue
    "resizemix":      PALETTE_DARK[3],   # red
    "saliencymix":    PALETTE_DARK[8],   # olive
    "smoothmix":      PALETTE_DARK[9],   # cyan
    "tokenmix":       PALETTE_DARK[4],   # purple
    "tla":            PALETTE_DARK[5],   # brown
    # LabelMix family — kept clearly distinct from the non-LabelMix block.
    "labelmix-mixed": PALETTE_DARK[1],   # orange
    "labelmix-pl":    PALETTE_DARK[0],   # blue
    "labelmix-sce":   PALETTE_DARK[3],   # red
}

def method_color_dark(method: str, default: str = REF_DARK_GREY) -> str:
    """Return the darker (line-plot) colour for ``method`` (or ``default``)."""
    return METHOD_COLORS_DARK.get(method, default)


# ---------------------------------------------------------------------------
# Canonical method → display-name mapping.
#
# Every paper plot must render the same internal key with the same
# user-facing string, so ``labelmix-sce`` always shows up as
# ``"TreemapMix (SCE)"`` in legends / titles / metadata, regardless of
# which figure it appears in.
#
# The three ``labelmix-*`` keys are *always* surfaced as ``"TreemapMix
# (...)"`` — the internal CSV keys stay untouched, only the display
# string is rewritten.
# ---------------------------------------------------------------------------

METHOD_DISPLAY: dict[str, str] = {
    # Non-LabelMix methods.
    "bare":           "Single-Image Aug",
    "noaug":          "No Aug",
    "baseline":       "Mixup+CutMix",
    "mixup":          "Mixup",
    "cutmix":         "CutMix",
    "mosaic":         "RICAP",
    "fmix":           "FMix",
    "gridmix":        "GridMix",
    "resizemix":      "ResizeMix",
    "saliencymix":    "SaliencyMix",
    "smoothmix":      "SmoothMix",
    "tokenmix":       "TokenMix",
    "tla":            "TLA",
    # LabelMix family — canonical TreemapMix naming for every paper plot.
    "labelmix-mixed": "TreemapMix-mixed",
    "labelmix-pl":    "TreemapMix-PL",
    "labelmix-sce":   "TreemapMix-SCE",
}


def method_display(method: str, default: str | None = None) -> str:
    """Return the canonical display string for ``method``.

    Falls back to ``default`` (or the raw key when ``default`` is None)
    so unknown keys are surfaced verbatim rather than silently dropped.
    """
    return METHOD_DISPLAY.get(method, default if default is not None else method)


# ---------------------------------------------------------------------------
# Semantic colour helpers.
#
# Plot scripts should call :func:`bar_color` when drawing filled bars
# (the lighter pastel palette) and :func:`line_color` when drawing
# strokes / markers (the darker seaborn-deep palette).  This mirrors the
# paper-wide rule "lines = dark hue, bars = light hue" without each
# call-site needing to remember which dict to import.
# ---------------------------------------------------------------------------

def bar_color(method: str, default: str = REF_MID_GREY) -> str:
    """Canonical fill colour for bar charts (pastel palette)."""
    return METHOD_COLORS.get(method, default)


def line_color(method: str, default: str = REF_DARK_GREY) -> str:
    """Canonical stroke colour for line plots (deep palette)."""
    return METHOD_COLORS_DARK.get(method, default)


_PAPER_RC = {
    "font.family": "serif",
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    # Keep the media box equal to ``figsize``. Tight bounding boxes make
    # native-size LaTeX placement depend on the labels surrounding the axes.
    "savefig.bbox": None,
    "pdf.fonttype": 42,   # embed TrueType so LaTeX can re-render cleanly
    "ps.fonttype": 42,
    # Use the seaborn ``pastel`` palette for matplotlib's default colour
    # cycle so plots that don't pick colours explicitly (e.g. the simple
    # grouped bar charts) still follow the paper-wide scheme.
    "axes.prop_cycle": cycler(color=PALETTE),
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
    """Save a figure as a PDF.

    Any suffix in ``out_path`` is normalised to ``.pdf`` so callers cannot
    accidentally emit raster plot files with inconsistent text rendering.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf")
