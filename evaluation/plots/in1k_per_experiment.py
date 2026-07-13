"""Plot: per-experiment Top-1 accuracy + ECE bars for ImageNet-1k.

Renders the aggregated output of
:mod:`evaluation.scripts.aggregate_per_experiment` as a 2-row grouped bar
chart:

* **Top row** -- Top-1 accuracy (%) per (model, experiment type), higher is
  better.
* **Bottom row** -- Expected Calibration Error at ``n_bins=15`` expressed
  in percentage points (``ECE x 100``), lower is better.

The figure itself is clean -- no in-figure legend -- and a matching
standalone legend PDF is written next to each panel figure so it can be
placed independently in the paper.

The three LabelMix variants use three standard, clearly distinct
tab10-palette colours (orange / green / red) so each variant is
immediately identifiable while still sitting together as a group in the
bar ordering.

Inputs (default run, ``--input`` overrides):
  - ``data/processed/in1k_per_experiment_short.csv``
  - ``data/processed/in1k_per_experiment_long.csv``

Outputs (per input):
  - ``figures/per_experiment/<stem>.pdf``

For the short-horizon CSV an additional companion figure is emitted in
the same run:
  - ``figures/per_experiment/<stem>_nll_brier.pdf``

The long-horizon run intentionally skips the NLL/Brier companion.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ..processing.aggregate import confidence_interval_95
from ._style import (
    DOUBLE_COL_WIDTH,
    METHOD_COLORS,
    METHOD_DISPLAY,
    REF_MID_GREY,
    apply_paper_style,
    bar_color,
    method_display,
    savefig,
)


_logger = logging.getLogger(__name__)

# Subdirectory for these paper figures.
_OUT_SUBDIR = "per_experiment"
_PANEL_WIDTH = DOUBLE_COL_WIDTH
_PANEL_HEIGHT = 4.0 * (DOUBLE_COL_WIDTH / (6.8 * 1.15))

# Reference ECE bin count (Guo et al., 2017).
_ECE_BINS = 15
_ECE_MEAN_COL = f"ece/n_bins={_ECE_BINS}_mean"
_ECE_STD_COL = f"ece/n_bins={_ECE_BINS}_std"

# Experiment types we never want to render. ``unbalanced-noaug`` is a
# diagnostic control that doesn't belong in the headline figure.
_EXCLUDED_TYPES: frozenset[str] = frozenset({"unbalanced-noaug"})

# Display order for the grouped bars. LabelMix variants sit together as a
# visually coherent block on the right-hand side.
_TYPE_ORDER: Sequence[str] = (
    "bare",
    "noaug",
    "baseline",
    "mixup",
    "cutmix",
    "mosaic",
    "fmix",
    "gridmix",
    "resizemix",
    "saliencymix",
    "smoothmix",
    "tokenmix",
    "tla",
    "labelmix-sce",
    "labelmix-mixed",
    "labelmix-pl",
)

# Per-type styling: friendly label and fill colour.  Labels come from
# :data:`evaluation.plots._style.METHOD_DISPLAY` and colours from
# :func:`evaluation.plots._style.bar_color` (the pastel palette), so
# every paper figure uses the same canonical method → (label, colour)
# mapping.  No hatches — colour alone disambiguates the bars.
_TYPE_STYLE: dict[str, dict[str, str | None]] = {
    key: {"label": method_display(key), "color": bar_color(key), "hatch": None}
    for key in METHOD_DISPLAY
}

# Friendly model labels (strip the patch/reg noise from the checkpoint name).
_MODEL_LABELS = {
    "vit_wee_patch16_reg1_gap_256": "ViT-Wee",
    "vit_little_patch16_reg4_gap_256": "ViT-Little",
    "vit_medium_patch16_reg1_gap_256": "ViT-Medium",
    "vit_betwixt_patch16_reg4_gap_256": "ViT-Betwixt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordered(values: Sequence[str], preferred: Sequence[str]) -> list[str]:
    """Return ``values`` sorted by ``preferred`` first, then alphabetically."""
    known = [v for v in preferred if v in values]
    extra = sorted(v for v in values if v not in set(preferred))
    return known + extra


def _style_for(t: str) -> dict[str, str | None]:
    """Return styling dict for type ``t`` (fallback to a neutral grey)."""
    return _TYPE_STYLE.get(t, {"label": t, "color": "#7f7f7f", "hatch": None})


def _ece_pp(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (mean, std) for ECE expressed in *percentage points* (x100)."""
    return df[_ECE_MEAN_COL] * 100.0, df[_ECE_STD_COL] * 100.0


def _filter_types(df: pd.DataFrame) -> pd.DataFrame:
    """Drop experiment types that should never appear in the paper figure."""
    return df[~df["type"].isin(_EXCLUDED_TYPES)].copy()


# ---------------------------------------------------------------------------
# Main panel plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, title: str | None = "ImageNet-1K") -> plt.Figure:
    """Render the 2-row (Top-1, ECE) grouped bar chart."""
    apply_paper_style()

    df = _filter_types(df)
    models = _ordered(df["model"].unique(), list(_MODEL_LABELS))
    types = _ordered(df["type"].unique(), _TYPE_ORDER)

    x = np.arange(len(models))
    # Slightly tighter group so bars within a model cluster visually.
    width = 0.82 / max(len(types), 1)

    # Visually separate the LabelMix family from the other methods by
    # leaving half a bar-width of empty space between the last non-LabelMix
    # bar and the first LabelMix bar. ``gap`` is added to every LabelMix
    # bar's offset; the group as a whole is then recentred so the cluster
    # still sits under its model tick.
    def _is_labelmix(t: str) -> bool:
        return t.startswith("labelmix")

    gap = 0.5 * width
    # Per-type positional index in the cluster, counting the gap as one
    # extra "virtual slot" inserted before the first LabelMix bar.
    positions: list[float] = []
    seen_labelmix = False
    for idx, t in enumerate(types):
        pos = float(idx)
        if _is_labelmix(t):
            if not seen_labelmix:
                seen_labelmix = True
            pos += gap / width  # advance by the fractional gap
        positions.append(pos)

    # Recentre the whole cluster (including the gap) on 0.
    centre = (positions[0] + positions[-1]) / 2.0
    offsets = [(p - centre) * width for p in positions]

    fig, (ax_top, ax_ece) = plt.subplots(
        2, 1,
        figsize=(_PANEL_WIDTH, _PANEL_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": 0.28},
    )

    for i, t in enumerate(types):
        sub = (
            df[df["type"] == t]
            .set_index("model")
            .reindex(models)
        )
        offset = offsets[i]
        style = _style_for(t)
        color = style["color"]
        hatch = style["hatch"]

        # Top-1 accuracy (already on a 0-100 scale).
        top1_ci = confidence_interval_95(sub["top1_acc_std"], sub["n_seeds"])
        ax_top.bar(
            x + offset,
            sub["top1_acc_mean"].values,
            width=width,
            yerr=top1_ci.values,
            capsize=3.0,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            error_kw={"elinewidth": 0.4, "capthick": 0.4, "ecolor": "black"},
            zorder=2,
        )

        # ECE in percentage points (4-decimal CSV precision -> ~0.01 pp).
        ece_mean_pp, ece_std_pp = _ece_pp(sub)
        ece_ci_pp = confidence_interval_95(ece_std_pp, sub["n_seeds"])
        ax_ece.bar(
            x + offset,
            ece_mean_pp.values,
            width=width,
            yerr=ece_ci_pp.values,
            capsize=3.0,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            error_kw={"elinewidth": 0.4, "capthick": 0.4, "ecolor": "black"},
            zorder=2,
        )

    # --- Axis cosmetics -----------------------------------------------------
    model_labels = [_MODEL_LABELS.get(m, m) for m in models]
    for ax in (ax_top, ax_ece):
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        # ``sharex=True`` hides upper tick labels by default; these stacked
        # panels are often placed one after another, so each panel repeats the
        # model names without repeating an extra x-axis label.
        ax.tick_params(axis="x", labelbottom=True)

    ax_top.set_ylabel("Accuracy (%)")
    ax_ece.set_ylabel("ECE (p.p.)")

    # Auto-zoom Top-1 so <1 pp differences remain visible.
    top1_vals = df["top1_acc_mean"].dropna()
    if not top1_vals.empty:
        lo = float(np.floor(top1_vals.min() - 0.5))
        hi = float(np.ceil(top1_vals.max() + 0.5))
        ax_top.set_ylim(lo, hi)

    for ax in (ax_top, ax_ece):
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)  # no x-ticks -- bars already group

    if title:
        ax_top.set_title(title, pad=6)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Standalone legend
# ---------------------------------------------------------------------------

def _legend_handles(types: Sequence[str]) -> list[Patch]:
    """Return legend handles matching the ordered types present in the data."""
    return [
        Patch(
            facecolor=_style_for(t)["color"],
            hatch=_style_for(t)["hatch"] or "",
            edgecolor="black",
            linewidth=0.5,
            label=_style_for(t)["label"],
        )
        for t in types
    ]


def plot_legend(types: Sequence[str]) -> plt.Figure:
    """Render the shared legend as a standalone horizontal figure."""
    apply_paper_style()

    handles = _legend_handles(types)
    labels = [h.get_label() for h in handles]

    ncol = min(len(handles), 4)
    width = DOUBLE_COL_WIDTH
    nrows = int(np.ceil(len(handles) / ncol)) if ncol else 1
    height = 0.35 + 0.28 * nrows

    fig = plt.figure(figsize=(width, height))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        handleheight=1.0,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_INPUTS: tuple[tuple[str, Path, Path, bool], ...] = (
    (
        "ImageNet-1K",
        PROCESSED_DIR / "in1k_per_experiment_short.csv",
        FIGURES_DIR / _OUT_SUBDIR / "in1k_per_experiment_short.pdf",
        True,  # also render NLL/Brier companion figure for the short CSV
    ),
    (
        "ImageNet-1K",
        PROCESSED_DIR / "in1k_per_experiment_long.csv",
        FIGURES_DIR / _OUT_SUBDIR / "in1k_per_experiment_long.pdf",
        False,  # no NLL/Brier companion for the long horizon
    ),
)


def _nll_brier_path_for(panel_path: Path) -> Path:
    """Companion NLL/Brier PDF next to ``panel_path`` (same stem + suffix)."""
    return panel_path.with_name(panel_path.stem + "_nll_brier.pdf")


def _legend_path_for(panel_path: Path) -> Path:
    return panel_path.with_name(panel_path.stem + "_legend.pdf")


def _render_one(
    title: str,
    in_path: Path,
    out_path: Path,
    *,
    with_nll_brier: bool = False,
) -> int:
    if not in_path.is_file():
        _logger.warning("Skipping %s (no such file)", in_path)
        return 0
    _logger.info("Reading %s", in_path)
    df = pd.read_csv(in_path)
    if df.empty:
        _logger.info("Skipping %s (empty CSV)", in_path)
        return 0

    if title == "ImageNet-1K" and "dataset" in df.columns:
        datasets = " ".join(str(v).lower() for v in df["dataset"].dropna().unique())
        if "cifar100" in datasets:
            title = "CIFAR100"

    fig = plot(df, title=title)
    savefig(fig, str(out_path))
    plt.close(fig)
    _logger.info("Wrote %s", out_path)

    types = _ordered(_filter_types(df)["type"].unique(), _TYPE_ORDER)
    leg_fig = plot_legend(types)
    leg_path = _legend_path_for(out_path)
    savefig(leg_fig, str(leg_path))
    plt.close(leg_fig)
    _logger.info("Wrote %s", leg_path)

    # Optional companion NLL/Brier figure (short CSV only by default).
    if with_nll_brier:
        # Local import keeps the two modules independent and avoids an
        # import cycle (the NLL/Brier module imports helpers from this one).
        from .in1k_per_experiment_nll_brier import plot as plot_nll_brier

        nb_path = _nll_brier_path_for(out_path)
        nb_fig = plot_nll_brier(df, title=title)
        savefig(nb_fig, str(nb_path))
        plt.close(nb_fig)
        _logger.info("Wrote %s", nb_path)

        nb_leg_fig = plot_legend(types)
        nb_leg_path = _legend_path_for(nb_path)
        savefig(nb_leg_fig, str(nb_leg_path))
        plt.close(nb_leg_fig)
        _logger.info("Wrote %s", nb_leg_path)

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=None,
        help=("Aggregated per-experiment CSV. If omitted, both the short- "
              "and long-horizon in1k CSVs are rendered."),
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output PDF path for the panel figure. "
             "Only honoured when --input is given.",
    )
    p.add_argument("--title", type=str, default="ImageNet-1K",
                   help="Figure title for the top panel.")
    p.add_argument(
        "--with-nll-brier", action="store_true",
        help=("Also render the NLL + Brier companion figure next to the "
              "Top-1/ECE panel. On by default for the short-horizon CSV "
              "when --input is omitted."),
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if args.input is None:
        exit_code = 0
        for title, in_path, out_path, with_nb in _DEFAULT_INPUTS:
            exit_code = (
                _render_one(title, in_path, out_path, with_nll_brier=with_nb)
                or exit_code
            )
        return exit_code

    out_path = args.output
    if out_path is None:
        out_path = FIGURES_DIR / _OUT_SUBDIR / (args.input.stem + ".pdf")
    return _render_one(
        args.title, args.input, out_path, with_nll_brier=args.with_nll_brier,
    )


if __name__ == "__main__":
    raise SystemExit(main())
