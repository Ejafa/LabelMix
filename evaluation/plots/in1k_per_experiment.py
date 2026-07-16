"""Plot: per-experiment Top-1 accuracy + ECE forests for ImageNet-1k.

Renders the aggregated output of
:mod:`evaluation.scripts.aggregate_per_experiment` as two stacked forest-plot
matrices:

* **Top section** -- Top-1 accuracy (%) per (model, experiment type), higher
  is better.
* **Bottom section** -- Expected Calibration Error at ``n_bins=15`` expressed
  in percentage points (``ECE x 100``), lower is better.

Methods form one shared row axis. Backbones use distinct markers and small
within-row offsets in the side-by-side Top-1 and ECE forests. Points denote
means and horizontal intervals denote +/- one standard deviation. Each metric
has a mean within-backbone rank column. TreemapMix variants are pinned to the
top; all other methods use one shared order based on their mean Top-1/ECE rank.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import (
    DOUBLE_COL_WIDTH,
    METHOD_DISPLAY,
    PALETTE_DARK,
    apply_paper_style,
    bar_color,
    method_color_dark,
    method_display,
    savefig,
)


_logger = logging.getLogger(__name__)

# Subdirectory for these paper figures.
_OUT_SUBDIR = "per_experiment"
_PANEL_WIDTH = DOUBLE_COL_WIDTH
_PANEL_HEIGHT = 4.05

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


def _filter_types(df: pd.DataFrame) -> pd.DataFrame:
    """Drop experiment types that should never appear in the paper figure."""
    return df[~df["type"].isin(_EXCLUDED_TYPES)].copy()


# ---------------------------------------------------------------------------
# Main panel plot
# ---------------------------------------------------------------------------

def _average_ranks(
    df: pd.DataFrame,
    models: Sequence[str],
    *,
    value_col: str,
    higher_is_better: bool,
) -> pd.Series:
    """Return each method's mean within-model rank (1 is best)."""
    wide = df.pivot_table(
        index="type", columns="model", values=value_col, aggfunc="mean",
    ).reindex(columns=models)
    return wide.rank(axis=0, ascending=not higher_is_better).mean(axis=1)


def _rank_order(types: Sequence[str], ranks: pd.Series) -> list[str]:
    """Sort by rank while keeping every TreemapMix variant at the top."""
    present = list(dict.fromkeys(types))

    def key(t: str) -> tuple[int, float, str]:
        rank = float(ranks.get(t, np.inf))
        return (0 if t.startswith("labelmix-") else 1, rank, t)

    return sorted(present, key=key)


def _plot_forest_section(
    axes: Sequence[plt.Axes],
    df: pd.DataFrame,
    models: Sequence[str],
    *,
    mean_col: str,
    std_col: str,
    scale: float,
    metric_title: str,
    higher_is_better: bool,
    types: Sequence[str] | None = None,
) -> None:
    """Draw one metric as four model forests plus an average-rank column."""
    ranks = _average_ranks(
        df, models, value_col=mean_col, higher_is_better=higher_is_better,
    )
    types = list(types) if types is not None else _rank_order(df["type"].unique(), ranks)
    y = np.arange(len(types), dtype=float)

    for model, ax in zip(models, axes[:-1]):
        sub = df[df["model"] == model].set_index("type").reindex(types)
        means = sub[mean_col].to_numpy(dtype=float) * scale
        stds = sub[std_col].to_numpy(dtype=float) * scale

        for row, method, mean, std in zip(y, types, means, stds):
            if not np.isfinite(mean):
                continue
            color = method_color_dark(method)
            ax.errorbar(
                mean,
                row,
                xerr=std if np.isfinite(std) else None,
                fmt="o",
                markersize=3.4,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.35,
                ecolor=color,
                elinewidth=1.15,
                capsize=0,
                zorder=3,
            )

        finite = np.isfinite(means) & np.isfinite(stds)
        if finite.any():
            low = float(np.min(means[finite] - stds[finite]))
            high = float(np.max(means[finite] + stds[finite]))
            span = max(high - low, 1e-6)
            ax.set_xlim(low - 0.08 * span, high + 0.08 * span)

        ax.set_title(_MODEL_LABELS.get(model, model), pad=4)
        ax.set_ylim(len(types) - 0.5, -0.5)
        ax.set_yticks(y)
        ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", pad=1)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    labels = [method_display(t) for t in types]
    axes[0].set_yticklabels(labels)
    for label, method in zip(axes[0].get_yticklabels(), types):
        if method.startswith("labelmix-"):
            label.set_fontweight("bold")
    for ax in axes[1:-1]:
        ax.set_yticklabels([])

    rank_ax = axes[-1]
    rank_ax.set_title("Avg.\nrank", pad=1)
    rank_ax.set_xlim(0.0, 1.0)
    rank_ax.set_ylim(len(types) - 0.5, -0.5)
    rank_ax.set_xticks([])
    rank_ax.set_yticks(y)
    rank_ax.set_yticklabels([])
    rank_ax.tick_params(axis="y", length=0)
    for spine in rank_ax.spines.values():
        spine.set_visible(False)
    for row, method in zip(y, types):
        rank = ranks.get(method, np.nan)
        if np.isfinite(rank):
            rank_ax.text(
                0.5,
                row,
                f"{rank:.1f}",
                ha="center",
                va="center",
                fontweight="bold" if method.startswith("labelmix-") else "normal",
            )

    # Put the metric label above the shared method-name column rather than
    # repeating it under every narrow model axis.
    axes[0].text(
        -0.03,
        1.05,
        metric_title,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def _plot_compact_forest(
    ax: plt.Axes,
    rank_ax: plt.Axes,
    df: pd.DataFrame,
    models: Sequence[str],
    types: Sequence[str],
    *,
    mean_col: str,
    std_col: str,
    scale: float,
    title: str,
    higher_is_better: bool,
    show_method_labels: bool,
) -> pd.Series:
    """Draw one metric with model-specific points offset within each row."""
    ranks = _average_ranks(
        df, models, value_col=mean_col, higher_is_better=higher_is_better,
    )
    y = np.arange(len(types), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(models))
    markers = ("o", "s", "^", "D")
    colors = (PALETTE_DARK[0], PALETTE_DARK[1], PALETTE_DARK[2], PALETTE_DARK[3])

    all_low: list[float] = []
    all_high: list[float] = []
    for model_index, (model, offset) in enumerate(zip(models, offsets)):
        sub = df[df["model"] == model].set_index("type").reindex(types)
        means = sub[mean_col].to_numpy(dtype=float) * scale
        stds = sub[std_col].to_numpy(dtype=float) * scale
        finite = np.isfinite(means)
        if finite.any():
            finite_std = np.where(np.isfinite(stds), stds, 0.0)
            all_low.extend((means[finite] - finite_std[finite]).tolist())
            all_high.extend((means[finite] + finite_std[finite]).tolist())
        ax.errorbar(
            means,
            y + offset,
            xerr=stds,
            fmt=markers[model_index],
            markersize=3.2,
            markerfacecolor=colors[model_index],
            markeredgecolor="black",
            markeredgewidth=0.3,
            ecolor=colors[model_index],
            elinewidth=0.9,
            capsize=0,
            linestyle="none",
            zorder=3,
        )

    if all_low and all_high:
        low, high = min(all_low), max(all_high)
        span = max(high - low, 1e-6)
        ax.set_xlim(low - 0.05 * span, high + 0.05 * span)
    ax.set_title(title, pad=4)
    ax.set_ylim(len(types) - 0.55, -0.55)
    ax.set_yticks(y)
    ax.tick_params(axis="y", length=0)
    if show_method_labels:
        ax.set_yticklabels([method_display(method) for method in types])
        for label, method in zip(ax.get_yticklabels(), types):
            if method.startswith("labelmix-"):
                label.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.35, color="0.9", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    rank_ax.set_title("Avg.\nrank", pad=1)
    rank_ax.set_xlim(0.0, 1.0)
    rank_ax.set_ylim(len(types) - 0.55, -0.55)
    rank_ax.set_xticks([])
    rank_ax.set_yticks([])
    for spine in rank_ax.spines.values():
        spine.set_visible(False)
    for row, method in zip(y, types):
        rank = ranks.get(method, np.nan)
        if np.isfinite(rank):
            rank_ax.text(
                0.5, row, f"{rank:.1f}", ha="center", va="center",
                fontweight="bold" if method.startswith("labelmix-") else "normal",
            )
    return ranks

def plot(df: pd.DataFrame, title: str | None = "ImageNet-1K") -> plt.Figure:
    """Render stacked Top-1 and ECE forest-plot matrices."""
    apply_paper_style()

    df = _filter_types(df)
    models = _ordered(df["model"].unique(), list(_MODEL_LABELS))
    top1_ranks = _average_ranks(
        df, models, value_col="top1_acc_mean", higher_is_better=True,
    )
    ece_ranks = _average_ranks(
        df, models, value_col=_ECE_MEAN_COL, higher_is_better=False,
    )
    overall_ranks = pd.concat(
        [top1_ranks.rename("top1"), ece_ranks.rename("ece")], axis=1,
    ).mean(axis=1)
    shared_types = _rank_order(df["type"].unique(), overall_ranks)

    fig = plt.figure(figsize=(_PANEL_WIDTH, _PANEL_HEIGHT))
    grid = fig.add_gridspec(
        1, 4,
        width_ratios=(1.0, 0.22, 1.0, 0.22),
        wspace=0.24,
        left=0.255,
        right=0.995,
        bottom=0.075,
        top=0.87,
    )
    top_ax = fig.add_subplot(grid[0, 0])
    top_rank_ax = fig.add_subplot(grid[0, 1])
    ece_ax = fig.add_subplot(grid[0, 2])
    ece_rank_ax = fig.add_subplot(grid[0, 3])

    _plot_compact_forest(
        top_ax,
        top_rank_ax,
        df,
        models,
        shared_types,
        mean_col="top1_acc_mean",
        std_col="top1_acc_std",
        scale=1.0,
        title=(
            f"{title} — Top-1 (%)"
            if title else "Top-1 accuracy (%)  |  higher is better"
        ),
        higher_is_better=True,
        show_method_labels=True,
    )
    _plot_compact_forest(
        ece_ax,
        ece_rank_ax,
        df,
        models,
        shared_types,
        mean_col=_ECE_MEAN_COL,
        std_col=_ECE_STD_COL,
        scale=100.0,
        title="ECE (p.p.)",
        higher_is_better=False,
        show_method_labels=False,
    )

    markers = ("o", "s", "^", "D")
    colors = (PALETTE_DARK[0], PALETTE_DARK[1], PALETTE_DARK[2], PALETTE_DARK[3])
    handles = [
        Line2D(
            [0], [0], marker=markers[index], color=colors[index],
            markeredgecolor="black", markeredgewidth=0.3, linewidth=1.0,
            label=_MODEL_LABELS.get(model, model),
        )
        for index, model in enumerate(models)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.61, 0.985),
        handlelength=1.4,
        columnspacing=1.0,
    )

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
