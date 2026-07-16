"""Plot: area-vs-logit diagnostic metrics across training configs.

Renders three paper-style figures from
``data/processed/diagnostic_area_logit_metrics.csv`` (one file per metric):

1. ``area_logit_spearman.pdf``
   Metric 1: mean Spearman correlation between class area ratios and
   per-class logits on the composed ImageNet-diagnostic dataset.
2. ``area_logit_pair_acc.pdf``
   Metric 2: present-class pairwise ranking accuracy with the standard
   ``|area_i - area_j| >= 0.05`` threshold.
3. ``area_logit_top_acc.pdf``
   Metric 3: largest-area top-logit accuracy.

Every figure is a single panel:
 * x-axis = composed-sample class count ``K`` (categorical-uniform
   spacing, values ``3, 4, 5, 6``),
 * y-axis = metric value,
 * one line-with-markers per training config.

The ``K == "all"`` aggregate row is not drawn on the canvas (it would
clutter an already eight-line plot) but is captured in the ``.meta.json``
sidecar so downstream tooling can quote headline numbers without
re-reading the CSV.

Styling is inherited from :mod:`evaluation.plots._style` (paper rcParams,
categorical x-axis, dotted horizontal grid, PDF output via
:func:`._style.savefig`).  The shared legend is rendered separately in
horizontal and vertical variants so the metric panels can be reused
without repeating the same legend.

Naming note: the ``labelmix-*`` runs in the source CSV are rendered as
``TreemapMix (...)`` in every user-facing string (legend labels, titles,
captions, metadata).  Internal CSV keys are untouched.

Input   : ``data/processed/diagnostic_area_logit_metrics.csv``
Output  : ``data/processed/figures/diagnostic_area_logit/area_logit_spearman.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_pair_acc.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_top_acc.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_spearman_with_std.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_pair_acc_with_std.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_top_acc_with_std.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_legend_horizontal.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_legend_vertical.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_heatmaps.pdf``
          (+ ``.meta.json`` sidecars for each)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import (
    DOUBLE_COL_WIDTH,
    THREE_PANEL_WIDTH,
    apply_paper_style,
    line_color,
    method_display,
    savefig,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_FILE = PROCESSED_DIR / "diagnostic_area_logit_metrics.csv"
DIAG_FIGURES_DIR = FIGURES_DIR / "diagnostic_area_logit"
OUTPUT_SPEARMAN = DIAG_FIGURES_DIR / "area_logit_spearman.pdf"
OUTPUT_PAIR_ACC = DIAG_FIGURES_DIR / "area_logit_pair_acc.pdf"
OUTPUT_TOP_ACC = DIAG_FIGURES_DIR / "area_logit_top_acc.pdf"
OUTPUT_SPEARMAN_STD = DIAG_FIGURES_DIR / "area_logit_spearman_with_std.pdf"
OUTPUT_PAIR_ACC_STD = DIAG_FIGURES_DIR / "area_logit_pair_acc_with_std.pdf"
OUTPUT_TOP_ACC_STD = DIAG_FIGURES_DIR / "area_logit_top_acc_with_std.pdf"
OUTPUT_LEGEND_HORIZONTAL = DIAG_FIGURES_DIR / "area_logit_legend_horizontal.pdf"
OUTPUT_LEGEND_VERTICAL = DIAG_FIGURES_DIR / "area_logit_legend_vertical.pdf"
OUTPUT_HEATMAPS = DIAG_FIGURES_DIR / "area_logit_heatmaps.pdf"

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
# CSV short name -> (display label, colour, linestyle).  Order here defines
# the left-to-right legend order and the stacking order in the figure.
#
# Every ``labelmix-*`` run is surfaced as ``TreemapMix (...)`` in the
# user-facing display string; the internal CSV key is left alone so the
# mapping YAML and the CSV stay stable.
@dataclass(frozen=True)
class ModelStyle:
    key: str
    display: str
    colour: str
    linestyle: str = "-"
    marker: str = "o"


MODEL_ORDER: Sequence[ModelStyle] = (
    ModelStyle("noaug",        method_display("noaug"),        line_color("noaug"),        "-", "s"),
    ModelStyle("bare",         method_display("bare"),         line_color("bare"),         "-", "P"),
    ModelStyle("baseline",     method_display("baseline"),     line_color("baseline"),     "-", "o"),
    ModelStyle("cutmix",       method_display("cutmix"),       line_color("cutmix"),       "-", "^"),
    ModelStyle("mixup",        method_display("mixup"),        line_color("mixup"),        "-", "v"),
    ModelStyle("mosaic",       method_display("mosaic"),       line_color("mosaic"),       "-", "D"),
    ModelStyle("fmix",         method_display("fmix"),         line_color("fmix"),         "-", "<"),
    ModelStyle("gridmix",      method_display("gridmix"),      line_color("gridmix"),      "-", ">"),
    ModelStyle("resizemix",    method_display("resizemix"),    line_color("resizemix"),    "-", "d"),
    ModelStyle("saliencymix",  method_display("saliencymix"),  line_color("saliencymix"),  "-", "p"),
    ModelStyle("smoothmix",    method_display("smoothmix"),    line_color("smoothmix"),    "-", "8"),
    ModelStyle("tokenmix",     method_display("tokenmix"),     line_color("tokenmix"),     "-", "H"),
    ModelStyle("tla",          method_display("tla"),          line_color("tla"),          "-", "+"),
    ModelStyle("labelmix-sce",   method_display("labelmix-sce"),   line_color("labelmix-sce"),   "-", "X"),
    ModelStyle("labelmix-mixed", method_display("labelmix-mixed"), line_color("labelmix-mixed"), "-", "h"),
    ModelStyle("labelmix-pl",    method_display("labelmix-pl"),    line_color("labelmix-pl"),    "-", "*"),
)

K_VALUES: Sequence[int] = (3, 4, 5, 6)
DIAG_FIG_WIDTH = THREE_PANEL_WIDTH
DIAG_FIG_HEIGHT = DIAG_FIG_WIDTH * (3.6 / 4.6)
K_DISPLAY_SPACING = 0.55
K_EDGE_PAD = 0.10

def _apply_diag_style() -> None:
    apply_paper_style()


def _k_positions() -> dict[int, float]:
    raw = np.arange(len(K_VALUES), dtype=float)
    centered = (raw - raw.mean()) * K_DISPLAY_SPACING + raw.mean()
    return {k: float(x) for k, x in zip(K_VALUES, centered)}


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=style.colour,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=3.8,
            lw=1.3,
            label=style.display,
        )
        for style in MODEL_ORDER
    ]


# metric key -> (csv column, pretty label, direction, y-pad fraction).
# Column names match the across-seed mean produced by
# ``evaluation/scripts/diag_area_logit_metrics.py``.
_METRICS = {
    "spearman": {
        "column": "spearman_mean_mean",
        "std_column": "spearman_mean_std",
        "label": r"Spearman $\rho$",
        "title": "Area-logit Spearman correlation",
        "direction": "higher is better",
        "filename": "area_logit_spearman",
        "out_path": OUTPUT_SPEARMAN,
        "out_path_std": OUTPUT_SPEARMAN_STD,
    },
    "pair_acc": {
        "column": "pair_acc_mean",
        "std_column": "pair_acc_std",
        "label": "Rank accuracy",
        "title": "Pairwise area-vs-logit ranking accuracy",
        "direction": "higher is better",
        "filename": "area_logit_pair_acc",
        "out_path": OUTPUT_PAIR_ACC,
        "out_path_std": OUTPUT_PAIR_ACC_STD,
    },
    "top_acc": {
        "column": "top_acc_mean",
        "std_column": "top_acc_std",
        "label": "Top accuracy",
        "title": "Largest-area top-logit accuracy",
        "direction": "higher is better",
        "filename": "area_logit_top_acc",
        "out_path": OUTPUT_TOP_ACC,
        "out_path_std": OUTPUT_TOP_ACC_STD,
    },
}

# Shading alpha for the mean +/- std band in the *_with_std variants.
STD_BAND_ALPHA = 0.15

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(input_path: Path) -> pd.DataFrame:
    """Read the metrics CSV, stripping the leading ``#`` banner."""
    df = pd.read_csv(input_path, comment="#")
    # Normalise types
    df["model"] = df["model"].astype(str)
    # ``k`` is mostly integer but contains the literal ``"all"`` aggregate.
    df["k"] = df["k"].astype(str)
    return df


def _per_k(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a (model, K) wide matrix restricted to K in ``K_VALUES``."""
    sub = df[df["k"].isin([str(k) for k in K_VALUES])].copy()
    sub["k_int"] = sub["k"].astype(int)
    wide = sub.pivot_table(index="model", columns="k_int", values=column, aggfunc="mean")
    wide = wide.reindex(columns=list(K_VALUES))
    return wide


def _all_row(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Return {model: value} for the ``K == "all"`` aggregate row."""
    sub = df[df["k"] == "all"]
    return {str(r["model"]): float(r[column]) for _, r in sub.iterrows()}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _plot_metric(
    ax: plt.Axes,
    wide: pd.DataFrame,
    *,
    ylabel: str,
    wide_std: pd.DataFrame | None = None,
) -> None:
    """Draw one metric panel (x = K, one line per model).

    When ``wide_std`` is provided, also draws a shaded mean +/- std band
    behind each line in the same colour at ``STD_BAND_ALPHA`` alpha.
    """
    pos = _k_positions()
    x = np.asarray([pos[k] for k in K_VALUES], dtype=float)

    for style in MODEL_ORDER:
        if style.key not in wide.index:
            continue
        y = wide.loc[style.key, list(K_VALUES)].to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue

        if wide_std is not None and style.key in wide_std.index:
            y_std = wide_std.loc[style.key, list(K_VALUES)].to_numpy(dtype=float)
            # Only draw the band where both mean and std are finite.
            mask = np.isfinite(y) & np.isfinite(y_std)
            if mask.any():
                lo = np.where(mask, y - y_std, np.nan)
                hi = np.where(mask, y + y_std, np.nan)
                ax.fill_between(
                    x, lo, hi,
                    color=style.colour,
                    alpha=STD_BAND_ALPHA,
                    linewidth=0,
                    zorder=1,
                )

        ax.plot(
            x, y,
            color=style.colour,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=3.8,
            lw=1.3,
            label=style.display,
            zorder=2,
        )

    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlim(x.min() - K_EDGE_PAD, x.max() + K_EDGE_PAD)
    ax.set_xlabel(r"Patches $K$")
    ax.set_ylabel(ylabel)
    # Grid intentionally disabled for the area-vs-logit diagnostic panels.
    ax.grid(False)
    ax.set_axisbelow(True)


def plot_metric(df: pd.DataFrame, metric: str, *, with_std: bool = False) -> plt.Figure:
    """Render a single-panel figure for ``metric`` (one of ``_METRICS``).

    If ``with_std`` is True, each line is accompanied by a shaded
    mean +/- 1 std band drawn from the ``*_std`` columns of the CSV.
    """
    cfg = _METRICS[metric]
    _apply_diag_style()

    wide = _per_k(df, cfg["column"])
    wide_std = _per_k(df, cfg["std_column"]) if with_std else None

    fig, ax = plt.subplots(figsize=(DIAG_FIG_WIDTH, DIAG_FIG_HEIGHT))
    _plot_metric(ax, wide, ylabel=cfg["label"], wide_std=wide_std)
    fig.tight_layout(pad=0.35)
    return fig


def _heatmap_order(df: pd.DataFrame) -> list[str]:
    """Return one shared order based on mean rank across all three metrics."""
    ranks: list[pd.Series] = []
    for cfg in _METRICS.values():
        values = pd.Series(_all_row(df, cfg["column"]), dtype=float)
        ranks.append(values.rank(ascending=False))
    average_rank = pd.concat(ranks, axis=1).mean(axis=1)
    present = [style.key for style in MODEL_ORDER if style.key in average_rank]
    return sorted(
        present,
        key=lambda method: (
            0 if method.startswith("labelmix-") else 1,
            float(average_rank.get(method, np.inf)),
            method,
        ),
    )


def plot_heatmaps(df: pd.DataFrame) -> plt.Figure:
    """Render the three diagnostic metrics as aligned paper heatmaps."""
    _apply_diag_style()
    methods = _heatmap_order(df)
    panels = (
        ("pair_acc", "Rank accuracy"),
        ("spearman", r"Spearman $\rho$"),
        ("top_acc", "Top accuracy"),
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE_COL_WIDTH, 3.55),
        gridspec_kw={"wspace": 0.18},
    )
    for panel_index, (metric, title) in enumerate(panels):
        ax = axes[panel_index]
        cfg = _METRICS[metric]
        wide = _per_k(df, cfg["column"]).reindex(methods)
        values = wide.to_numpy(dtype=float)
        image = ax.imshow(
            values, cmap="viridis", aspect="auto", interpolation="none",
        )

        ax.set_title(title, pad=4)
        ax.set_xticks(np.arange(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel(r"Patches $K$")
        ax.set_yticks(np.arange(len(methods)))
        ax.tick_params(axis="both", length=0)
        if panel_index == 0:
            ax.set_yticklabels([method_display(method) for method in methods])
            for label, method in zip(ax.get_yticklabels(), methods):
                if method.startswith("labelmix-"):
                    label.set_fontweight("bold")
        else:
            ax.set_yticklabels([])

        ax.set_xticks(np.arange(-0.5, len(K_VALUES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)

        colorbar = fig.colorbar(
            image, ax=ax, orientation="horizontal",
            fraction=0.045, pad=0.07, aspect=14,
        )
        colorbar.ax.tick_params(length=2, pad=1)
        colorbar.locator = MaxNLocator(3)
        colorbar.update_ticks()

    fig.subplots_adjust(left=0.255, right=0.99, top=0.92, bottom=0.12)
    return fig


def plot_legend(orientation: str) -> plt.Figure:
    """Render the shared model legend as a standalone figure."""
    _apply_diag_style()

    handles = _legend_handles()
    labels = [h.get_label() for h in handles]

    if orientation == "horizontal":
        figsize = (DOUBLE_COL_WIDTH, 1.35)
        ncol = 4
        handlelength = 1.4
        columnspacing = 0.9
    elif orientation == "vertical":
        figsize = (2.8, 2.6)
        ncol = 1
        handlelength = 1.5
        columnspacing = 0.8
    else:
        raise ValueError(f"Unknown legend orientation: {orientation}")

    fig = plt.figure(figsize=figsize)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=handlelength,
        columnspacing=columnspacing,
        borderaxespad=0.0,
    )
    return fig


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------


def build_metric_metadata(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    with_std: bool = False,
) -> dict:
    cfg = _METRICS[metric]
    wide = _per_k(df, cfg["column"])
    wide_std = _per_k(df, cfg["std_column"])
    all_row = _all_row(df, cfg["column"])
    all_row_std = _all_row(df, cfg["std_column"])

    per_model: list[dict] = []
    for style in MODEL_ORDER:
        if style.key not in wide.index:
            continue
        values = wide.loc[style.key, list(K_VALUES)].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        stds = (
            wide_std.loc[style.key, list(K_VALUES)].to_numpy(dtype=float)
            if style.key in wide_std.index
            else np.full(len(K_VALUES), np.nan)
        )
        per_model.append({
            "model_key": style.key,
            "display": style.display,
            "per_k": {
                int(k): (float(v) if np.isfinite(v) else None)
                for k, v in zip(K_VALUES, values)
            },
            "per_k_std": {
                int(k): (float(s) if np.isfinite(s) else None)
                for k, s in zip(K_VALUES, stds)
            },
            "aggregate_all_k": all_row.get(style.key),
            "aggregate_all_k_std": all_row_std.get(style.key),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
        })

    # Ranking of models by the k="all" aggregate (higher is better for all
    # three current metrics).
    ranked = sorted(
        (entry for entry in per_model if entry["aggregate_all_k"] is not None),
        key=lambda e: e["aggregate_all_k"],
        reverse=True,
    )

    plot_id_suffix = "_with_std" if with_std else ""
    kind = "line_per_model_with_std_band" if with_std else "line_per_model"
    caption_std = (
        " Shaded bands show mean \u00b1 1 std across 3 training seeds "
        "(42, 43, 44)."
        if with_std else ""
    )

    meta = {
        "plot_id": f"diagnostic_{cfg['filename']}{plot_id_suffix}",
        "title": cfg["title"] + (" (mean \u00b1 std)" if with_std else ""),
        "kind": kind,
        "shows_std_band": with_std,
        "panels": [{
            "axis": "single",
            "metric": cfg["label"],
            "direction": cfg["direction"],
            "yscale": "linear",
        }],
        "x": {
            "name": "K",
            "scale": "categorical_uniform",
            "values": list(K_VALUES),
        },
        "renaming": {
            "labelmix-mixed": method_display("labelmix-mixed"),
            "labelmix-pl": method_display("labelmix-pl"),
            "labelmix-sce": method_display("labelmix-sce"),
        },
        "source_csv": str(INPUT_FILE),
        "per_model_stats": per_model,
        "ranked_all_k": [
            {
                "rank": i + 1,
                "model_key": entry["model_key"],
                "display": entry["display"],
                "aggregate_all_k": entry["aggregate_all_k"],
                "aggregate_all_k_std": entry["aggregate_all_k_std"],
            }
            for i, entry in enumerate(ranked)
        ],
        "caption_hint": (
            f"{cfg['title']} on the composed ImageNet diagnostic dataset "
            "(ViT-Betwixt). Each line traces one training configuration "
            "across the composed-sample class count "
            f"K \u2208 {list(K_VALUES)}; the ``K=all'' aggregate is "
            "reported in the metadata sidecar, not plotted. ``TreemapMix'' "
            "replaces the internal name ``LabelMix''."
            + caption_std
        ),
        "file": {
            "pdf": out_path.with_suffix(".pdf").name,
            "directory": str(out_path.parent),
        },
    }
    return meta


def build_legend_metadata(out_path: Path, orientation: str) -> dict:
    return {
        "plot_id": f"diagnostic_area_logit_legend_{orientation}",
        "title": f"Area-logit diagnostic legend ({orientation})",
        "kind": "standalone_legend",
        "orientation": orientation,
        "entries": [
            {
                "model_key": style.key,
                "display": style.display,
                "colour": style.colour,
                "linestyle": style.linestyle,
                "marker": style.marker,
            }
            for style in MODEL_ORDER
        ],
        "renaming": {
            "labelmix-pl": method_display("labelmix-pl"),
            "labelmix-sce": method_display("labelmix-sce"),
        },
        "file": {
            "pdf": out_path.with_suffix(".pdf").name,
            "directory": str(out_path.parent),
        },
    }


def _write_metadata(meta: dict, out_path: Path) -> Path:
    """Write a ``<stem>.meta.json`` sidecar next to ``out_path``."""
    meta = {
        **meta,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "evaluation.plots.area_logit_diagnostic",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
        f.write("\n")
    return meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT_FILE)
    p.add_argument("--out-spearman", type=Path, default=OUTPUT_SPEARMAN)
    p.add_argument("--out-pair-acc", type=Path, default=OUTPUT_PAIR_ACC)
    p.add_argument("--out-top-acc", type=Path, default=OUTPUT_TOP_ACC)
    p.add_argument("--out-spearman-std", type=Path, default=OUTPUT_SPEARMAN_STD,
                   help="Spearman panel with mean +/- std band.")
    p.add_argument("--out-pair-acc-std", type=Path, default=OUTPUT_PAIR_ACC_STD,
                   help="Pair-acc panel with mean +/- std band.")
    p.add_argument("--out-top-acc-std", type=Path, default=OUTPUT_TOP_ACC_STD,
                   help="Top-acc panel with mean +/- std band.")
    p.add_argument("--out-legend-horizontal", type=Path, default=OUTPUT_LEGEND_HORIZONTAL)
    p.add_argument("--out-legend-vertical", type=Path, default=OUTPUT_LEGEND_VERTICAL)
    p.add_argument("--out-heatmaps", type=Path, default=OUTPUT_HEATMAPS)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if not args.input.is_file():
        _logger.error("Input CSV not found: %s", args.input)
        return 1

    _logger.info("Reading %s", args.input)
    df = _load(args.input)

    args.out_heatmaps.parent.mkdir(parents=True, exist_ok=True)
    heatmap_fig = plot_heatmaps(df)
    savefig(heatmap_fig, str(args.out_heatmaps))
    plt.close(heatmap_fig)
    _logger.info("Wrote %s", args.out_heatmaps)

    outputs = {
        "spearman": args.out_spearman,
        "pair_acc": args.out_pair_acc,
        "top_acc":  args.out_top_acc,
    }
    for metric, out_path in outputs.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plot_metric(df, metric)
        savefig(fig, str(out_path))
        meta = build_metric_metadata(df, metric, out_path)
        meta_path = _write_metadata(meta, out_path)
        _logger.info("Wrote %s", out_path)
        _logger.info("Wrote %s", meta_path)
        plt.close(fig)

    outputs_std = {
        "spearman": args.out_spearman_std,
        "pair_acc": args.out_pair_acc_std,
        "top_acc":  args.out_top_acc_std,
    }
    for metric, out_path in outputs_std.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plot_metric(df, metric, with_std=True)
        savefig(fig, str(out_path))
        meta = build_metric_metadata(df, metric, out_path, with_std=True)
        meta_path = _write_metadata(meta, out_path)
        _logger.info("Wrote %s", out_path)
        _logger.info("Wrote %s", meta_path)
        plt.close(fig)

    legends = {
        "horizontal": args.out_legend_horizontal,
        "vertical": args.out_legend_vertical,
    }
    for orientation, out_path in legends.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plot_legend(orientation)
        savefig(fig, str(out_path))
        meta = build_legend_metadata(out_path, orientation)
        meta_path = _write_metadata(meta, out_path)
        _logger.info("Wrote %s", out_path)
        _logger.info("Wrote %s", meta_path)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
