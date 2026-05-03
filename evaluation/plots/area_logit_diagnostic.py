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
 * x-axis = composed-sample class count ``k`` (categorical-uniform
   spacing, values ``3, 4, 5, 6``),
 * y-axis = metric value,
 * one line-with-markers per training config.

The ``k == "all"`` aggregate row is not drawn on the canvas (it would
clutter an already eight-line plot) but is captured in the ``.meta.json``
sidecar so downstream tooling can quote headline numbers without
re-reading the CSV.

Styling is inherited from :mod:`evaluation.plots._style` (paper rcParams,
categorical x-axis, dotted horizontal grid, ``.pdf`` + ``.png`` siblings
via :func:`._style.savefig`).  The shared legend is rendered separately in
horizontal and vertical variants so the three metric panels can be reused
without repeating the same legend.

Naming note: the ``labelmix-*`` runs in the source CSV are rendered as
``TreemapMix (...)`` in every user-facing string (legend labels, titles,
captions, metadata).  Internal CSV keys are untouched.

Input   : ``data/processed/diagnostic_area_logit_metrics.csv``
Output  : ``data/processed/figures/diagnostic_area_logit/area_logit_spearman.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_pair_acc.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_top_acc.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_legend_horizontal.pdf``
          ``data/processed/figures/diagnostic_area_logit/area_logit_legend_vertical.pdf``
          (+ ``.png`` and ``.meta.json`` siblings for each)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import apply_paper_style, savefig


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_FILE = PROCESSED_DIR / "diagnostic_area_logit_metrics.csv"
DIAG_FIGURES_DIR = FIGURES_DIR / "diagnostic_area_logit"
OUTPUT_SPEARMAN = DIAG_FIGURES_DIR / "area_logit_spearman.pdf"
OUTPUT_PAIR_ACC = DIAG_FIGURES_DIR / "area_logit_pair_acc.pdf"
OUTPUT_TOP_ACC = DIAG_FIGURES_DIR / "area_logit_top_acc.pdf"
OUTPUT_LEGEND_HORIZONTAL = DIAG_FIGURES_DIR / "area_logit_legend_horizontal.pdf"
OUTPUT_LEGEND_VERTICAL = DIAG_FIGURES_DIR / "area_logit_legend_vertical.pdf"

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
    ModelStyle("noaug",        "No augmentation",           "#404040", "--", "s"),
    ModelStyle("bare",         "Single image aug",          "#8c8c8c", "-",  "s"),
    ModelStyle("baseline",     "Baseline (Mixup+CutMix)",   "#000000", "-",  "o"),
    ModelStyle("cutmix",       "CutMix only",               "#ff7f0e", "-",  "^"),
    ModelStyle("mixup",        "MixUp only",                "#9467bd", "-",  "v"),
    ModelStyle("mosaic",       "RICAP",                    "#2ca02c", "-",  "D"),
    ModelStyle("labelmix-sce", "TreemapMix (SCE)",          "#d62728", "-",  "o"),
    ModelStyle("labelmix-pl",  "TreemapMix (PL)",           "#1f77b4", "-",  "o"),
)

K_VALUES: Sequence[int] = (3, 4, 5, 6)
DIAG_FIG_WIDTH = 4.6
K_DISPLAY_SPACING = 0.55
K_EDGE_PAD = 0.10

DIAG_RC = {
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
}


def _apply_diag_style() -> None:
    apply_paper_style()
    plt.rcParams.update(DIAG_RC)


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


# metric key -> (csv column, pretty label, direction, y-pad fraction)
_METRICS = {
    "spearman": {
        "column": "spearman_mean",
        "label": r"Mean Spearman $\rho$ ($\uparrow$)",
        "title": "Area-logit Spearman correlation",
        "direction": "higher is better",
        "filename": "area_logit_spearman",
        "out_path": OUTPUT_SPEARMAN,
    },
    "pair_acc": {
        "column": "pair_acc",
        "label": r"Pair ranking accuracy ($\uparrow$)",
        "title": "Pairwise area-vs-logit ranking accuracy",
        "direction": "higher is better",
        "filename": "area_logit_pair_acc",
        "out_path": OUTPUT_PAIR_ACC,
    },
    "top_acc": {
        "column": "top_acc",
        "label": r"Largest area $=$ top logit ($\uparrow$)",
        "title": "Largest-area top-logit accuracy",
        "direction": "higher is better",
        "filename": "area_logit_top_acc",
        "out_path": OUTPUT_TOP_ACC,
    },
}

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
    """Return a (model, k) wide matrix restricted to k in ``K_VALUES``."""
    sub = df[df["k"].isin([str(k) for k in K_VALUES])].copy()
    sub["k_int"] = sub["k"].astype(int)
    wide = sub.pivot_table(index="model", columns="k_int", values=column, aggfunc="mean")
    wide = wide.reindex(columns=list(K_VALUES))
    return wide


def _all_row(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Return {model: value} for the ``k == "all"`` aggregate row."""
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
) -> None:
    """Draw one metric panel (x = k, one line per model)."""
    pos = _k_positions()
    x = np.asarray([pos[k] for k in K_VALUES], dtype=float)

    for style in MODEL_ORDER:
        if style.key not in wide.index:
            continue
        y = wide.loc[style.key, list(K_VALUES)].to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue
        ax.plot(
            x, y,
            color=style.colour,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=3.8,
            lw=1.3,
            label=style.display,
        )

    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlim(x.min() - K_EDGE_PAD, x.max() + K_EDGE_PAD)
    ax.set_xlabel(r"Number of patches $k$ per image")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def plot_metric(df: pd.DataFrame, metric: str) -> plt.Figure:
    """Render a single-panel figure for ``metric`` (one of ``_METRICS``)."""
    cfg = _METRICS[metric]
    _apply_diag_style()

    wide = _per_k(df, cfg["column"])

    fig, ax = plt.subplots(figsize=(DIAG_FIG_WIDTH, 3.6))
    _plot_metric(ax, wide, ylabel=cfg["label"])
    fig.tight_layout()
    return fig


def plot_legend(orientation: str) -> plt.Figure:
    """Render the shared model legend as a standalone figure."""
    _apply_diag_style()

    handles = _legend_handles()
    labels = [h.get_label() for h in handles]

    if orientation == "horizontal":
        figsize = (5.2, 1.3)
        ncol = int(np.ceil(len(handles) / 2))
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


def build_metric_metadata(df: pd.DataFrame, metric: str, out_path: Path) -> dict:
    cfg = _METRICS[metric]
    wide = _per_k(df, cfg["column"])
    all_row = _all_row(df, cfg["column"])

    per_model: list[dict] = []
    for style in MODEL_ORDER:
        if style.key not in wide.index:
            continue
        values = wide.loc[style.key, list(K_VALUES)].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        per_model.append({
            "model_key": style.key,
            "display": style.display,
            "per_k": {
                int(k): (float(v) if np.isfinite(v) else None)
                for k, v in zip(K_VALUES, values)
            },
            "aggregate_all_k": all_row.get(style.key),
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

    meta = {
        "plot_id": f"diagnostic_{cfg['filename']}",
        "title": cfg["title"],
        "kind": "line_per_model",
        "panels": [{
            "axis": "single",
            "metric": cfg["label"],
            "direction": cfg["direction"],
            "yscale": "linear",
        }],
        "x": {
            "name": "k",
            "scale": "categorical_uniform",
            "values": list(K_VALUES),
        },
        "renaming": {
            "labelmix-pl": "TreemapMix (PL)",
            "labelmix-sce": "TreemapMix (SCE)",
        },
        "source_csv": str(INPUT_FILE),
        "per_model_stats": per_model,
        "ranked_all_k": [
            {
                "rank": i + 1,
                "model_key": entry["model_key"],
                "display": entry["display"],
                "aggregate_all_k": entry["aggregate_all_k"],
            }
            for i, entry in enumerate(ranked)
        ],
        "caption_hint": (
            f"{cfg['title']} on the composed ImageNet diagnostic dataset "
            "(ViT-Betwixt, seed 42). Each line traces one training "
            "configuration across the composed-sample class count "
            f"k \u2208 {list(K_VALUES)}; the ``k=all'' aggregate is "
            "reported in the metadata sidecar, not plotted. ``TreemapMix'' "
            "replaces the internal name ``LabelMix''."
        ),
        "file": {
            "pdf": out_path.with_suffix(".pdf").name,
            "png": out_path.with_suffix(".png").name,
            "directory": str(out_path.parent),
        },
    }
    return meta


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
            "labelmix-pl": "TreemapMix (PL)",
            "labelmix-sce": "TreemapMix (SCE)",
        },
        "file": {
            "pdf": out_path.with_suffix(".pdf").name,
            "png": out_path.with_suffix(".png").name,
            "directory": str(out_path.parent),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT_FILE)
    p.add_argument("--out-spearman", type=Path, default=OUTPUT_SPEARMAN)
    p.add_argument("--out-pair-acc", type=Path, default=OUTPUT_PAIR_ACC)
    p.add_argument("--out-top-acc", type=Path, default=OUTPUT_TOP_ACC)
    p.add_argument("--out-legend-horizontal", type=Path, default=OUTPUT_LEGEND_HORIZONTAL)
    p.add_argument("--out-legend-vertical", type=Path, default=OUTPUT_LEGEND_VERTICAL)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if not args.input.is_file():
        _logger.error("Input CSV not found: %s", args.input)
        return 1

    _logger.info("Reading %s", args.input)
    df = _load(args.input)

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
