"""Plot: per-experiment NLL + Brier score bars for ImageNet-1k.

Companion to :mod:`evaluation.plots.in1k_per_experiment`. Same grouped
bar-chart layout, same palette, same standalone-legend export, but
the two panels show proper-scoring-rule losses instead of accuracy / ECE:

* **Top row** -- Negative log-likelihood (NLL), lower is better.
* **Bottom row** -- Brier score, lower is better.

Only the *short*-horizon CSV is rendered.

Input (default run, ``--input`` overrides):
  - ``data/processed/in1k_per_experiment_short.csv``

Outputs:
  - ``figures/per_experiment/in1k_per_experiment_nll_brier_short.pdf``
  - ``figures/per_experiment/in1k_per_experiment_nll_brier_short_legend.pdf``
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ..processing.aggregate import confidence_interval_95
from ._style import DOUBLE_COL_WIDTH, apply_paper_style, savefig
from .in1k_per_experiment import (
    _OUT_SUBDIR,
    _MODEL_LABELS,
    _TYPE_ORDER,
    _filter_types,
    _ordered,
    _style_for,
    plot_legend,
)


_logger = logging.getLogger(__name__)

_PANEL_WIDTH = DOUBLE_COL_WIDTH
_PANEL_HEIGHT = 3.8 * (DOUBLE_COL_WIDTH / (6.8 * 1.15))

_NLL_MEAN_COL = "nll_mean"
_NLL_STD_COL = "nll_std"
_BRIER_MEAN_COL = "brier_mean"
_BRIER_STD_COL = "brier_std"


# ---------------------------------------------------------------------------
# Main panel plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, title: str | None = "ImageNet-1k") -> plt.Figure:
    """Render the 2-row (NLL, Brier) grouped bar chart."""
    apply_paper_style()

    df = _filter_types(df)
    models = _ordered(df["model"].unique(), list(_MODEL_LABELS))
    types = _ordered(df["type"].unique(), _TYPE_ORDER)

    x = np.arange(len(models))
    width = 0.82 / max(len(types), 1)

    # Same LabelMix / non-LabelMix grouping gap as the ECE figure so the
    # two panels can be placed side-by-side in the paper.
    def _is_labelmix(t: str) -> bool:
        return t.startswith("labelmix")

    gap = 0.5 * width
    positions: list[float] = []
    seen_labelmix = False
    for idx, t in enumerate(types):
        pos = float(idx)
        if _is_labelmix(t):
            if not seen_labelmix:
                seen_labelmix = True
            pos += gap / width
        positions.append(pos)

    centre = (positions[0] + positions[-1]) / 2.0
    offsets = [(p - centre) * width for p in positions]

    fig, (ax_nll, ax_brier) = plt.subplots(
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

        # NLL (natural units, lower is better).
        nll_ci = confidence_interval_95(sub[_NLL_STD_COL], sub["n_seeds"])
        ax_nll.bar(
            x + offset,
            sub[_NLL_MEAN_COL].values,
            width=width,
            yerr=nll_ci.values,
            capsize=3.0,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            error_kw={"elinewidth": 0.4, "capthick": 0.4, "ecolor": "black"},
            zorder=2,
        )

        # Brier score (lower is better).
        brier_ci = confidence_interval_95(sub[_BRIER_STD_COL], sub["n_seeds"])
        ax_brier.bar(
            x + offset,
            sub[_BRIER_MEAN_COL].values,
            width=width,
            yerr=brier_ci.values,
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
    for ax in (ax_nll, ax_brier):
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        # ``sharex=True`` hides upper tick labels by default; these stacked
        # panels are often placed one after another, so each panel repeats the
        # model names without repeating an extra x-axis label.
        ax.tick_params(axis="x", labelbottom=True)

    ax_nll.set_ylabel("NLL")
    ax_brier.set_ylabel("Brier score")

    # Auto-zoom both panels so small but meaningful gaps stay readable.
    def _zoom(ax: plt.Axes, values: pd.Series, pad_frac: float = 0.05) -> None:
        vals = values.dropna()
        if vals.empty:
            return
        lo = float(vals.min())
        hi = float(vals.max())
        span = max(hi - lo, 1e-6)
        ax.set_ylim(max(0.0, lo - pad_frac * span), hi + pad_frac * span)

    _zoom(ax_nll, df[_NLL_MEAN_COL])
    _zoom(ax_brier, df[_BRIER_MEAN_COL])

    for ax in (ax_nll, ax_brier):
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.8", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    if title:
        ax_nll.set_title(title, pad=6)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_INPUT: tuple[str, Path, Path] = (
    "ImageNet-1k",
    PROCESSED_DIR / "in1k_per_experiment_short.csv",
    FIGURES_DIR / _OUT_SUBDIR / "in1k_per_experiment_nll_brier_short.pdf",
)


def _render_one(title: str, in_path: Path, out_path: Path) -> int:
    if not in_path.is_file():
        _logger.warning("Skipping %s (no such file)", in_path)
        return 0
    _logger.info("Reading %s", in_path)
    df = pd.read_csv(in_path)
    if df.empty:
        _logger.info("Skipping %s (empty CSV)", in_path)
        return 0

    if title == "ImageNet-1k" and "dataset" in df.columns:
        datasets = " ".join(str(v).lower() for v in df["dataset"].dropna().unique())
        if "cifar100" in datasets:
            title = "CIFAR100"

    fig = plot(df, title=title)
    savefig(fig, str(out_path))
    plt.close(fig)
    _logger.info("Wrote %s", out_path)

    types = _ordered(_filter_types(df)["type"].unique(), _TYPE_ORDER)
    leg_fig = plot_legend(types)
    leg_path = out_path.with_name(out_path.stem + "_legend.pdf")
    savefig(leg_fig, str(leg_path))
    plt.close(leg_fig)
    _logger.info("Wrote %s", leg_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=None,
        help=("Aggregated per-experiment CSV. If omitted, the short-horizon "
              "in1k CSV is rendered. The long-horizon variant is "
              "intentionally not produced here."),
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output PDF path for the panel figure. "
             "Only honoured when --input is given.",
    )
    p.add_argument("--title", type=str, default="ImageNet-1k",
                   help="Figure title for the top panel.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if args.input is None:
        title, in_path, out_path = _DEFAULT_INPUT
        return _render_one(title, in_path, out_path)

    out_path = args.output
    if out_path is None:
        out_path = FIGURES_DIR / _OUT_SUBDIR / (args.input.stem + "_nll_brier.pdf")
    return _render_one(args.title, args.input, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
