"""Plot: ECE bars — LabelMix vs baseline, per model.

Shows calibration error (lower is better) with one bar per method inside
each model group.  Reads ECE at ``n_bins=15`` by default.

Input   : ``data/processed/calibration_by_model_method.csv``
          Required columns: ``model``, ``method``, ``ece/n_bins=15_mean``,
          ``ece/n_bins=15_std``, ``n_seeds``.
Output  : ``data/processed/figures/calibration_ece_bars.pdf``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ..processing.aggregate import confidence_interval_95
from ._style import (
    DOUBLE_COL_WIDTH,
    apply_paper_style,
    bar_color,
    method_display,
    savefig,
)


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "calibration_by_model_method.csv"
OUTPUT_FILE = FIGURES_DIR / "calibration_ece_bars.pdf"

_MEAN_COL = "ece/n_bins=15_mean"
_STD_COL = "ece/n_bins=15_std"


def plot(df: pd.DataFrame) -> plt.Figure:
    apply_paper_style()

    models = sorted(df["model"].unique())
    methods = sorted(df["method"].unique())
    x = np.arange(len(models))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, 2.6 * (DOUBLE_COL_WIDTH / 6.8))
    )

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("model").reindex(models)
        offset = (i - (len(methods) - 1) / 2) * width
        ci = confidence_interval_95(sub[_STD_COL], sub["n_seeds"])
        ax.bar(
            x + offset,
            sub[_MEAN_COL].values,
            width=width,
            yerr=ci.values,
            capsize=2,
            color=bar_color(method),
            edgecolor="black",
            linewidth=0.5,
            label=method_display(method),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("ECE ($n_{bins}=15$, lower is better)")
    ax.set_xlabel("Model")
    fig.tight_layout()
    return fig


def plot_legend(methods: list[str]) -> plt.Figure:
    apply_paper_style()
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=bar_color(method), edgecolor="black", linewidth=0.5,
              label=method_display(method))
        for method in methods
    ]
    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 0.55 * (DOUBLE_COL_WIDTH / 6.8)))
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center",
        ncol=min(len(handles), 3),
        frameon=False,
        borderaxespad=0.0,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT_FILE)
    p.add_argument("--output", type=Path, default=OUTPUT_FILE)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    _logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input)
    fig = plot(df)
    savefig(fig, str(args.output))
    plt.close(fig)
    _logger.info("Wrote %s", args.output)
    leg_path = args.output.with_name(args.output.stem + "_legend.pdf")
    leg_fig = plot_legend(sorted(df["method"].unique()))
    savefig(leg_fig, str(leg_path))
    plt.close(leg_fig)
    _logger.info("Wrote %s", leg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
