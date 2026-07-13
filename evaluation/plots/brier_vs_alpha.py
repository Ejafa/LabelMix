"""Plot: Brier score as a function of the mixed-alpha coefficient.

Scans LabelMix runs indexed by their ``ma`` (``labelmix_mixed_alpha``)
hyper-parameter and plots Brier (lower is better) with seed CIs.

Input   : ``data/processed/brier_vs_alpha.csv``
          Required columns: ``ma``, ``brier_mean``, ``brier_std``, ``n_seeds``,
          ``model`` (series key).
Output  : ``data/processed/figures/brier_vs_alpha.pdf``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ..processing.aggregate import confidence_interval_95
from ._style import (
    SINGLE_COL_WIDTH,
    apply_paper_style,
    line_color,
    method_display,
    savefig,
)


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "brier_vs_alpha.csv"
OUTPUT_FILE = FIGURES_DIR / "brier_vs_alpha.pdf"


def plot(df: pd.DataFrame) -> plt.Figure:
    apply_paper_style()
    fig, ax = plt.subplots(
        figsize=(SINGLE_COL_WIDTH, 2.4 * (SINGLE_COL_WIDTH / 3.3))
    )

    df = df.copy()
    df["ma"] = pd.to_numeric(df["ma"], errors="coerce")
    df = df.dropna(subset=["ma"]).sort_values("ma")

    for model, sub in df.groupby("model"):
        ci = confidence_interval_95(sub["brier_std"], sub["n_seeds"])
        ax.errorbar(
            sub["ma"].values,
            sub["brier_mean"].values,
            yerr=ci.values,
            marker="o",
            capsize=2,
            color=line_color(model),
            label=method_display(model),
        )

    ax.set_xlabel(r"Mixed-alpha coefficient $\alpha_{\mathrm{mix}}$")
    ax.set_ylabel("Brier score (lower is better)")
    fig.tight_layout()
    return fig


def plot_legend(models: list[str]) -> plt.Figure:
    apply_paper_style()
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=line_color(model), marker="o",
               label=method_display(model))
        for model in models
    ]
    fig = plt.figure(figsize=(SINGLE_COL_WIDTH, 0.55 * (SINGLE_COL_WIDTH / 3.3)))
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center",
        ncol=min(len(handles), 2),
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
    leg_fig = plot_legend(sorted(df["model"].unique()))
    savefig(leg_fig, str(leg_path))
    plt.close(leg_fig)
    _logger.info("Wrote %s", leg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
