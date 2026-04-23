"""Plot: Top-1 accuracy — LabelMix vs baseline, per model.

Produces a grouped bar chart with one group per model (e.g. ResNet-50,
ResNet-101) and one bar per method (baseline vs LabelMix variants).  Error
bars show the 95% CI over seeds.

Input   : ``data/processed/accuracy_by_model_method.csv``
          Required columns: ``model``, ``method``, ``top1_acc_mean``,
          ``top1_acc_std``, ``n_seeds``.
Output  : ``data/processed/figures/accuracy_vs_baseline.pdf`` (and ``.png``).
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
from ._style import DOUBLE_COL_WIDTH, apply_paper_style, savefig


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "accuracy_by_model_method.csv"
OUTPUT_FILE = FIGURES_DIR / "accuracy_vs_baseline.pdf"


def plot(df: pd.DataFrame) -> plt.Figure:
    """Render the grouped bar chart.  Does not save the figure."""
    apply_paper_style()

    models = sorted(df["model"].unique())
    methods = sorted(df["method"].unique())

    x = np.arange(len(models))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.6))

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("model").reindex(models)
        offset = (i - (len(methods) - 1) / 2) * width
        ci = confidence_interval_95(sub["top1_acc_std"], sub["n_seeds"])
        ax.bar(
            x + offset,
            sub["top1_acc_mean"].values,
            width=width,
            yerr=ci.values,
            capsize=2,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xlabel("Model")
    ax.legend(ncol=min(len(methods), 3), loc="lower right")
    fig.tight_layout()
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
    _logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
