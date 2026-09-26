"""Render aligned per-architecture tables for Top-1 accuracy and ECE@15.

Both metrics are displayed as percentages.  Each cell reports cross-seed
mean +/- standard deviation followed by the percentage-point change from the
same model's baseline.  The best mean in each row is highlighted (highest
Top-1, lowest ECE).

Input:
  ``data/processed/architectures_per_experiment_short.csv``

Outputs:
  ``data/processed/figures/architectures/architectures_top1_ece_table.pdf``
  ``data/processed/figures/architectures/architectures_top1_ece_table.png``
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import BASE_FONT_SIZE, DOUBLE_COL_WIDTH, apply_paper_style, bar_color, method_display, savefig
from .architectures_comparison_table import MODEL_LABELS, TYPE_ORDER, seed_description


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "architectures_per_experiment_short.csv"
OUTPUT_FILE = FIGURES_DIR / "architectures" / "architectures_top1_ece_table.pdf"


def _ordered(values: Sequence[str], preferred: Sequence[str]) -> list[str]:
    present = list(dict.fromkeys(str(value) for value in values))
    return [value for value in preferred if value in present] + sorted(
        value for value in present if value not in preferred
    )


def _draw_metric_table(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    mean_column: str,
    std_column: str,
    scale: float,
    title: str,
    higher_is_better: bool,
    model_labels: Mapping[str, str] = MODEL_LABELS,
) -> None:
    """Draw one model-by-method metric table on ``ax``."""
    types = _ordered(df["type"], TYPE_ORDER)
    models = _ordered(df["model"], tuple(model_labels))
    means = df.pivot(index="model", columns="type", values=mean_column) * scale
    stds = df.pivot(index="model", columns="type", values=std_column) * scale

    cell_text: list[list[str]] = []
    best_columns: list[int] = []
    for model in models:
        baseline = float(means.loc[model, "baseline"]) if "baseline" in types else np.nan
        values = means.loc[model, types].to_numpy(dtype=float)
        best_columns.append(
            int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
        )
        row: list[str] = []
        for method in types:
            mean = float(means.loc[model, method])
            std = float(stds.loc[model, method])
            change = "reference" if method == "baseline" else f"{mean - baseline:+.2f} pp"
            row.append(f"{mean:.2f} ± {std:.2f}\n{change}")
        cell_text.append(row)

    ax.axis("off")
    table = ax.table(
        cellText=[[model_labels.get(model, model).replace("ConvNeXt V2 Nano", "ConvNeXt V2\nNano"), *row]
                  for model, row in zip(models, cell_text, strict=True)],
        colLabels=["Backbone", *[method_display(method).replace("TreemapMix-", "TreemapMix\n") for method in types]],
        colWidths=[0.24, *[0.76 / len(types)] * len(types)],
        cellLoc="center",
        rowLoc="right",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(BASE_FONT_SIZE)

    table[(0, 0)].set_edgecolor("white")
    table[(0, 0)].set_facecolor("#F2F2F2")
    table[(0, 0)].set_text_props(weight="bold")

    for column, method in enumerate(types):
        cell = table[(0, column + 1)]
        cell.set_facecolor(bar_color(method))
        cell.set_text_props(weight="bold")
        cell.set_edgecolor("white")

    for row, best_column in enumerate(best_columns, start=1):
        label_cell = table[(row, 0)]
        label_cell.set_facecolor("#F2F2F2")
        label_cell.set_text_props(weight="bold")
        label_cell.set_edgecolor("white")
        for column in range(len(types)):
            cell = table[(row, column + 1)]
            cell.set_edgecolor("white")
            if column == best_column:
                cell.set_facecolor("#D9F2D9")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#FAFAFA" if row % 2 else "#F3F3F3")

    ax.set_title(title, pad=2, weight="bold")


def plot(
    df: pd.DataFrame,
    *,
    title: str = "ImageNet-1K results by architecture",
    model_labels: Mapping[str, str] = MODEL_LABELS,
) -> plt.Figure:
    """Render the combined Top-1 and ECE comparison figure."""
    required = {
        "model",
        "type",
        "top1_acc_mean",
        "top1_acc_std",
        "ece/n_bins=15_mean",
        "ece/n_bins=15_std",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL_WIDTH, 4.1))
    _draw_metric_table(
        axes[0],
        df,
        mean_column="top1_acc_mean",
        std_column="top1_acc_std",
        scale=1.0,
        title="Top-1 accuracy (%) — higher is better",
        higher_is_better=True,
        model_labels=model_labels,
    )
    _draw_metric_table(
        axes[1],
        df,
        mean_column="ece/n_bins=15_mean",
        std_column="ece/n_bins=15_std",
        scale=100.0,
        title="Expected calibration error, 15 bins (%) — lower is better",
        higher_is_better=False,
        model_labels=model_labels,
    )
    fig.suptitle(title, y=0.995, weight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.12, hspace=0.27)
    fig.text(
        0.5,
        0.018,
        f"Mean ± sample std over {seed_description(df)}.\n"
        "Second line: change from Mixup+CutMix (pp); green: best mean.",
        ha="center",
        va="bottom",
        fontsize=BASE_FONT_SIZE,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    _logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input)
    fig = plot(df)
    savefig(fig, str(args.output))
    png_path = args.output.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    _logger.info("Wrote %s and %s", args.output.with_suffix(".pdf"), png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
