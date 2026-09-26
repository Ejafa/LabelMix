"""Render a per-architecture Top-1 comparison table.

Each row is a model and each method column reports cross-seed mean +/- standard
deviation.  The second line is the percentage-point change from that model's
baseline, and the highest mean in every row is highlighted.

Input:
  ``data/processed/architectures_per_experiment_short.csv``

Outputs:
  ``data/processed/figures/architectures/architectures_top1_table.pdf``
  ``data/processed/figures/architectures/architectures_top1_table.png``
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
from ._style import (
    apply_paper_style,
    bar_color,
    method_display,
    savefig,
)


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "architectures_per_experiment_short.csv"
OUTPUT_FILE = FIGURES_DIR / "architectures" / "architectures_top1_table.pdf"

TYPE_ORDER: Sequence[str] = (
    "baseline",
    "mosaic",
    "labelmix-sce",
    "labelmix-pl",
)

MODEL_LABELS = {
    "convnextv2_nano": "ConvNeXt V2 Nano",
    "deit_tiny_patch16_224": "DeiT Tiny",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv4_hybrid_medium": "MobileNetV4 Hybrid M",
    "swin_tiny_patch4_window7_224": "Swin Tiny",
}


def _ordered_types(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["type"].astype(str)))
    return [t for t in TYPE_ORDER if t in present] + sorted(
        t for t in present if t not in TYPE_ORDER
    )


def _ordered_models(
    df: pd.DataFrame,
    model_labels: Mapping[str, str] = MODEL_LABELS,
) -> list[str]:
    present = list(dict.fromkeys(df["model"].astype(str)))
    known = [m for m in model_labels if m in present]
    return known + sorted(m for m in present if m not in model_labels)


def seed_description(df: pd.DataFrame) -> str:
    """Describe the common seed set recorded by the aggregation pipeline."""
    if "seeds_present" not in df.columns:
        return "available seeds"
    seed_sets = {
        tuple(part.strip() for part in str(value).split(",") if part.strip())
        for value in df["seeds_present"].dropna()
    }
    if len(seed_sets) != 1:
        return "available complete seeds"
    seeds = next(iter(seed_sets))
    if not seeds:
        return "available seeds"
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    if len(seeds) == 2:
        return f"seeds {seeds[0]} and {seeds[1]}"
    return f"seeds {', '.join(seeds[:-1])}, and {seeds[-1]}"


def plot(
    df: pd.DataFrame,
    *,
    title: str = "ImageNet-1K Top-1 accuracy by architecture",
    model_labels: Mapping[str, str] = MODEL_LABELS,
) -> plt.Figure:
    """Render the comparison table without saving it."""
    required = {"model", "type", "top1_acc_mean", "top1_acc_std"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    apply_paper_style()
    types = _ordered_types(df)
    models = _ordered_models(df, model_labels)
    means = df.pivot(index="model", columns="type", values="top1_acc_mean")
    stds = df.pivot(index="model", columns="type", values="top1_acc_std")

    cell_text: list[list[str]] = []
    best_columns: list[int] = []
    for model in models:
        baseline = float(means.loc[model, "baseline"]) if "baseline" in types else np.nan
        values = means.loc[model, types].to_numpy(dtype=float)
        best_columns.append(int(np.nanargmax(values)))
        row: list[str] = []
        for method in types:
            mean = float(means.loc[model, method])
            std = float(stds.loc[model, method])
            if method == "baseline" or not np.isfinite(baseline):
                comparison = "reference"
            else:
                comparison = f"{mean - baseline:+.2f} pp"
            row.append(f"{mean:.2f} ± {std:.2f}\n{comparison}")
        cell_text.append(row)

    row_labels = [model_labels.get(model, model) for model in models]
    col_labels = [method_display(method) for method in types]
    # This is a standalone comparison artifact rather than a paper-column
    # panel, so allow enough width for full architecture names.
    fig, ax = plt.subplots(figsize=(7.5, 2.75))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        colLoc="center",
        bbox=[0.21, 0.15, 0.78, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    # Header colours follow the project-wide method palette.
    for column, method in enumerate(types):
        cell = table[(0, column)]
        cell.set_facecolor(bar_color(method))
        cell.set_text_props(weight="bold")
        cell.set_edgecolor("white")

    # Use quiet grid lines, a neutral model-label column, and a distinct best
    # cell per model. Row 0 is the header, hence the +1 offset.
    for row, best_column in enumerate(best_columns, start=1):
        label_cell = table[(row, -1)]
        label_cell.set_facecolor("#F2F2F2")
        label_cell.set_text_props(weight="bold")
        label_cell.set_edgecolor("white")
        for column in range(len(types)):
            cell = table[(row, column)]
            cell.set_edgecolor("white")
            if column == best_column:
                cell.set_facecolor("#D9F2D9")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#FAFAFA" if row % 2 else "#F3F3F3")

    ax.set_title(title, pad=4, weight="bold")
    fig.text(
        0.5,
        0.035,
        f"Mean ± std over {seed_description(df)}; second line is change from baseline.\n"
        "Green marks the highest mean per model (higher is better).",
        ha="center",
        va="bottom",
        fontsize=6.5,
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
