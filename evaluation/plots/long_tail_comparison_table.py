"""Render a per-architecture Top-1 comparison table for ImageNet-LT.

Input:
  ``data/processed/long_tail_per_experiment_short.csv``

Outputs:
  ``data/processed/figures/long_tail/long_tail_top1_table.pdf``
  ``data/processed/figures/long_tail/long_tail_top1_table.png``
  ``data/processed/figures/long_tail/long_tail_top1_ece_table.pdf``
  ``data/processed/figures/long_tail/long_tail_top1_ece_table.png``
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import savefig
from .architectures_comparison_table import plot as comparison_plot
from .architectures_top1_ece_table import plot as top1_ece_plot


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "long_tail_per_experiment_short.csv"
OUTPUT_FILE = FIGURES_DIR / "long_tail" / "long_tail_top1_table.pdf"
ECE_OUTPUT_FILE = FIGURES_DIR / "long_tail" / "long_tail_top1_ece_table.pdf"

# Insertion order determines the rows, from the smallest to largest variant.
MODEL_LABELS = {
    "vit_wee_patch16_reg1_gap_256": "ViT-Wee",
    "vit_little_patch16_reg4_gap_256": "ViT-Little",
    "vit_medium_patch16_reg1_gap_256": "ViT-Medium",
    "vit_betwixt_patch16_reg4_gap_256": "ViT-Betwixt",
}


def plot(df: pd.DataFrame) -> plt.Figure:
    """Render the ImageNet-LT comparison table without saving it."""
    return comparison_plot(
        df,
        title="ImageNet-LT Top-1 accuracy by architecture",
        model_labels=MODEL_LABELS,
    )


def plot_top1_ece(df: pd.DataFrame) -> plt.Figure:
    """Render aligned Top-1 and ECE@15 comparison tables."""
    return top1_ece_plot(
        df,
        title="ImageNet-LT results by architecture",
        model_labels=MODEL_LABELS,
    )


def _save_pdf_and_png(fig: plt.Figure, output: Path) -> None:
    savefig(fig, str(output))
    png_path = output.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    _logger.info("Wrote %s and %s", output.with_suffix(".pdf"), png_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--ece-output", type=Path, default=ECE_OUTPUT_FILE)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    _logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input)
    _save_pdf_and_png(plot(df), args.output)
    _save_pdf_and_png(plot_top1_ece(df), args.ece_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
