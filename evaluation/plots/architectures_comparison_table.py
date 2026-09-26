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
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import (
    apply_paper_style,
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
    """Render the Top-1 panel using the shared publication-size table layout."""
    from ._style import BASE_FONT_SIZE, DOUBLE_COL_WIDTH
    from .architectures_top1_ece_table import _draw_metric_table

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.25))
    _draw_metric_table(
        ax, df, mean_column="top1_acc_mean", std_column="top1_acc_std",
        scale=1.0, title=title, higher_is_better=True, model_labels=model_labels,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.21)
    fig.text(
        0.5, 0.025,
        f"Mean ± sample std over {seed_description(df)}.\n"
        "Second line: change from Mixup+CutMix (pp); green: best mean.",
        ha="center", va="bottom", fontsize=BASE_FONT_SIZE,
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
