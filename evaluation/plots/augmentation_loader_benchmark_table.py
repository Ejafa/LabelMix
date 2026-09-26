"""Render a highlighted summary table for the augmentation-loader benchmark.

Input:
  ``data/processed/augmentation_loader_benchmark.csv``

Outputs:
  ``data/processed/figures/runtime/augmentation_loader_benchmark_table.pdf``
  ``data/processed/figures/runtime/augmentation_loader_benchmark_table.png``
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import BASE_FONT_SIZE, DOUBLE_COL_WIDTH, apply_paper_style, method_display, savefig


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "augmentation_loader_benchmark.csv"
OUTPUT_FILE = FIGURES_DIR / "runtime" / "augmentation_loader_benchmark_table.pdf"

CONFIGURATION_ORDER = (
    "no_augmentation",
    "single_image",
    "mixup_cutmix",
    "treemapmix_sce",
    "treemapmix_pl",
    "mosaic",
)

CONFIGURATION_LABELS = {
    "no_augmentation": method_display("noaug"),
    "single_image": method_display("bare"),
    "mixup_cutmix": method_display("baseline"),
    "treemapmix_sce": method_display("labelmix-sce") + "\n(K=4)",
    "treemapmix_pl": method_display("labelmix-pl") + "\n(K=6)",
    "mosaic": "Mosaic (recorded)",
}

# Stored memory values are MiB, converted to GiB for publication.
DISPLAY_METRICS = (
    ("throughput_images_per_second", "Throughput\n(images/s)", 1, True),
    ("peak_rss_mb", "Peak RSS\n(GiB)", 2, False),
    ("peak_uss_mb", "Peak USS\n(GiB)", 2, False),
)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    rank = {name: index for index, name in enumerate(CONFIGURATION_ORDER)}
    return (
        df.assign(_rank=df["configuration"].map(rank).fillna(len(rank)))
        .sort_values(["_rank", "configuration"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def _format(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _common_integer(df: pd.DataFrame, column: str) -> int:
    """Return a benchmark setting shared by every aggregated row."""
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")
    values = df[column].dropna().astype(int).unique()
    if len(values) != 1:
        raise ValueError(f"Expected one common {column}, found {values.tolist()}")
    return int(values[0])


def plot(df: pd.DataFrame) -> plt.Figure:
    """Render the benchmark table without saving it."""
    required = {"configuration", "runs", "seeds_present"}
    for metric, _, _, _ in DISPLAY_METRICS:
        required.update((f"{metric}_mean", f"{metric}_std"))
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    apply_paper_style()
    data = _ordered(df)
    warmup_batches = _common_integer(data, "warmup_batches")
    measured_batches = _common_integer(data, "measured_batches")
    loader_batch_size = _common_integer(data, "batch_size")
    workers = _common_integer(data, "workers")
    runs = _common_integer(data, "runs")
    reference = data.loc[data["configuration"] == "single_image", "throughput_images_per_second_mean"]
    if len(reference) != 1 or reference.iloc[0] <= 0:
        raise ValueError("Expected one positive single-image throughput reference")
    reduction = 100.0 * (1.0 - data["throughput_images_per_second_mean"] / reference.iloc[0])
    cell_text = []
    for position, (_, row) in enumerate(data.iterrows()):
        values = [CONFIGURATION_LABELS.get(row["configuration"], row["configuration"])]
        for metric, _, decimals, _ in DISPLAY_METRICS:
            scale = 1024.0 if metric.startswith("peak_") else 1.0
            values.append(_format(row[f"{metric}_mean"] / scale, row[f"{metric}_std"] / scale, decimals))
        values.insert(2, f"{reduction.iloc[position]:+.1f}%")
        cell_text.append(values)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.75))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Method", DISPLAY_METRICS[0][1], "Throughput\nreduction", DISPLAY_METRICS[1][1], DISPLAY_METRICS[2][1]],
        colWidths=[0.28, 0.21, 0.17, 0.17, 0.17],
        cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(BASE_FONT_SIZE)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#DCE6F1" if row == 0 else ("#FAFAFA" if row % 2 else "#F2F2F2"))
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.20)
    fig.text(
        0.5, 0.025,
        f"{runs} runs; batch {loader_batch_size}; {workers} workers; "
        f"{warmup_batches} warm-up + {measured_batches} measured batches.\n"
        "Mean ± sample std; throughput reduction relative to single-image recipe.",
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
    fig = plot(pd.read_csv(args.input))
    savefig(fig, str(args.output))
    png_path = args.output.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    _logger.info("Wrote %s and %s", args.output.with_suffix(".pdf"), png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
