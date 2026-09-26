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
import numpy as np
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from ._style import apply_paper_style, savefig


_logger = logging.getLogger(__name__)

INPUT_FILE = PROCESSED_DIR / "augmentation_loader_benchmark.csv"
OUTPUT_FILE = FIGURES_DIR / "runtime" / "augmentation_loader_benchmark_table.pdf"

BASE_RECIPE_LABEL = "ViT-Little base recipe"
GLOBAL_TRAINING_BATCH_SIZE = 1024

CONFIGURATION_ORDER = (
    "no_augmentation",
    "single_image",
    "mixup_cutmix",
    "treemapmix_sce",
    "treemapmix_pl",
)

CONFIGURATION_LABELS = {
    "no_augmentation": "No augmentation",
    "single_image": "Single-image recipe",
    "mixup_cutmix": "Mixup + CutMix",
    "treemapmix_sce": "TreemapMix-SCE (K=4)",
    "treemapmix_pl": "TreemapMix-PL (K=6)",
}

# key, heading, decimals, whether a larger value is preferable
DISPLAY_METRICS = (
    ("throughput_images_per_second", "Throughput\n(images/s) ↑", 0, True),
    ("first_batch_seconds", "First batch\n(s) ↓", 2, False),
    ("total_batch_processing_seconds", "Measured batches\n(s) ↓", 2, False),
    ("latency_p95_ms", "p95 latency\n(ms) ↓", 1, False),
    ("peak_uss_mb", "Peak USS\n(GiB) ↓", 2, False),
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
    balanced_mode = _common_integer(data, "balanced_mode")
    row_labels = [
        CONFIGURATION_LABELS.get(name, name.replace("_", " ").title())
        for name in data["configuration"].astype(str)
    ]
    cell_text: list[list[str]] = []
    best_rows: list[int] = []

    for metric, _, decimals, higher_is_better in DISPLAY_METRICS:
        means = data[f"{metric}_mean"].to_numpy(dtype=float)
        # Memory is stored in MiB; present it in GiB for a compact table.
        scale = 1024.0 if metric == "peak_uss_mb" else 1.0
        means = means / scale
        stds = data[f"{metric}_std"].to_numpy(dtype=float) / scale
        values = [
            _format(mean, std, decimals) for mean, std in zip(means, stds, strict=True)
        ]
        cell_text.append(values)
        chooser = np.nanargmax if higher_is_better else np.nanargmin
        best_rows.append(int(chooser(means)))

    # Matplotlib tables consume rows first, while the construction above is
    # metric-major to make best-value selection straightforward.
    cell_text_by_row = np.asarray(cell_text, dtype=object).T.tolist()
    headers = [heading for _, heading, _, _ in DISPLAY_METRICS]
    headers[2] = f"{measured_batches} batches\n(s) ↓"

    fig, ax = plt.subplots(figsize=(7.7, 2.75))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text_by_row,
        rowLabels=row_labels,
        colLabels=headers,
        cellLoc="center",
        rowLoc="right",
        colLoc="center",
        bbox=[0.25, 0.18, 0.74, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)

    for column in range(len(DISPLAY_METRICS)):
        cell = table[(0, column)]
        cell.set_facecolor("#DCE6F1")
        cell.set_text_props(weight="bold")
        cell.set_edgecolor("white")

    for row in range(1, len(data) + 1):
        label_cell = table[(row, -1)]
        label_cell.set_facecolor("#ECECEC")
        label_cell.set_text_props(weight="bold")
        label_cell.set_edgecolor("white")
        for column in range(len(DISPLAY_METRICS)):
            cell = table[(row, column)]
            cell.set_edgecolor("white")
            if best_rows[column] == row - 1:
                cell.set_facecolor("#D9F2D9")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#FAFAFA" if row % 2 else "#F2F2F2")

    seed_sets = data["seeds_present"].astype(str).unique()
    run_counts = data["runs"].astype(int).unique()
    seed_text = seed_sets[0] if len(seed_sets) == 1 else "available seeds"
    run_text = str(run_counts[0]) if len(run_counts) == 1 else "varying n"
    ax.set_title("Augmentation data-loader runtime", pad=5, weight="bold")
    fig.text(
        0.5,
        0.025,
        f"{BASE_RECIPE_LABEL}; balanced loader (mode {balanced_mode}); "
        f"{warmup_batches} warmup + {measured_batches} measured batches; "
        f"loader batch {loader_batch_size} (training global batch "
        f"{GLOBAL_TRAINING_BATCH_SIZE}).\n"
        f"Mean ± sample std across seeds {seed_text} (n={run_text}). "
        "Green marks the best result per column.",
        ha="center",
        va="bottom",
        fontsize=6.8,
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
