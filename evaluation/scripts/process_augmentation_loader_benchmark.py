"""Aggregate augmentation-loader benchmark runs across random seeds.

Input:
  ``data/raw/runtime/augmentation_loader_benchmark.csv``

Output:
  ``data/processed/augmentation_loader_benchmark.csv``

Every measured metric is reported as a cross-seed mean and sample standard
deviation (pandas ``std``, i.e. ``ddof=1``).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..common import PROCESSED_DIR, RAW_DIR, setup_logging


_logger = logging.getLogger(__name__)

INPUT_FILE = RAW_DIR / "runtime" / "augmentation_loader_benchmark.csv"
OUTPUT_FILE = PROCESSED_DIR / "augmentation_loader_benchmark.csv"

GROUP_COLUMNS = [
    "configuration",
    "description",
    "loss",
    "mix_k",
    "balanced_mode",
    "batch_size",
    "workers",
    "warmup_batches",
    "measured_batches",
]

METRIC_COLUMNS = [
    "first_batch_seconds",
    "total_batch_processing_seconds",
    "throughput_images_per_second",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_rss_mb",
    "peak_uss_mb",
]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per configuration with mean and sample std metrics."""
    required = set(GROUP_COLUMNS + METRIC_COLUMNS + ["seed"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Retain configurations without a mix_k value (the non-TreemapMix rows).
    grouped = df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    summary = grouped[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    run_info = grouped["seed"].agg(
        runs="size",
        seeds_present=lambda values: ", ".join(
            str(value) for value in sorted(values.astype(int).unique())
        ),
    ).reset_index()
    summary = summary.merge(run_info, on=GROUP_COLUMNS, how="left", validate="1:1")

    leading = GROUP_COLUMNS + ["runs", "seeds_present"]
    metric_columns = [column for column in summary.columns if column not in leading]
    return summary[leading + metric_columns]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    _logger.info("Reading %s", args.input)
    summary = aggregate(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    _logger.info("Wrote %s (%d configurations)", args.output, len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
