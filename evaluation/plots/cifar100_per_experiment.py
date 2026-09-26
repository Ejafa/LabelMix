"""Render the appendix CIFAR-100 accuracy/ECE forest using the main-paper style."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..common import FIGURES_DIR, PROCESSED_DIR, setup_logging
from .in1k_per_experiment import _render_one, plot as _plot


def plot(df: pd.DataFrame) -> plt.Figure:
    """Return the CIFAR-100 accuracy/ECE forest without saving it."""
    return _plot(df, title="CIFAR-100")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "cifar100_per_experiment_short.csv")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "per_experiment" / "cifar100_per_experiment_short.pdf")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    return _render_one("CIFAR-100", args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
