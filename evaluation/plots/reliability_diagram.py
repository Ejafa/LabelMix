"""Plot: reliability diagram for a single run (confidence vs accuracy).

Reads a raw logits dump written by the offline evaluator
(``cfg.save_raw_logits=True``) and produces the classic Guo et al. (2017)
reliability diagram with equal-width bins.

Input   : ``data/raw/logits/<slug>.pt`` holding ``{"logits", "targets"}``.
Output  : ``data/processed/figures/reliability_<slug>.pdf``.

Note: this is the one plot that reads from ``data/raw/`` because it
requires per-sample confidences, not the aggregated metrics CSV.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..common import FIGURES_DIR, RAW_LOGITS_DIR, setup_logging
from ._style import SINGLE_COL_WIDTH, apply_paper_style, savefig


_logger = logging.getLogger(__name__)


def _reliability_bins(
    logits: torch.Tensor, targets: torch.Tensor, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(bin_centers, accuracy_per_bin, fraction_per_bin)``."""
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).numpy()
    acc = np.full(n_bins, np.nan)
    frac = np.zeros(n_bins)
    total = conf.numel()

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        n_in = int(in_bin.sum().item())
        frac[i] = n_in / total
        if n_in > 0:
            acc[i] = float(correct[in_bin].mean().item())
    return centers, acc, frac


def plot(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> plt.Figure:
    apply_paper_style()
    centers, acc, frac = _reliability_bins(logits, targets, n_bins)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="gray", label="Perfect")
    width = 1.0 / n_bins
    ax.bar(centers, acc, width=width * 0.95, edgecolor="black", linewidth=0.5, label="Accuracy")
    ax.bar(centers, frac, width=width * 0.95, alpha=0.25, color="tab:orange", label="Fraction of samples")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy / fraction")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help=f"Path to a .pt dump under {RAW_LOGITS_DIR}.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PDF; defaults to figures/reliability_<stem>.pdf.")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    data = torch.load(args.input, map_location="cpu")
    logits, targets = data["logits"], data["targets"]
    output = args.output or (FIGURES_DIR / f"reliability_{args.input.stem}.pdf")

    fig = plot(logits, targets, n_bins=args.n_bins)
    savefig(fig, str(output))
    _logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
