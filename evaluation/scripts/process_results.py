"""CLI: turn raw evaluation CSVs into tidy tables under ``data/processed/``.

Usage::

    python -m evaluation.scripts.process_results \\
        --inputs evaluation/data/raw/eval_csv/*.csv \\
        --output-dir evaluation/data/processed

Produces three canonical outputs (all consumed by the plots subpackage):

* ``all_runs.csv``                       -- every raw row, with parsed columns.
* ``accuracy_by_model_method.csv``       -- mean/std top-1 & top-5 per
                                            (model, method), with seed counts.
* ``calibration_by_model_method.csv``    -- mean/std ECE/Brier/NLL per
                                            (model, method).
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import List

import pandas as pd

from ..common import PROCESSED_DIR, ensure_dirs, setup_logging
from ..processing import (
    add_parsed_columns,
    aggregate_over_seeds,
    load_eval_csvs,
)


_logger = logging.getLogger(__name__)


def _expand_globs(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        out.extend(matches if matches else [pat])
    return out


def _derive_method(df: pd.DataFrame) -> pd.Series:
    """Heuristic ``method`` column derived from parsed ``tags``.

    Runs tagged ``"baseline"`` become ``method="baseline"``; everything else
    is labelled ``"labelmix"``.  Override by writing a ``method`` column into
    the raw CSV instead.
    """
    if "method" in df.columns and df["method"].notna().any():
        return df["method"]

    def _classify(tags) -> str:
        if isinstance(tags, list):
            joined = " ".join(str(t) for t in tags)
        else:
            joined = str(tags)
        return "baseline" if "baseline" in joined else "labelmix"

    return df["tags"].apply(_classify) if "tags" in df.columns else pd.Series("labelmix", index=df.index)


def _safe_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _logger.info("Wrote %s (%d rows)", path, len(df))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Raw eval CSV paths or globs.")
    p.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    ensure_dirs()

    paths = _expand_globs(args.inputs)
    _logger.info("Loading %d raw CSV file(s)", len(paths))
    df = load_eval_csvs(paths)
    if df.empty:
        _logger.error("No rows loaded from %s", paths)
        return 2

    df = add_parsed_columns(df, name_col="name")
    df["method"] = _derive_method(df)

    _safe_write(df, args.output_dir / "all_runs.csv")

    # Accuracy table.
    acc_cols = [c for c in ("top1_acc", "top5_acc") if c in df.columns]
    if acc_cols:
        acc_tbl = aggregate_over_seeds(
            df, group_by=("model", "method"), metric_cols=acc_cols,
        )
        _safe_write(acc_tbl, args.output_dir / "accuracy_by_model_method.csv")

    # Calibration table (ECE variants + Brier + NLL if present).
    calib_cols = [
        c for c in df.columns
        if c.startswith("ece/") or c in ("brier", "nll")
    ]
    if calib_cols:
        calib_tbl = aggregate_over_seeds(
            df, group_by=("model", "method"), metric_cols=calib_cols,
        )
        _safe_write(calib_tbl, args.output_dir / "calibration_by_model_method.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
