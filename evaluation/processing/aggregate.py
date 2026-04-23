"""Aggregate per-run metrics into mean ± std tables over seeds."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _numeric_metric_columns(df: pd.DataFrame, skip: Iterable[str]) -> list[str]:
    """Columns that are numeric and not listed in ``skip``."""
    skip_set = set(skip)
    return [
        c for c in df.columns
        if c not in skip_set and pd.api.types.is_numeric_dtype(df[c])
    ]


def aggregate_over_seeds(
    df: pd.DataFrame,
    group_by: Sequence[str],
    metric_cols: Sequence[str] | None = None,
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Return mean, std, and seed count per ``group_by`` key.

    The output has a flat column layout::

        <group_by>... | <metric>_mean | <metric>_std | n_seeds

    Args:
        df: Tidy table (typically the output of :mod:`processing.load` after
            :func:`parse_run_name.add_parsed_columns`).
        group_by: Columns that identify a model/method combination, e.g.
            ``("model", "method")``.
        metric_cols: Subset of metric columns to aggregate.  Defaults to all
            numeric columns not in ``group_by`` and not equal to ``seed_col``.
        seed_col: Column used to count independent repeats.
    """
    if metric_cols is None:
        metric_cols = _numeric_metric_columns(df, skip=list(group_by) + [seed_col])

    agg_spec = {c: ["mean", "std"] for c in metric_cols}
    if seed_col in df.columns:
        agg_spec[seed_col] = "nunique"

    grouped = df.groupby(list(group_by), dropna=False).agg(agg_spec)

    # Flatten the MultiIndex columns.
    flat_cols: list[str] = []
    for col in grouped.columns.to_flat_index():
        base, stat = col
        if base == seed_col and stat == "nunique":
            flat_cols.append("n_seeds")
        else:
            flat_cols.append(f"{base}_{stat}")
    grouped.columns = flat_cols

    return grouped.reset_index()


def confidence_interval_95(std: pd.Series, n: pd.Series) -> pd.Series:
    """Symmetric 95% CI half-width assuming a normal sampling distribution.

    Suitable for "mean ± CI" error bars when ``n`` (seed count) is small but
    > 1; falls back to ``NaN`` for ``n < 2``.
    """
    n = n.astype(float)
    ci = 1.96 * std / np.sqrt(n)
    return ci.where(n >= 2, other=np.nan)
