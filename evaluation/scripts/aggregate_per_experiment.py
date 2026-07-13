"""CLI: sanitize raw per-seed eval rows and aggregate them by (type, dataset, model, long_horizon).

Pipeline
--------

1. **Sanitize columns.** Drop every column that is not an evaluation metric or
   one of the kept identity columns (``name``, ``model``, ``dataset``).  Rename
   ``name`` -> ``type`` and rewrite each value with :func:`sanitize_name` to
   one of a small fixed vocabulary of canonical experiment types.  Additionally
   derive a boolean ``long_horizon`` column via :func:`is_long_horizon`:
   ``True`` iff the raw name contains the delimited token ``__300__``
   (i.e. the 300-epoch schedule).


   =====================================  =========================
   name contains                          canonical ``type``
   =====================================  =========================
   ``baseline``                            ``baseline``
   ``mixup_switch_prob=1.0``               ``cutmix``
   ``mixup_switch_prob=0.0``               ``mixup``
   ``noaug``                               ``noaug``
   ``mosaic``                              ``mosaic``
   ``openmixup-fmix``                      ``fmix``
   ``openmixup-gridmix``                   ``gridmix``
   ``openmixup-resizemix``                 ``resizemix``
   ``openmixup-saliencymix``               ``saliencymix``
   ``openmixup-smoothmix``                 ``smoothmix``
   ``openmixup-tokenmix``                  ``tokenmix``
   ``openmixup-tla``                       ``tla``
   ``mixed``                               ``labelmix-mixed``
   ``pl-loss``                             ``labelmix-pl``
   ``soft-ce``                             ``labelmix-sce``
   ``bare``                                ``bare``
   ``unbalanced`` (+ secondary keyword k)  ``unbalanced-<k>``
   ``unbalanced`` (no secondary keyword)   ``unbalanced-noaug`` (default)
   =====================================  =========================

   ``unbalanced`` is evaluated first so an "unbalanced mosaic" run becomes
   ``unbalanced-mosaic`` rather than plain ``mosaic``.

2. **Aggregate** by the composite key ``(type, dataset, model, long_horizon)``:
   for each numeric metric column compute mean & std across all seeds available
   for that group. Groups with fewer than 3 seeds are tagged
   ``status=unfinished`` with the list of missing seeds (from {42, 43, 44});
   full groups are ``complete``. **Unfinished groups are dropped from the
   output** (they are listed in the log only).

3. **Split by training horizon.** The aggregated frame is partitioned on
   ``long_horizon``; the short-horizon (90-epoch) rows and long-horizon
   (300-epoch) rows are written to two separate CSVs. The ``--output``
   argument is treated as a *base path*: given
   ``.../in1k_per_experiment.csv`` the script writes
   ``.../in1k_per_experiment_short.csv`` and
   ``.../in1k_per_experiment_long.csv``.

4. **Report**: print a Markdown table with every metric requested via
   ``--metrics`` (all of ``_DEFAULT_METRICS`` by default) for full
   interpretation.

Usage::

    python -m evaluation.scripts.aggregate_per_experiment \
        --input  evaluation/data/raw/eval_csv/in1k.csv \
        --output evaluation/data/processed/in1k_per_experiment.csv \
        --table-out evaluation/data/processed/in1k_per_experiment.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column policy
# ---------------------------------------------------------------------------

# Identity columns kept alongside the metric columns after sanitization.
# ``name`` is renamed to ``type`` (and value-remapped by ``sanitize_name``).
_IDENTITY_COLUMNS: Sequence[str] = ("name", "model", "dataset")

# Evaluation metric columns we want aggregated (in display order).
_DEFAULT_METRICS: Sequence[str] = (
    "top1_acc",
    "top5_acc",
    "nll",
    "brier",
    "ece/n_bins=5",
    "ece/n_bins=10",
    "ece/n_bins=15",
    "ece/n_bins=20",
    "eval_time_s",
)

_EXPECTED_SEEDS: Sequence[int] = (42, 43, 44)

# Rounding policy for the aggregated CSV.
#
# The project-wide default is 4 decimal places -- small cross-seed differences
# matter for every calibration/loss metric we track, so under-rounding is
# strictly worse than a slightly wider CSV.
#
# ``_PERCENT_METRICS`` are on a 0-100 scale; ``_HIGH_PRECISION_METRICS`` are
# on a 0-1 scale but are typically reported x100 in the paper (e.g. ECE in
# percentage points, NLL and Brier at higher precision). Both get the same
# 4-decimal budget as the default; the separate constants are kept so a
# future caller can tighten one bucket without touching the others.
_PERCENT_METRICS: frozenset[str] = frozenset({"top1_acc", "top5_acc"})
_HIGH_PRECISION_METRICS: frozenset[str] = frozenset({
    "nll",
    "brier",
    "ece/n_bins=5",
    "ece/n_bins=10",
    "ece/n_bins=15",
    "ece/n_bins=20",
})
_DEFAULT_DECIMALS: int = 4
_PERCENT_DECIMALS: int = 4
_HIGH_PRECISION_DECIMALS: int = 4

# Ordered primary-keyword table. ``unbalanced`` is handled separately and
# first; the remaining rules are applied in this order — first match wins.
# The order is chosen so that specific/long tokens dominate ambiguous ones
# (e.g. a ``*mixup_switch_prob=*`` string also contains ``mixup`` but we want
# the explicit switch-prob rule to win).
_KEYWORD_RULES: Sequence[tuple[str, str]] = (
    ("baseline", "baseline"),
    ("mixup_switch_prob=1.0", "cutmix"),
    ("mixup_switch_prob=0.0", "mixup"),
    ("openmixup-fmix", "fmix"),
    ("openmixup-gridmix", "gridmix"),
    ("openmixup-resizemix", "resizemix"),
    ("openmixup-saliencymix", "saliencymix"),
    ("openmixup-smoothmix", "smoothmix"),
    ("openmixup-tokenmix", "tokenmix"),
    ("openmixup-tla", "tla"),
    ("cutmix", "cutmix"),
    ("mixup", "mixup"),
    ("pl-loss", "labelmix-pl"),
    ("soft-ce", "labelmix-sce"),
    ("mosaic", "mosaic"),
    ("mixed", "labelmix-mixed"),
    ("noaug", "noaug"),
    ("bare", "bare"),
)

# Secondary keywords attachable to ``unbalanced-*``; probed in order.
_UNBALANCED_SECONDARIES: Sequence[tuple[str, str]] = (
    ("mixup_switch_prob=1.0", "cutmix"),
    ("mixup_switch_prob=0.0", "mixup"),
    ("openmixup-fmix", "fmix"),
    ("openmixup-gridmix", "gridmix"),
    ("openmixup-resizemix", "resizemix"),
    ("openmixup-saliencymix", "saliencymix"),
    ("openmixup-smoothmix", "smoothmix"),
    ("openmixup-tokenmix", "tokenmix"),
    ("openmixup-tla", "tla"),
    ("cutmix", "cutmix"),
    ("mixup", "mixup"),
    ("pl-loss", "labelmix-pl"),
    ("soft-ce", "labelmix-sce"),
    ("mosaic", "mosaic"),
    ("mixed", "labelmix-mixed"),
    ("noaug", "noaug"),
    ("bare", "bare"),
    ("baseline", "baseline"),
)
_UNBALANCED_DEFAULT = "noaug"

# Marker in the raw run name indicating a 300-epoch (long-horizon) schedule.
# Using the delimited ``__300__`` token avoids accidental matches against
# other numeric fragments that may contain the digit sequence '300'.
_LONG_HORIZON_MARKER = "__300__"


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def is_long_horizon(name: str) -> bool:
    """True iff ``name`` denotes a long-horizon (300-epoch) run."""
    if not isinstance(name, str):
        return False
    return _LONG_HORIZON_MARKER in name


def sanitize_name(name: str) -> str:
    """Map a raw run ``name`` to a canonical experiment ``type`` token.

    See module docstring for the full rule table. Returns ``"unknown"`` for
    inputs that match none of the rules (callers should inspect these).
    """
    if not isinstance(name, str):
        return "unknown"
    n = name.lower()
    if "unbalanced" in n:
        for needle, label in _UNBALANCED_SECONDARIES:
            if needle in n:
                return f"unbalanced-{label}"
        return f"unbalanced-{_UNBALANCED_DEFAULT}"
    for needle, label in _KEYWORD_RULES:
        if needle in n:
            return label
    return "unknown"


def sanitize_dataframe(
    df: pd.DataFrame,
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> pd.DataFrame:
    """Keep only ``(name, model, dataset, seed)`` + metric columns, and rewrite
    ``name`` -> ``type`` via :func:`sanitize_name`.

    ``seed`` is retained because aggregation still needs it; it is dropped from
    the final aggregated output. Missing input columns are ignored silently
    (the caller gets whatever the raw CSV provided).
    """
    keep = [c for c in ("name", "model", "dataset", "seed") if c in df.columns]
    keep += [m for m in metrics if m in df.columns]
    out = df[keep].copy()

    if "name" in out.columns:
        out["type"] = out["name"].map(sanitize_name)
        out["long_horizon"] = out["name"].map(is_long_horizon)
        out = out.drop(columns=["name"])
        # Reorder so type/model/dataset/long_horizon come first.
        front = [c for c in ("type", "model", "dataset", "long_horizon", "seed") if c in out.columns]
        out = out[front + [c for c in out.columns if c not in front]]
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _decimals_for(metric: str) -> int:
    """Return the number of decimal places to use for ``metric``.

    All metrics currently use 4 decimal places. The dispatch on bucket is
    preserved so a future caller can lower precision for one family (e.g.
    accuracy) without touching the calibration metrics.

    - ``top1_acc`` / ``top5_acc``: ``_PERCENT_DECIMALS`` (4).
    - ``nll``, ``brier``, ``ece/n_bins=*``: ``_HIGH_PRECISION_DECIMALS`` (4).
      Stored on a 0-1 scale; 4 decimals here => 2 decimals after the
      conventional x100 conversion (e.g. ``0.0456`` -> ``4.56 pp``).
    - Everything else: ``_DEFAULT_DECIMALS`` (4).
    """
    if metric in _PERCENT_METRICS:
        return _PERCENT_DECIMALS
    if metric in _HIGH_PRECISION_METRICS:
        return _HIGH_PRECISION_DECIMALS
    return _DEFAULT_DECIMALS


def round_aggregated(agg: pd.DataFrame) -> pd.DataFrame:
    """Round every ``<metric>_mean`` / ``<metric>_std`` pair in-place.

    Every metric is rounded to 4 decimal places (see :func:`_decimals_for`
    for the per-bucket dispatch). The frame is copied so the caller's
    object is not mutated.
    """
    out = agg.copy()
    for col in out.columns:
        for suffix in ("_mean", "_std"):
            if not col.endswith(suffix):
                continue
            metric = col[: -len(suffix)]
            out[col] = pd.to_numeric(out[col], errors="coerce").round(
                _decimals_for(metric)
            )
    return out


def aggregate_by_type(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Group sanitized rows by ``(type, dataset, model)`` and compute mean/std
    of each metric, plus seed completeness metadata.
    """
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise SystemExit(f"No expected metric columns found. Have: {list(df.columns)}")
    required = {"type", "model", "dataset", "long_horizon"}
    missing_id = required.difference(df.columns)
    if missing_id:
        raise SystemExit(f"Missing required identity columns: {sorted(missing_id)}")

    df = df.copy()
    # Normalise long_horizon to a plain bool so groupby keys are stable.
    df["long_horizon"] = df["long_horizon"].astype(bool)
    for m in metrics:
        df[m] = _numeric(df[m])
    if "seed" in df.columns:
        df["seed"] = _numeric(df["seed"]).astype("Int64")

    out_rows: List[dict] = []
    for (t, dset, model, lh), group in df.groupby(
        ["type", "dataset", "model", "long_horizon"], sort=True
    ):
        if "seed" in group.columns:
            present_seeds = sorted({int(s) for s in group["seed"].dropna().tolist()})
        else:
            present_seeds = []
        missing = [s for s in _EXPECTED_SEEDS if s not in present_seeds]
        row: dict = {
            "type": t,
            "dataset": dset,
            "model": model,
            "long_horizon": bool(lh),
            "n_seeds": len(present_seeds),
            "seeds_present": ",".join(str(s) for s in present_seeds),
            "missing_seeds": ",".join(str(s) for s in missing) if missing else "",
            "status": "complete" if not missing and len(present_seeds) >= len(_EXPECTED_SEEDS) else "unfinished",
        }
        for m in metrics:
            vals = group[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_table(agg: pd.DataFrame, metrics: Sequence[str] | None = None) -> str:
    """Render ``agg`` as a Markdown table.

    If ``metrics`` is ``None`` every metric present in ``agg`` (i.e. every
    column with a matching ``<metric>_mean``) is rendered, preserving their
    original column order. Pass an explicit list to restrict to a subset.
    """
    if metrics is None:
        preview_metrics = [
            c[: -len("_mean")]
            for c in agg.columns
            if c.endswith("_mean") and f"{c[: -len('_mean')]}_std" in agg.columns
        ]
    else:
        preview_metrics = [m for m in metrics if f"{m}_mean" in agg.columns]
    cols = ["type", "model", "dataset", "schedule", "status", "n_seeds"] + list(preview_metrics)

    df = agg.sort_values(
        ["dataset", "model", "long_horizon", "type", "status"]
    ).reset_index(drop=True)

    def _fmt(row: pd.Series, m: str) -> str:
        mean, std = row[f"{m}_mean"], row[f"{m}_std"]
        if pd.isna(mean):
            return "-"
        d = _decimals_for(m)
        return f"{mean:.{d}f} +/- {std:.{d}f}"

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        parts = [
            str(row["type"]),
            str(row["model"]),
            str(row["dataset"]),
            "300ep" if bool(row["long_horizon"]) else "90ep",
            str(row["status"]),
            str(int(row["n_seeds"])),
        ]
        for m in preview_metrics:
            parts.append(_fmt(row, m))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help=("Base path for the aggregated CSV. Two files are written, "
                         "derived from this path by inserting ``_short`` / ``_long`` "
                         "before the extension (e.g. ``foo.csv`` -> ``foo_short.csv`` "
                         "and ``foo_long.csv``)."))
    p.add_argument("--sanitized-out", type=Path, default=None,
                   help="Optional path to dump the sanitized (non-aggregated) CSV.")
    p.add_argument("--metrics", nargs="*", default=list(_DEFAULT_METRICS))
    p.add_argument("--table-out", type=Path, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.input.is_file():
        _logger.error("Input CSV not found: %s", args.input)
        return 2

    df = pd.read_csv(args.input)
    _logger.info("Loaded %d rows from %s", len(df), args.input)

    sanitized = sanitize_dataframe(df, args.metrics)

    # Warn about any name that didn't match any rule.
    if "type" in sanitized.columns:
        unknown_mask = sanitized["type"] == "unknown"
        if unknown_mask.any():
            unknown_names = df.loc[unknown_mask, "name"].unique().tolist()
            _logger.warning(
                "Found %d row(s) with unmapped name (type='unknown'): %s",
                int(unknown_mask.sum()),
                unknown_names[:10],
            )

    if args.sanitized_out is not None:
        args.sanitized_out.parent.mkdir(parents=True, exist_ok=True)
        sanitized.to_csv(args.sanitized_out, index=False)
        _logger.info("Wrote sanitized rows -> %s", args.sanitized_out)

    agg = aggregate_by_type(sanitized, args.metrics)

    # Drop unfinished groups from the output (log which ones were removed so
    # the user still gets visibility into what is still running).
    unfinished_mask = agg["status"] == "unfinished"
    if unfinished_mask.any():
        unfinished_preview = (
            agg.loc[unfinished_mask, ["type", "model", "dataset", "long_horizon",
                                      "n_seeds", "missing_seeds"]]
            .to_dict(orient="records")
        )
        _logger.info(
            "Dropping %d unfinished group(s) from output: %s",
            int(unfinished_mask.sum()),
            unfinished_preview,
        )
    agg = agg.loc[~unfinished_mask].reset_index(drop=True)

    # Split by training horizon and write one CSV per half.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = args.output
    short_path = base.with_name(f"{base.stem}_short{base.suffix}")
    long_path = base.with_name(f"{base.stem}_long{base.suffix}")

    short_df = agg.loc[~agg["long_horizon"].astype(bool)].reset_index(drop=True)
    long_df = agg.loc[agg["long_horizon"].astype(bool)].reset_index(drop=True)

    # Round metric columns for human-readable CSV output: every metric gets
    # 4 decimal places (see ``_decimals_for`` for the per-bucket dispatch).
    short_df = round_aggregated(short_df)
    long_df = round_aggregated(long_df)

    short_df.to_csv(short_path, index=False)
    long_df.to_csv(long_path, index=False)
    _logger.info(
        "Wrote %d short-horizon row(s) -> %s", len(short_df), short_path,
    )
    _logger.info(
        "Wrote %d long-horizon row(s) -> %s", len(long_df), long_path,
    )

    table = _render_table(agg, metrics=args.metrics)
    print("\n" + table + "\n")
    if args.table_out is not None:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(table + "\n")
        _logger.info("Wrote Markdown preview table -> %s", args.table_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
