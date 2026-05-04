"""CLI: build the per-run optimizer comparison CSV.

Pipeline
--------

1. Read the raw per-run CSV produced by ``run_eval`` over the W&B
   ``optimizers`` group (one row per run, seed 42 only).

2. **Sanitize** each row with :func:`aggregate_per_experiment.sanitize_name`
   (-> canonical ``type``) and
   :func:`aggregate_per_experiment.is_long_horizon` (-> ``long_horizon``),
   preserving the default metric columns.

3. Add an ``optimizer`` column by parsing the raw run ``name``: the model
   segment encodes the optimizer as ``vit-<size>-<opt>`` where ``opt`` is
   ``muon`` or ``adamw`` (e.g. ``vit-betwixt-muon``). Runs whose name does
   not carry ``-muon`` or ``-adamw`` abort the pipeline loudly — per user
   instruction we do **not** guess.

4. **No aggregation.** Each optimizer run is emitted as-is (single seed).

5. For every optimizer row, look up a **matching nadamw baseline** from
   ``evaluation/data/raw/eval_csv/in1k.csv`` on the strict composite key
   ``(type, model, dataset, long_horizon, seed)``. If no match is found
   the script **errors out** and halts (non-zero exit) so the user can
   intervene. Matched nadamw rows are appended with ``optimizer='nadamw'``.

6. Write a single CSV with columns:
   ``type, model, dataset, long_horizon, seed, optimizer, <metrics...>``.

Usage::

    python -m evaluation.scripts.aggregate_optimizers \\
        --input       evaluation/data/raw/eval_csv/optimizers.csv \\
        --nadamw-ref  evaluation/data/raw/eval_csv/in1k.csv \\
        --output      evaluation/data/processed/optimizers_per_experiment.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .aggregate_per_experiment import (
    _DEFAULT_METRICS,
    is_long_horizon,
    sanitize_name,
)

_logger = logging.getLogger(__name__)

# Canonical optimizer tokens we expect to find in the run name.
_OPTIMIZER_TOKENS: Sequence[str] = ("muon", "adamw")
# Optimizer tag used for rows pulled from the nadamw reference table.
_REFERENCE_OPTIMIZER = "nadamw"

# Columns that uniquely identify a run up to optimizer choice.
_JOIN_KEY: Sequence[str] = ("type", "model", "dataset", "long_horizon", "seed")


def _detect_optimizer(name: str) -> str | None:
    """Return ``"muon"`` / ``"adamw"`` based on the run name, else ``None``.

    The optimizer appears inside the model segment of the run name as
    ``vit-<size>-<opt>`` (e.g. ``vit-betwixt-muon``), so we match the
    delimited token ``-<opt>`` to avoid picking up stray substrings.
    """
    if not isinstance(name, str):
        return None
    n = name.lower()
    for tok in _OPTIMIZER_TOKENS:
        if f"-{tok}" in n:
            return tok
    return None


def _sanitize_optimizers(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Project + rewrite the raw optimizer eval frame.

    Output columns: ``type, model, dataset, long_horizon, seed, optimizer``
    followed by the subset of ``metrics`` that is actually present in ``df``.
    Aborts (via ``SystemExit``) if any row's optimizer cannot be detected.
    """
    missing_required = {"name", "model", "dataset", "seed"}.difference(df.columns)
    if missing_required:
        raise SystemExit(
            f"Input is missing required columns: {sorted(missing_required)}"
        )

    present_metrics = [m for m in metrics if m in df.columns]
    if not present_metrics:
        raise SystemExit(
            f"Input has none of the expected metric columns. "
            f"Expected any of: {list(metrics)}; have: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "type": df["name"].map(sanitize_name),
        "model": df["model"],
        "dataset": df["dataset"],
        "long_horizon": df["name"].map(is_long_horizon),
        "seed": pd.to_numeric(df["seed"], errors="coerce").astype("Int64"),
        "optimizer": df["name"].map(_detect_optimizer),
    })
    for m in present_metrics:
        out[m] = pd.to_numeric(df[m], errors="coerce")

    unknown_type = out["type"] == "unknown"
    if unknown_type.any():
        bad = df.loc[unknown_type, "name"].unique().tolist()
        raise SystemExit(
            f"Cannot canonicalise ``type`` for {int(unknown_type.sum())} row(s). "
            f"Examples: {bad[:5]}"
        )

    missing_opt = out["optimizer"].isna()
    if missing_opt.any():
        bad = df.loc[missing_opt, "name"].unique().tolist()
        raise SystemExit(
            f"Cannot detect optimizer (muon/adamw) for "
            f"{int(missing_opt.sum())} row(s). Examples: {bad[:5]}"
        )

    return out


def _load_nadamw_reference(
    path: Path,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Load + sanitize the nadamw reference CSV with the same schema."""
    df = pd.read_csv(path)
    _logger.info("Loaded %d nadamw reference row(s) from %s", len(df), path)

    missing_required = {"name", "model", "dataset", "seed"}.difference(df.columns)
    if missing_required:
        raise SystemExit(
            f"Reference CSV {path} missing columns: {sorted(missing_required)}"
        )

    present_metrics = [m for m in metrics if m in df.columns]
    ref = pd.DataFrame({
        "type": df["name"].map(sanitize_name),
        "model": df["model"],
        "dataset": df["dataset"],
        "long_horizon": df["name"].map(is_long_horizon),
        "seed": pd.to_numeric(df["seed"], errors="coerce").astype("Int64"),
    })
    for m in present_metrics:
        ref[m] = pd.to_numeric(df[m], errors="coerce")

    # Drop rows the reference table cannot contribute (e.g. type==unknown).
    ref = ref.loc[ref["type"] != "unknown"].reset_index(drop=True)
    return ref


def _match_nadamw_rows(
    optimizer_rows: pd.DataFrame,
    nadamw_ref: pd.DataFrame,
) -> pd.DataFrame:
    """For each optimizer row, find the single nadamw reference row on the
    composite ``_JOIN_KEY``.

    Aborts (via ``SystemExit``) if any target key is missing from the
    reference frame — per user instruction we stop so they can intervene.
    Duplicate matches (more than one nadamw row for the same key) are also
    fatal: it means the reference table is ambiguous for our purposes.
    """
    # Distinct keys we need to find in the reference.
    wanted = (
        optimizer_rows[list(_JOIN_KEY)]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Inner-join to discover hits and miss counts.
    ref_keyed = nadamw_ref.set_index(list(_JOIN_KEY), drop=False)
    matched_rows: list[pd.Series] = []
    missing_keys: list[dict] = []
    ambiguous_keys: list[dict] = []

    for _, key_row in wanted.iterrows():
        key = tuple(key_row[c] for c in _JOIN_KEY)
        try:
            hit = ref_keyed.loc[[key]]
        except KeyError:
            missing_keys.append(key_row.to_dict())
            continue
        if len(hit) != 1:
            ambiguous_keys.append({**key_row.to_dict(), "n_hits": int(len(hit))})
            continue
        matched_rows.append(hit.iloc[0])

    if missing_keys:
        preview = missing_keys[:10]
        _logger.error(
            "STOP: %d optimizer key(s) have no matching nadamw row in the "
            "reference table. First %d: %s",
            len(missing_keys), len(preview), preview,
        )
        raise SystemExit(
            f"Aborting: {len(missing_keys)} optimizer key(s) have no nadamw "
            f"counterpart in the reference CSV. Inspect the cases above and "
            f"re-run once the reference table is complete."
        )
    if ambiguous_keys:
        preview = ambiguous_keys[:10]
        _logger.error(
            "STOP: %d optimizer key(s) match multiple nadamw reference rows. "
            "First %d: %s",
            len(ambiguous_keys), len(preview), preview,
        )
        raise SystemExit(
            f"Aborting: {len(ambiguous_keys)} optimizer key(s) have >1 "
            f"nadamw row in the reference CSV; join is ambiguous."
        )

    matched = pd.DataFrame(matched_rows).reset_index(drop=True)
    matched["optimizer"] = _REFERENCE_OPTIMIZER
    return matched


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Raw per-run eval CSV for the optimizers W&B group "
                        "(produced by evaluation.scripts.run_eval).")
    p.add_argument("--nadamw-ref", type=Path, required=True,
                   help="Reference eval CSV carrying the nadamw baseline rows "
                        "(conventionally evaluation/data/raw/eval_csv/in1k.csv).")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination CSV. Parent dirs are created as needed.")
    p.add_argument("--metrics", nargs="*", default=list(_DEFAULT_METRICS),
                   help="Metric columns to carry through (default: the same "
                        "set used by aggregate_per_experiment).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.input.is_file():
        _logger.error("Input CSV not found: %s", args.input)
        return 2
    if not args.nadamw_ref.is_file():
        _logger.error("nadamw reference CSV not found: %s", args.nadamw_ref)
        return 2

    raw = pd.read_csv(args.input)
    _logger.info("Loaded %d optimizer row(s) from %s", len(raw), args.input)

    optimizer_rows = _sanitize_optimizers(raw, args.metrics)
    _logger.info(
        "Sanitized %d optimizer row(s); optimizer counts: %s",
        len(optimizer_rows),
        optimizer_rows["optimizer"].value_counts().to_dict(),
    )

    nadamw_ref = _load_nadamw_reference(args.nadamw_ref, args.metrics)
    nadamw_rows = _match_nadamw_rows(optimizer_rows, nadamw_ref)
    _logger.info(
        "Matched %d nadamw reference row(s) for %d optimizer key(s).",
        len(nadamw_rows),
        len(optimizer_rows[list(_JOIN_KEY)].drop_duplicates()),
    )

    present_metrics = [m for m in args.metrics if m in optimizer_rows.columns]
    front = ["type", "model", "dataset", "long_horizon", "seed", "optimizer"]
    final_cols = front + present_metrics
    combined = pd.concat([optimizer_rows, nadamw_rows], ignore_index=True)
    # Ensure every expected column exists (metrics missing from the reference
    # frame already come through as NaN via the sanitizer).
    for c in final_cols:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[final_cols].sort_values(
        ["type", "model", "dataset", "long_horizon", "seed", "optimizer"]
    ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    _logger.info("Wrote %d row(s) -> %s", len(combined), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
