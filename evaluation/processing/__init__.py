"""Turn raw evaluation CSVs into tidy, paper-ready tables.

The processing stage is strictly offline: it reads from ``data/raw/`` and
writes to ``data/processed/``.  It never touches checkpoints or GPUs.
"""
from .aggregate import aggregate_over_seeds, confidence_interval_95
from .load import load_eval_csv, load_eval_csvs
from .parse_run_name import add_parsed_columns, parse_many, parse_run_name

__all__ = [
    "load_eval_csv",
    "load_eval_csvs",
    "parse_run_name",
    "parse_many",
    "add_parsed_columns",
    "aggregate_over_seeds",
    "confidence_interval_95",
]
