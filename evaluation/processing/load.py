"""Load raw evaluation CSVs into tidy :class:`pandas.DataFrame` objects.

The offline evaluator writes one CSV per invocation.  ``load_eval_csv``
reads a single file, ``load_eval_csvs`` stitches a directory together and
tags each row with its source so downstream processing can deduplicate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

PathLike = Union[str, Path]


def load_eval_csv(path: PathLike) -> pd.DataFrame:
    """Load a single evaluation CSV written by :mod:`evaluation.offline_eval`."""
    df = pd.read_csv(path)
    df["_source_csv"] = str(path)
    return df


def load_eval_csvs(paths: Iterable[PathLike]) -> pd.DataFrame:
    """Concatenate multiple evaluation CSVs; no deduplication is applied."""
    frames = [load_eval_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
