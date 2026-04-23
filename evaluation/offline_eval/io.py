"""I/O helpers for the offline evaluator (args.yaml, mapping files, CSV writes)."""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Mapping, Sequence

import yaml

from ..config import EvalConfig
from ..types import EvalResult


# ---------------------------------------------------------------------------
# args.yaml
# ---------------------------------------------------------------------------

def load_args_yaml(run_dir: str) -> Dict[str, Any]:
    """Load the training-time ``args.yaml`` snapshot from ``run_dir``."""
    path = os.path.join(run_dir, "args.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"args.yaml not found in {run_dir}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"args.yaml in {run_dir} did not parse to a dict.")
    return data


# ---------------------------------------------------------------------------
# Mapping files (CLI input)
# ---------------------------------------------------------------------------

def load_mapping(path: str) -> Dict[str, str]:
    """Load a ``{name: run_dir}`` mapping from YAML or JSON.

    Supports both the short form::

        name1: /path/to/run
        name2: /path/to/other/run

    and the richer dict form::

        name1:
            path: /path/to/run
            # ... extra keys ignored
    """
    with open(path, "r") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Mapping file {path} must be a top-level dict of {{name: run_dir}}."
        )

    resolved: Dict[str, str] = {}
    for name, v in data.items():
        if isinstance(v, str):
            resolved[name] = v
        elif isinstance(v, dict) and "path" in v:
            resolved[name] = v["path"]
        else:
            raise ValueError(
                f"Entry for '{name}' must be a path string or a dict with 'path'."
            )
    return resolved


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _flatten(v: Any) -> Any:
    """CSV-safe rendering for nested yaml values (lists / dicts)."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        return ";".join(f"{k}={vv}" for k, vv in v.items())
    return v


def write_results_csv(
    path: str,
    results: Sequence[EvalResult],
    cfg: EvalConfig,
    errors: Mapping[str, str],
    mapping: Mapping[str, str],
) -> None:
    """Serialize evaluation results + per-run errors to a single CSV.

    Column order is::

        cfg.fixed_columns + cfg.args_columns + <metric cols> + ["eval_time_s", "error"]

    Metric columns are the stable sorted union across all runs, so adding a
    new metric to ``METRIC_REGISTRY`` extends the CSV automatically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    metric_keys = sorted({k for r in results for k in r.metrics.keys()})
    header = (
        list(cfg.fixed_columns)
        + list(cfg.args_columns)
        + metric_keys
        + ["eval_time_s", "error"]
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for r in results:
            row: Dict[str, Any] = {
                "name": r.name,
                "run_dir": r.run_dir,
                "checkpoint": r.checkpoint,
                "eval_time_s": f"{r.eval_time:.2f}",
                "error": "",
            }
            for col in cfg.args_columns:
                row[col] = _flatten(r.args.get(col))
            for k in metric_keys:
                row[k] = r.metrics.get(k, "")
            writer.writerow(row)

        # One row per failed run so the CSV matches the input mapping 1:1.
        for name, err in errors.items():
            row = {k: "" for k in header}
            row["name"] = name
            row["run_dir"] = mapping.get(name, "")
            row["error"] = err
            writer.writerow(row)
