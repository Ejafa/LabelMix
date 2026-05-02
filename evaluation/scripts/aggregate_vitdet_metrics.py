#!/usr/bin/env python3
"""Aggregate COCO detection metrics across vitdet training runs.

Reads training outputs from ``evaluation/data/raw/vitdet_output/`` and
produces two CSV files inside ``evaluation/data/processed/``:

* ``vitdet_metrics_per_run.csv``    -- one row per run, raw final eval values
* ``vitdet_metrics_aggregated.csv`` -- mean/std across ``ptseed`` per (model, checkpoint-type)

Only generic evaluation criteria are kept (``AP``, ``AP50``, ``AP75``,
``APs``, ``APm``, ``APl`` for both ``bbox`` and ``segm``). Per-class
``AP-<classname>`` entries are filtered out.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev, stdev

# ``evaluation/`` package root (this file lives at evaluation/scripts/...).
_EVAL_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR  = _EVAL_ROOT / "data" / "raw"       / "vitdet_output"
RESULTS_DIR = _EVAL_ROOT / "data" / "processed"

# Generic evaluation criteria to keep (drop per-class AP-<name>).
METRIC_KEYS = [
    "bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl",
    "segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl",
]

CKPT_TYPE_RENAME = {
    "baseline": "baseline",
    "pl-loss":  "labelmix-pl",
    "soft-ce":  "labelmix-sce",
    "mosaic":   "mosaic",
}


def parse_folder_name(name: str) -> dict | None:
    """Extract (model, checkpoint-type, ptseed, trseed) from folder name.

    Expected layout: ``<model>__coco__img<res>__<ckpt_type>__ptseed<N>__trseed<M>__<hash>``.
    """
    m = re.match(
        r"^(?P<model>[^_]+(?:-[^_]+)*)__coco__img\d+__"
        r"(?P<ckpt>baseline|pl-loss|soft-ce|mosaic)__"
        r"ptseed(?P<ptseed>\d+)__trseed(?P<trseed>\d+)__[A-Za-z0-9]+$",
        name,
    )
    if not m:
        return None
    return {
        "model":           m.group("model"),
        "checkpoint-type": CKPT_TYPE_RENAME[m.group("ckpt")],
        "ptseed":          int(m.group("ptseed")),
        "trseed":          int(m.group("trseed")),
    }


def last_eval_line(metrics_path: Path) -> dict | None:
    """Return the last JSON line that contains ``bbox/AP`` (final evaluation)."""
    last = None
    with metrics_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "bbox/AP" in obj:
                last = obj
    return last


def collect_runs() -> list[dict]:
    rows = []
    for sub in sorted(OUTPUT_DIR.iterdir()):
        if not sub.is_dir():
            continue
        meta = parse_folder_name(sub.name)
        if meta is None:
            print(f"[skip] unrecognised folder: {sub.name}")
            continue
        metrics_file = sub / "metrics.json"
        if not metrics_file.exists():
            print(f"[skip] missing metrics.json: {sub.name}")
            continue
        last = last_eval_line(metrics_file)
        if last is None:
            print(f"[skip] no final evaluation found: {sub.name}")
            continue
        row = {"run": sub.name, **meta}
        for k in METRIC_KEYS:
            row[k] = last.get(k)
        rows.append(row)
    return rows


def write_per_run_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["run", "model", "checkpoint-type", "ptseed", "trseed"] + METRIC_KEYS
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate mean/std across ``ptseed`` per (model, checkpoint-type)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["checkpoint-type"]), []).append(r)

    out = []
    for (model, ckpt), group in sorted(groups.items()):
        agg = {
            "model":           model,
            "checkpoint-type": ckpt,
            "n_seeds":         len(group),
            "ptseeds":         ",".join(str(x["ptseed"]) for x in sorted(group, key=lambda y: y["ptseed"])),
        }
        for k in METRIC_KEYS:
            vals = [r[k] for r in group if r[k] is not None]
            if not vals:
                agg[f"{k}_mean"] = None
                agg[f"{k}_std"]  = None
                continue
            agg[f"{k}_mean"] = mean(vals)
            agg[f"{k}_std"]  = stdev(vals) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def write_agg_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["model", "checkpoint-type", "n_seeds", "ptseeds"]
    for k in METRIC_KEYS:
        fieldnames += [f"{k}_mean", f"{k}_std"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    rows = collect_runs()
    rows.sort(key=lambda r: (r["model"], r["checkpoint-type"], r["ptseed"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_run_path = RESULTS_DIR / "vitdet_metrics_per_run.csv"
    agg_path     = RESULTS_DIR / "vitdet_metrics_aggregated.csv"
    write_per_run_csv(rows, per_run_path)
    write_agg_csv(aggregate(rows), agg_path)

    print(f"[done] {len(rows)} runs -> {per_run_path}")
    print(f"[done] aggregated      -> {agg_path}")


if __name__ == "__main__":
    main()
