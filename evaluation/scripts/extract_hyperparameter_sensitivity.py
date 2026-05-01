"""Extract ViT-Wee LabelMix hyperparameter-sweep data for the alpha/k sensitivity plot.

For each eligible wandb run this script:

1. Reads ``config.yaml`` and filters so that the run holds alpha and k constant
   (``labelmix_alpha_min == labelmix_alpha_max`` and
   ``labelmix_k_min == labelmix_k_max``, both schedules ``fixed``, ``labelmix=True``).
2. Reads ``summary.json`` to obtain ``eval_ece`` and ``eval_top1``.
3. Emits one CSV row per qualifying run.

In addition, it reads the already-aggregated baseline CSV
(``in1k_per_experiment_short.csv``) and emits a second "plot-ready" CSV that
concatenates the LabelMix grid with two baseline reference rows ('baseline' =
mixup+cutmix, 'bare' = single-image aug only) for the ``vit_wee`` model, with
means and stds across seeds. ECE values from the baseline CSV are stored as
fractions (0..1) while the sensitivity CSV stores percent (0..100); the
combined CSV harmonises everything to percent.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(
    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix"
)
WANDB_ROOT = REPO_ROOT / "evaluation" / "data" / "raw" / "wandb"

SOURCES: list[tuple[str, set[int]]] = [
    ("labelmix_ejafa_hyperparameter_sweep", {7, 8, 9, 10}),
    ("hyperparameter_sweep", {3, 4, 5, 6}),
]

TARGET_ALPHAS: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
ALPHA_TOL = 1e-6

OUT_CSV = (
    REPO_ROOT
    / "evaluation"
    / "data"
    / "processed"
    / "hyperparameter_sensitivity_vit_wee.csv"
)

# Baseline aggregates produced by evaluation/scripts/aggregate_per_experiment.py
BASELINE_CSV = (
    REPO_ROOT
    / "evaluation"
    / "data"
    / "processed"
    / "in1k_per_experiment_short.csv"
)

# Combined CSV consumed by the plotting script.
COMBINED_CSV = (
    REPO_ROOT
    / "evaluation"
    / "data"
    / "processed"
    / "hyperparameter_sensitivity_vit_wee_with_baselines.csv"
)

# Which baseline 'type' values to pull, and how to rename them in the combined CSV.
BASELINE_TYPES: dict[str, str] = {
    "baseline": "baseline",  # mixup + cutmix
    "bare": "bare",          # single-image aug only
}

TARGET_BASELINE_MODEL = "vit_wee_patch16_reg1_gap_256"


def _snap_alpha(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    for a in TARGET_ALPHAS:
        if abs(xf - a) < ALPHA_TOL:
            return a
    return None


def _load_run(run_dir: Path):
    meta_p = run_dir / "metadata.json"
    cfg_p = run_dir / "config.yaml"
    sum_p = run_dir / "summary.json"
    if not (meta_p.is_file() and cfg_p.is_file()):
        return None
    with meta_p.open() as f:
        meta = json.load(f)
    with cfg_p.open() as f:
        cfg = yaml.safe_load(f)
    summary: dict = {}
    if sum_p.is_file():
        with sum_p.open() as f:
            summary = json.load(f)
    return meta, cfg, summary


def _row_from_run(run_dir: Path, source_dir: str, target_ks: set[int]):
    loaded = _load_run(run_dir)
    if loaded is None:
        return None
    meta, cfg, summary = loaded

    if cfg.get("labelmix") is not True:
        return None
    if meta.get("state") != "finished":
        return None
    # alpha must be held constant (i.e. no alpha scheduling)
    if cfg.get("labelmix_schedule") != "fixed":
        return None

    kmin, kmax = cfg.get("labelmix_k_min"), cfg.get("labelmix_k_max")
    if kmin is None or kmax is None or kmin != kmax:
        return None
    if int(kmin) not in target_ks:
        return None
    # k is effectively constant only if k_min == k_max (already checked above).
    # `labelmix_k_schedule` may be 'linear' in the clean imagenet1k grid because
    # scheduling is a no-op when k_min == k_max, so we do not filter on it.

    amin, amax = cfg.get("labelmix_alpha_min"), cfg.get("labelmix_alpha_max")
    if amin is None or amax is None:
        return None
    try:
        if abs(float(amin) - float(amax)) > ALPHA_TOL:
            return None
    except (TypeError, ValueError):
        return None

    alpha = _snap_alpha(amin)
    if alpha is None:
        return None

    # No reverse-scheduling variants
    if cfg.get("labelmix_reverse") is True:
        return None
    if cfg.get("labelmix_k_reverse") is True:
        return None
    # For the hyperparameter_sweep directory we want the clean imagenet1k baseline
    # grid: no aspect-ratio sampling filter, no min-side-px filter.  The
    # labelmix_ejafa_hyperparameter_sweep runs predate these keys, so only
    # apply this constraint if the field is explicitly True.
    if cfg.get("labelmix_sampling") is True:
        return None

    eval_top1 = summary.get("eval_top1")
    eval_ece = summary.get("eval_ece")
    if eval_top1 is None or eval_ece is None:
        return None

    k = int(kmin)
    model = cfg.get("model")
    dataset = cfg.get("dataset")
    loss = cfg.get("labelmix_loss")
    seed = cfg.get("seed")

    model_tag = "vit-wee" if isinstance(model, str) and "vit_wee" in model else str(model)
    dataset_tag = "in1k" if isinstance(dataset, str) and "imagenet-1k" in dataset else str(dataset)

    rowname = f"{model_tag}__{dataset_tag}__k{k}__a{alpha:g}__{loss}"

    return {
        "rowname": rowname,
        "model": model,
        "dataset": dataset,
        "loss": loss,
        "alpha": alpha,
        "k": k,
        "seed": seed,
        "eval_top1": eval_top1,
        "eval_ece": eval_ece,
        "source_dir": source_dir,
        "run_id": run_dir.name,
        "display_name": meta.get("display_name"),
    }


def extract():
    rows = []
    for subdir, target_ks in SOURCES:
        base = WANDB_ROOT / subdir
        if not base.is_dir():
            print(f"[WARN] missing source dir: {base}", file=sys.stderr)
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            try:
                row = _row_from_run(child, subdir, target_ks)
            except Exception as e:
                print(f"[WARN] skipping {child.name}: {e}", file=sys.stderr)
                continue
            if row is not None:
                rows.append(row)
    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rowname",
        "model",
        "dataset",
        "loss",
        "alpha",
        "k",
        "seed",
        "eval_top1",
        "eval_ece",
        "source_dir",
        "run_id",
        "display_name",
    ]
    rows_sorted = sorted(
        rows,
        key=lambda r: (str(r["loss"]), int(r["k"]), float(r["alpha"]), str(r["run_id"])),
    )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)


def print_summary(rows):
    from collections import Counter

    print(f"total rows extracted: {len(rows)}", flush=True)
    print(f"by loss: {Counter(r['loss'] for r in rows).most_common()}", flush=True)
    print("by (source_dir, loss):", flush=True)
    for key, v in sorted(Counter((r["source_dir"], r["loss"]) for r in rows).items()):
        print(f"  {key} -> {v}", flush=True)

    TARGET_KS_ALL = {3, 4, 5, 6, 7, 8, 9, 10}
    present = {(r["loss"], r["k"], r["alpha"]) for r in rows}
    for loss in sorted({r["loss"] for r in rows}):
        missing = [
            (k, a)
            for k in sorted(TARGET_KS_ALL)
            for a in TARGET_ALPHAS
            if (loss, k, a) not in present
        ]
        if missing:
            print(
                f"[{loss}] MISSING {len(missing)} target (k, alpha) combos:",
                flush=True,
            )
            for k, a in missing:
                print(f"    k={k}, alpha={a}", flush=True)
        else:
            print(f"[{loss}] full target grid covered.", flush=True)


def _load_baseline_rows(baseline_csv: Path) -> list[dict]:
    """Read `in1k_per_experiment_short.csv` and return the baseline/bare rows
    for the vit_wee model, harmonising units with the sensitivity CSV.

    Returns rows with the schema used by the combined CSV (see
    `_write_combined_csv`).
    """
    if not baseline_csv.is_file():
        print(f"[WARN] missing baseline CSV: {baseline_csv}", file=sys.stderr)
        return []

    with baseline_csv.open() as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    out: list[dict] = []
    for row in all_rows:
        if row.get("model") != TARGET_BASELINE_MODEL:
            continue
        t = row.get("type")
        if t not in BASELINE_TYPES:
            continue
        try:
            top1_mean = float(row["top1_acc_mean"])
            top1_std = float(row["top1_acc_std"])
            # ECE in baseline CSV is a fraction (0..1); sensitivity uses percent.
            ece_mean = float(row["ece/n_bins=15_mean"]) * 100.0
            ece_std = float(row["ece/n_bins=15_std"]) * 100.0
        except (KeyError, TypeError, ValueError) as e:
            print(f"[WARN] skipping baseline row {t}: {e}", file=sys.stderr)
            continue

        out.append(
            {
                "kind": "baseline",
                "label": BASELINE_TYPES[t],
                "loss": "",
                "alpha": "",
                "k": "",
                "n_seeds": int(row.get("n_seeds", 0)) if row.get("n_seeds") else "",
                "eval_top1_mean": top1_mean,
                "eval_top1_std": top1_std,
                "eval_ece_mean": ece_mean,
                "eval_ece_std": ece_std,
            }
        )
    return out


def _sensitivity_rows_for_combined(rows: list[dict]) -> list[dict]:
    """Reshape per-run sensitivity rows to the combined-CSV schema. One seed
    per (loss, k, alpha) cell => std columns are empty.
    """
    out = []
    for r in rows:
        out.append(
            {
                "kind": "labelmix",
                "label": r["loss"],
                "loss": r["loss"],
                "alpha": r["alpha"],
                "k": r["k"],
                "n_seeds": 1,
                "eval_top1_mean": r["eval_top1"],
                "eval_top1_std": "",
                "eval_ece_mean": r["eval_ece"],
                "eval_ece_std": "",
            }
        )
    return out


def _write_combined_csv(sens_rows: list[dict], baseline_rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",            # 'labelmix' | 'baseline'
        "label",           # loss name, or 'baseline' / 'bare'
        "loss",            # sensitivity-only
        "alpha",           # sensitivity-only
        "k",               # sensitivity-only
        "n_seeds",
        "eval_top1_mean",
        "eval_top1_std",
        "eval_ece_mean",   # percent
        "eval_ece_std",    # percent
    ]
    rows = baseline_rows + sens_rows  # baselines first for easy visual inspection
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    rows = extract()
    write_csv(rows, OUT_CSV)
    print_summary(rows)
    print(f"CSV written to: {OUT_CSV}", flush=True)

    baseline_rows = _load_baseline_rows(BASELINE_CSV)
    sens_rows = _sensitivity_rows_for_combined(rows)
    _write_combined_csv(sens_rows, baseline_rows, COMBINED_CSV)
    print(
        f"Combined CSV (with {len(baseline_rows)} baseline rows) written to: {COMBINED_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
