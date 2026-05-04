"""CLI: compute area-vs-logit diagnostic metrics across multiple seeds.

Pipeline (always multi-seed):

    1. Load ``area_logit_models.yaml`` which maps ``short_name -> run_dir``.
       Every ``run_dir`` is pinned at ``seed=42``; other seeds are obtained by
       string-substituting ``seed=42 -> seed=<S>``, matching the training
       directory naming convention.
    2. For each ``(short_name, seed)``:
       a. If ``<logits-dir>/seed<S>/<short_name>/logits.pt`` is missing or
          ``--force``, run inference over the composed manifest
          (reuses :func:`evaluation.diagnostic.infer.export_logits_for_run`).
       b. Otherwise reuse the cached artifact.
    3. Compute per-k and aggregated metrics via
       :func:`evaluation.diagnostic.area_logit_metrics.compute_metrics_for_model`.
    4. Emit a single CSV at ``diagnostic_area_logit_metrics.csv`` with one
       row per (model, k) reporting the across-seed mean and std of each
       metric.

The script fails hard if any (model, seed) combination cannot produce
metrics; every listed model must successfully evaluate on *every*
requested seed.

Defaults:
    --manifest   evaluation/data/raw/diagnostic/composed/manifest.jsonl
    --logits-dir evaluation/data/raw/diagnostic/logits_multiseed
    --out-csv    evaluation/data/processed/diagnostic_area_logit_metrics.csv
    --mapping    evaluation/diagnostic/area_logit_models.yaml
    --seeds      42 43 44
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from evaluation.common import (
    DIAGNOSTIC_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    setup_logging,
)
from evaluation.diagnostic.area_logit_metrics import (
    K_ALLOWED,
    MIN_AREA_DIFF,
    ModelMetricsRow,
    compute_metrics_for_model,
)
from evaluation.diagnostic.infer import (
    InferConfig,
    _load_manifest,
    _manifest_sha256,
    _slug,
    export_logits_for_run,
)
from evaluation.offline_eval.io import load_mapping

_logger = logging.getLogger("evaluation.scripts.diag_area_logit_metrics")


_DEFAULT_MAPPING = (
    Path(__file__).resolve().parent.parent / "diagnostic" / "area_logit_models.yaml"
)
_DEFAULT_MANIFEST = DIAGNOSTIC_DIR / "composed" / "manifest.jsonl"
_DEFAULT_LOGITS_DIR = DIAGNOSTIC_DIR / "logits_multiseed"
_DEFAULT_OUT_CSV = PROCESSED_DIR / "diagnostic_area_logit_metrics.csv"
_DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)
_REQUIRED_NUM_SEEDS = 3  # we always want 3 seeds of data; errors otherwise
_ANCHOR_SEED = 42  # run_dirs in the mapping yaml are pinned at seed=42


# ---------------------------------------------------------------------------
# Seed path handling + logits caching
# ---------------------------------------------------------------------------


def _seeded_run_dir(run_dir: str, seed: int) -> str:
    """Return ``run_dir`` with ``seed=42`` replaced by ``seed=<seed>``."""
    anchor = f"seed={_ANCHOR_SEED}"
    if anchor not in run_dir:
        raise ValueError(
            f"run_dir {run_dir!r} does not contain '{anchor}'; the mapping "
            "yaml must use seed=42 as the anchor for every entry."
        )
    return run_dir.replace(anchor, f"seed={int(seed)}")


def _logits_path(logits_dir: Path, seed: int, short_name: str) -> Path:
    return logits_dir / f"seed{int(seed)}" / _slug(short_name) / "logits.pt"


def _ensure_logits(
    short_name: str,
    seeded_run_dir: str,
    seed: int,
    *,
    manifest_path: Path,
    logits_dir: Path,
    infer_cfg_kwargs: Mapping[str, object],
    force: bool,
) -> Path:
    """Return the path of ``logits.pt`` for ``(short_name, seed)``; generate if needed."""
    out = _logits_path(logits_dir, seed, short_name)
    if out.is_file() and not force:
        _logger.info("[%s|seed=%d] reusing existing logits at %s",
                     short_name, seed, out)
        return out

    per_seed_out_dir = logits_dir / f"seed{int(seed)}"
    _logger.info("[%s|seed=%d] running inference -> %s",
                 short_name, seed, out)
    cfg = InferConfig(
        manifest_path=str(manifest_path),
        out_dir=str(per_seed_out_dir),
        **dict(infer_cfg_kwargs),
    )
    rows, _meta = _load_manifest(cfg.manifest_path)
    manifest_sha = _manifest_sha256(cfg.manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(cfg.manifest_path))
    written = export_logits_for_run(
        short_name, seeded_run_dir, rows, cfg,
        manifest_dir=manifest_dir, manifest_sha=manifest_sha,
    )
    return Path(written)


# ---------------------------------------------------------------------------
# Across-seed aggregation
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(vals) < 2:
        return float("nan")
    return float(statistics.stdev(vals))


def _k_sort_key(k: str) -> Tuple[int, int]:
    """Sort keys so numeric k's come first (ascending) and ``"all"`` comes last."""
    try:
        return (0, int(k))
    except ValueError:
        return (1, 0)


def _aggregate_across_seeds(
    per_seed_rows: Mapping[int, List[ModelMetricsRow]],
    *,
    model_order: Sequence[str],
) -> List[Dict[str, object]]:
    """Produce one row per (model, k) with mean/std across seeds."""
    # (model, k) -> list of rows (one per seed that produced it).
    groups: Dict[Tuple[str, str], List[ModelMetricsRow]] = {}
    for seed, rows in per_seed_rows.items():
        for r in rows:
            groups.setdefault((r.model, r.k), []).append(r)

    # Deterministic order: follow mapping order for models, ascending k, then "all".
    order_index = {name: i for i, name in enumerate(model_order)}
    sorted_keys = sorted(
        groups.keys(),
        key=lambda mk: (order_index.get(mk[0], len(order_index)), _k_sort_key(mk[1])),
    )

    out: List[Dict[str, object]] = []
    for (model, k_label) in sorted_keys:
        rs = groups[(model, k_label)]
        sp = [r.spearman_mean for r in rs]
        pa = [r.pair_acc for r in rs]
        ta = [r.top_acc for r in rs]
        out.append({
            "model": model,
            "k": k_label,
            "n_seeds": len(rs),
            "spearman_mean_mean": _mean(sp),
            "spearman_mean_std": _std(sp),
            "pair_acc_mean": _mean(pa),
            "pair_acc_std": _std(pa),
            "top_acc_mean": _mean(ta),
            "top_acc_std": _std(ta),
            "n_samples_mean": _mean([r.n_samples for r in rs]),
            "spearman_n_mean": _mean([r.spearman_n for r in rs]),
            "pair_n_mean": _mean([r.pair_n for r in rs]),
            "top_n_mean": _mean([r.top_n for r in rs]),
        })
    return out


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


_AGG_COLUMNS: Tuple[str, ...] = (
    "model", "k", "n_seeds",
    "spearman_mean_mean", "spearman_mean_std",
    "pair_acc_mean",      "pair_acc_std",
    "top_acc_mean",       "top_acc_std",
    "n_samples_mean", "spearman_n_mean", "pair_n_mean", "top_n_mean",
)


_AGG_HEADER_TEMPLATE = """\
# Mean/std of area-vs-logit diagnostic metrics across seeds.
# Generated by ``evaluation/scripts/diag_area_logit_metrics.py``.
#
# One row per (model, k). ``k == "all"`` aggregates over k in {k_list}.
# Seeds evaluated: {seed_list}.
# Standard deviations are sample std (ddof=1); empty when n_seeds < 2.
# Metric 2 pair-inclusion threshold: |area_i - area_j| >= {min_area_diff}.
#
# Columns:
#   model, k, n_seeds,
#   spearman_mean_mean / spearman_mean_std  (Metric 1 mean / std across seeds)
#   pair_acc_mean      / pair_acc_std       (Metric 2 mean / std across seeds)
#   top_acc_mean       / top_acc_std        (Metric 3 mean / std across seeds)
#   n_samples_mean, spearman_n_mean, pair_n_mean, top_n_mean
#       (averaged sample counts, reported for reference)
#
# Base mapping (short_name -> seed=42 run_dir):
{model_lines}#
"""


def _build_header(
    template: str,
    models: Mapping[str, str],
    seeds: Sequence[int],
    *,
    k_allowed: Sequence[int],
    min_area_diff: float,
) -> str:
    k_list = "{" + ", ".join(str(int(k)) for k in k_allowed) + "}"
    seed_list = ", ".join(str(int(s)) for s in seeds)
    model_lines = "".join(
        f"#   {name:>14s}  {run_dir}\n" for name, run_dir in models.items()
    )
    return template.format(
        k_list=k_list,
        seed_list=seed_list,
        min_area_diff=min_area_diff,
        model_lines=model_lines,
    )


def _format_number(x: object) -> object:
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        # 6 decimal places is plenty and still round-trips cleanly.
        return f"{x:.6f}"
    return x


def _write_agg_csv(
    rows: List[Dict[str, object]],
    out_csv: Path,
    *,
    header_text: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    passthrough = {"model", "k", "n_seeds"}
    with out_csv.open("w", newline="") as f:
        f.write(header_text)
        writer = csv.DictWriter(f, fieldnames=list(_AGG_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (r[k] if k in passthrough else _format_number(r[k]))
                for k in _AGG_COLUMNS
            })
    _logger.info("Wrote %d aggregated rows to %s", len(rows), out_csv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    mapping_path: Path,
    manifest_path: Path,
    logits_dir: Path,
    out_csv: Path,
    *,
    seeds: Sequence[int],
    force: bool,
    infer_cfg_kwargs: Mapping[str, object],
    k_allowed: Sequence[int],
    min_area_diff: float,
) -> None:
    ensure_dirs()
    mapping: Dict[str, str] = dict(load_mapping(str(mapping_path)))
    _logger.info("Mapping has %d models: %s", len(mapping), list(mapping.keys()))
    _logger.info("Evaluating seeds: %s", list(seeds))

    # Hard requirement: we always want exactly _REQUIRED_NUM_SEEDS seeds.
    if len(seeds) != _REQUIRED_NUM_SEEDS:
        raise ValueError(
            f"Expected exactly {_REQUIRED_NUM_SEEDS} seeds but got "
            f"{len(seeds)}: {list(seeds)}. Pass --seeds with "
            f"{_REQUIRED_NUM_SEEDS} values."
        )

    seeds_required = len(seeds)
    per_seed_rows: Dict[int, List[ModelMetricsRow]] = {int(s): [] for s in seeds}
    # Track, per model, how many seeds produced metrics so we can error out
    # if any model ended up with fewer than the requested number of seeds.
    model_seed_count: Dict[str, int] = {name: 0 for name in mapping}

    for short_name, run_dir_base in mapping.items():
        for seed in seeds:
            seed = int(seed)
            seeded_run_dir = _seeded_run_dir(run_dir_base, seed)
            if not os.path.isdir(seeded_run_dir):
                raise FileNotFoundError(
                    f"[{short_name}|seed={seed}] run_dir not found: "
                    f"{seeded_run_dir}"
                )

            logits_path = _ensure_logits(
                short_name, seeded_run_dir, seed,
                manifest_path=manifest_path, logits_dir=logits_dir,
                infer_cfg_kwargs=infer_cfg_kwargs, force=force,
            )

            blob = torch.load(str(logits_path), map_location="cpu",
                              weights_only=False)
            model_rows = compute_metrics_for_model(
                blob,
                model_name=short_name,
                k_allowed=k_allowed,
                min_area_diff=min_area_diff,
            )
            per_seed_rows[seed].extend(model_rows)
            model_seed_count[short_name] += 1

            agg_row = next((r for r in model_rows if r.k == "all"), None)
            if agg_row is not None:
                _logger.info(
                    "[%s|seed=%d] all k: n=%d | spearman=%.4f (n=%d) | "
                    "pair=%.4f (n=%d) | top=%.4f (n=%d)",
                    short_name, seed, agg_row.n_samples,
                    agg_row.spearman_mean, agg_row.spearman_n,
                    agg_row.pair_acc, agg_row.pair_n,
                    agg_row.top_acc, agg_row.top_n,
                )

    # Hard check: every model must have contributed exactly `seeds_required`
    # seeds. In practice we can only get here with fewer if an inner step
    # silently returned an empty set, but we still verify.
    short_models = [m for m, n in model_seed_count.items() if n != seeds_required]
    if short_models:
        details = ", ".join(
            f"{m}: {model_seed_count[m]}/{seeds_required}" for m in short_models
        )
        raise RuntimeError(
            f"Expected {seeds_required} seeds per model, but got a different "
            f"count for: {details}. All models must evaluate on every "
            f"requested seed."
        )

    # Single headline CSV: across-seed mean + std.
    agg_rows = _aggregate_across_seeds(
        per_seed_rows, model_order=list(mapping.keys()),
    )
    _write_agg_csv(
        agg_rows,
        out_csv,
        header_text=_build_header(
            _AGG_HEADER_TEMPLATE, mapping, seeds,
            k_allowed=k_allowed, min_area_diff=min_area_diff,
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING,
                   help="YAML/JSON {short_name: seed=42 run_dir}.")
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST,
                   help="Composed dataset manifest.jsonl (shared across seeds).")
    p.add_argument("--logits-dir", type=Path, default=_DEFAULT_LOGITS_DIR,
                   help="Root to cache <seed<S>>/<name>/logits.pt artifacts.")
    p.add_argument("--out-csv", type=Path, default=_DEFAULT_OUT_CSV,
                   help="Output CSV path (across-seed mean/std).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(_DEFAULT_SEEDS),
                   help="Exactly 3 seeds to evaluate (default: 42 43 44). "
                        "The script errors out if a different number of "
                        "seeds is given, or if any model fails to evaluate "
                        "on every requested seed.")
    p.add_argument("--force", action="store_true",
                   help="Re-run inference even when logits.pt exists.")
    # Inference knobs (mirroring evaluation/diagnostic/infer.py defaults).
    p.add_argument("--checkpoint-name", default="model_best.pth.tar")
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    # Metric knobs.
    p.add_argument("--k-allowed", type=int, nargs="+", default=list(K_ALLOWED),
                   help="Restrict to these k values (default: 3 4 5 6).")
    p.add_argument("--min-area-diff", type=float, default=MIN_AREA_DIFF,
                   help="Metric 2 pair-inclusion threshold (default: 0.05).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(level="INFO" if args.verbose else "WARNING")

    infer_cfg_kwargs = dict(
        checkpoint_name=args.checkpoint_name,
        use_ema=args.use_ema,
        batch_size=int(args.batch_size),
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    run(
        mapping_path=args.mapping,
        manifest_path=args.manifest,
        logits_dir=args.logits_dir,
        out_csv=args.out_csv,
        seeds=tuple(int(s) for s in args.seeds),
        force=bool(args.force),
        infer_cfg_kwargs=infer_cfg_kwargs,
        k_allowed=tuple(int(k) for k in args.k_allowed),
        min_area_diff=float(args.min_area_diff),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
