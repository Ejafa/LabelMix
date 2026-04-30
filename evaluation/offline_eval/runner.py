"""Top-level orchestration: ``evaluate_run`` and ``evaluate_mapping``."""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import suppress
from functools import partial
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from ..config import EvalConfig
from ..types import EvalResult
from .data_builder import build_val_loader
from .inference import collect_logits, run_metrics
from .io import load_args_yaml, write_results_csv
from .model_builder import build_model


_logger = logging.getLogger("evaluation.offline_eval.runner")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "run"


def evaluate_run(name: str, run_dir: str, cfg: EvalConfig) -> EvalResult:
    """Evaluate a single run directory and return its metric bundle.

    Side effects:
        * Loads ``args.yaml`` and the checkpoint from ``run_dir``.
        * Optionally writes ``<raw_logits_dir>/<slug(name)>.pt`` containing
          ``{"logits": ..., "targets": ...}`` when ``cfg.save_raw_logits``.
    """
    t0 = time.time()
    train_args = load_args_yaml(run_dir)
    checkpoint_path = os.path.join(run_dir, cfg.checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{cfg.checkpoint_name}' not found in {run_dir}"
        )

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if cfg.amp:
        amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
    else:
        amp_autocast = suppress

    model = build_model(train_args, device, checkpoint_path, cfg.use_ema)
    loader, _data_config = build_val_loader(train_args, model, cfg, device)

    _logger.info(
        "[%s] running eval on %d batches (batch_size=%d)",
        name, len(loader), cfg.batch_size,
    )
    logits, targets = collect_logits(model, loader, device, amp_autocast)

    if cfg.save_raw_logits and cfg.raw_logits_dir:
        os.makedirs(cfg.raw_logits_dir, exist_ok=True)
        out_path = os.path.join(cfg.raw_logits_dir, f"{_slugify(name)}.pt")
        torch.save({"logits": logits, "targets": targets, "name": name}, out_path)
        _logger.info("[%s] wrote raw logits to %s", name, out_path)

    metrics = run_metrics(logits, targets, cfg.metrics)

    # Free GPU before the next run.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    _logger.info(
        "[%s] done in %.1fs | %s",
        name, elapsed,
        " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
    )

    return EvalResult(
        name=name,
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        args=dict(train_args),
        metrics=metrics,
        eval_time=elapsed,
    )


def _load_done_names(output_csv: str,
                     extra_csvs: Iterable[str] = ()) -> set[str]:
    """Names that should be skipped on the next invocation.

    A row counts as "done" when it has a non-empty ``name`` AND its ``error``
    column is empty: failed rows are retried on resume so transient errors
    (e.g. a checkpoint that wasn't yet synced) don't get permanently baked in.

    ``extra_csvs`` lets callers point at other result files (e.g. the merged
    ``in1k.csv``) whose names should also be skipped. Missing files are
    silently ignored so globs that match nothing are safe.
    """
    import csv
    done: set[str] = set()
    for path in (output_csv, *extra_csvs):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "name" not in reader.fieldnames:
                    continue
                has_error_col = "error" in reader.fieldnames
                for row in reader:
                    name = row.get("name")
                    if not name:
                        continue
                    if has_error_col and (row.get("error") or "").strip():
                        continue
                    done.add(name)
        except Exception:  # noqa: BLE001 - corrupt file -> ignore just that file.
            _logger.warning("Could not parse existing CSV %s; ignoring.", path)
    return done


def evaluate_mapping(
    mapping: Mapping[str, str],
    output_csv: str,
    cfg: Optional[EvalConfig] = None,
    continue_on_error: bool = True,
    skip_names: Optional[Iterable[str]] = None,
    skip_from_csvs: Iterable[str] = (),
) -> List[EvalResult]:
    """Evaluate ``{name: run_dir}`` and write a single CSV.

    CSV columns are ``fixed_columns + args_columns + metric_cols + [eval_time_s, error]``.
    Adding a new metric to :data:`METRIC_REGISTRY` automatically extends the
    CSV without any pipeline change.

    The CSV is rewritten after every run so the pipeline is crash-safe and
    resumable: names already present in ``output_csv`` (with no error) are
    skipped on the next invocation.

    Extra skip sources:
        * ``skip_names``: explicit names to skip (no I/O).
        * ``skip_from_csvs``: additional result CSVs whose ``name`` column is
          also treated as "already done". Useful to avoid re-evaluating runs
          that are already covered by a merged ``in1k.csv`` or sibling shards.

    Failed rows (``error`` column non-empty) are *not* considered done, so
    resuming automatically retries them.
    """
    cfg = cfg or EvalConfig()
    results: List[EvalResult] = []
    errors: Dict[str, str] = {}

    done = _load_done_names(output_csv, skip_from_csvs)
    if skip_names:
        done |= set(skip_names)
    if done:
        _logger.info("Resuming: %d entries already done will be skipped.",
                     len(done))

    total = len(mapping)
    skipped = 0
    for idx, (name, run_dir) in enumerate(mapping.items(), start=1):
        if name in done:
            skipped += 1
            _logger.info("[%d/%d] skip (already done): %s", idx, total, name)
            continue
        try:
            results.append(evaluate_run(name, run_dir, cfg))
        except Exception as exc:  # noqa: BLE001
            _logger.exception("[%s] FAILED: %s", name, exc)
            if not continue_on_error:
                raise
            errors[name] = f"{type(exc).__name__}: {exc}"

        # Persist after every run so a mid-shard crash doesn't lose progress.
        # When resuming, merge with rows already on disk.
        try:
            write_results_csv(output_csv, results, cfg, errors, mapping,
                              merge_existing=True)
        except TypeError:
            # Older write_results_csv without merge support: full rewrite.
            write_results_csv(output_csv, results, cfg, errors, mapping)

    _logger.info(
        "Wrote %d result row(s) to %s (%d skipped, %d new, %d failed)",
        len(results) + len(errors), output_csv,
        skipped, len(results), len(errors),
    )
    return results