"""Top-level orchestration: ``evaluate_run`` and ``evaluate_mapping``."""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import suppress
from functools import partial
from typing import Dict, List, Mapping, Optional

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


def evaluate_mapping(
    mapping: Mapping[str, str],
    output_csv: str,
    cfg: Optional[EvalConfig] = None,
    continue_on_error: bool = True,
) -> List[EvalResult]:
    """Evaluate ``{name: run_dir}`` and write a single CSV.

    CSV columns are ``fixed_columns + args_columns + metric_cols + [eval_time_s, error]``.
    Adding a new metric to :data:`METRIC_REGISTRY` automatically extends the
    CSV without any pipeline change.
    """
    cfg = cfg or EvalConfig()
    results: List[EvalResult] = []
    errors: Dict[str, str] = {}

    for name, run_dir in mapping.items():
        try:
            results.append(evaluate_run(name, run_dir, cfg))
        except Exception as exc:  # noqa: BLE001
            _logger.exception("[%s] FAILED: %s", name, exc)
            if not continue_on_error:
                raise
            errors[name] = f"{type(exc).__name__}: {exc}"

    write_results_csv(output_csv, results, cfg, errors, mapping)
    _logger.info(
        "Wrote %d result row(s) to %s",
        len(results) + len(errors), output_csv,
    )
    return results
