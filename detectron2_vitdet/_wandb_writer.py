"""Weights & Biases integration for the ViTDet training loop.

This module adds a minimal ``WandbWriter`` that plugs into detectron2's
``PeriodicWriter`` hook (same contract as ``CommonMetricPrinter``,
``JSONWriter`` and ``TensorboardXWriter``). It also exposes a tiny
``init_wandb_from_cfg`` helper to spin up a run once at the start of
training.

Design notes
------------
* Main-rank only: we deliberately rely on the caller (``train_net.py``) to
  guard construction behind ``comm.is_main_process()``. DDP workers never
  touch ``wandb``.
* Fully optional: if the ``wandb`` package is missing, if
  ``WANDB_MODE=disabled``, or if ``wandb.init`` raises (e.g. offline node),
  we log a warning and return a no-op writer. Detection training must
  never crash because of logging.
* Run name = ``os.path.basename(cfg.train.output_dir)``. This matches the
  job names emitted by ``generate_vitdet_jobs.py`` (e.g.
  ``vitdet-wee__coco__img256__seed42``), so W&B runs line up 1-to-1 with
  scheduler jobs.
"""
from __future__ import annotations

import atexit
import logging
import os
from typing import Any, Dict, Optional

from detectron2.utils.events import EventWriter, get_event_storage

_logger = logging.getLogger("detectron2.wandb")

# Populated by ``init_wandb_from_cfg``. Kept at module scope so that both the
# writer and the atexit hook see the same handle.
_WANDB_RUN: Any = None


def _import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception as e:  # pragma: no cover - defensive
        _logger.warning("wandb not importable (%s); W&B logging disabled.", e)
        return None


def _cfg_to_flat_dict(cfg: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Best-effort flattening of a LazyConfig / OmegaConf / dict tree.

    We only keep JSON-serialisable leaves; everything else is stringified so
    that W&B's ``config`` panel does not blow up on e.g. Python callables
    stored inside LazyConfig nodes.
    """
    if out is None:
        out = {}
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore
        if isinstance(cfg, (DictConfig, ListConfig)):
            cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            _cfg_to_flat_dict(v, f"{prefix}{k}.", out)
    elif isinstance(cfg, (list, tuple)):
        # Store as a single stringified blob to keep the panel readable.
        out[prefix.rstrip(".")] = str(cfg)[:500]
    else:
        key = prefix.rstrip(".")
        if isinstance(cfg, (str, int, float, bool)) or cfg is None:
            out[key] = cfg
        else:
            out[key] = str(cfg)[:500]
    return out


def init_wandb_from_cfg(cfg: Any, *, extra_tags: Optional[list] = None) -> Any:
    """Initialise a W&B run from a LazyConfig. Safe to call multiple times.

    Returns the live ``wandb`` run, or ``None`` if W&B is unavailable /
    disabled. Honours the ``WANDB_MODE`` env var (``disabled``/``offline``/
    ``online``) and the ``WANDB_PROJECT`` / ``WANDB_ENTITY`` overrides.
    """
    global _WANDB_RUN
    if _WANDB_RUN is not None:
        return _WANDB_RUN

    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode == "disabled":
        _logger.info("WANDB_MODE=disabled; skipping W&B init.")
        return None

    wandb = _import_wandb()
    if wandb is None:
        return None

    output_dir = str(getattr(cfg.train, "output_dir", "") or "./output")
    run_name = os.path.basename(os.path.normpath(output_dir)) or "vitdet-run"
    project = os.environ.get("WANDB_PROJECT", "labelmix-vitdet")
    entity = os.environ.get("WANDB_ENTITY") or None
    tags = ["vitdet"] + list(extra_tags or [])

    try:
        _WANDB_RUN = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            dir=output_dir,
            tags=tags,
            config=_cfg_to_flat_dict(cfg),
            resume="allow",
            reinit=False,
        )
    except Exception as e:
        _logger.warning("wandb.init failed (%s); continuing without W&B.", e)
        _WANDB_RUN = None
        return None

    # Make sure the run is flushed / closed even on crashes.
    def _finish_run():  # pragma: no cover - best-effort teardown
        try:
            if _WANDB_RUN is not None:
                _WANDB_RUN.finish()
        except Exception:
            pass

    atexit.register(_finish_run)
    _logger.info("W&B run initialised: project=%s name=%s", project, run_name)
    return _WANDB_RUN


class WandbWriter(EventWriter):
    """Mirror detectron2's ``EventStorage`` scalars to the active W&B run.

    Mirrors the ``JSONWriter`` / ``TensorboardXWriter`` contract:
    ``write()`` is called by ``PeriodicWriter`` at the configured period,
    and reads from the current ``EventStorage``. All logs are keyed by
    ``train/iter`` so that W&B's step axis matches detectron2's iteration.
    """

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._last_write = -1

    def write(self) -> None:
        if _WANDB_RUN is None:
            return
        storage = get_event_storage()

        to_log: Dict[str, float] = {}
        new_last_write = self._last_write
        for k, (v, iter_) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter_ <= self._last_write:
                continue
            to_log[k] = float(v)
            new_last_write = max(new_last_write, int(iter_))

        # Histograms / images are ignored on purpose — scalars are plenty.
        if to_log:
            try:
                _WANDB_RUN.log(to_log, step=int(storage.iter))
            except Exception as e:  # pragma: no cover - defensive
                _logger.warning("wandb.log failed (%s); disabling further logs.", e)
                self._disable()
        self._last_write = new_last_write

    def close(self) -> None:
        # The atexit hook finishes the run globally; nothing to do here.
        pass

    @staticmethod
    def _disable() -> None:
        global _WANDB_RUN
        _WANDB_RUN = None
