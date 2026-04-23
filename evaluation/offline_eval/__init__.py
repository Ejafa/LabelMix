"""Offline evaluation of trained LabelMix checkpoints.

The subpackage turns a ``{name: run_dir}`` mapping into a single CSV (plus,
optionally, raw logits tensors) by:

1. rebuilding the training-time model (``model_builder``),
2. rebuilding the validation loader (``data_builder``),
3. forwarding the whole eval set (``inference``),
4. applying every registered metric,
5. serializing results (``io``) from a thin orchestrator (``runner``).
"""
from .io import load_args_yaml, load_mapping, write_results_csv
from .runner import evaluate_mapping, evaluate_run

__all__ = [
    "evaluate_run",
    "evaluate_mapping",
    "load_mapping",
    "load_args_yaml",
    "write_results_csv",
]
