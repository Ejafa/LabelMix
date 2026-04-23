"""CLI: run the offline evaluator over a ``{name: run_dir}`` mapping.

Usage::

    python -m evaluation.scripts.run_eval \\
        --mapping my_runs.yaml \\
        --output-csv evaluation/data/raw/eval_csv/my_eval.csv \\
        [--save-raw-logits] [--checkpoint-name model_best.pth.tar] ...

``my_runs.yaml`` is a ``{name: run_dir}`` map.  The richer dict form
``{name: {path: ..., ...}}`` is also supported.  The same file can be
provided as JSON (``*.json``).
"""
from __future__ import annotations

import argparse
import sys

from ..common import RAW_LOGITS_DIR, ensure_dirs, setup_logging
from ..config import EvalConfig
from ..metrics import METRIC_REGISTRY
from ..offline_eval import evaluate_mapping, load_mapping


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline metric evaluation for trained runs.")
    p.add_argument("--mapping", required=True,
                   help="YAML/JSON file mapping {name: run_dir}.")
    p.add_argument("--output-csv", required=True,
                   help="Single CSV file to write all results to "
                        "(conventionally under evaluation/data/raw/eval_csv/).")
    p.add_argument("--checkpoint-name", default="model_best.pth.tar")
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                   help="Load EMA weights if present (default: True).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--data-dir", default=None,
                   help="Override data_dir from args.yaml.")
    p.add_argument("--dataset", default=None,
                   help="Override dataset spec from args.yaml.")
    p.add_argument("--val-split", default=None,
                   help="Override val_split from args.yaml.")
    p.add_argument("--metrics", nargs="*", default=None,
                   help=f"Subset of metrics to run. Default: all registered "
                        f"({list(METRIC_REGISTRY.keys())}).")
    p.add_argument("--args-columns", nargs="*", default=None,
                   help="Override the args.yaml keys surfaced as CSV columns.")
    p.add_argument("--save-raw-logits", action="store_true",
                   help="Also dump {logits, targets} to data/raw/logits/ per run.")
    p.add_argument("--raw-logits-dir", default=str(RAW_LOGITS_DIR),
                   help=f"Where to write raw logits (default: {RAW_LOGITS_DIR}).")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort immediately on the first failing run.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    ensure_dirs()

    mapping = load_mapping(args.mapping)
    if not mapping:
        print(f"Mapping {args.mapping} is empty.", file=sys.stderr)
        return 2

    cfg_kwargs = dict(
        checkpoint_name=args.checkpoint_name,
        use_ema=args.use_ema,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        data_dir_override=args.data_dir,
        dataset_override=args.dataset,
        val_split_override=args.val_split,
        metrics=args.metrics,
        save_raw_logits=args.save_raw_logits,
        raw_logits_dir=args.raw_logits_dir,
    )
    if args.args_columns is not None:
        cfg_kwargs["args_columns"] = args.args_columns
    cfg = EvalConfig(**cfg_kwargs)

    evaluate_mapping(
        mapping=mapping,
        output_csv=args.output_csv,
        cfg=cfg,
        continue_on_error=not args.stop_on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
