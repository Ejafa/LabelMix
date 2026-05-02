"""CLI: run trained models over the composed diagnostic set and dump logits.

Usage::

    python -m evaluation.scripts.diag_export_logits \\
        --manifest evaluation/data/raw/diagnostic/composed/manifest.jsonl \\
        --mapping  my_runs.yaml \\
        --out-dir  evaluation/data/raw/diagnostic/logits
"""
from __future__ import annotations

import sys

from evaluation.diagnostic.infer import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
