"""CLI: compute Gradient x Input attribution maps for composed samples.

Usage::

    python -m evaluation.scripts.diag_gradient_x_input \\
        --manifest evaluation/data/raw/diagnostic/composed/manifest.jsonl \\
        --mapping  my_runs.yaml \\
        --out-dir  evaluation/data/raw/diagnostic/attributions \\
        --max-samples-per-k 20
"""
from __future__ import annotations

import sys

from evaluation.diagnostic.attribution import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
