"""CLI: overlay Gradient x Input heatmaps on the composed inputs and
draw a red box around the patch(es) belonging to each target class.

Consumes the artifacts produced by
``evaluation.scripts.diag_gradient_x_input`` + the dataset manifest.

Usage::

    python -m evaluation.scripts.diag_visualize_attributions \\
        --manifest evaluation/data/raw/diagnostic/composed/manifest.jsonl \\
        --attributions-dir evaluation/data/raw/diagnostic/attributions \\
        -v
"""
from __future__ import annotations

import sys

from evaluation.diagnostic.visualize import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
