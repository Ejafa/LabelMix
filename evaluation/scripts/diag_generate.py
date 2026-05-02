"""CLI: generate the composed diagnostic evaluation dataset.

Usage::

    python -m evaluation.scripts.diag_generate \\
        --out-dir evaluation/data/raw/diagnostic/composed \\
        --k-values 3 4 5 6 \\
        --samples-per-k 500 \\
        --alpha 0.5 \\
        --sampling-max-aspect 15 \\
        --split validation
"""
from __future__ import annotations

import sys

from evaluation.diagnostic.generate import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
