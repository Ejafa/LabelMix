"""Paper-ready plots, one file per figure.

Design rules for this subpackage:

1. **One plot per file.** Each module exposes a single ``plot(df) -> Figure``
   function plus a ``main()`` CLI entry point.
2. **Reads only from ``data/processed/``.** The one exception is
   :mod:`reliability_diagram`, which needs per-sample logits and therefore
   reads ``data/raw/logits/``.  No plot script may read a metric CSV from
   ``data/raw/``.
3. **Writes only to ``data/processed/figures/``.** Both PDF and PNG copies
   are emitted via :func:`._style.savefig`.
4. **Consistent style.** Every script starts with
   :func:`._style.apply_paper_style` so all figures share fonts, colors
   and sizes.
"""
from . import _style

__all__ = ["_style"]
