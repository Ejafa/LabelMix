"""Paper-ready plots, one file per figure.

Design rules for this subpackage:

1. **One plot per file.** Most modules expose a single ``plot(df) -> Figure``
   function plus a ``main()`` CLI entry point; :mod:`augmentation_showcase`
   is a raw image renderer with a CLI-only entry point.
2. **Reads only from ``data/processed/``.** Exceptions are
   :mod:`reliability_diagram`, which needs per-sample logits and therefore
   reads ``data/raw/logits/``, and :mod:`augmentation_showcase`, which renders
   input-space augmentation illustrations from source images.  No plot script
   may read a metric CSV from ``data/raw/``.
3. **Writes only to ``data/processed/figures/``.** Both PDF and PNG copies
   are emitted via :func:`._style.savefig`.
4. **Consistent style.** Every script starts with
   :func:`._style.apply_paper_style` so all figures share fonts, colors
   and sizes.

Run augmentation showcase with a local source-image pool::

    python evaluation/plots/augmentation_showcase.py \
        --source-dir evaluation/data/raw/hf_imagenet1k_animal_samples \
        --out-dir evaluation/data/processed/figures/augmentation_showcase_hf_imagenet1k_animals
"""
from . import _style

__all__ = ["_style"]
