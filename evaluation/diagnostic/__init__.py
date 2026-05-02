"""Diagnostic evaluation pipeline for the LabelMix image augmentation.

Three independent stages, each addressable via :mod:`evaluation.scripts`:

    1. :mod:`evaluation.diagnostic.generate`  -- synthesize composed samples
       (and patch masks + metadata) from ImageNet-1k held-out images.
    2. :mod:`evaluation.diagnostic.infer`     -- run trained models on the
       composed samples and dump raw logits + per-sample context.
    3. :mod:`evaluation.diagnostic.attribution` -- compute Gradient x Input
       attribution maps per present class, for a subset of samples.

The stages are intentionally decoupled so a single composed dataset can be
evaluated by many models, and attribution can be recomputed without
regenerating images.  The final statistical analysis is out of scope for
this module.
"""
from __future__ import annotations

from .compose import ComposedSample, compose_labelmix_with_mask
from .source_pool import SourcePoolEntry, load_source_pool

__all__ = [
    "ComposedSample",
    "compose_labelmix_with_mask",
    "SourcePoolEntry",
    "load_source_pool",
]
