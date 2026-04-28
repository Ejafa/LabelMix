"""Paper-figure generators for the LabelMix paper.

Each module in this package renders one self-contained figure (or a set of
figures that belong together) to ``evaluation/data/processed/figures/``.
Unlike ``evaluation.plots.*``, these are not consumers of the eval pipeline:
they render *input-space* illustrations (what the augmentations look like)
and therefore operate directly on source images + the augmentation code
in ``timm.data``.
"""
