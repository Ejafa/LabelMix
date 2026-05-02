"""CLI entry-point: stitch per-class ``class_<id>_boxed.png`` tiles across the
trained models into one compound PNG per (sample, class).

Delegates to :mod:`evaluation.diagnostic.compose_boxed_grid`.
"""
from evaluation.diagnostic.compose_boxed_grid import main


if __name__ == "__main__":
    main()
