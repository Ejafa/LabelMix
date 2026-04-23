"""CLI: render every paper plot in one go.

Discovers every non-private module under :mod:`evaluation.plots` and invokes
its ``main()``.  Plot scripts that fail (e.g. missing input) are reported
but do not abort the run.

Usage::

    python -m evaluation.scripts.make_all_plots
"""
from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import sys

from ..common import ensure_dirs, setup_logging
from .. import plots as _plots_pkg


_logger = logging.getLogger(__name__)


def _iter_plot_modules() -> list[str]:
    out: list[str] = []
    for info in pkgutil.iter_modules(_plots_pkg.__path__):
        if info.ispkg or info.name.startswith("_"):
            continue
        out.append(f"{_plots_pkg.__name__}.{info.name}")
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None,
                   help="Subset of plot module names (without the package "
                        "prefix) to render. Default: all.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    ensure_dirs()

    modules = _iter_plot_modules()
    if args.only:
        allow = set(args.only)
        modules = [m for m in modules if m.rsplit(".", 1)[-1] in allow]

    if not modules:
        _logger.warning("No plot modules found.")
        return 1

    exit_code = 0
    for mod_name in modules:
        short = mod_name.rsplit(".", 1)[-1]
        _logger.info("=== %s ===", short)
        try:
            mod = importlib.import_module(mod_name)
            rc = mod.main([]) if hasattr(mod, "main") else 0
            if rc:
                _logger.error("[%s] returned non-zero exit code %s", short, rc)
                exit_code = rc
        except SystemExit as e:
            # argparse-driven scripts may SystemExit; only non-zero is an error.
            if e.code:
                _logger.error("[%s] SystemExit(%s)", short, e.code)
                exit_code = int(e.code) if isinstance(e.code, int) else 1
        except Exception as exc:  # noqa: BLE001
            _logger.exception("[%s] failed: %s", short, exc)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
