"""Experiment family packages.

Each sub-package under ``experiments/`` represents one *research campaign*
(experiment family).  A family defines:

- model config YAMLs
- common training overrides
- search space (as plain Python lists)
- a ``get_model_configs()`` function
- a ``build_common_overrides()`` function

Jobs are generated via ``jobdaemon.py generate`` and executed by the daemon.
"""
