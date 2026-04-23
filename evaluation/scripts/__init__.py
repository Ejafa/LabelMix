"""Thin CLI wrappers for the evaluation package.

Each script here is a **pure entry point** — it only parses CLI args and
calls functions defined elsewhere.  No analysis logic lives here.

Canonical invocations::

    python -m evaluation.scripts.run_eval --mapping my_runs.yaml \\
        --output-csv evaluation/data/raw/eval_csv/my_eval.csv

    python -m evaluation.scripts.process_results \\
        --inputs evaluation/data/raw/eval_csv/*.csv \\
        --output-dir evaluation/data/processed

    python -m evaluation.scripts.make_all_plots

    python -m evaluation.scripts.backup_runs \\
        --backup-root ../backup
"""
