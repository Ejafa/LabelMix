"""Thin CLI wrappers for the evaluation package.

Each script here is a **pure entry point** — it only parses CLI args and
calls functions defined elsewhere.  No analysis logic lives here.

The scripts below are listed in roughly the order you would run them in
a typical end-to-end pipeline (sync W&B -> build mapping -> evaluate ->
aggregate -> make plots -> back up).

Canonical invocations::

    # 1. Pull W&B runs into ``evaluation/data/raw/wandb/`` (incremental).
    python -m evaluation.scripts.sync_wandb \\
        --entity my-team --project labelmix --tags in1k

    # 2. Build a ``{run_name: run_dir}`` YAML mapping for a W&B group.
    python -m evaluation.scripts.build_eval_mapping \\
        --output evaluation/data/raw/in1k_mapping.yaml \\
        --checkpoint-name model_best.pth.tar

    # 3. (Optional) Shard the mapping for parallel offline evaluation.
    python -m evaluation.scripts.shard_mapping \\
        --mapping evaluation/data/raw/in1k_mapping.yaml \\
        --n-shards 4 \\
        --out-prefix evaluation/data/raw/in1k_mapping_shard

    # 4. Run the offline evaluator over a mapping (one CSV per invocation).
    python -m evaluation.scripts.run_eval \\
        --mapping evaluation/data/raw/in1k_mapping.yaml \\
        --output-csv evaluation/data/raw/eval_csv/in1k.csv

    # 5. Sanitize + aggregate per-seed eval rows into per-experiment tables.
    #    Writes two CSVs: ``<stem>_short.csv`` and ``<stem>_long.csv``.
    python -m evaluation.scripts.aggregate_per_experiment \\
        --input  evaluation/data/raw/eval_csv/in1k.csv \\
        --output evaluation/data/processed/in1k_per_experiment.csv \\
        --table-out evaluation/data/processed/in1k_per_experiment.md

    # 5b. Build the muon/adamw-vs-nadamw comparison table for the W&B
    #     ``optimizers`` group. All runs are single-seed (seed 42), so no
    #     aggregation happens: each optimizer row is emitted as-is and
    #     joined against the nadamw baseline in ``in1k.csv`` on the strict
    #     key ``(type, model, dataset, long_horizon, seed)``. The script
    #     aborts loudly if any optimizer key has no (or an ambiguous)
    #     nadamw counterpart, so the reference table can be fixed before
    #     re-running. Output columns mirror ``in1k.csv`` (raw per-run
    #     metrics, no ``_mean``/``_std``) plus a new ``optimizer`` column
    #     taking values ``muon``, ``adamw`` or ``nadamw``.
    #
    #     Prerequisite: run steps 2 + 4 pointed at the ``optimizers``
    #     group, i.e. ``--wandb-dir evaluation/data/raw/wandb/optimizers
    #     --group optimizers`` for ``build_eval_mapping``, then
    #     ``--output-csv evaluation/data/raw/eval_csv/optimizers.csv`` for
    #     ``run_eval``.
    python -m evaluation.scripts.aggregate_optimizers \\
        --input      evaluation/data/raw/eval_csv/optimizers.csv \\
        --nadamw-ref evaluation/data/raw/eval_csv/in1k.csv \\
        --output     evaluation/data/processed/optimizers_per_experiment.csv

    # 6. Turn raw eval CSVs into tidy per-(model, method) tables.
    python -m evaluation.scripts.process_results \\
        --inputs evaluation/data/raw/eval_csv/*.csv \\
        --output-dir evaluation/data/processed

    # 7. Render every paper plot in one go.
    python -m evaluation.scripts.make_all_plots

    # 8. Back up finished training runs (W&B/timm + vitdet/detectron2).
    python -m evaluation.scripts.backup_runs \\\\
        --backup-root ../backup

    # --- Diagnostic (composed ImageNet) pipeline --------------------------

    # D1. Compose the diagnostic evaluation set from ImageNet val.
    python -m evaluation.scripts.diag_generate \\\\
        --out-dir evaluation/data/raw/diagnostic/composed \\\\
        --k-values 3 4 5 6 \\\\
        --samples-per-k 500 \\\\
        --alpha 0.5 \\\\
        --sampling-max-aspect 15 \\\\
        --split validation

    # D2. Run the trained models over the composed set and dump ``logits.pt``.
    python -m evaluation.scripts.diag_export_logits \\\\
        --manifest evaluation/data/raw/diagnostic/composed/manifest.jsonl \\\\
        --mapping  evaluation/diagnostic/area_logit_models.yaml \\\\
        --out-dir  evaluation/data/raw/diagnostic/logits

    # D3. Compute area-vs-logit diagnostic metrics across seeds
    #     (per-k and aggregated, reported as mean +/- std). Always multi-seed;
    #     exactly 3 seeds are required (default ``--seeds 42 43 44``) and
    #     every model must evaluate successfully on every seed -- the script
    #     errors out otherwise. Writes a single CSV:
    #     ``diagnostic_area_logit_metrics.csv``.
    python -m evaluation.scripts.diag_area_logit_metrics \\\\
        --mapping    evaluation/diagnostic/area_logit_models.yaml \\\\
        --manifest   evaluation/data/raw/diagnostic/composed/manifest.jsonl \\\\
        --logits-dir evaluation/data/raw/diagnostic/logits_multiseed \\\\
        --seeds 42 43 44 \\\\
        --out-csv    evaluation/data/processed/diagnostic_area_logit_metrics.csv
"""