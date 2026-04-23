# `evaluation/` — offline evaluation, processing, and plotting

Four-stage pipeline for the paper: **(sync) → evaluate → process → plot**.
Each stage reads from the previous stage's directory and writes to its own,
so intermediate state is always inspectable and re-runnable independently.

```
 W&B cloud ──▶ data/raw/wandb ──┐
                                ├──▶ data/processed ──▶ data/processed/figures
 checkpoints ──▶ data/raw/eval_csv, data/raw/logits ──┘
```

## Layout

```
evaluation/
├── config.py              EvalConfig, default CSV column sets
├── types.py               EvalResult
│
├── common/                Shared helpers (paths, logging)
├── metrics/               One metric per file; plugin registry
├── offline_eval/          Checkpoint → raw CSV + (optional) logits
├── processing/            Raw CSV → tidy paper tables
├── plots/                 One figure per file, reads only from data/processed/
├── scripts/               Thin CLI wrappers (no analysis logic here)
├── wandb_sync/            Incremental downloader for W&B runs
│
└── data/
    ├── raw/               Raw outputs of offline_eval / wandb_sync (do NOT edit by hand)
    │   ├── eval_csv/      Per-invocation CSVs
    │   ├── logits/        Per-run {logits, targets} .pt dumps
    │   └── wandb/         One subdir per W&B run_id + _manifest.json cache
    └── processed/         Tidy tables that plots consume
        └── figures/       Final PDF + PNG figures
```

## Typical workflow

0. **(Optional) Sync** W&B run histories into the local cache:

   ```bash
   python -m evaluation.scripts.sync_wandb \
       --entity my-team --project labelmix --tags in1k
   ```

   Re-running the same command only fetches runs whose server-side
   ``heartbeatAt`` / ``updatedAt`` has advanced since the last sync.

1. **Evaluate** a batch of training runs:

   ```bash
   python -m evaluation.scripts.run_eval \
       --mapping my_runs.yaml \
       --output-csv evaluation/data/raw/eval_csv/my_eval.csv \
       --save-raw-logits
   ```

2. **Process** all raw CSVs into tidy tables:

   ```bash
   python -m evaluation.scripts.process_results \
       --inputs 'evaluation/data/raw/eval_csv/*.csv'
   ```

3. **Plot** every paper figure:

   ```bash
   python -m evaluation.scripts.make_all_plots
   ```

   Or render a single figure:

   ```bash
   python -m evaluation.plots.calibration_ece_bars
   ```

## Adding a new metric

1. Create `evaluation/metrics/my_metric.py`:

   ```python
   from .registry import register_metric

   @register_metric("my_metric")
   def my_metric(logits, targets):
       return float(...)
   ```

2. Import it in `evaluation/metrics/__init__.py` so it gets auto-registered.
3. Re-run `scripts.run_eval`; the CSV gets a new column automatically.

## Adding a new plot

1. Create `evaluation/plots/<my_plot>.py` exposing ``plot(df)`` and ``main()``.
2. Read **only** from `data/processed/` (or, for per-sample work,
   `data/raw/logits/`).
3. Write the figure via `._style.savefig(fig, out)` so it gets both a PDF
   and a PNG copy under `data/processed/figures/`.
4. No action needed for `make_all_plots` — it auto-discovers the module.

## Hygiene rules

* **Raw ≠ processed.** Nothing under `data/processed/` should be written
  by `offline_eval`; nothing under `data/raw/` should be written by
  `processing` or `plots`.
* **One figure per file** in `plots/`, with one `plot()` function and one
  `main()`.
* **Scripts are dumb.** `scripts/*.py` only parse args and dispatch; all
  analysis code lives in importable modules.
* **Shared style.** Every plot calls `_style.apply_paper_style()` so the
  paper has uniform typography and sizing.
