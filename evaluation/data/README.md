# `evaluation/data/` — data lifecycle

Strict three-tier split.  Tools read from one tier, write to the next, and
never cross over.

* **`raw/`** — direct outputs of the offline evaluator.
  Contents are reproducible from checkpoints + `args.yaml` alone, so this
  tree can be regenerated at any time.
  * `raw/eval_csv/<run>.csv` — one CSV per `scripts.run_eval` invocation.
  * `raw/logits/<slug>.pt`   — optional `{logits, targets}` tensor dumps
    (enabled with `--save-raw-logits`).  Needed only for reliability
    diagrams / calibration analyses that require per-sample confidences.
  * `raw/wandb/<run_id>/`    — per-run materialization pulled from W&B by
    `scripts.sync_wandb` (metadata, config, summary, history, optional
    files).  The sibling `_manifest.json` file is the incremental cache
    and must not be edited by hand.

* **`processed/`** — tidy tables derived by `scripts.process_results`.
  Human-readable, stable schema, safe for downstream analysis and paper
  tables.  Every file here is a deterministic function of `raw/`.

* **`processed/figures/`** — final paper figures (PDF + PNG per plot).
  Regenerated from `processed/` by the `plots/` subpackage.

## Regeneration order

```
raw/            ← scripts.run_eval
processed/      ← scripts.process_results
processed/figures/ ← plots.<name> / scripts.make_all_plots
```

Never edit any file under `raw/` or `processed/` by hand: the next pipeline
re-run will silently overwrite your changes.  Put manual analyses in
notebooks that consume `processed/*.csv` instead.
