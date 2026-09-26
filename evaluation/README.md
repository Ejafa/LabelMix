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

   The augmentation showcase can be rendered without a local ImageNet cache by
   first pulling a small Lorem Picsum image pool:

   ```bash
   python evaluation/scripts/download_picsum_images.py
   python evaluation/plots/augmentation_showcase.py \
       --source-dir evaluation/data/raw/picsum_augmentation_showcase_diverse
   ```

   By default the downloader samples 128 images from a larger Picsum catalog
   window and balances the selection across authors.  Change ``--seed`` for a
   different reproducible pool, or increase ``--catalog-pages`` to sample from
   a wider catalog slice.

   To use animal-focused ImageNet-1k samples without downloading the full
   dataset, stream only the required samples from Hugging Face:

   ```bash
   python evaluation/scripts/download_hf_imagenet_samples.py
   python evaluation/plots/augmentation_showcase.py \
       --source-dir evaluation/data/raw/hf_imagenet1k_animal_samples \
       --out-dir evaluation/data/processed/figures/augmentation_showcase_hf_imagenet1k_animals
   ```

   A normal ``augmentation_showcase.py`` run renders all showcase image groups,
   including one basic ``aug_comparison_clean/`` example per augmentation and
   ``treemapmix_alpha_sweep/`` with the default alpha values
   ``0.05 0.1 0.3 0.5 1.0 1.5 3.0 5.0``.
   When the source directory contains ``imagenet1k_####.jpg`` files, the
   showcase places the preferred animal sample ids
   ``81 18 6 124 2 122 119 118 4`` first in the shared source-image pool.
   Override with ``--preferred-source-ids`` or pass that flag with no ids to
   keep pure filename ordering.

   This requires the ``datasets`` package and access to the gated
   ``ILSVRC/imagenet-1k`` dataset.  Pass ``--token`` or set ``HF_TOKEN`` if
   your Hugging Face login is not already configured.  The downloader defaults
   to the animal class preset (ImageNet labels 0..397), saves at most two
   images per class for diversity, and filters before decoding images.  Set
   ``--num-images`` lower for a smaller smoke test, or use
   ``--class-preset all`` to disable the animal filter.

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


## Rebuttal results used by the paper

The existing ImageNet summaries already include all seven OpenMixup methods.
`in1k_merged_openmixup.csv` reproduces their metrics. Its two additional complete
`unbalanced-noaug` groups are diagnostic controls excluded by the main plot.
Explicit `bare` recipes remain single-image augmentation even when their names
contain disabled settings such as `cutmix=0` and `mixup=0`. These rows must
not be pooled with CutMix. The long-horizon paper table reports only ViT-Wee
and ViT-Little with single-image augmentation, Mixup+CutMix, SCE, and PL.

Regenerate ImageNet, CIFAR-100, and loader summaries with:

```bash
python -m evaluation.scripts.aggregate_per_experiment \
  --input evaluation/data/raw/eval_csv/in1k_merged_openmixup.csv \
  --output evaluation/data/processed/in1k_per_experiment.csv \
  --table-out evaluation/data/processed/in1k_per_experiment.md
python -m evaluation.scripts.aggregate_per_experiment \
  --input evaluation/data/raw/eval_csv/cifar100.csv \
  --output evaluation/data/processed/cifar100_per_experiment.csv \
  --table-out evaluation/data/processed/cifar100_per_experiment.md
python -m evaluation.scripts.process_augmentation_loader_benchmark
```

The paper uses the supplied `long_tail_per_experiment_short.csv` (seeds 42
and 43) and `architectures_per_experiment_short.csv` (seeds 42, 43, and 44).
Do not describe the long-tail results as three-seed estimates. Classification
ECE is stored as a fraction and displayed in percentage points. Runtime memory
columns named `_mb` store MiB and are displayed as GiB. Throughput reduction is
computed from the ratio of configuration mean throughput to the single-image
reference mean; it is not an end-to-end training slowdown.

Export native LaTeX tables and render the comparison figures with:

```bash
python -m evaluation.scripts.export_paper_result_tables
python -m evaluation.scripts.make_all_plots --only \
  in1k_per_experiment cifar100_per_experiment
```

All paper tables use native LaTeX `tabular` and `booktabs` rules. The result
table exporter reads the processed CSVs and writes the eight result tables in
`paper/fig/`; no rendered table images are used by the paper.

The runtime CSV contains a sixth configuration labeled `mosaic` that is absent
from the current benchmark runner. Preserve its recorded label rather than
claiming a separately verified RICAP runtime measurement. The data summarize
five runs, batch size 128, eight workers, and 100 measured batches after 20
warm-up batches (plus an initial first-batch measurement). They measure loading
and augmentation without model forward or backward passes.
