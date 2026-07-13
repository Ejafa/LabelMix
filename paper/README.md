# Paper

The LaTeX source lives in this directory, while experiment data and generated
figures remain owned by the evaluation pipeline:

- processed tables: `../evaluation/data/processed/`
- paper-ready figures: `../evaluation/data/processed/figures/`

Do not copy generated plots or CSV snapshots into `paper/`. Regenerate plots
from the repository root with:

```bash
python -m evaluation.scripts.make_all_plots
```

The qualitative assets have dedicated entry points:

```bash
python -m evaluation.plots.augmentation_showcase --source-dir <image-directory>
python -m evaluation.scripts.diag_paper_figures
```

The augmentation exporter writes matching PNG and PDF files; LaTeX references
only the PDF variants.

Compile the paper from this directory so section, bibliography, and style-file
paths resolve consistently:

```bash
cd paper
latexmk -pdf main.tex
```

The paper uses PDF plot assets. LaTeX figure paths should therefore target the
PDF outputs in the evaluation tree.
