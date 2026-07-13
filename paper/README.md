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

The paper uses PDF plot assets at their native page size. LaTeX figure imports
must not pass `width`, `height`, `scale`, or other resizing options to
`\includegraphics`; size changes belong in the plot exporter so the generated
PDF is already at its final publication dimensions.

The exporters use the NeurIPS 5.5-inch text block as their sizing contract:
full-width plots are 5.5 inches wide, the area-logit panels are 1.76 inches
wide for the three-across row, augmentation tiles are approximately 1.32
inches for four-across rows, and Figure 1 tiles are approximately 1.045 inches
for its five-across row. Width and height are reduced by the same factor. All
plot text is 9 point to match the NeurIPS LaTeX `\small` font size.
