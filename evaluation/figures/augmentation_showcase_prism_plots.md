# Augmentation Showcase Prism Plot Brief

Use the 1024px PNGs for final figures; use 256px PNGs only for previews.
All source images come from the same pool passed to `augmentation_showcase.py`.
For `hf_imagenet1k_animal_samples`, the preferred pool order starts with ids
`14`, `20`, `22`, `30`, `35`, `43`, `58`, `64`, `67`, `68`.

## 1. Augmentation Comparison

- Source directory: `aug_comparison_clean/`
- Layout: one column with rows `none`, `mixup`, `cutmix`, `mosaic`, `labelmix`.
- Purpose: compare each augmentation in its basic form, without additional single-image training augmentations.
- Parameters: Mixup `alpha=1.0`; CutMix `alpha=1.0`; Mosaic center jitter `(0.8, 1.2)`, post-scale `(0.6, 0.9)`, grey filler; LabelMix `alpha=1.0`, `k=4`.
- Files: `{aug}__panel1_1024.png`

## 2. LabelMix Randomness

- Source directory: `labelmix_randomness/`
- Layout: one row, panels ordered `panel01` through `panel08`.
- Purpose: show layout randomness only; all panels use the same source images and hyperparameters.
- Files: `panel{01..08}_1024.png`

## 3. LabelMix K Sweep

- Source directory: `labelmix_k_sweep/`
- Layout: one row ordered by increasing `k`.
- Purpose: show how the number of mixed images changes the composite; source
  images are size-ranked, so earlier/repeated images are larger than newly
  introduced images.
- Parameters: LabelMix `alpha=5.0`, max aspect filter `15`.
- Files: `k02_1024.png` through `k10_1024.png`

## 4. LabelMix Alpha Sweep

- Source directory: `labelmix_alpha_sweep/`
- Layout: one row ordered by alpha: `0.05`, `0.1`, `0.3`, `0.5`, `1.0`, `1.5`, `3.0`, `5.0`.
- Purpose: show how alpha changes region sizes while source images and `k=5` stay fixed; source-image size ranking is held fixed from largest to smallest.
- Parameters: max aspect filter `10`.
- Files: `alpha0_05_1024.png`, `alpha0_1_1024.png`, `alpha0_3_1024.png`, `alpha0_5_1024.png`, `alpha1_1024.png`, `alpha1_5_1024.png`, `alpha3_1024.png`, `alpha5_1024.png`
