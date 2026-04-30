# Augmentation Showcase Prism Plot Brief

Use the 1024px PNGs for final figures; use 256px PNGs only for previews.
All source images come from the same pool passed to `augmentation_showcase.py`.

## 1. Augmentation Comparison, Clean

- Source directory: `aug_comparison_clean/`
- Layout: grid with rows `none`, `mixup`, `cutmix`, `mosaic`, `labelmix`; columns `panel1` through `panel4`.
- Purpose: compare augmentation methods without single-image training augmentations.
- Files: `{aug}__panel{1..4}_1024.png`

## 2. Augmentation Comparison, ViT-Wee

- Source directory: `aug_comparison_vit_wee/`
- Layout: same grid as clean comparison.
- Purpose: compare the same methods after ViT-wee ImageNet training augmentations.
- Files: `{aug}__panel{1..4}_1024.png`

## 3. LabelMix Randomness

- Source directory: `labelmix_randomness/`
- Layout: one row, panels ordered `panel01` through `panel08`.
- Purpose: show layout randomness only; all panels use the same source images and hyperparameters.
- Files: `panel{01..08}_1024.png`

## 4. LabelMix K Sweep

- Source directory: `labelmix_k_sweep/`
- Layout: one row ordered by increasing `k`.
- Purpose: show how the number of mixed images changes the composite.
- Files: `k02_1024.png` through `k10_1024.png`

## 5. LabelMix Alpha Sweep

- Source directory: `labelmix_alpha_sweep/`
- Layout: one row ordered by alpha: `0.05`, `0.1`, `0.3`, `0.5`, `1.0`, `1.5`, `3.0`, `5.0`.
- Purpose: show how Dirichlet concentration alpha changes region sizes while source images and `k=6` stay fixed.
- Files: `alpha0_05_1024.png`, `alpha0_1_1024.png`, `alpha0_3_1024.png`, `alpha0_5_1024.png`, `alpha1_1024.png`, `alpha1_5_1024.png`, `alpha3_1024.png`, `alpha5_1024.png`
