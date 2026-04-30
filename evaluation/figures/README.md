## evaluation/figures/ — Paper augmentation-image generators

The scripts in this directory emit **raw per-image PNGs** used to illustrate
what the different data augmentations look like in input space.  Figure
composition (captions, sub-figure grids, etc.) is done later in LaTeX —
this folder only produces the image assets and a `metadata.json` per
image group.

Output root: `evaluation/data/processed/figures/augmentation_showcase/`

### augmentation_showcase.py

Produces three image sets, one per subdirectory.  Every PNG is saved at
both `256x256` and `1024x1024`; existing files are skipped on re-run.

1. **`aug_comparison_{clean,vit_wee}/`** — one PNG per
   `(augmentation, panel)` tile, with
   `augmentation ∈ {none, mixup, cutmix, mosaic, labelmix}`.
   - `clean` applies only a resize + center-crop before augmentation.
   - `vit_wee` additionally applies the single-image augmentations from
     the ImageNet-1k vit-wee training config
     (`rand-m6-inc1-mstd1.0-n3`, `RandomResizedCrop`, `hflip`,
     `reprob=0.2`).
   - Filenames: `<aug>__panel<P>_<size>.png`
     (e.g. `labelmix__panel3_1024.png`).
   - Defaults: `--num-comparison-panels 4`, `labelmix α=0.5, k=6,
     aspect≤20`.

2. **`labelmix_randomness/`** — N independent LabelMix draws
   (`α=0.5, k=6, aspect≤20`), each under a different seed. No
   single-image augmentations are applied.
   - Filenames: `panel<PP>_<size>.png` (zero-padded for sorting).
   - Defaults: `--num-randomness-panels 8`.

3. **`labelmix_k_sweep/`** — one LabelMix PNG per `k ∈ [2, 10]`, fixed
   `α=0.5` and `aspect≤20`. No single-image augmentations.
   - Filenames: `k<KK>_<size>.png`.

Each of the three subdirectories also contains a concise
`metadata.json` capturing the group's purpose, the source dataset,
numeric hyperparameters, resolutions, and a `files` list describing
each image. The JSON is designed to be pasted into an LLM prompt to
generate fitting LaTeX captions / file names.

### Usage

```bash
# Recommended: read images from the shared HuggingFace `datasets` Arrow
# cache.  The default --hfds-cache-dir already points at:
#   /apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/data/imagenet-1k
python -m evaluation.figures.augmentation_showcase --verbose

# Use the validation split instead of train:
python -m evaluation.figures.augmentation_showcase \
    --hfds-split validation --verbose

# Alternative: point at a local directory containing >= 128 images
python -m evaluation.figures.augmentation_showcase \
    --source-dir /path/to/images --verbose

# Alternative: go through timm's create_dataset
python -m evaluation.figures.augmentation_showcase \
    --use-timm-dataset hfds/ILSVRC/imagenet-1k \
    --timm-data-dir /dev/shm/imagenet-1k --verbose
```

### Design guarantees

- **Determinism.** Same `--seed` + same image source ⇒ identical PNGs.
  Each panel of each augmentation uses an independent
  `panel_seed + p*1000 + aug_salt` RNG stream.
- **Idempotent re-runs.** Each PNG is skipped if already present;
  `metadata.json` is always refreshed.
- **Matched resolutions.** Every composite is first built at
  `--img-size` (default 256) and then upsampled with PIL bicubic to each
  `--tile-sizes` entry (default 256 and 1024), so the two resolutions
  are pixel-identical up to resampling.
- **Faithful LabelMix.** The compositing reuses
  `timm.data.labelmix_layout.squarify_core` together with
  `balanced_dataset._layout_to_pixel_boxes` / `_apply_box_symmetry`,
  so the visualised layouts match exactly what the training loop
  produces.
