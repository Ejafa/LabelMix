# ViTDet Object Detection for LabelMix ViTs

This folder evaluates the LabelMix pre-trained ViT backbones (`vit-wee` and
`vit-betwixt` -- the only two backbones used for downstream object detection)
on COCO object detection + instance segmentation, using the official
[Detectron2 ViTDet recipe](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).

It provides:

* `install.sh` – installs Detectron2 and fetches the reference ViTDet code.
* `download_coco.sh` – downloads COCO 2017 (train/val + annotations).
* `convert_timm_to_vitdet.py` – converts a LabelMix timm checkpoint to the
  state-dict layout expected by Detectron2's `ViT` backbone (drops the
  classification head + register tokens, fuses LayerScale into the adjacent
  projections, etc.).
* `configs/COCO/mask_rcnn_vitdet_<variant>_30ep.py` – LazyConfig training
  recipes for each supported backbone (`wee`, `betwixt`). 30 epochs of
  cosine-decayed LR at batch 256 / 256x256 inputs, suitable for
  ablation-style sweeps. Override ``train.max_iter`` / ``lr_multiplier``
  via ``build_schedule(epochs=100)`` for a full confirmation run.
* `train_net.py` – training / evaluation driver built on Detectron2's
  `lazyconfig_train_net.py`.

---

## 1. Install

```bash
cd detectron2_vitdet
bash install.sh            # pre-built wheel if available, source build otherwise
# or force a source build (safer for custom CUDA/Torch):
bash install.sh --dev
```

The script installs Detectron2, core dependencies (`pycocotools`, `fvcore`,
etc.), and fetches the upstream ViTDet project tree into
`detectron2_vitdet/upstream_vitdet/` for reference.

## 2. Download COCO 2017

```bash
# Default location: ./datasets/coco (relative to this folder)
bash download_coco.sh

# Or specify an explicit directory:
bash download_coco.sh /path/to/coco
```

After it finishes, export:

```bash
export DETECTRON2_DATASETS=/path/to/parent_of_coco   # contains a "coco" subdir
```

## 3. Convert a LabelMix checkpoint

The LabelMix ViTs have four differences from the ViT that Detectron2 expects:

| LabelMix (timm)                    | Detectron2 `ViT`                       |
|------------------------------------|----------------------------------------|
| `init_values=1e-5` (LayerScale)    | no LayerScale                          |
| `class_token=False`, register tokens | no prefix tokens                     |
| classification head + final norm   | backbone-only                          |
| trained at 256×256                 | 1024×1024 at detection time            |

`convert_timm_to_vitdet.py` handles all of these:

```bash
python convert_timm_to_vitdet.py \
    --input ../mixed_loss/vit-wee__in1k__img256__k4-4_a0.1-0.5_mixed_ma0.9_as-cosine_scheduling__seed=42/model_best.pth.tar \
    --output ./converted/vit_wee_in1k.pth
```

Use `--use-ema` to export the EMA weights (present for `vit-wee` only).

The script folds `ls1.gamma` into `attn.proj.{weight,bias}` and `ls2.gamma`
into `mlp.fc2.{weight,bias}` (mathematically equivalent because LayerScale is a
channel-wise diagonal scaling that commutes with the following residual add).

## 4. Train ViTDet

Single node, 4 GPUs:

```bash
python train_net.py \
    --config-file configs/COCO/mask_rcnn_vitdet_wee_30ep.py \
    --num-gpus 4 \
    train.init_checkpoint=./converted/vit_wee_in1k.pth \
    train.output_dir=./output/vit_wee_vitdet
```

Swap the config for `mask_rcnn_vitdet_betwixt_30ep.py` to train the other
supported backbone.

### Standard ViTDet recipe (adapted for ablation sweeps)

| Hyper-parameter              | Value                                   |
|------------------------------|-----------------------------------------|
| Image size                   | 256 × 256 with Large-Scale Jittering    |
| Batch size                   | 256                                     |
| Max iterations               | 13 830 (≈ 30 epochs)                    |
| Optimizer                    | AdamW                                   |
| LR schedule                  | Cosine × Warmup (250 iters)             |
| LR decay endpoint            | 0.01 × base_lr at training end          |
| Layer-wise LR decay          | 0.7                                     |
| `pos_embed` weight decay     | 0.0                                     |
| Window size                  | 14 (global attn every `depth/4` blocks) |

For a full 100-epoch confirmation run, override from the CLI:

```bash
python train_net.py --config-file configs/COCO/mask_rcnn_vitdet_betwixt_30ep.py \
    --num-gpus 4 \
    train.init_checkpoint=./converted/vit_betwixt_in1k.pth \
    train.max_iter=46094 \
    dataloader.train.total_batch_size=256 \
    optimizer.lr=2e-4
```

## 5. Evaluate only

```bash
python train_net.py \
    --config-file configs/COCO/mask_rcnn_vitdet_wee_30ep.py \
    --num-gpus 4 --eval-only \
    train.init_checkpoint=./output/vit_wee_vitdet/model_final.pth
```

## 6. Batch evaluation of many backed-up runs

If you have a directory of LabelMix runs produced by
[`evaluation/scripts/backup_runs.py`](../evaluation/scripts/backup_runs.py)
— i.e. folders containing `args.yaml` + `model_best.pth.tar` — the
`eval_all.py` driver automates the full convert → ViTDet-train → eval →
aggregate pipeline in one shot:

```bash
# Train + evaluate every backed-up run (4 GPUs per run):
python eval_all.py --backup-root ../backup --num-gpus 4

# Dry-run first, to see what would be launched:
python eval_all.py --backup-root ../backup --dry-run

# Only re-aggregate metrics from already-trained runs (no GPU work):
python eval_all.py --backup-root ../backup --summary-only

# Restrict to one variant / pattern, keep going on individual failures:
python eval_all.py --backup-root ../backup \
    --include-variants wee \
    --include-pattern "seed=42" \
    --continue-on-error
```

What it does per-run:

1. Parses `args.yaml` to pick the matching backbone variant
   (`vit_wee` / `vit_betwixt`) and its
   `configs/COCO/mask_rcnn_vitdet_<variant>_30ep.py` recipe.
2. Calls `convert_timm_to_vitdet.py` (cached in `./converted/`).
3. Runs `torchrun ... train_net.py ...` with
   `train.init_checkpoint` / `train.output_dir` set appropriately.
   If `model_final.pth` already exists for the run, it automatically
   falls back to `--eval-only` (use `--force-retrain` to override).
4. Reads the final `bbox/AP*` and `segm/AP*` from each run's
   `metrics.json` and appends them to `./output/vitdet_eval_summary.csv`.

Useful knobs: `--use-ema`, `--eval-only`, `--force-convert`,
`--extra-override dataloader.train.total_batch_size=32` (repeatable),
`--master-port-base 30000` (avoid DDP port clashes).

---

## Backbone hyper-parameters (for reference)

| Variant    | `embed_dim` | `depth` | `num_heads` | `mlp_ratio` | reg. tokens |
|------------|------------:|--------:|------------:|------------:|------------:|
| vit-wee    |         256 |      14 |           4 |         5.0 |           1 |
| vit-betwixt|         640 |      12 |          10 |         4.0 |           4 |

(Both variants use `patch_size=16`, absolute position embedding, `qkv_bias=True`,
and were pre-trained at 256×256 on ImageNet-1k.)

## Troubleshooting

* **`KeyError: norm.weight` when loading** – the converter drops the final
  `norm` layer. The Detectron2 backbone replaces it with LayerNorms from
  `SimpleFeaturePyramid`. Make sure `matching_heuristics=True` is kept in the
  converted checkpoint (default).
* **`pos_embed` size mismatch** – expected. Detectron2 interpolates the
  position embedding from 256/16=16² tokens to 1024/16=64² tokens on the fly
  (`get_abs_pos`), **provided** you set `pretrain_use_cls_token=False`
  (already set in the common config).
* **Slow start** – the first training iterations compile the FPN/LN kernels;
  subsequent iterations run at normal speed.
