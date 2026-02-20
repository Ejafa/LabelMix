# Object Detection Evaluation

This folder contains `run.py`, a driver that evaluates timm backbones with MMDetection configs. It registers a timm backbone on the fly, patches the config, and can run multiple configs and checkpoints in a single sweep.

**Key Features**
- Multi-config and multi-checkpoint runs.
- Optional fine-tuning (`--run train` / `--run train+test`).
- JSONL results file per run.
- Automatic SimpleFPN for single-scale backbones (typical ViTs).
- Default `--data-root` is `./data` (pass your dataset root explicitly if different).

**Install (Example, CUDA 12.1 + Torch 2.4)**
```bash
conda create -n mmdet311 python=3.11 -y
conda activate mmdet311
pip install -U pip setuptools

# Install torch first (adjust CUDA/Torch to your environment)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install mmcv from the matching OpenMMLab wheel index (must match Torch/CUDA)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html --only-binary=:all:

# Install the remaining dependencies
pip install -r requirements.txt
```

**Usage**
```bash
python run.py \
  --configs /path/to/coco_config.py,/path/to/voc_config.py \
  --timm-model mobilenetv4_conv_small \
  --backbone-checkpoints /path/to/ckpt1.pth,/path/to/ckpt2.pth \
  --data-root /path/to/datasets \
  --run train+test \
  --work-dir output_test/mmdet_eval \
  --results-file output_test/mmdet_eval/results.jsonl
```

**Batch Usage (CNN + ViT)**
Create a job file (see `jobs.example.yaml`) and run:
```bash
python run_batch.py --jobs jobs.example.yaml
```

**Results**
Each run appends a JSON line to `--results-file` with metadata (config, checkpoint, work_dir) and metrics returned by MMDetection.

**Notes for ViTs**
If the backbone only produces a single feature scale, the script swaps the neck to `SimpleFPN` automatically and uses the deepest output as the feature source.
