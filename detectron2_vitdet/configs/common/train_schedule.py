"""Common training schedule shared by all LabelMix ViTDet configs.

Tuned for 256x256 inputs at batch size 256 (COCO 2017 train has 118k images,
so ~461 iters/epoch). We run a shortened 30-epoch schedule for ablation-style
sweeps: it costs ~30% of the canonical 100-ep ViTDet recipe, but rank-orders
variants (losses / learning rates) essentially identically -- which is all we
need when comparing pretraining losses on downstream detection.

Use ``build_schedule(epochs=...)`` to derive ``(max_iter, lr_multiplier)`` for
any other epoch budget (e.g. 100 ep for a final confirmation run).
"""

from functools import partial

from fvcore.common.param_scheduler import CosineParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate


# ---------------------------------------------------------------------------
# Trainer defaults
# ---------------------------------------------------------------------------
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True


# ---------------------------------------------------------------------------
# Learning-rate schedule (cosine decay with linear warmup)
# ---------------------------------------------------------------------------
# COCO 2017 train: ~118k images, batch size 256 -> ~461 iters/epoch.
_ITERS_PER_EPOCH = 461
_WARMUP_ITERS = 250


def build_schedule(epochs: int = 30, warmup_iters: int = _WARMUP_ITERS):
    """Return ``(max_iter, lr_multiplier)`` for a cosine schedule at ``epochs``.

    The cosine decays smoothly from 1.0x at warmup-end to 0.01x at training-end,
    after a linear warmup of ``warmup_iters`` iterations. Cosine is used instead
    of the upstream MultiStep schedule because its decay shape is scale-free --
    a 30-ep schedule and a 100-ep schedule share the same shape without any
    per-schedule milestone retuning.
    """
    max_iter = _ITERS_PER_EPOCH * epochs
    lr_multiplier = L(WarmupParamScheduler)(
        scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=0.01),
        warmup_length=warmup_iters / max_iter,
        warmup_factor=0.001,
    )
    return max_iter, lr_multiplier


# Default: 30-epoch ablation recipe (cosine, batch 256 at 256x256 inputs).
train.max_iter, lr_multiplier = build_schedule(epochs=30)

# Evaluation / checkpointing: 4 times per run (every 25% of training).
train.eval_period = train.max_iter // 4
train.checkpointer.period = train.max_iter // 4
train.checkpointer.max_to_keep = 5


# ---------------------------------------------------------------------------
# Optimizer (AdamW with per-layer lr decay -- critical for ViT detection)
# ---------------------------------------------------------------------------
def build_optimizer(num_layers: int, lr_decay_rate: float = 0.7, lr: float = 2e-4):
    """AdamW with per-layer lr decay.

    The default ``lr=2e-4`` follows the square-root scaling rule from the
    upstream ViTDet default (``1e-4`` at batch size 64) to our batch size of
    256 (``1e-4 * sqrt(256/64) = 2e-4``). Square-root is preferred over
    linear scaling for AdamW on ViT detection, which is sensitive to LR.
    """
    optimizer = model_zoo.get_config("common/optim.py").AdamW
    optimizer.lr = lr
    optimizer.params.lr_factor_func = partial(
        get_vit_lr_decay_rate,
        num_layers=num_layers,
        lr_decay_rate=lr_decay_rate,
    )
    optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
    return optimizer
