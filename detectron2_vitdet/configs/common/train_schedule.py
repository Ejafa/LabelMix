"""Common training schedule shared by all LabelMix ViTDet configs.

Tuned for 256x256 inputs at batch size 256 (4x the upstream ViTDet default
of 64 at 1024x1024, since the smaller images free up ~16x activation memory).
Iterations are scaled down 4x to preserve the 100-epoch recipe. Override
``train.max_iter`` and ``lr_multiplier`` on a per-experiment basis if needed.
"""

from functools import partial

from fvcore.common.param_scheduler import MultiStepParamScheduler

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

# 100 ep = 46094 iters * 256 images/iter / 118000 images/ep
train.max_iter = 46094
# Evaluation / checkpointing cadence (~ every 10 ep)
train.eval_period = 4609
train.checkpointer.period = 4609
train.checkpointer.max_to_keep = 5


# ---------------------------------------------------------------------------
# Learning-rate schedule (same MultiStep as upstream ViTDet)
# ---------------------------------------------------------------------------
# Milestones keep the same 88.9% / 96.3% decay points as the upstream schedule.
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[40972, 44387],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)


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
