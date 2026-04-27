"""ViTDet Mask R-CNN with the LabelMix vit_wee backbone.

Architecture (from timm):
    embed_dim=256, depth=14, num_heads=4, mlp_ratio=5, patch_size=16.

Runs on a 30-epoch cosine schedule (see ``configs/common/train_schedule.py``);
suitable for ablation-style sweeps. For a full 100-epoch confirmation run,
override ``train.max_iter`` and ``lr_multiplier`` via::

    from detectron2_vitdet.configs.common.train_schedule import build_schedule
    train.max_iter, lr_multiplier = build_schedule(epochs=100)

Set ``train.init_checkpoint`` to a converted timm checkpoint produced by
``detectron2_vitdet/convert_timm_to_vitdet.py``.
"""

from ..common.labelmix_vitdet import build_vitdet_model
from ..common.coco_loader_lsj import dataloader
from ..common.train_schedule import train, lr_multiplier, build_optimizer


model = build_vitdet_model(
    embed_dim=256,
    depth=14,
    num_heads=4,
    mlp_ratio=5.0,
    drop_path_rate=0.1,
    use_act_checkpoint=True,
)

train.init_checkpoint = ""  # set via CLI: train.init_checkpoint=/path/to/vit_wee.pth
optimizer = build_optimizer(num_layers=14, lr_decay_rate=0.7)
