"""ViTDet Mask R-CNN with the LabelMix vit_betwixt backbone.

Architecture: embed_dim=640, depth=12, num_heads=10, mlp_ratio=4, patch=16.

Runs on a 30-epoch cosine schedule (see ``configs/common/train_schedule.py``);
suitable for ablation-style sweeps. For a full 100-epoch confirmation run,
override ``train.max_iter`` and ``lr_multiplier`` via::

    from detectron2_vitdet.configs.common.train_schedule import build_schedule
    train.max_iter, lr_multiplier = build_schedule(epochs=100)
"""

from ..common.labelmix_vitdet import build_vitdet_model
from ..common.coco_loader_lsj import dataloader
from ..common.train_schedule import train, lr_multiplier, build_optimizer


model = build_vitdet_model(
    embed_dim=640,
    depth=12,
    num_heads=10,
    mlp_ratio=4.0,
    drop_path_rate=0.1,
    use_act_checkpoint=True,
)

train.init_checkpoint = ""
optimizer = build_optimizer(num_layers=12, lr_decay_rate=0.7)
