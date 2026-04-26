"""ViTDet Mask R-CNN with the LabelMix vit_medium backbone.

Architecture: embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, patch=16.
"""

from ..common.labelmix_vitdet import build_vitdet_model
from ..common.coco_loader_lsj import dataloader
from ..common.train_schedule import train, lr_multiplier, build_optimizer


model = build_vitdet_model(
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4.0,
    drop_path_rate=0.1,
)

train.init_checkpoint = ""
optimizer = build_optimizer(num_layers=12, lr_decay_rate=0.7)
