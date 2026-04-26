"""ViTDet Mask R-CNN with the LabelMix vit_little backbone.

Architecture: embed_dim=320, depth=14, num_heads=5, mlp_ratio=5.6, patch=16.
"""

from ..common.labelmix_vitdet import build_vitdet_model
from ..common.coco_loader_lsj import dataloader
from ..common.train_schedule import train, lr_multiplier, build_optimizer


model = build_vitdet_model(
    embed_dim=320,
    depth=14,
    num_heads=5,
    mlp_ratio=5.6,
    drop_path_rate=0.1,
)

train.init_checkpoint = ""
optimizer = build_optimizer(num_layers=14, lr_decay_rate=0.7)
