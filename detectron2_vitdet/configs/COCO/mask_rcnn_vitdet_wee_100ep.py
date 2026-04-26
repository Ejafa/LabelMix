"""ViTDet Mask R-CNN with the LabelMix vit_wee backbone.

Architecture (from timm):
    embed_dim=256, depth=14, num_heads=4, mlp_ratio=5, patch_size=16.

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
)

train.init_checkpoint = ""  # set via CLI: train.init_checkpoint=/path/to/vit_wee.pth
optimizer = build_optimizer(num_layers=14, lr_decay_rate=0.7)
