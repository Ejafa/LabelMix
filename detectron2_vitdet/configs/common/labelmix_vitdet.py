"""Shared ViTDet Mask R-CNN backbone + head definitions for the LabelMix ViTs.

This is a LazyConfig module. It builds on top of detectron2's model zoo
configuration ``common/models/mask_rcnn_vitdet.py`` and only overrides the
backbone hyper-parameters per model variant.
"""

from functools import partial

import torch.nn as nn

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool


def _window_block_indexes(depth: int) -> list:
    """Replicate the ViTDet heuristic (every depth/4-th block is global).

    Keeps the last layer of each quarter at global attention, just like the
    official ViT-B recipe which keeps blocks 2, 5, 8, 11 global at depth 12.
    """
    period = max(depth // 4, 1)
    global_idxs = set(range(period - 1, depth, period))
    global_idxs.add(depth - 1)
    return sorted(i for i in range(depth) if i not in global_idxs)


def build_vitdet_model(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    drop_path_rate: float = 0.1,
    window_size: int = 14,
    pretrain_img_size: int = 256,
):
    """Return a LazyConfig model tree configured with a LabelMix ViT backbone.

    The rest of Mask R-CNN (RPN / ROI heads / losses) is kept at the ViTDet
    defaults.
    """

    model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

    model.backbone = L(SimpleFeaturePyramid)(
        net=L(ViT)(
            img_size=256,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            window_size=window_size,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=_window_block_indexes(depth),
            residual_block_indexes=[],
            use_rel_pos=True,
            # LabelMix ViTs use no_embed_class=True, so the pretrained pos_embed
            # has *no* class-token slot.
            pretrain_use_cls_token=False,
            pretrain_img_size=pretrain_img_size,
            out_feature="last_feat",
        ),
        in_feature="${.net.out_feature}",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=L(LastLevelMaxPool)(),
        norm="LN",
        square_pad=256,
    )

    # ViTDet Mask R-CNN head tweaks (same as upstream)
    model.roi_heads.box_head.conv_norm = "LN"
    model.roi_heads.mask_head.conv_norm = "LN"
    model.proposal_generator.head.conv_dims = [-1, -1]
    model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
    model.roi_heads.box_head.fc_dims = [1024]
    return model
