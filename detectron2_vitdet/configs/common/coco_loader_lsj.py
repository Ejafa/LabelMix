"""COCO dataloader with Large-Scale Jittering (same recipe as upstream ViTDet).

The images are resized & cropped to 1024x1024, which matches the ViT backbone
``img_size`` used at detection time.
"""

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

image_size = 256

dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0,
        target_height=image_size, target_width=image_size,
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
# With images resized to 256x256 (~16x less activation memory than 1024),
# we can comfortably 4x the batch size from the upstream ViTDet default of 64.
dataloader.train.total_batch_size = 256
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]
