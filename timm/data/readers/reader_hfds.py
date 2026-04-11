""" Dataset reader that wraps Hugging Face datasets

Hacked together by / Copyright 2022 Ross Wightman
"""
import io
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
from PIL import Image

try:
    import datasets
except ImportError as e:
    print("Please install Hugging Face datasets package `pip install datasets`.")
    raise e
from .class_map import load_class_map
from .reader import Reader


def _try_load_arrow_disk(root: Optional[str], split: str):
    """Try to load a pre-saved Arrow dataset from ``<root>/arrow/<split>/``.

    Returns the loaded dataset or *None* if the directory does not exist.
    These directories are created by ``copy_data_to_ram.py`` via
    ``datasets.save_to_disk()`` and can be loaded instantly with
    ``datasets.load_from_disk()`` — no fingerprint matching required.
    """
    if root is None:
        return None
    arrow_dir = os.path.join(root, "arrow", split)
    if not os.path.isdir(arrow_dir):
        return None
    return datasets.load_from_disk(arrow_dir)


def get_class_labels(info, label_key='label'):
    if 'label' not in info.features:
        return {}
    class_label = info.features[label_key]
    class_to_idx = {n: class_label.str2int(n) for n in class_label.names}
    return class_to_idx


class ReaderHfds(Reader):

    def __init__(
            self,
            name: str,
            root: Optional[str] = None,
            split: str = 'train',
            class_map: dict = None,
            input_key: str = 'image',
            target_key: str = 'label',
            additional_features: Optional[list[str]] = None,
            download: bool = False,
            trust_remote_code: bool = False,
    ):
        super().__init__()
        self.root = root
        self.split = split

        # Prefer pre-saved Arrow datasets (created by copy_data_to_ram.py)
        # which bypass HF's fingerprint-based cache and load instantly.
        ds = _try_load_arrow_disk(self.root, split)
        if ds is not None:
            self.dataset = ds
        else:
            self.dataset = datasets.load_dataset(
                name,  # 'name' maps to path arg in hf datasets
                split=split,
                cache_dir=self.root,
                trust_remote_code=trust_remote_code
            )

        # leave decode for caller, plus we want easy access to original path names...
        self.dataset = self.dataset.cast_column(input_key, datasets.Image(decode=False))

        self.image_key = input_key
        self.label_key = target_key
        self.remap_class = False
        if class_map:
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = get_class_labels(self.dataset.info, self.label_key)

        # load_from_disk datasets may not have split metadata in info.splits,
        # so fall back to len(dataset) for num_samples.
        self.split_info = None
        if self.dataset.info.splits and split in self.dataset.info.splits:
            self.split_info = self.dataset.info.splits[split]
            self.num_samples = self.split_info.num_examples
        else:
            self.num_samples = len(self.dataset)

        if additional_features is not None:
            if isinstance(additional_features, list):
                self.additional_features = additional_features
            else:
                self.additional_features = [additional_features]
        else:
            self.additional_features = None

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[self.image_key]

        if 'bytes' in image and image['bytes']:
            image = io.BytesIO(image['bytes'])
        else:
            assert 'path' in image and image['path']
            image = open(image['path'], 'rb')

        label = item[self.label_key]
        if self.remap_class:
            label = self.class_to_idx[label]

        if self.additional_features is not None:
            features = [item[feat] for feat in self.additional_features]
            return image, label, *features
        else:
            return image, label

    def __len__(self):
        return len(self.dataset)

    def _filename(self, index, basename=False, absolute=False):
        item = self.dataset[index]
        return item[self.image_key]['path']
