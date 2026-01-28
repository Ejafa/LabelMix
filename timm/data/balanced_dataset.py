import io
import os
import pickle
import random
from collections import defaultdict

import torch
import torch.distributed as dist
from PIL import Image, UnidentifiedImageError
from torch.utils.data import IterableDataset, get_worker_info


class BalancedBucketDataset(IterableDataset):
    """
    A high-performance, DDP-safe IterableDataset for class-balanced training.

    Features:
    - Pure PyTorch vectorization for index generation.
    - Supports Standard ImageFolder AND Hugging Face Datasets.
    - Encoded buffering (bytes/PIL) for low RAM usage.
    - Automatic RAM caching for small minority classes.
    """
    def __init__(
            self,
            base_dataset,
            transform=None,
            mode='max',
            buffer_size=4096,
            cache_path='class_buckets.pkl',
            cache_small_classes_threshold=100,
            input_key='image',
            target_key='label',
    ):
        """
        Args:
            base_dataset: Standard ImageFolder or timm ImageDataset.
            transform: The augmentation transform to apply AFTER buffering.
            mode: 'min' (undersample), 'max' (oversample), or int (fixed cycles).
            buffer_size: Size of the shuffle buffer (in images).
            cache_path: Path to cache the class buckets.
            cache_small_classes_threshold: If a class has fewer images than this, cache them in RAM (encoded).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.mode = mode
        self.buffer_size = int(buffer_size)
        self.cache_threshold = int(cache_small_classes_threshold)
        self.input_key = input_key
        self.target_key = target_key
        self._epoch = 0

        if self.buffer_size < 1:
            raise ValueError('buffer_size must be >= 1')

        # --- 1. Efficient Indexing ---
        self.reader = getattr(base_dataset, 'reader', None)
        self.is_hf = hasattr(base_dataset, 'features') or hasattr(base_dataset, 'column_names')
        self._raw_samples = self._get_samples_list(base_dataset)
        self.buckets = self._load_or_build_buckets(cache_path)

        # Vectorization Prep: Convert to Torch Tensors
        self.classes = torch.tensor(sorted(list(self.buckets.keys())), dtype=torch.int64)
        self.num_classes = len(self.classes)
        if self.num_classes == 0:
            raise ValueError('No classes found in dataset')
        self.bucket_arrays = [torch.tensor(self.buckets[int(c)], dtype=torch.int64) for c in self.classes]
        self.bucket_lengths = torch.tensor([len(b) for b in self.bucket_arrays], dtype=torch.int64)
        if torch.any(self.bucket_lengths == 0):
            raise ValueError('One or more classes have zero samples')

        self.bucket_len_map = {int(c): int(l) for c, l in zip(self.classes, self.bucket_lengths)}

        # --- 2. Calculate M (Target Cycles) ---
        if isinstance(mode, int):
            self.M = int(mode)
        elif mode == 'min':
            self.M = int(torch.min(self.bucket_lengths).item())
        else:  # max
            self.M = int(torch.max(self.bucket_lengths).item())

        if self.M < 1:
            raise ValueError('mode must result in at least 1 cycle per class')

        self.total_images_global = int(self.num_classes * self.M)

        # --- 3. Minority Class Caching Init ---
        # We allow workers to populate this locally
        self.local_byte_cache = {}

    def _get_samples_list(self, base_dataset):
        if hasattr(base_dataset, 'parser'):
            return base_dataset.parser.samples
        if hasattr(base_dataset, 'samples'):
            return base_dataset.samples
        if self.reader is not None and hasattr(self.reader, 'samples'):
            return self.reader.samples
        return None

    def _get_targets(self):
        # 1. Try fast raw samples access (ImageFolder/timm)
        if self._raw_samples is not None:
            return [s[1] for s in self._raw_samples]

        # 2. Try Hugging Face Column Access
        if self.is_hf:
            try:
                # This loads the label column. Efficient for arrow datasets.
                return self.base_dataset[self.target_key]
            except KeyError:
                raise AttributeError(
                    f'HF dataset missing target_key="{self.target_key}". '
                    f'Available: {self.base_dataset.column_names}'
                )
            except Exception as e:
                raise AttributeError(f'Failed to load targets from HF dataset: {e}')

        # 3. Try timm Reader
        if self.reader is not None and hasattr(self.reader, 'dataset'):
            label_key = getattr(self.reader, 'label_key', self.target_key)
            try:
                targets = self.reader.dataset[label_key]
            except Exception as e:
                raise AttributeError(f'Dataset missing label_key="{label_key}"') from e
            if getattr(self.reader, 'remap_class', False):
                class_to_idx = getattr(self.reader, 'class_to_idx', None)
                if class_to_idx is not None:
                    targets = [class_to_idx[t] for t in targets]
            return targets

        # 4. Fallback attributes
        if hasattr(self.base_dataset, 'targets'):
            return self.base_dataset.targets

        raise AttributeError(
            'base_dataset must provide targets, samples, or be a Hugging Face dataset for balanced loading'
        )

    def _is_primary(self):
        return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _load_or_build_buckets(self, cache_path):
        # Attempt Load
        if cache_path and os.path.exists(cache_path):
            if self._is_primary():
                print(f'[BalancedDataset] Loading buckets from {cache_path}')
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                print('[BalancedDataset] Cache corrupt or unreadable, rebuilding...')

        if self._is_primary():
            print('[BalancedDataset] Indexing dataset (one-time setup)...')

        buckets = defaultdict(list)
        targets = self._get_targets()

        # Validation: Ensure targets align with dataset length
        try:
            total_len = len(self.base_dataset)
            if len(targets) != total_len:
                print(
                    f'[BalancedDataset] Warning: Target length ({len(targets)}) '
                    f'!= Dataset length ({total_len}).'
                )
        except Exception:
            pass

        for idx, target in enumerate(targets):
            buckets[int(target)].append(idx)

        # Atomic Save
        if cache_path and self._is_primary():
            try:
                tmp_path = f'{cache_path}.tmp'
                with open(tmp_path, 'wb') as f:
                    pickle.dump(dict(buckets), f)
                os.replace(tmp_path, cache_path)
            except Exception as e:
                print(f'[BalancedDataset] Failed to save cache: {e}')

        return buckets

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        if dist.is_available() and dist.is_initialized():
            return self.total_images_global // dist.get_world_size()
        return self.total_images_global

    def __iter__(self):
        # --- Worker & DDP Split ---
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        base_seed = worker_info.seed if worker_info else torch.initial_seed()

        # Global Stride
        total_workers = num_replicas * num_workers
        global_worker_id = (rank * num_workers) + worker_id

        # Seed torch RNG for this worker + epoch
        seed = (base_seed + self._epoch * 10007 + rank * 1000003) % (2 ** 32 - 1)
        rng = torch.Generator()
        rng.manual_seed(int(seed))

        # Assign Cycles to Worker
        worker_cycles = torch.arange(global_worker_id, self.M, total_workers, dtype=torch.int64)
        num_worker_cycles = int(worker_cycles.numel())

        if num_worker_cycles == 0:
            return

        # --- 4. VECTORIZED Scheduler ---
        # Create a matrix of indices: (Cycles, Classes)
        schedule = torch.empty((num_worker_cycles, self.num_classes), dtype=torch.int64)

        for i in range(self.num_classes):
            indices = self.bucket_arrays[i]
            n_imgs = int(indices.numel())

            if self.mode == 'min':
                # Deterministic striding
                selected_pos = worker_cycles % n_imgs
                schedule[:, i] = indices[selected_pos]
            else:
                # Random sampling with replacement
                rand_pos = torch.randint(n_imgs, (num_worker_cycles,), generator=rng)
                schedule[:, i] = indices[rand_pos]

        # Shuffle columns (Classes) within each row (Cycle)
        noise = torch.rand(schedule.shape, generator=rng)
        permutations = torch.argsort(noise, dim=1)

        row_indices = torch.arange(num_worker_cycles, dtype=torch.int64).unsqueeze(1)
        shuffled_schedule = schedule[row_indices, permutations]

        # Flatten to 1D stream
        flat_indices = shuffled_schedule.flatten()

        # --- 5. High-Performance Yield Loop ---
        buffer = []
        raw_samples = self._raw_samples

        for idx in flat_indices:
            idx_int = int(idx)
            cached = self.local_byte_cache.get(idx_int)
            # Hit Cache
            if cached is not None:
                img_obj, target = cached
                buffer.append((img_obj, target))
            else:
                # Miss Cache - Fetch
                try:
                    # A. Standard ImageFolder (Fastest)
                    if raw_samples is not None:
                        path, target = raw_samples[idx_int][0], raw_samples[idx_int][1]
                        with open(path, 'rb') as f:
                            img_obj = f.read()

                    # B. Hugging Face / Map-Style
                    else:
                        item = self.base_dataset[idx_int]
                        if self.is_hf:
                            img_obj = item[self.input_key]
                            target = item[self.target_key]
                        elif isinstance(item, (tuple, list)) and len(item) >= 2:
                            img_obj, target = item[0], item[1]
                        elif isinstance(item, dict):
                            img_obj = item[self.input_key]
                            target = item[self.target_key]
                        else:
                            # If we hit this, the dataset format is unknown.
                            # We raise error here because catching it loop-wide is dangerous.
                            raise ValueError(f'Unknown sample format at index {idx_int}: {type(item)}')

                    # Populate Cache (Small classes only)
                    if self.bucket_len_map.get(int(target), 0) <= self.cache_threshold:
                        self.local_byte_cache[idx_int] = (img_obj, target)

                    buffer.append((img_obj, target))

                except (IOError, OSError, UnidentifiedImageError, IndexError, ValueError):
                    # Only catch predictable data errors.
                    # Let KeyErrors (config errors) crash the script so you see them.
                    continue

            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for b in buffer:
                    yield self._process_item(b[0], b[1])
                buffer = []

        # Flush remaining
        if buffer:
            random.shuffle(buffer)
            for b in buffer:
                yield self._process_item(b[0], b[1])

    def _process_item(self, img_obj, target):
        try:
            if isinstance(img_obj, bytes):
                img = Image.open(io.BytesIO(img_obj)).convert('RGB')
            elif isinstance(img_obj, Image.Image):
                img = img_obj.convert('RGB')
            else:
                # Fallback for numpy arrays (often used in TFDS/custom datasets)
                img = Image.fromarray(img_obj).convert('RGB')
        except Exception:
            # If decoding fails here, we return a black image to keep batch size consistent
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)
        return img, target
