#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import copy
import glob
import importlib
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.utils
import yaml

from timm.data import BalancedBucketDataset
from timm import utils
from timm.data import create_dataset, create_loader, create_naflex_loader, resolve_data_config, \
    create_transform, Mixup, FastCollateMixup, AugMixDataset
from timm.data.loader import PrefetchLoader, _worker_init, fast_collate
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import (
    JsdCrossEntropy,
    SoftTargetCrossEntropy,
    LabelMixSoftTargetCrossEntropy,
    LabelMixPlackettLuceLoss,
    BinaryCrossEntropy,
    LabelSmoothingCrossEntropy,
)
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler
from timm.task import (
    ClassificationTask,
    LogitDistillationTask,
    FeatureDistillationTask,
    TokenDistillationTask,
)

try:
    mp.set_sharing_strategy("file_system")
except Exception as exc:
    logging.warning("Unable to set multiprocessing sharing strategy: %s", exc)

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

_WANDB_RUN_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"wandb:\s+setting up run\s+([A-Za-z0-9]+)"),
    re.compile(r"/runs/([A-Za-z0-9]+)\b"),
    re.compile(r"run-\d{8}_\d{6}-([A-Za-z0-9]+)\b"),
]


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------
# The module-level ``config_parser`` handles only ``-c / --config`` so a YAML
# file can be loaded before the main parser runs.  The main ``parser`` carries
# every training argument.  Both are created here at module scope so that the
# existing CLI entrypoint (``python train.py …``) keeps working exactly as
# before.
#
# For programmatic / Ray-based usage, call ``create_parsers()`` to get a *fresh*
# pair with no shared state, or use ``build_args()`` which internally creates
# fresh parsers each time.
# ---------------------------------------------------------------------------

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
group.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
group.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--train-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in train split, for IterableDatasets.')
group.add_argument('--val-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in validation split, for IterableDatasets.')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
group.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
group.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
group.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')
group.add_argument('--dataset-trust-remote-code', action='store_true', default=False,
                   help='Allow huggingface dataset import to execute code downloaded from the dataset\'s repo.')
group.add_argument('--balanced-mode', default='', type=str,
                   help='Enable balanced loading: "min", "max", or int (samples per class). Default: disabled.')
group.add_argument('--balanced-buffer', default=4096, type=int,
                   help='Buffer size for balanced loading (default: 4096)')
group.add_argument('--balanced-buffer-steps', default='0', type=str,
                   help='If > 0, set balanced buffer size to steps * batch_size '
                        '(or global batch size when centralized LabelMix is enabled).')
group.add_argument('--balanced-cache-threshold-steps', default='0', type=str,
                   help='If > 0, set balanced cache threshold to steps * batch_size '
                        '(or global batch size when centralized LabelMix is enabled).')
group.add_argument('--balanced-cache-path', default='', type=str,
                   help='Cache path for balanced class buckets '
                        '(default: <output>/<experiment>/class_buckets.pkl if --experiment is set, '
                        'else <data-dir>/class_buckets.pkl)')
group.add_argument('--balanced-cache-threshold', default=256, type=int,
                   help='Cache classes with <= N samples in RAM for balanced loading (default: 256)')
group.add_argument('--balanced-input-key', default=None, type=str,
                   help='Input key override for balanced loading (default: --input-key or "image")')
group.add_argument('--balanced-target-key', default=None, type=str,
                   help='Target key override for balanced loading (default: --target-key or "label")')
group.add_argument('--labelmix', action='store_true', default=False,
                   help='Enable LabelMix augmentation for balanced loading.')
group.add_argument('--labelmix-loss', default='soft_ce', type=str,
                   choices=['soft_ce', 'pl_loss'],
                   help='Loss for LabelMix targets: "soft_ce" or "pl_loss".')
group.add_argument('--labelmix-mix-k', default=5, type=int,
                   help='LabelMix K (number of source images per output, >= 1).')
group.add_argument('--labelmix-k-min', default=None, type=int,
                   help='LabelMix K minimum for K scheduler (default: --labelmix-mix-k).')
group.add_argument('--labelmix-k-max', default=None, type=int,
                   help='LabelMix K maximum for K scheduler (default: --labelmix-mix-k).')
group.add_argument('--labelmix-k-schedule', default='linear', type=str,
                   choices=['fixed', 'linear', 'cosine'],
                   help='LabelMix K schedule over epochs.')
group.add_argument('--labelmix-k-reverse', action='store_true', default=False,
                   help='Reverse LabelMix K schedule (max->min).')
group.add_argument('--labelmix-k-warmup-epochs', default=0, type=int,
                   help='Warmup epochs for LabelMix K schedule.')
group.add_argument('--labelmix-k-total-epochs', default=None, type=int,
                   help='Total epochs for LabelMix K schedule (default: inferred from total run length).')
group.add_argument('--labelmix-alpha-min', default=0.1, type=float,
                   help='LabelMix Dirichlet alpha min.')
group.add_argument('--labelmix-alpha-max', default=1.0, type=float,
                   help='LabelMix Dirichlet alpha max.')
group.add_argument('--labelmix-schedule', default='linear', type=str,
                   choices=['fixed', 'linear', 'cosine'],
                   help='LabelMix alpha schedule.')
group.add_argument('--labelmix-reverse', action='store_true', default=False,
                   help='Reverse LabelMix alpha schedule (max->min).')
group.add_argument('--labelmix-step-mode', default='total', type=str,
                   choices=['epoch', 'total'],
                   help='LabelMix alpha step mode.')
group.add_argument('--labelmix-warmup-steps', default=0, type=int,
                   help='LabelMix warmup steps for alpha schedule.')
group.add_argument('--labelmix-total-epochs', default=None, type=int,
                   help='LabelMix total epochs for schedule (required for step-mode=total).')
group.add_argument('--labelmix-total-steps', default=None, type=int,
                   help='LabelMix total steps for schedule (alternative to total-epochs).')
group.add_argument('--labelmix-sampling', action='store_true', default=False,
                   help='Enable minimum-weight sampling for LabelMix weights.')
group.add_argument('--labelmix-sampling-min-side-px', default=6, type=int,
                   help='Minimum side length (px) allowed for any slot in base layout.')
group.add_argument('--labelmix-sampling-max-aspect', default=10.0, type=float,
                   help='Maximum aspect ratio allowed for any slot in base layout.')
group.add_argument('--labelmix-sampling-bins', default=16, type=int,
                   help='Number of alpha bins for sampling cache.')
group.add_argument('--labelmix-sampling-pool-size', default=128, type=int,
                   help='Per-bin sampling pool size.')
group.add_argument('--labelmix-sampling-low-watermark', default=32, type=int,
                   help='Refill threshold for sampling pool.')
group.add_argument('--labelmix-sampling-max-attempts', default=200, type=int,
                   help='Max attempts per sampled weight.')
group.add_argument('--labelmix-producer-rank', default=-1, type=int,
                   help='Centralize LabelMix on a single rank and scatter batches. '
                        'Set to rank id (e.g. 0) to enable, -1 disables.')
group.add_argument('--labelmix-producer-workers', default=0, type=int,
                   help='Num DataLoader workers for centralized LabelMix producer. '
                        'If <=0, falls back to --workers.')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--pretrained-path', default=None, type=str,
                   help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')
group.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
group.add_argument('--amp', action='store_true', default=False,
                   help='use AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--model-dtype', default=None, type=str,
                   help='Model dtype override (non-AMP) (default: float32)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--layer-decay-min-scale', type=float, default=0,
                   help='layer-wise lr decay minimum scale clamp (default: 0)')
group.add_argument('--layer-decay-no-opt-scale', type=float, default=None,
                   help='layer-wise lr decay no optimization scale (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off step percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--num-steps', type=int, default=None, metavar='N',
                   help='number of optimization steps to train (default: auto)')
group.add_argument('--start-step', default=None, type=int, metavar='N',
                   help='manual step number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay step indices for multistep lr. must be increasing')
group.add_argument('--decay-steps', type=float, default=90, metavar='N',
                   help='step interval to decay LR')
group.add_argument('--warmup-steps', type=int, default=5, metavar='N',
                   help='steps to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-steps', type=int, default=0, metavar='N',
                   help='steps to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-steps', type=int, default=10, metavar='N',
                   help='patience steps for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--train-crop-mode', type=str, default=None,
                   help='Crop-mode in train'),
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying any color jitter.')
group.add_argument('--grayscale-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying random grayscale conversion.')
group.add_argument('--gaussian-blur-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying gaussian blur.')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-sum', action='store_true', default=False,
                   help='Sum over classes when using BCE loss.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled).')
group.add_argument('--bce-pos-weight', type=float, default=None,
                   help='Positive weighting for BCE loss.')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-step', default=0, type=int, metavar='N',
                   help='Turn off mixup after this step, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                   help='Enable synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each validation interval ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights.')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='Decay factor for model weights moving average (default: 0.9998)')
group.add_argument('--model-ema-warmup', action='store_true',
                   help='Enable warmup for model EMA decay.')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many steps to wait before logging training status')
group.add_argument('--num-logs', type=int, default=None, metavar='N',
                   help='Number of logging events over the total training duration (overrides log-interval)')
group.add_argument('--val-interval', type=int, default=1, metavar='N',
                   help='how many steps between validation')
group.add_argument('--num-evals', type=int, default=None, metavar='N',
                   help='Number of evaluations over the total training duration (overrides val-interval)')
group.add_argument('--num-saves', type=int, default=None, metavar='N',
                   help='Number of checkpoint saves over the total training duration (decoupled from eval)')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many steps to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--loader-prefetch-factor', type=int, default=None, metavar='N',
                   help='DataLoader prefetch factor per worker when --workers > 0 (default: PyTorch default)')
group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input batches every log interval for debugging')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--log-dir', default='', type=str, metavar='DIR',
                   help='directory for training log files (default: none, no file logging)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
group.add_argument('--wandb-project', default=None, type=str,
                   help='wandb project name')
group.add_argument('--wandb-tags', default=[], type=str, nargs='+',
                   help='wandb tags')
group.add_argument('--wandb-resume-id', default='', type=str, metavar='ID',
                   help='If resuming a run, the id of the run in wandb')
group.add_argument('--check-resume', action='store_true', default=False,
                   help='Auto-detect --resume checkpoint and wandb resume id from local metadata.')
group.add_argument('--check-resume-status-file', default='', type=str, metavar='PATH',
                   help='Status YAML used to recover wandb id when --check-resume is enabled '
                        '(default: <output>/<experiment>/run_status.yaml).')
group.add_argument('--check-resume-log-dir', default='./logs', type=str, metavar='DIR',
                   help='Log root scanned for wandb id fallback when --check-resume is enabled.')
group.add_argument('--check-resume-search-limit', default=30, type=int, metavar='N',
                   help='Max number of recent log files to inspect for wandb id during --check-resume.')

# NaFlex scheduled loader arguments
group.add_argument('--naflex-loader', action='store_true', default=False,
                   help='Use NaFlex loader (Requires NaFlex compatible model)')
group.add_argument('--naflex-train-seq-lens', type=int, nargs='+', default=[128, 256, 576, 784, 1024],
                   help='Sequence lengths to use for NaFlex loader')
group.add_argument('--naflex-max-seq-len', type=int, default=576,
                   help='Fixed maximum sequence length for NaFlex loader (validation)')
group.add_argument('--naflex-patch-sizes', type=int, nargs='+', default=None,
                   help='List of patch sizes for variable patch size training (e.g., 8 12 16 24 32)')
group.add_argument('--naflex-patch-size-probs', type=float, nargs='+', default=None,
                   help='Probabilities for each patch size (must sum to 1.0, uniform if not specified)')
group.add_argument('--naflex-loss-scale', default='linear', type=str,
                   help='Scale loss (gradient) by batch_size ("none", "sqrt", or "linear")')

# Knowledge Distillation parameters
parser.add_argument('--kd-model-name', default=None, type=str,
                    help='Name of teacher model for knowledge distillation')
parser.add_argument('--kd-distill-type', default='logit', type=str, choices=['logit', 'feature', 'token'],
                    help='Type of distillation: "logit" for output distillation, "feature" for intermediate features, "token" for models with distillation heads (default: logit)')
parser.add_argument('--kd-loss-type', default='kl', type=str, choices=['kl', 'plackett_luce'],
                    help='Loss function for logit distillation (default: kl): "kl" or "plackett_luce". '
                         '"plackett_luce" uses only distillation loss as total loss.')
parser.add_argument('--distill-loss-weight', default=None, type=float,
                    help='Weight for distillation loss. If both weights specified: loss = task_weight * task + distill_weight * distill. '
                         'If only task_weight: loss = task_weight * task + (1-task_weight) * distill. '
                         'Ignored when --kd-loss-type=plackett_luce.')
parser.add_argument('--task-loss-weight', default=None, type=float,
                    help='Weight for task (classification) loss. See --distill-loss-weight for weighting modes. '
                         'Ignored when --kd-loss-type=plackett_luce.')
parser.add_argument('--kd-temperature', default=4.0, type=float,
                    help='Temperature for softmax in distillation (default: 4.0, typical range: 1-4)')
parser.add_argument('--kd-student-feature-dim', default=None, type=int,
                    help='Student model feature dimension (auto-detected from model.head_hidden_size or model.num_features if not specified)')
parser.add_argument('--kd-teacher-feature-dim', default=None, type=int,
                    help='Teacher model feature dimension (auto-detected from model.head_hidden_size or model.num_features if not specified)')
parser.add_argument('--kd-token-distill-type', default='soft', type=str, choices=['soft', 'hard'],
                    help='Token distillation type: "soft" for KL-div with temperature, "hard" for CE with teacher argmax (default: soft)')


# ---------------------------------------------------------------------------
# Public API for programmatic use (Ray, notebooks, tests)
# ---------------------------------------------------------------------------

def create_parsers():
    """Return a *fresh* ``(config_parser, main_parser)`` pair.

    Every call builds new ``ArgumentParser`` instances so callers (e.g. Ray
    trials running concurrently) never share mutable state.

    *config_parser* handles only ``-c / --config``.
    *main_parser* contains every training argument.
    """
    import copy as _copy
    cfg_p = argparse.ArgumentParser(description='Training Config', add_help=False)
    cfg_p.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                       help='YAML config file specifying default arguments')
    # Deep-copy the module-level parser so all arguments are present but
    # set_defaults() in one trial cannot leak into another.
    main_p = _copy.deepcopy(parser)
    return cfg_p, main_p


def build_args(
    config_path: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
    dict_overrides: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Construct training args programmatically.

    Config precedence (highest wins):
        1. YAML config (``config_path``)
        2. ``dict_overrides`` (common Python overrides)
        3. ``cli_overrides`` (trial-specific overrides as a list of CLI tokens)

    Returns ``(args, args_text)`` identical in shape to the old
    ``_parse_args()`` return value.
    """
    import copy as _copy
    p = _copy.deepcopy(parser)

    # 1. YAML config defaults
    if config_path:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg:
            p.set_defaults(**cfg)

    # 2. dict overrides (applied as defaults so CLI can still override)
    if dict_overrides:
        p.set_defaults(**dict_overrides)

    # 3. CLI-style overrides
    args = p.parse_args(cli_overrides or [])

    # Attach config path for downstream use
    if config_path and not getattr(args, 'config', ''):
        args.config = config_path

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def validate_args(args) -> None:
    """Run the validation checks that ``main()`` performs on parsed args.

    Raises ``SystemExit`` (via ``parser.error``) or ``ValueError`` on
    invalid combinations.  Call this *after* ``build_args`` or ``_parse_args``
    and *before* ``run_training``.
    """
    if getattr(args, 'check_resume_search_limit', 1) < 1:
        raise ValueError('--check-resume-search-limit must be >= 1')

    if getattr(args, 'labelmix', False):
        if not getattr(args, 'balanced_mode', ''):
            raise ValueError('--labelmix requires --balanced-mode to be set')
        if getattr(args, 'labelmix_mix_k', 1) < 1:
            raise ValueError('--labelmix-mix-k must be >= 1')
        k_min = getattr(args, 'labelmix_k_min', None)
        k_max = getattr(args, 'labelmix_k_max', None)
        if k_min is not None and k_min < 1:
            raise ValueError('--labelmix-k-min must be >= 1')
        if k_max is not None and k_max < 1:
            raise ValueError('--labelmix-k-max must be >= 1')
        if k_min is not None and k_max is not None and k_max < k_min:
            raise ValueError('--labelmix-k-max must be >= --labelmix-k-min')

    bm = getattr(args, 'balanced_mode', '')
    if bm:
        if isinstance(bm, str):
            mode = bm.strip().lower()
            if mode.isdigit():
                mode = int(mode)
            elif mode not in ('min', 'max'):
                raise ValueError('--balanced-mode must be "min", "max", or a positive int')
        elif isinstance(bm, int):
            mode = bm
        else:
            raise ValueError('--balanced-mode must be "min", "max", or a positive int')
        if isinstance(mode, int) and mode < 1:
            raise ValueError('--balanced-mode int value must be >= 1')




def _extract_wandb_id_from_text(text: str) -> Optional[str]:
    for pattern in _WANDB_RUN_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _extract_wandb_id_from_file(path: str, max_bytes: int = 1_048_576) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_bytes)
    except Exception:
        return None
    return _extract_wandb_id_from_text(content)


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def _find_wandb_id_in_status_file(
    status_file: str,
    output_dir: Optional[str],
    experiment: Optional[str],
) -> Optional[str]:
    if not status_file or not os.path.exists(status_file):
        return None
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    output_norm = _normalized_path(output_dir) if output_dir else None

    def _entry_wandb_id(entry: Dict[str, Any]) -> Optional[str]:
        value = str(entry.get("wandb_id") or "").strip()
        if value:
            return value
        log_path = str(entry.get("log") or "").strip()
        return _extract_wandb_id_from_file(log_path)

    if output_norm:
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            entry_output = str(entry.get("output") or "").strip()
            if not entry_output:
                continue
            try:
                same_output = _normalized_path(entry_output) == output_norm
            except Exception:
                same_output = False
            if not same_output:
                continue
            found = _entry_wandb_id(entry)
            if found:
                return found

    exp_name = str(experiment or "").strip()
    if not exp_name:
        return None

    best_id = None
    best_stamp = ""
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name") or "").strip() != exp_name:
            continue
        found = _entry_wandb_id(entry)
        if not found:
            continue
        stamp = str(entry.get("updated_at") or entry.get("created_at") or "")
        if best_id is None or stamp >= best_stamp:
            best_id = found
            best_stamp = stamp
    return best_id


def _find_wandb_id_in_logs(log_root: str, experiment: Optional[str], search_limit: int) -> Optional[str]:
    if not log_root or search_limit < 1:
        return None
    exp_name = str(experiment or "").strip()
    if not exp_name:
        return None

    pattern = os.path.join(log_root, "**", f"{exp_name}*.txt")
    try:
        files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    except Exception:
        return None
    if not files:
        return None

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for path in files[:search_limit]:
        found = _extract_wandb_id_from_file(path)
        if found:
            return found
    return None


def _resolve_check_resume_status_file(args) -> str:
    status_file = str(getattr(args, "check_resume_status_file", "") or "").strip()
    if status_file:
        return status_file

    exp_name = str(getattr(args, "experiment", "") or "").strip()
    out_root = str(getattr(args, "output", "") or "./output/train")
    if exp_name:
        return os.path.join(out_root, exp_name, "run_status.yaml")

    resume_path = str(getattr(args, "resume", "") or "").strip()
    if resume_path:
        resume_dir = os.path.dirname(resume_path)
        if resume_dir:
            return os.path.join(resume_dir, "run_status.yaml")

    return os.path.join(out_root, "run_status.yaml")


def _auto_configure_resume(args) -> bool:
    if not args.check_resume:
        return False

    args.check_resume_status_file = _resolve_check_resume_status_file(args)

    changed = False
    if not args.resume:
        resume_candidates = []
        if args.experiment:
            out_root = args.output if args.output else "./output/train"
            resume_candidates.append(os.path.join(out_root, args.experiment, "last.pth.tar"))
        elif args.output:
            resume_candidates.append(os.path.join(args.output, "last.pth.tar"))

        for candidate in resume_candidates:
            if os.path.isfile(candidate):
                args.resume = candidate
                changed = True
                _logger.info("Auto-resume: using checkpoint %s", candidate)
                break

    if args.log_wandb and args.resume and not args.wandb_resume_id:
        resume_output_dir = os.path.dirname(args.resume)
        wandb_id = _find_wandb_id_in_status_file(
            args.check_resume_status_file,
            resume_output_dir,
            args.experiment,
        )
        if not wandb_id:
            wandb_id = _find_wandb_id_in_logs(
                args.check_resume_log_dir,
                args.experiment,
                int(args.check_resume_search_limit),
            )
        if wandb_id:
            args.wandb_resume_id = wandb_id
            changed = True
            _logger.info("Auto-resume: using wandb run id %s", wandb_id)
        else:
            _logger.warning(
                "Auto-resume: checkpoint found but wandb run id not found. "
                "Training resume will continue without wandb run resume."
            )

    return changed


def _resolve_balanced_cache_path(args) -> str:
    balanced_cache_path = str(getattr(args, "balanced_cache_path", "") or "").strip()
    if balanced_cache_path:
        return balanced_cache_path

    exp_name = str(getattr(args, "experiment", "") or "").strip()
    if exp_name:
        out_root = str(getattr(args, "output", "") or "./output/train")
        return os.path.join(out_root, exp_name, "class_buckets.pkl")

    base_dir = getattr(args, "data_dir", None) or getattr(args, "data", None) or "."
    return os.path.join(base_dir, "class_buckets.pkl")


def _status_key_for_experiment(exp_name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(exp_name)).strip("_")
    return f"experiment_{safe or 'default'}"


def _load_status_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in obj.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _save_status_yaml(path: str, status_data: Dict[str, Dict[str, Any]]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(status_data, f, sort_keys=True, default_flow_style=False)
    os.replace(tmp_path, path)


def _update_training_status(
    args,
    *,
    exp_name: str,
    output_dir: str,
    state: str,
    wandb_id: Optional[str] = None,
    start_step: Optional[int] = None,
    last_step: Optional[int] = None,
    launch_increment: bool = False,
    last_exit_code: Optional[int] = None,
) -> None:
    if not getattr(args, "check_resume", False):
        return

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    status_file = _resolve_check_resume_status_file(args)
    args.check_resume_status_file = status_file
    data = _load_status_yaml(status_file)
    key = _status_key_for_experiment(exp_name)

    entry: Dict[str, Any] = data.get(key, {}) if isinstance(data.get(key), dict) else {}
    if not entry.get("created_at"):
        entry["created_at"] = now

    entry["name"] = exp_name
    entry["output"] = output_dir
    entry["config"] = str(getattr(args, "config", "") or "")
    entry["status"] = state
    entry["updated_at"] = now
    entry["resume_checkpoint"] = args.resume if args.resume else None
    entry["resume_used"] = bool(args.resume)
    entry["finished"] = bool(state == "finished")

    if start_step is not None:
        entry["start_step"] = int(start_step)
    if last_step is not None:
        entry["last_step"] = int(last_step)
    if last_exit_code is not None:
        entry["last_exit_code"] = int(last_exit_code)
    if launch_increment:
        entry["launch_count"] = int(entry.get("launch_count", 0)) + 1
        entry["last_started_at"] = now
    if state in ("finished", "failed", "interrupted"):
        entry["last_finished_at"] = now

    if wandb_id:
        entry["wandb_id"] = str(wandb_id)
    elif args.wandb_resume_id and not entry.get("wandb_id"):
        entry["wandb_id"] = str(args.wandb_resume_id)

    data[key] = entry
    _save_status_yaml(status_file, data)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class RepeatingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.epoch = 0
        self._set_epoch(self.epoch)
        self.data_iter = iter(self.loader)

    def _set_epoch(self, epoch):
        if hasattr(self.loader, 'sampler') and hasattr(self.loader.sampler, 'set_epoch'):
            self.loader.sampler.set_epoch(epoch)
        if hasattr(self.loader, 'dataset') and hasattr(self.loader.dataset, 'set_epoch'):
            self.loader.dataset.set_epoch(epoch)

    def __iter__(self):
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._set_epoch(self.epoch)
        self.data_iter = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.epoch += 1
            self._set_epoch(self.epoch)
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _NoOp:
    def set_epoch(self, epoch: int) -> None:
        return None


class LabelMixBroadcastLoader:
    def __init__(
        self,
        loader,
        batch_size: int,
        mix_k: int,
        producer_rank: int,
        pin_memory: bool,
        group=None,
        log_fn=None,
    ) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("LabelMixBroadcastLoader requires distributed training.")
        self.loader = loader
        self.batch_size = int(batch_size)
        self.mix_k = int(mix_k)
        self.producer_rank = int(producer_rank)
        self.pin_memory = bool(pin_memory)
        self.group = group
        self.log_fn = log_fn
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._meta_ready = False
        self._input_buf: Optional[torch.Tensor] = None
        self._labels_buf: Optional[torch.Tensor] = None
        self._weights_buf: Optional[torch.Tensor] = None
        self._iter = None
        self._steps_per_epoch = self._broadcast_len()
        self.dataset = self.loader.dataset if self.rank == self.producer_rank and self.loader is not None else _NoOp()
        self.sampler = self.loader.sampler if self.rank == self.producer_rank and self.loader is not None else _NoOp()

    def _broadcast_len(self) -> int:
        length = 0
        if self.rank == self.producer_rank and self.loader is not None:
            try:
                length = len(self.loader)
            except TypeError:
                length = 0
        length_t = torch.tensor([length], dtype=torch.int64)
        dist.broadcast(length_t, src=self.producer_rank, group=self.group)
        return int(length_t.item())

    def __len__(self) -> int:
        return self._steps_per_epoch

    def __iter__(self):
        if self.rank == self.producer_rank:
            self._iter = None
        # Target shape can change across epochs (e.g. K scheduling), so refresh
        # scatter metadata and buffers at each new epoch iterator.
        self._meta_ready = False
        self._input_buf = None
        self._labels_buf = None
        self._weights_buf = None
        return self

    def __getattr__(self, name):
        if self.rank == self.producer_rank and self.loader is not None:
            return getattr(self.loader, name)
        raise AttributeError(name)


    @staticmethod
    def _dtype_from_string(value: str) -> torch.dtype:
        value = value.replace("torch.", "")
        return getattr(torch, value)

    def _broadcast_meta(self, input_tensor, labels, weights) -> None:
        if self.rank == self.producer_rank:
            if input_tensor is None or labels is None or weights is None:
                raise RuntimeError("Producer must supply tensors for meta broadcast.")
            if input_tensor.shape[0] % self.world_size != 0:
                raise ValueError("Global batch size must be divisible by world size.")
            per_rank_bs = input_tensor.shape[0] // self.world_size
            meta = {
                "input_shape": (per_rank_bs,) + tuple(input_tensor.shape[1:]),
                "input_dtype": str(input_tensor.dtype),
                "labels_shape": (per_rank_bs,) + tuple(labels.shape[1:]),
                "labels_dtype": str(labels.dtype),
                "weights_shape": (per_rank_bs,) + tuple(weights.shape[1:]),
                "weights_dtype": str(weights.dtype),
            }
        else:
            meta = None
        obj_list = [meta]
        dist.broadcast_object_list(obj_list, src=self.producer_rank, group=self.group)
        meta = obj_list[0]
        input_dtype = self._dtype_from_string(meta["input_dtype"])
        labels_dtype = self._dtype_from_string(meta["labels_dtype"])
        weights_dtype = self._dtype_from_string(meta["weights_dtype"])
        self._input_buf = torch.empty(
            meta["input_shape"],
            dtype=input_dtype,
            pin_memory=self.pin_memory,
        )
        self._labels_buf = torch.empty(
            meta["labels_shape"],
            dtype=labels_dtype,
            pin_memory=self.pin_memory,
        )
        self._weights_buf = torch.empty(
            meta["weights_shape"],
            dtype=weights_dtype,
            pin_memory=self.pin_memory,
        )
        self._meta_ready = True
        if self.log_fn is not None and self.rank == self.producer_rank:
            self.log_fn(
                f"[LabelMixProducer] broadcast meta: per_rank_bs={meta['input_shape'][0]} "
                f"mix_k={meta['labels_shape'][1]}"
            )

    @staticmethod
    def _split_target(target):
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            raise ValueError("LabelMix producer expected target as (labels, weights).")
        return target[0], target[1]

    def _scatter(self, input_tensor, labels, weights) -> None:
        if self.rank == self.producer_rank:
            input_chunks = list(input_tensor.chunk(self.world_size, dim=0))
            labels_chunks = list(labels.chunk(self.world_size, dim=0))
            weights_chunks = list(weights.chunk(self.world_size, dim=0))
            dist.scatter(self._input_buf, scatter_list=input_chunks, src=self.producer_rank, group=self.group)
            dist.scatter(self._labels_buf, scatter_list=labels_chunks, src=self.producer_rank, group=self.group)
            dist.scatter(self._weights_buf, scatter_list=weights_chunks, src=self.producer_rank, group=self.group)
        else:
            dist.scatter(self._input_buf, scatter_list=None, src=self.producer_rank, group=self.group)
            dist.scatter(self._labels_buf, scatter_list=None, src=self.producer_rank, group=self.group)
            dist.scatter(self._weights_buf, scatter_list=None, src=self.producer_rank, group=self.group)

    def __next__(self):
        eof_t = torch.zeros(1, dtype=torch.uint8)
        if self.rank == self.producer_rank:
            if self.loader is None:
                raise RuntimeError("LabelMix producer rank has no loader.")
            try:
                if self._iter is None:
                    self._iter = iter(self.loader)
                batch = next(self._iter)
                eof_t[0] = 0
            except StopIteration:
                self._iter = None
                eof_t[0] = 1
                batch = None
            dist.broadcast(eof_t, src=self.producer_rank, group=self.group)
            if eof_t.item():
                raise StopIteration
            input_tensor, target = batch
            labels, weights = self._split_target(target)
            if not self._meta_ready:
                self._broadcast_meta(input_tensor, labels, weights)
            self._scatter(input_tensor, labels, weights)
        else:
            dist.broadcast(eof_t, src=self.producer_rank, group=self.group)
            if eof_t.item():
                raise StopIteration
            if not self._meta_ready:
                self._broadcast_meta(None, None, None)
            self._scatter(None, None, None)
        return self._input_buf, (self._labels_buf, self._weights_buf)


def run_training(args=None, args_text=None):
    """Main training entry point.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Pre-built args (e.g. from ``build_args``).  If *None*, args are
        parsed from the command line via ``_parse_args()`` (original behavior).
    args_text : str, optional
        YAML-serialized args text saved alongside checkpoints.  Generated
        automatically when *args* is *None*.
    """
    log_path = ''
    if getattr(args, 'log_dir', ''):
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        exp_name_for_log = getattr(args, 'experiment', '') or 'train'
        log_path = os.path.join(log_dir, f'{exp_name_for_log}.log')
    utils.setup_default_logging(log_path=log_path)
    if args is None:
        args, args_text = _parse_args()
    elif args_text is None:
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    initial_status_file = str(getattr(args, "check_resume_status_file", "") or "").strip()

    def _parse_step_setting(value: str, name: str) -> tuple[int, bool]:
        raw = str(value).strip().lower()
        if raw in ('', '0'):
            return 0, False
        if raw in ('auto', '-1'):
            return 0, True
        if raw.isdigit():
            return int(raw), False
        parser.error(f'{name} must be a non-negative int or "auto"')
        return 0, False

    if args.check_resume_search_limit < 1:
        parser.error('--check-resume-search-limit must be >= 1')

    auto_resume_changed = _auto_configure_resume(args)
    status_file_changed = (
        str(getattr(args, "check_resume_status_file", "") or "").strip() != initial_status_file
    )
    if auto_resume_changed or status_file_changed:
        # Keep saved args.yaml aligned with any auto-resume updates.
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)

    if args.labelmix and args.prefetcher:
        _logger.info('Disabling prefetcher for LabelMix (target format not supported).')
        args.prefetcher = False

    if args.labelmix_total_steps is None and args.num_steps:
        args.labelmix_total_steps = args.num_steps

    if args.balanced_mode:
        if isinstance(args.balanced_mode, str):
            mode = args.balanced_mode.strip().lower()
            if mode.isdigit():
                mode = int(mode)
            elif mode not in ('min', 'max'):
                parser.error('--balanced-mode must be "min", "max", or a positive int')
        elif isinstance(args.balanced_mode, int):
            mode = args.balanced_mode
        else:
            parser.error('--balanced-mode must be "min", "max", or a positive int')
        if isinstance(mode, int) and mode < 1:
            parser.error('--balanced-mode int value must be >= 1')
        args.balanced_mode = mode

    if args.balanced_mode and args.prefetcher:
        _logger.info('Disabling prefetcher for balanced dataset (CPU-side normalization expected).')
        args.prefetcher = False

    if args.balanced_cache_threshold < 0:
        parser.error('--balanced-cache-threshold must be >= 0')
    balanced_buffer_steps, balanced_buffer_steps_auto = _parse_step_setting(
        args.balanced_buffer_steps, '--balanced-buffer-steps')
    balanced_cache_steps, balanced_cache_steps_auto = _parse_step_setting(
        args.balanced_cache_threshold_steps, '--balanced-cache-threshold-steps')
    if args.labelmix_producer_workers < 0:
        parser.error('--labelmix-producer-workers must be >= 0')

    if args.labelmix:
        if not args.balanced_mode:
            parser.error('--labelmix requires --balanced-mode to be set')
        if args.labelmix_mix_k < 1:
            parser.error('--labelmix-mix-k must be >= 1')
        if args.labelmix_k_min is not None and args.labelmix_k_min < 1:
            parser.error('--labelmix-k-min must be >= 1')
        if args.labelmix_k_max is not None and args.labelmix_k_max < 1:
            parser.error('--labelmix-k-max must be >= 1')
        if (
            args.labelmix_k_min is not None
            and args.labelmix_k_max is not None
            and args.labelmix_k_max < args.labelmix_k_min
        ):
            parser.error('--labelmix-k-max must be >= --labelmix-k-min')
        if args.labelmix_k_warmup_epochs < 0:
            parser.error('--labelmix-k-warmup-epochs must be >= 0')
        if args.labelmix_k_total_epochs is not None and args.labelmix_k_total_epochs <= 0:
            parser.error('--labelmix-k-total-epochs must be > 0')
        if args.labelmix_step_mode == 'total' and not (args.labelmix_total_epochs or args.labelmix_total_steps):
            parser.error('--labelmix-step-mode=total requires --labelmix-total-epochs or --labelmix-total-steps')
    elif args.labelmix_producer_rank >= 0:
        parser.error('--labelmix-producer-rank requires --labelmix')

    if args.balanced_mode and args.naflex_loader:
        parser.error('--balanced-mode is not compatible with --naflex-loader')
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    central_labelmix = args.labelmix and args.labelmix_producer_rank >= 0
    labelmix_cpu_group = None
    if central_labelmix:
        if not args.distributed:
            parser.error('--labelmix-producer-rank requires distributed training')
        if args.labelmix_producer_rank >= args.world_size or args.labelmix_producer_rank < 0:
            parser.error('--labelmix-producer-rank must be within [0, world_size)')
        try:
            labelmix_cpu_group = dist.new_group(backend='gloo')
        except Exception as exc:
            raise RuntimeError('Failed to create gloo process group for LabelMix producer.') from exc
        if utils.is_primary(args):
            _logger.info(
                f'LabelMix centralized producer enabled on rank {args.labelmix_producer_rank} '
                f'(world_size={args.world_size}).'
            )

    buffer_batch = args.batch_size * args.world_size if central_labelmix else args.batch_size
    balanced_buffer = args.balanced_buffer
    if balanced_buffer_steps_auto:
        balanced_buffer_steps = int(math.ceil(float(balanced_buffer) / float(buffer_batch)))
        if utils.is_primary(args):
            _logger.info(
                f'Balanced buffer steps (auto) resolved to {balanced_buffer_steps} '
                f'from buffer_size={balanced_buffer} (batch={buffer_batch}).'
            )
    if balanced_buffer_steps > 0:
        balanced_buffer = int(balanced_buffer_steps) * int(buffer_batch)
        if balanced_buffer < 1:
            parser.error('--balanced-buffer-steps yields buffer_size < 1')
        if utils.is_primary(args):
            mode = "auto" if balanced_buffer_steps_auto else "manual"
            _logger.info(
                f'Balanced buffer steps ({mode})={balanced_buffer_steps} '
                f'=> buffer_size={balanced_buffer} (batch={buffer_batch}).'
            )

    balanced_cache_threshold = args.balanced_cache_threshold
    if balanced_cache_steps_auto:
        balanced_cache_steps = int(math.ceil(float(balanced_cache_threshold) / float(buffer_batch)))
        if utils.is_primary(args):
            _logger.info(
                f'Balanced cache threshold steps (auto) resolved to {balanced_cache_steps} '
                f'from threshold={balanced_cache_threshold} (batch={buffer_batch}).'
            )
    if balanced_cache_steps > 0:
        balanced_cache_threshold = int(balanced_cache_steps) * int(buffer_batch)
        if balanced_cache_threshold < 0:
            parser.error('--balanced-cache-threshold-steps yields threshold < 0')
        if utils.is_primary(args):
            mode = "auto" if balanced_cache_steps_auto else "manual"
            _logger.info(
                f'Balanced cache threshold steps ({mode})={balanced_cache_steps} '
                f'=> threshold={balanced_cache_threshold} (batch={buffer_batch}).'
            )

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning('float16 is not recommended for training, for half precision bfloat16 is recommended.')

    # resolve AMP arguments based on PyTorch availability
    amp_dtype = torch.float16
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    )
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    # Create training task (classification or distillation)
    task = None

    param_counts = sum(m.numel() for m in model.parameters())

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{param_counts}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    model_patch_size = None
    if args.naflex_loader:
        # NaFlexVit models have embeds.patch_size. Needs to be extracted here before mutating the model.
        model_patch_size = getattr(getattr(model, "embeds", None), "patch_size", None)

    if args.torchscript:
        assert not args.torchcompile
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.weight_decay  # this isn't stored in optimizer.defaults
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
        )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type in ('cuda',) and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    # optionally resume from a checkpoint
    resume_step = None
    if args.resume:
        resume_step = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(
                model_ema,
                backend=args.torchcompile,
                mode=args.torchcompile_mode,
            )

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    dataset_eval = None
    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
            trust_remote_code=args.dataset_trust_remote_code,
        )

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    # Check if we should use the NaFlex scheduled loader
    loader_prefetch_factor = None
    if args.loader_prefetch_factor is not None and args.loader_prefetch_factor > 0:
        loader_prefetch_factor = args.loader_prefetch_factor
    prefetch_label = str(loader_prefetch_factor) if loader_prefetch_factor is not None else "default"
    _logger.info("DataLoader prefetch_factor=%s (workers=%s)", prefetch_label, args.workers)
    common_loader_kwargs = dict(
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        distributed=args.distributed,
        use_prefetcher=args.prefetcher,
        prefetch_factor=loader_prefetch_factor,
        persistent_workers=args.workers > 0,
    )

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        num_workers=args.workers,
        worker_seeding=args.worker_seeding,
    )
    train_common_loader_kwargs = dict(common_loader_kwargs)
    train_common_loader_kwargs['pin_memory'] = args.pin_mem
    train_common_loader_kwargs['persistent_workers'] = args.workers > 0

    mixup_fn = None
    mixup_args = {}
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    labelmix_train_dataset = None
    labelmix_k_total_epochs_auto = False
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )

    naflex_mode = False
    if args.naflex_loader:
        if utils.is_primary(args):
            _logger.info('Using NaFlex loader')

        assert num_aug_splits <= 1, 'Augmentation splits not supported in NaFlex mode'
        naflex_mixup_fn = None
        if mixup_active:
            from timm.data import NaFlexMixup
            mixup_args.pop('mode')  # not supported
            mixup_args.pop('cutmix_minmax')  # not supported
            naflex_mixup_fn = NaFlexMixup(**mixup_args)

        # Check if we have model's patch size for NaFlex mode
        if model_patch_size is None:
            # Fallback to default
            model_patch_size = (16, 16)
            if utils.is_primary(args):
                _logger.warning(f'Could not determine model patch size, using default: {model_patch_size}')

        # Configure patch sizes for NaFlex loader
        patch_loader_kwargs = {}
        if args.naflex_patch_sizes:
            # Variable patch size mode
            patch_loader_kwargs['patch_size_choices'] = args.naflex_patch_sizes
            if args.naflex_patch_size_probs:
                if len(args.naflex_patch_size_probs) != len(args.naflex_patch_sizes):
                    parser.error('--naflex-patch-size-probs must have same length as --naflex-patch-sizes')
                patch_loader_kwargs['patch_size_choice_probs'] = args.naflex_patch_size_probs
            if utils.is_primary(args):
                _logger.info(f'Using variable patch sizes: {args.naflex_patch_sizes}')
        else:
            # Single patch size mode - use model's patch size
            patch_loader_kwargs['patch_size'] = model_patch_size
            if utils.is_primary(args):
                _logger.info(f'Using model patch size: {model_patch_size}')

        naflex_mode = True
        loader_train = create_naflex_loader(
            dataset=dataset_train,
            train_seq_lens=args.naflex_train_seq_lens,
            mixup_fn=naflex_mixup_fn,
            rank=args.rank,
            world_size=args.world_size,
            **patch_loader_kwargs,
            **train_common_loader_kwargs,
            **train_loader_kwargs,
        )
    else:
        # setup mixup / cutmix
        collate_fn = None
        if mixup_active:
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        if args.balanced_mode:
            if num_aug_splits > 1:
                parser.error('--balanced-mode is not compatible with --aug-splits > 1')
            if args.aug_repeats:
                parser.error('--balanced-mode is not compatible with --aug-repeats')

            re_num_splits = 0
            if args.resplit:
                re_num_splits = num_aug_splits or 2

            train_transform = create_transform(
                input_size=data_config['input_size'],
                is_training=True,
                no_aug=args.no_aug,
                train_crop_mode=args.train_crop_mode,
                scale=args.scale,
                ratio=args.ratio,
                hflip=args.hflip,
                vflip=args.vflip,
                color_jitter=args.color_jitter,
                color_jitter_prob=args.color_jitter_prob,
                grayscale_prob=args.grayscale_prob,
                gaussian_blur_prob=args.gaussian_blur_prob,
                auto_augment=args.aa,
                interpolation=train_interpolation,
                mean=data_config['mean'],
                std=data_config['std'],
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                re_num_splits=re_num_splits,
                use_prefetcher=args.prefetcher,
                separate=num_aug_splits > 0,
            )

            balanced_input_key = args.balanced_input_key
            if balanced_input_key is None:
                balanced_input_key = args.input_key if args.input_key is not None else 'image'
            balanced_target_key = args.balanced_target_key
            if balanced_target_key is None:
                balanced_target_key = args.target_key if args.target_key is not None else 'label'
            balanced_cache_path = _resolve_balanced_cache_path(args)
            balanced_cache_dir = os.path.dirname(balanced_cache_path)
            if balanced_cache_dir:
                os.makedirs(balanced_cache_dir, exist_ok=True)

            labelmix_batch_size = args.batch_size * args.world_size if central_labelmix else args.batch_size
            # Some step-based configs do not define --epochs; keep K scheduling robust.
            default_train_epochs = getattr(args, 'epochs', None)
            if default_train_epochs is None:
                default_train_epochs = args.labelmix_total_epochs
            if default_train_epochs is None:
                default_train_epochs = 1
            labelmix_k_min = args.labelmix_k_min if args.labelmix_k_min is not None else args.labelmix_mix_k
            labelmix_k_max = args.labelmix_k_max if args.labelmix_k_max is not None else args.labelmix_mix_k
            explicit_k_total_epochs = (
                args.labelmix_k_total_epochs
                if args.labelmix_k_total_epochs is not None
                else args.labelmix_total_epochs
            )
            labelmix_k_total_epochs_auto = explicit_k_total_epochs is None
            labelmix_k_total_epochs = (
                explicit_k_total_epochs
                if explicit_k_total_epochs is not None
                else default_train_epochs
            )
            labelmix_kwargs = {
                'mix_k': args.labelmix_mix_k,
                'k_min': labelmix_k_min,
                'k_max': labelmix_k_max,
                'k_schedule': args.labelmix_k_schedule,
                'k_reverse': args.labelmix_k_reverse,
                'k_warmup_epochs': args.labelmix_k_warmup_epochs,
                'k_total_epochs': labelmix_k_total_epochs,
                'train_epochs': default_train_epochs,
                'alpha_min': args.labelmix_alpha_min,
                'alpha_max': args.labelmix_alpha_max,
                'schedule': args.labelmix_schedule,
                'reverse': args.labelmix_reverse,
                'step_mode': args.labelmix_step_mode,
                'warmup_steps': args.labelmix_warmup_steps,
                'total_epochs': args.labelmix_total_epochs,
                'total_steps': args.labelmix_total_steps,
                'batch_size': labelmix_batch_size,
                'sampling': args.labelmix_sampling,
                'sampling_min_side_px': args.labelmix_sampling_min_side_px,
                'sampling_max_aspect': args.labelmix_sampling_max_aspect,
                'sampling_bins': args.labelmix_sampling_bins,
                'sampling_pool_size': args.labelmix_sampling_pool_size,
                'sampling_low_watermark': args.labelmix_sampling_low_watermark,
                'sampling_max_attempts': args.labelmix_sampling_max_attempts,
            }

            dataset_train = BalancedBucketDataset(
                base_dataset=dataset_train,
                transform=train_transform,
                mode=args.balanced_mode,
                buffer_size=balanced_buffer,
                cache_path=balanced_cache_path,
                cache_small_classes_threshold=balanced_cache_threshold,
                input_key=balanced_input_key,
                target_key=balanced_target_key,
                labelmix=args.labelmix,
                labelmix_kwargs=labelmix_kwargs,
                dist_rank_override=0 if central_labelmix else None,
                dist_world_size_override=1 if central_labelmix else None,
            )
            labelmix_train_dataset = dataset_train if args.labelmix else None

            if collate_fn is None:
                collate_fn = fast_collate if args.prefetcher else torch.utils.data.dataloader.default_collate

            if central_labelmix:
                producer_workers = (
                    args.labelmix_producer_workers
                    if args.labelmix_producer_workers > 0
                    else args.workers
                )
                producer_loader = None
                if args.rank == args.labelmix_producer_rank:
                    producer_prefetch = (
                        str(loader_prefetch_factor)
                        if loader_prefetch_factor is not None
                        else "default"
                    )
                    _logger.info(
                        "LabelMix producer DataLoader prefetch_factor=%s (workers=%s)",
                        producer_prefetch,
                        producer_workers,
                    )
                    producer_loader_kwargs = dict(
                        dataset=dataset_train,
                        batch_size=args.batch_size * args.world_size,
                        num_workers=producer_workers,
                        collate_fn=collate_fn,
                        pin_memory=args.pin_mem,
                        drop_last=True,
                        worker_init_fn=partial(_worker_init, worker_seeding=args.worker_seeding),
                        persistent_workers=producer_workers > 0,
                    )
                    if producer_workers > 0 and loader_prefetch_factor is not None:
                        producer_loader_kwargs['prefetch_factor'] = loader_prefetch_factor
                    producer_loader = torch.utils.data.DataLoader(**producer_loader_kwargs)
                loader_train = LabelMixBroadcastLoader(
                    producer_loader,
                    batch_size=args.batch_size,
                    mix_k=args.labelmix_mix_k,
                    producer_rank=args.labelmix_producer_rank,
                    pin_memory=args.pin_mem,
                    group=labelmix_cpu_group,
                    log_fn=_logger.info if args.rank == args.labelmix_producer_rank else None,
                )
            else:
                loader_train_kwargs = dict(
                    dataset=dataset_train,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    collate_fn=collate_fn,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                    worker_init_fn=partial(_worker_init, worker_seeding=args.worker_seeding),
                    persistent_workers=args.workers > 0,
                )
                if args.workers > 0 and loader_prefetch_factor is not None:
                    loader_train_kwargs['prefetch_factor'] = loader_prefetch_factor
                loader_train = torch.utils.data.DataLoader(**loader_train_kwargs)

                if args.prefetcher:
                    prefetch_re_prob = args.reprob if not args.no_aug else 0.
                    loader_train = PrefetchLoader(
                        loader_train,
                        mean=data_config['mean'],
                        std=data_config['std'],
                        channels=data_config['input_size'][0],
                        device=device,
                        img_dtype=model_dtype or torch.float32,
                        re_prob=prefetch_re_prob,
                        re_mode=args.remode,
                        re_count=args.recount,
                        re_num_splits=re_num_splits,
                    )
        else:
            # wrap dataset in AugMix helper
            if num_aug_splits > 1:
                dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

            # Use standard loader
            loader_train = create_loader(
                dataset_train,
                input_size=data_config['input_size'],
                collate_fn=collate_fn,
                use_multi_epochs_loader=args.use_multi_epochs_loader,
                **train_common_loader_kwargs,
                **train_loader_kwargs,
            )

    loader_train = RepeatingLoader(loader_train)

    loader_eval = None
    if args.val_split:
        assert dataset_eval is not None
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)

        eval_loader_kwargs = dict(
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            num_workers=eval_workers,
            crop_pct=data_config['crop_pct'],
        )

        if args.naflex_loader:
            # Use largest sequence length for validation
            loader_eval = create_naflex_loader(
                dataset=dataset_eval,
                patch_size=model_patch_size,  # Use model's native patch size (already determined above)
                max_seq_len=args.naflex_max_seq_len,
                **common_loader_kwargs,
                **eval_loader_kwargs
            )
        else:
            # Use standard loader
            loader_eval = create_loader(
                dataset_eval,
                input_size=data_config['input_size'],
                **common_loader_kwargs,
                **eval_loader_kwargs,
            )

    # setup loss function
    if args.labelmix:
        if args.labelmix_loss == 'pl_loss':
            train_loss_fn = LabelMixPlackettLuceLoss()
        else:
            train_loss_fn = LabelMixSoftTargetCrossEntropy()
    elif args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # Setup training task (classification or distillation)
    if args.kd_model_name is not None:
        # Create distillation task (teacher created internally from model name)
        if args.kd_distill_type == 'logit':
            task = LogitDistillationTask(
                student_model=model,
                teacher_model=args.kd_model_name,
                criterion=train_loss_fn,
                loss_type=args.kd_loss_type,
                distill_loss_weight=args.distill_loss_weight,
                task_loss_weight=args.task_loss_weight,
                temperature=args.kd_temperature,
                device=device,
                dtype=model_dtype,
                verbose=utils.is_primary(args),
            )
        elif args.kd_distill_type == 'feature':
            task = FeatureDistillationTask(
                student_model=model,
                teacher_model=args.kd_model_name,
                criterion=train_loss_fn,
                distill_loss_weight=args.distill_loss_weight,
                task_loss_weight=args.task_loss_weight,
                student_feature_dim=args.kd_student_feature_dim,
                teacher_feature_dim=args.kd_teacher_feature_dim,
                device=device,
                dtype=model_dtype,
                verbose=utils.is_primary(args),
            )
        elif args.kd_distill_type == 'token':
            task = TokenDistillationTask(
                student_model=model,
                teacher_model=args.kd_model_name,
                criterion=train_loss_fn,
                distill_type=args.kd_token_distill_type,
                distill_loss_weight=args.distill_loss_weight,
                task_loss_weight=args.task_loss_weight,
                temperature=args.kd_temperature,
                device=device,
                dtype=model_dtype,
                verbose=utils.is_primary(args),
            )
        else:
            raise ValueError(f"Unknown distillation type: {args.kd_distill_type}")
    else:
        # Standard classification task
        task = ClassificationTask(
            model=model,
            criterion=train_loss_fn,
            device=device,
            dtype=model_dtype,
            verbose=utils.is_primary(args),
        )

    # Prepare task for distributed training
    if args.distributed:
        if utils.is_primary(args):
            _logger.info("Preparing task for distributed training")
        task.prepare_distributed(device_ids=[device])

    # Compile task if requested (should be done after DDP)
    if args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        if utils.is_primary(args):
            _logger.info(f"Compiling task with backend={args.torchcompile}, mode={args.torchcompile_mode}")
        task = torch.compile(task, backend=args.torchcompile, mode=args.torchcompile_mode)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_step = None
    saver = None
    output_dir = None
    exp_name = None
    summary_path = None
    summary_header_written = False
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        summary_path = os.path.join(output_dir, 'summary.csv')
        summary_header_written = os.path.exists(summary_path)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        _update_training_status(
            args,
            exp_name=exp_name,
            output_dir=output_dir,
            state='setup',
        )

        if args.log_wandb:
            if has_wandb:
                assert not args.wandb_resume_id or args.resume
                wandb.init(
                    project=args.wandb_project,
                    name=exp_name,
                    config=args,
                    tags=args.wandb_tags,
                    resume="must" if args.wandb_resume_id else None,
                    id=args.wandb_resume_id if args.wandb_resume_id else None,
                )
                wandb.config.update({'param_counts': param_counts}, allow_val_change=True)
                run = getattr(wandb, "run", None)
                if run is not None:
                    run.summary['param_counts'] = param_counts
                run_id = getattr(run, "id", None) if run is not None else None
                if run_id:
                    _update_training_status(
                        args,
                        exp_name=exp_name,
                        output_dir=output_dir,
                        state='setup',
                        wandb_id=str(run_id),
                    )
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting step
    steps_per_epoch = None
    try:
        steps_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    except TypeError:
        steps_per_epoch = None

    if args.num_steps is None or args.num_steps <= 0:
        if steps_per_epoch is None:
            raise ValueError('num_steps must be specified when dataloader length is unknown.')
        base_epochs = getattr(args, 'epochs', 300)
        args.num_steps = steps_per_epoch * base_epochs
        if utils.is_primary(args):
            _logger.info(
                f'num_steps not set, using {args.num_steps} '
                f'(steps_per_epoch={steps_per_epoch}, epochs={base_epochs}).')

    lr_scheduler, num_steps = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=1,
    )
    if num_steps != args.num_steps:
        args.num_steps = num_steps

    # Default K schedule horizon follows total run length so K progresses over all epochs.
    if args.labelmix and labelmix_k_total_epochs_auto and steps_per_epoch is not None and labelmix_train_dataset is not None:
        effective_total_epochs = max(1, int(math.ceil(args.num_steps / steps_per_epoch)))
        if getattr(labelmix_train_dataset, 'k_scheduler', None) is not None:
            labelmix_train_dataset.k_scheduler.total_epochs = effective_total_epochs
            labelmix_train_dataset.set_epoch(getattr(labelmix_train_dataset, '_epoch', 0))
            if utils.is_primary(args):
                _logger.info(
                    "LabelMix K schedule total_epochs auto-set to %d (num_steps=%d, steps_per_epoch=%d).",
                    effective_total_epochs,
                    args.num_steps,
                    steps_per_epoch,
                )

    log_interval = args.log_interval
    if args.num_logs is not None:
        if args.num_logs <= 0:
            raise ValueError('--num-logs must be > 0')
        log_interval = max(1, num_steps // args.num_logs)
        args.log_interval = log_interval

    val_interval = args.val_interval
    if args.num_evals is not None:
        if args.num_evals <= 0:
            raise ValueError('--num-evals must be > 0')
        val_interval = max(1, num_steps // args.num_evals)
        args.val_interval = val_interval

    save_interval = val_interval
    if args.num_saves is not None:
        if args.num_saves <= 0:
            raise ValueError('--num-saves must be > 0')
        save_interval = max(1, num_steps // args.num_saves)

    start_step = 0
    if args.start_step is not None:
        # a specified start_step will always override the resume step
        start_step = args.start_step
    elif resume_step is not None:
        start_step = resume_step
    if lr_scheduler is not None and start_step > 0:
        if lr_scheduler.t_in_epochs:
            lr_scheduler.step(start_step, metric=None)
        else:
            lr_scheduler.step_update(start_step)

    # Make dataset-side schedules (alpha / K) resume-aware by restoring epoch
    # from step count when possible.
    if start_step > 0 and steps_per_epoch is not None:
        resume_epoch = max(0, int(start_step // steps_per_epoch))
        if hasattr(loader_train, 'set_epoch'):
            loader_train.set_epoch(resume_epoch)
            if utils.is_primary(args):
                _logger.info(
                    "Resumed loader/dataset epoch=%d from start_step=%d (steps_per_epoch=%d).",
                    resume_epoch,
                    start_step,
                    steps_per_epoch,
                )

    if utils.is_primary(args) and output_dir is not None and exp_name is not None:
        _update_training_status(
            args,
            exp_name=exp_name,
            output_dir=output_dir,
            state='running',
            start_step=start_step,
            launch_increment=True,
        )

    if utils.is_primary(args):
        if args.warmup_prefix:
            sched_explain = '(warmup_steps + num_steps + cooldown_steps). Warmup added to total when warmup_prefix=True'
        else:
            sched_explain = '(num_steps + cooldown_steps). Warmup within num_steps when warmup_prefix=False'
        _logger.info(
            f'Scheduled steps: {num_steps} {sched_explain}. '
            f'LR stepped per {"step" if lr_scheduler.t_in_epochs else "update"}.')

    results = []
    train_state = TrainState()
    optimizer.zero_grad()
    global_step = start_step
    latest_metric = None
    interrupted = False
    run_error: Optional[Exception] = None
    run_error_tb = None
    try:
        while global_step < num_steps:
            step = global_step + 1

            train_metrics = train_step(
                step,
                model,
                loader_train,
                optimizer,
                args,
                task=task,
                device=device,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                naflex_mode=naflex_mode,
                train_state=train_state,
                num_steps=num_steps,
                output_dir=output_dir,
                saver=saver,
                log_console=False,
            )

            global_step = step

            if lr_scheduler is not None:
                if lr_scheduler.t_in_epochs:
                    lr_scheduler.step(global_step, metric=latest_metric)
                else:
                    lr_scheduler.step_update(num_updates=global_step, metric=train_state.losses_m.avg)

            summary_epoch = global_step
            if steps_per_epoch:
                summary_epoch = global_step // steps_per_epoch

            do_log = (global_step % log_interval == 0) or (global_step == num_steps) or (global_step == 1)
            do_eval = (global_step % val_interval == 0) or (global_step == num_steps) or (global_step == 1)
            do_log = do_log or do_eval
            eval_metrics = None
            if do_eval:
                if hasattr(optimizer, 'sync_lookahead'):
                    optimizer.sync_lookahead()

                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if utils.is_primary(args):
                        _logger.info("Distributing BatchNorm running means and vars")
                    utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

                if loader_eval is not None:
                    eval_metrics = validate(
                        model,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        model_dtype=model_dtype,
                    )

                    if model_ema is not None and not args.model_ema_force_cpu:
                        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                            utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                        ema_eval_metrics = validate(
                            model_ema,
                            loader_eval,
                            validate_loss_fn,
                            args,
                            device=device,
                            amp_autocast=amp_autocast,
                            log_suffix=' (EMA)',
                        )
                        eval_metrics = ema_eval_metrics

                if eval_metrics is not None:
                    latest_metric = eval_metrics[eval_metric]
                elif loader_eval is None:
                    latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                if (global_step % save_interval == 0) or (global_step == num_steps):
                    best_metric, best_step = saver.save_checkpoint(global_step, metric=latest_metric)

            if do_log:
                if output_dir is not None:
                    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                    utils.update_summary(
                        summary_epoch,
                        train_metrics,
                        eval_metrics if do_eval else None,
                        filename=summary_path,
                        lr=sum(lrs) / len(lrs),
                        write_header=not summary_header_written,
                        log_wandb=args.log_wandb and has_wandb,
                        step=global_step,
                    )
                    summary_header_written = True
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                loss_avg, loss_now = train_state.losses_m.avg, train_state.losses_m.val
                if args.distributed:
                    loss_avg_t = torch.tensor([loss_avg], device=device, dtype=torch.float32)
                    loss_now_t = torch.tensor([loss_now], device=device, dtype=torch.float32)
                    loss_avg = utils.reduce_tensor(loss_avg_t, args.world_size).item()
                    loss_now = utils.reduce_tensor(loss_now_t, args.world_size).item()

                if utils.is_primary(args):
                    pct = 0.0 if not num_steps else 100. * global_step / num_steps
                    _logger.info(
                        f'Train: {global_step} [{global_step:>6d}/{num_steps} ({pct:>3.0f}%)]  '
                        f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                        f'Time: {train_state.update_time_m.val:.3f}s, '
                        f'{train_state.update_sample_count / train_state.update_time_m.val:>7.2f}/s  '
                        f'({train_state.update_time_m.avg:.3f}s, '
                        f'{train_state.update_sample_count / train_state.update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {train_state.data_time_m.val:.3f} ({train_state.data_time_m.avg:.3f})'
                    )
                train_state.reset_interval()

            latest_results = {
                'step': global_step,
                'train': train_metrics,
            }
            if eval_metrics is not None:
                latest_results['validation'] = eval_metrics
            results.append(latest_results)

    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        run_error = exc
        run_error_tb = exc.__traceback__

    if utils.is_primary(args) and output_dir is not None and exp_name is not None:
        run = getattr(wandb, "run", None) if (args.log_wandb and has_wandb) else None
        run_id = getattr(run, "id", None) if run is not None else None
        if run_error is not None:
            _update_training_status(
                args,
                exp_name=exp_name,
                output_dir=output_dir,
                state='failed',
                wandb_id=str(run_id) if run_id else None,
                last_step=global_step,
                last_exit_code=1,
            )
        elif interrupted:
            _update_training_status(
                args,
                exp_name=exp_name,
                output_dir=output_dir,
                state='interrupted',
                wandb_id=str(run_id) if run_id else None,
                last_step=global_step,
                last_exit_code=130,
            )
        else:
            _update_training_status(
                args,
                exp_name=exp_name,
                output_dir=output_dir,
                state='finished',
                wandb_id=str(run_id) if run_id else None,
                last_step=global_step,
                last_exit_code=0,
            )

    if args.distributed:
        torch.distributed.destroy_process_group()

    if run_error is not None:
        raise run_error.with_traceback(run_error_tb)

    if best_metric is not None:
        # log best metric as tracked by checkpoint saver
        _logger.info('*** Best metric: {0} (step {1})'.format(best_metric, best_step))

    if utils.is_primary(args):
        # for parsable results display, dump top-10 summaries to avoid excess console spam
        display_results = sorted(
            results,
            key=lambda x: x.get('validation', x.get('train')).get(eval_metric, 0),
            reverse=decreasing_metric,
        )
        print(f'--result\n{json.dumps(display_results[-10:], indent=4)}')


class TrainState:
    def __init__(self):
        self.update_time_m = utils.AverageMeter()
        self.data_time_m = utils.AverageMeter()
        self.losses_m = utils.AverageMeter()
        self.ce_loss_m = utils.AverageMeter()
        self.grad_norm_m = utils.AverageMeter()
        self.update_sample_count = 0
        now = time.time()
        self.data_start_time = now
        self.update_start_time = now

    def reset_interval(self):
        self.update_time_m.reset()
        self.data_time_m.reset()
        self.losses_m.reset()
        self.ce_loss_m.reset()
        self.grad_norm_m.reset()
        self.update_sample_count = 0
        now = time.time()
        self.data_start_time = now
        self.update_start_time = now


def train_step(
        step,
        model,
        loader,
        optimizer,
        args,
        task=None,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        model_ema=None,
        mixup_fn=None,
        naflex_mode=False,
        train_state=None,
        num_steps=None,
        output_dir=None,
        saver=None,
        log_console=False,
):
    if train_state is None:
        train_state = TrainState()

    if args.mixup_off_step and step >= args.mixup_off_step:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")

    model.train()

    accum_steps = args.grad_accum_steps
    for accum_idx in range(accum_steps):
        need_update = accum_idx == (accum_steps - 1)

        input, target = next(loader)

        if not args.prefetcher:
            input = input.to(device=device, dtype=model_dtype)
            if args.labelmix and isinstance(target, (tuple, list)) and len(target) == 2:
                target = (
                    target[0].to(device=device),
                    target[1].to(device=device),
                )
            else:
                target = target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        train_state.data_time_m.update(accum_steps * (time.time() - train_state.data_start_time))

        def _forward():
            with amp_autocast():
                # Task handles the complete forward pass and loss computation
                result = task(input, target)
                _loss = result['loss']

            if accum_steps > 1:
                _loss /= accum_steps
            return _loss, result

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if naflex_mode:
            assert isinstance(input, dict)
            batch_size = input['patches'].shape[0]

            # scale gradient vs the minimum batch size (for max seq len)
            if not args.naflex_loss_scale or args.naflex_loss_scale == 'none':
                local_scale = 1.0
            else:
                local_scale = (batch_size / args.batch_size)
                if local_scale == 'sqrt':
                    local_scale = local_scale ** 0.5

            if args.distributed:
                # scale gradient btw distributed ranks, each one can have different batch size
                global_batch_size = utils.reduce_tensor(
                    torch.tensor(batch_size, device=device, dtype=torch.float32),
                    1 # SUM
                )
                dist_scale = args.world_size * batch_size / global_batch_size
            else:
                dist_scale = None
                global_batch_size = batch_size

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, result = _forward()
                    scaled_loss = local_scale * loss
                    if dist_scale is not None:
                        scaled_loss *= dist_scale
                    _backward(scaled_loss)
            else:
                loss, result = _forward()
                scaled_loss = local_scale * loss
                if dist_scale is not None:
                    scaled_loss *= dist_scale
                _backward(scaled_loss)
        else:
            global_batch_size = batch_size = input.shape[0]
            if args.distributed:
                global_batch_size *= args.world_size

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, result = _forward()
                    _backward(loss)
            else:
                loss, result = _forward()
                _backward(loss)

        train_state.losses_m.update(loss.item() * accum_steps, batch_size)
        train_state.update_sample_count += global_batch_size

        with torch.no_grad():
            if args.labelmix and isinstance(target, (tuple, list)) and len(target) == 2:
                labels, weights = target
                if labels.ndim == 1:
                    labels = labels.unsqueeze(0)
                if weights.ndim == 1:
                    weights = weights.unsqueeze(0)
                # Use highest-weight label for metrics
                max_idx = weights.argmax(dim=1)
                clean_target = labels.gather(1, max_idx.unsqueeze(1)).squeeze(1)
            elif target.ndim > 1:
                clean_target = target.argmax(dim=1)
            else:
                clean_target = target
            output = result['output']
            ce_loss = nn.functional.cross_entropy(output, clean_target)
            train_state.ce_loss_m.update(ce_loss.item(), batch_size)

        if need_update:
            with torch.no_grad():
                parameters = list(model_parameters(model, exclude_head='agc' in args.clip_mode))
                grads = [p.grad for p in parameters if p.grad is not None]
                if grads:
                    grad_norm = torch.norm(
                        torch.stack([torch.norm(g.detach(), 2) for g in grads]),
                        2
                    )
                    train_state.grad_norm_m.update(float(grad_norm))
                else:
                    train_state.grad_norm_m.update(0.0)

        if not need_update:
            train_state.data_start_time = time.time()
            continue

        if model_ema is not None:
            model_ema.update(model, step=step)

        if args.synchronize_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()
        time_now = time.time()

        train_state.update_time_m.update(time.time() - train_state.update_start_time)
        train_state.update_start_time = time_now

        if (step % args.log_interval == 0 or (num_steps is not None and step == num_steps)):
            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, f'train-step-{step}.jpg'),
                    padding=0,
                    normalize=True
                )

        if log_console and (step % args.log_interval == 0 or (num_steps is not None and step == num_steps)):
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = train_state.losses_m.avg, train_state.losses_m.val
            if args.distributed:
                # synchronize current step and avg loss, each process keeps its own running avg
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()

            if utils.is_primary(args):
                pct = 0.0 if not num_steps else 100. * step / num_steps
                _logger.info(
                    f'Train: {step} [{step:>6d}/{num_steps} ({pct:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Time: {train_state.update_time_m.val:.3f}s, '
                    f'{train_state.update_sample_count / train_state.update_time_m.val:>7.2f}/s  '
                    f'({train_state.update_time_m.avg:.3f}s, '
                    f'{train_state.update_sample_count / train_state.update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {train_state.data_time_m.val:.3f} ({train_state.data_time_m.avg:.3f})'
                )

            train_state.update_sample_count = 0

        if saver is not None and args.recovery_interval and (step % args.recovery_interval == 0):
            saver.save_recovery(step, batch_idx=0)

        train_state.data_start_time = time.time()
        optimizer.zero_grad()

    loss_avg = train_state.losses_m.avg
    ce_loss_avg = train_state.ce_loss_m.avg
    grad_norm_avg = train_state.grad_norm_m.avg
    if args.distributed:
        # synchronize avg loss, each process keeps its own running avg
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()
        ce_loss_avg = torch.tensor([ce_loss_avg], device=device, dtype=torch.float32)
        ce_loss_avg = utils.reduce_tensor(ce_loss_avg, args.world_size).item()
        grad_norm_avg = torch.tensor([grad_norm_avg], device=device, dtype=torch.float32)
        grad_norm_avg = utils.reduce_tensor(grad_norm_avg, args.world_size).item()
    return OrderedDict([
        ('loss', loss_avg),
        ('cross_entropy', ce_loss_avg),
        ('grad_norm', grad_norm_avg),
    ])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        model_dtype=None,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    ece_m = utils.ECEMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            ece_m.update(output, target)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == "npu":
                torch.npu.synchronize()

            batch_size = output.shape[0]
            losses_m.update(reduced_loss.item(), batch_size)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([
        ('loss', losses_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg),
        ('ece', ece_m.compute()),
    ])

    return metrics


def main():
    """CLI entry point — preserves the original ``python train.py`` and
    ``torchrun ... train.py`` workflows exactly.
    """
    run_training()


if __name__ == '__main__':
    main()
