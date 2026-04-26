#!/usr/bin/env python3
"""LazyConfig-based training entry point for LabelMix ViT backbones on COCO.

This is a minimal port of ``detectron2/tools/lazyconfig_train_net.py`` that
also adds convenient overrides for ``train.init_checkpoint`` (pointing to a
converted LabelMix ViT checkpoint) and ``train.output_dir``.

Example (single-node, 8 GPUs)::

    python train_net.py \\
        --config-file configs/COCO/mask_rcnn_vitdet_wee_100ep.py \\
        --num-gpus 8 \\
        train.init_checkpoint=./converted/vit_wee_in1k.pth \\
        train.output_dir=./output/vit_wee_vitdet

Example (evaluation only)::

    python train_net.py \\
        --config-file configs/COCO/mask_rcnn_vitdet_wee_100ep.py \\
        --num-gpus 8 --eval-only \\
        train.init_checkpoint=./output/vit_wee_vitdet/model_final.pth
"""

import logging
import os
import sys

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    start_iter = trainer.iter + 1 if args.resume and checkpointer.has_checkpoint() else 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    # Make the configs/ package importable (it lives next to this script).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = default_argument_parser()
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
