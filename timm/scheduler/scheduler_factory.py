""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import List, Optional, Union

from torch.optim import Optimizer

from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler


def scheduler_kwargs(cfg, decreasing_metric: Optional[bool] = None):
    """ cfg/argparse to kwargs helper
    Refactored to prefer step-based arguments if available.
    """
    eval_metric = getattr(cfg, 'eval_metric', 'top1')
    if decreasing_metric is not None:
        plateau_mode = 'min' if decreasing_metric else 'max'
    else:
        plateau_mode = 'min' if 'loss' in eval_metric else 'max'

    # Support both step-based and epoch-based arguments
    num_steps = getattr(cfg, 'num_steps', getattr(cfg, 'epochs', 300))
    warmup_steps = getattr(cfg, 'warmup_steps', getattr(cfg, 'warmup_epochs', 5))
    decay_steps = getattr(cfg, 'decay_steps', getattr(cfg, 'decay_epochs', 90))
    cooldown_steps = getattr(cfg, 'cooldown_steps', getattr(cfg, 'cooldown_epochs', 0))
    patience_steps = getattr(cfg, 'patience_steps', getattr(cfg, 'patience_epochs', 10))
    decay_milestones = getattr(cfg, 'decay_milestones', [90, 180, 270])

    kwargs = dict(
        sched=cfg.sched,
        num_steps=num_steps,
        decay_steps=decay_steps,
        decay_milestones=decay_milestones,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        patience_steps=patience_steps,
        decay_rate=getattr(cfg, 'decay_rate', 0.1),
        min_lr=getattr(cfg, 'min_lr', 0.),
        warmup_lr=getattr(cfg, 'warmup_lr', 1e-5),
        warmup_prefix=getattr(cfg, 'warmup_prefix', False),
        noise=getattr(cfg, 'lr_noise', None),
        noise_pct=getattr(cfg, 'lr_noise_pct', 0.67),
        noise_std=getattr(cfg, 'lr_noise_std', 1.),
        noise_seed=getattr(cfg, 'seed', 42),
        cycle_mul=getattr(cfg, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(cfg, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(cfg, 'lr_cycle_limit', 1),
        k_decay=getattr(cfg, 'lr_k_decay', 1.0),
        plateau_mode=plateau_mode,
        step_on_epochs=not getattr(cfg, 'sched_on_updates', False),
    )
    return kwargs


def create_scheduler(
        args,
        optimizer: Optimizer,
        updates_per_epoch: int = 0,
):
    return create_scheduler_v2(
        optimizer=optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )


def create_scheduler_v2(
        optimizer: Optimizer,
        sched: str = 'cosine',
        num_steps: int = 300,
        decay_steps: int = 90,
        decay_milestones: List[int] = (90, 180, 270),
        cooldown_steps: int = 0,
        patience_steps: int = 10,
        decay_rate: float = 0.1,
        min_lr: float = 0.,
        warmup_lr: float = 1e-5,
        warmup_steps: int = 0,
        warmup_prefix: bool = False,
        noise: Union[float, List[float]] = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.,
        noise_seed: int = 42,
        cycle_mul: float = 1.,
        cycle_decay: float = 0.1,
        cycle_limit: int = 1,
        k_decay: float = 1.0,
        plateau_mode: str = 'max',
        step_on_epochs: bool = True,
        updates_per_epoch: int = 0,
):
    # Initialize "time" variables (can be epochs or steps)
    t_initial = num_steps
    warmup_t = warmup_steps
    decay_t = decay_steps
    cooldown_t = cooldown_steps

    if not step_on_epochs:
        # If explicitly step-based, we assume inputs are already steps.
        # Only scale if legacy behavior (updates_per_epoch > 1) is detected.
        if updates_per_epoch > 1:
            t_initial = t_initial * updates_per_epoch
            warmup_t = warmup_t * updates_per_epoch
            decay_t = decay_t * updates_per_epoch
            decay_milestones = [d * updates_per_epoch for d in decay_milestones]
            cooldown_t = cooldown_t * updates_per_epoch

    # warmup args
    warmup_args = dict(
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_t,
        warmup_prefix=warmup_prefix,
    )

    # setup noise args for supporting schedulers
    if noise is not None:
        if isinstance(noise, (list, tuple)):
            noise_range = [n * t_initial for n in noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = noise * t_initial
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=noise_pct,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    # setup cycle args for supporting schedulers
    cycle_args = dict(
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
    )

    lr_scheduler = None
    if sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_rate,
            t_in_epochs=step_on_epochs,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_milestones,
            decay_rate=decay_rate,
            t_in_epochs=step_on_epochs,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'plateau':
        warmup_args.pop('warmup_prefix', False)
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=decay_rate,
            patience_t=patience_steps,
            cooldown_t=0,
            **warmup_args,
            lr_min=min_lr,
            mode=plateau_mode,
            **noise_args,
        )
    elif sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            k_decay=k_decay,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )

    if hasattr(lr_scheduler, 'get_cycle_length'):
        # For cycle based schedulers (cosine, tanh, poly) recalculate total steps w/ cycles & cooldown
        # NOTE: Warmup prefix added in get_cycle_lengths() if enabled
        t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
        if step_on_epochs:
            num_steps = t_with_cycles_and_cooldown
        else:
            if updates_per_epoch > 1:
                num_steps = t_with_cycles_and_cooldown // updates_per_epoch
            else:
                num_steps = t_with_cycles_and_cooldown
    else:
        if warmup_prefix:
            num_steps += warmup_steps

    return lr_scheduler, num_steps
