#!/usr/bin/env python3
"""Benchmark the ImageNet balanced-loader cost of the paper configurations.

The benchmark deliberately excludes the model and loss computation.  It times
the production of a training-ready CPU batch, including post-collation
Mixup/CutMix when that configuration is selected.

Example::

    python experiments/benchmark_augmentation_loader.py \
        --data-dir /dev/shm/imagenet-1k \
        --config experiments/labelmix_imagenet1k/configs/vit-little.yaml

The default batch size is 128, matching the per-GPU batch size of the 1024
global-batch ImageNet recipe on eight GPUs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import resource
import statistics
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from timm.data import (  # noqa: E402
    BalancedBucketDataset,
    Mixup,
    create_dataset,
    create_transform,
    resolve_data_config,
)
from train import build_args  # noqa: E402

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal environments
    psutil = None


CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "no_augmentation": {
        "description": "No train-time augmentation",
        "no_aug": True,
        "mixup": False,
        "labelmix": False,
        "loss": "cross_entropy",
    },
    "single_image": {
        "description": "Single-image recipe only",
        "no_aug": False,
        "mixup": False,
        "labelmix": False,
        "loss": "label_smoothing_cross_entropy",
    },
    "mixup_cutmix": {
        "description": "Baseline Mixup+CutMix",
        "no_aug": False,
        "mixup": True,
        "labelmix": False,
        "loss": "soft_target_cross_entropy",
    },
    "treemapmix_sce": {
        "description": "TreemapMix-SCE (K=4)",
        "no_aug": False,
        "mixup": False,
        "labelmix": True,
        "mix_k": 4,
        "loss": "soft_ce",
    },
    "treemapmix_pl": {
        "description": "TreemapMix-PL (K=6)",
        "no_aug": False,
        "mixup": False,
        "labelmix": True,
        "mix_k": 6,
        "loss": "pl_loss",
    },
}


@dataclass
class BenchmarkResult:
    configuration: str
    description: str
    loss: str
    mix_k: Optional[int]
    balanced_mode: int
    batch_size: int
    workers: int
    warmup_batches: int
    measured_batches: int
    first_batch_seconds: float
    throughput_images_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    peak_rss_mb: float
    peak_uss_mb: Optional[float]


def percentile(values: Iterable[float], q: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _process_tree_memory_bytes() -> tuple[int, Optional[int]]:
    """Return summed RSS/USS for this process and its DataLoader workers."""
    if psutil is None:
        # Linux reports KiB; macOS reports bytes. This project runs on Linux.
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return rss, None

    root = psutil.Process(os.getpid())
    processes = [root, *root.children(recursive=True)]
    rss = 0
    uss = 0
    has_uss = True
    for process in processes:
        try:
            rss += int(process.memory_info().rss)
            try:
                uss += int(process.memory_full_info().uss)
            except (AttributeError, psutil.AccessDenied):
                has_uss = False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, uss if has_uss else None


class MemorySampler:
    """Sample process-tree memory while workers are active."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss = 0
        self.peak_uss: Optional[int] = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        rss, uss = _process_tree_memory_bytes()
        self.peak_rss = max(self.peak_rss, rss)
        if uss is None:
            self.peak_uss = None
        elif self.peak_uss is not None:
            self.peak_uss = max(self.peak_uss, uss)

    def start(self) -> None:
        self._sample()

        def run() -> None:
            while not self._stop.wait(self.interval_seconds):
                self._sample()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _build_transform(args, data_config: Dict[str, Any], no_aug: bool):
    return create_transform(
        input_size=data_config["input_size"],
        is_training=True,
        no_aug=no_aug,
        train_crop_mode=args.train_crop_mode,
        scale=tuple(args.scale),
        ratio=tuple(args.ratio),
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        interpolation=args.train_interpolation or data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=0,
        use_prefetcher=False,
        separate=False,
    )


def _labelmix_kwargs(args, mix_k: int) -> Dict[str, Any]:
    return {
        "mix_k": mix_k,
        "k_min": mix_k,
        "k_max": mix_k,
        "k_schedule": "fixed",
        "k_reverse": False,
        "k_warmup_epochs": 0,
        "k_total_epochs": 110,
        "labelmix_k_cooldown_epochs": None,
        "train_epochs": 110,
        "alpha_min": 0.1,
        "alpha_max": 0.5,
        "schedule": "cosine",
        "reverse": False,
        "step_mode": "total",
        "warmup_steps": 0,
        "total_epochs": 110,
        "total_steps": None,
        "batch_size": args.batch_size,
        # Match experiments.generate_jobs defaults.
        "sampling": True,
        "sampling_min_side_px": 0,
        "sampling_max_aspect": 20.0,
        "sampling_bins": 16,
        "sampling_pool_size": 128,
        "sampling_low_watermark": 32,
        "sampling_max_attempts": 200,
    }


def build_loader(args, configuration: str):
    spec = CONFIGURATIONS[configuration]
    data_config = resolve_data_config(vars(args), model=None)
    base_dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=False,
        input_img_mode=args.input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        trust_remote_code=False,
    )
    transform = _build_transform(args, data_config, no_aug=bool(spec["no_aug"]))
    mix_k = int(spec.get("mix_k", 1))
    dataset = BalancedBucketDataset(
        base_dataset=base_dataset,
        transform=transform,
        mode=args.balanced_mode,
        buffer_size=args.balanced_buffer_steps * args.batch_size,
        cache_path=args.balanced_cache_path,
        cache_small_classes_threshold=args.balanced_cache_threshold_steps * args.batch_size,
        input_key=args.input_key,
        target_key=args.target_key,
        labelmix=bool(spec["labelmix"]),
        labelmix_kwargs=_labelmix_kwargs(args, mix_k) if spec["labelmix"] else None,
        seed=args.seed,
    )

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": torch.utils.data.dataloader.default_collate,
        "drop_last": True,
        "pin_memory": bool(args.pin_mem),
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    mixup_fn = None
    if spec["mixup"]:
        if args.batch_size % 2:
            raise ValueError("Mixup+CutMix requires an even --batch-size")
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
    return dataset, loader, mixup_fn


def benchmark_configuration(args, configuration: str) -> BenchmarkResult:
    spec = CONFIGURATIONS[configuration]
    dataset, loader, mixup_fn = build_loader(args, configuration)
    sampler = MemorySampler(args.memory_sample_interval)
    sampler.start()
    iterator = None

    def consume_one():
        nonlocal iterator
        batch, iterator = _next_batch(iterator, loader)
        images, targets = batch
        if mixup_fn is not None:
            images, targets = mixup_fn(images.float(), targets)
        return int(images.shape[0])

    try:
        iterator = iter(loader)
        start = time.perf_counter()
        consume_one()
        first_batch_seconds = time.perf_counter() - start

        for _ in range(args.warmup_batches):
            consume_one()

        latencies = []
        total_images = 0
        measured_start = time.perf_counter()
        for _ in range(args.measured_batches):
            batch_start = time.perf_counter()
            total_images += consume_one()
            latencies.append(time.perf_counter() - batch_start)
        measured_seconds = time.perf_counter() - measured_start
    finally:
        sampler.stop()
        del iterator, loader, dataset
        gc.collect()

    mib = 1024.0 * 1024.0
    return BenchmarkResult(
        configuration=configuration,
        description=str(spec["description"]),
        loss=str(spec["loss"]),
        mix_k=int(spec["mix_k"]) if "mix_k" in spec else None,
        balanced_mode=args.balanced_mode,
        batch_size=args.batch_size,
        workers=args.workers,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
        first_batch_seconds=first_batch_seconds,
        throughput_images_per_second=total_images / measured_seconds,
        latency_p50_ms=1000.0 * statistics.median(latencies),
        latency_p95_ms=1000.0 * percentile(latencies, 0.95),
        peak_rss_mb=sampler.peak_rss / mib,
        peak_uss_mb=None if sampler.peak_uss is None else sampler.peak_uss / mib,
    )


def _isolated_benchmark_worker(args, configuration: str, result_queue) -> None:
    """Run one configuration in a clean process for comparable peak memory."""
    try:
        result_queue.put((True, benchmark_configuration(args, configuration)))
    except BaseException:
        result_queue.put((False, traceback.format_exc()))


def benchmark_configuration_isolated(args, configuration: str) -> BenchmarkResult:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_isolated_benchmark_worker,
        args=(args, configuration, result_queue),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"Benchmark {configuration!r} exited with status {process.exitcode}"
        )
    ok, payload = result_queue.get(timeout=5)
    if not ok:
        raise RuntimeError(f"Benchmark {configuration!r} failed:\n{payload}")
    return payload


def _write_results(results: list[BenchmarkResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_results(results: list[BenchmarkResult]) -> None:
    header = f"{'configuration':<20} {'img/s':>10} {'p50 ms':>10} {'p95 ms':>10} {'RSS MiB':>10}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.configuration:<20} "
            f"{result.throughput_images_per_second:>10.1f} "
            f"{result.latency_p50_ms:>10.1f} "
            f"{result.latency_p95_ms:>10.1f} "
            f"{result.peak_rss_mb:>10.1f}"
        )


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Benchmark five augmentation configurations with the balanced ImageNet loader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="experiments/labelmix_imagenet1k/configs/vit-little.yaml",
        help="Training YAML supplying the single-image transform recipe.",
    )
    parser.add_argument("--data-dir", default="/dev/shm/imagenet-1k")
    parser.add_argument("--dataset", default="hfds/ILSVRC/imagenet-1k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--input-key", default="image")
    parser.add_argument("--target-key", default="label")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--balanced-mode", type=int, default=1280)
    parser.add_argument("--balanced-buffer-steps", type=int, default=4)
    parser.add_argument("--balanced-cache-threshold-steps", type=int, default=3)
    parser.add_argument("--balanced-cache-path", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measured-batches", type=int, default=100)
    parser.add_argument("--memory-sample-interval", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configuration",
        action="append",
        choices=tuple(CONFIGURATIONS),
        help="Configuration to run; repeat as needed. All five run by default.",
    )
    parser.add_argument(
        "--output",
        default="results/augmentation_loader_benchmark.json",
        help="JSON output path; a CSV is written alongside it.",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run configurations in this process (useful for debugging, less reliable for memory comparison).",
    )
    cli = parser.parse_args(argv)

    if cli.batch_size < 1 or cli.workers < 0:
        parser.error("--batch-size must be positive and --workers must be non-negative")
    if cli.warmup_batches < 0 or cli.measured_batches < 1:
        parser.error("--warmup-batches must be non-negative and --measured-batches positive")
    if cli.balanced_buffer_steps < 1 or cli.balanced_cache_threshold_steps < 0:
        parser.error("balanced buffer steps must be positive and cache threshold steps non-negative")

    training_args, _ = build_args(
        config_path=cli.config,
        dict_overrides={
            "data_dir": cli.data_dir,
            "dataset": cli.dataset,
            "train_split": cli.train_split,
            "input_key": cli.input_key,
            "target_key": cli.target_key,
            "num_classes": cli.num_classes,
            "batch_size": cli.batch_size,
            "workers": cli.workers,
            "seed": cli.seed,
        },
    )
    # Benchmark-only arguments intentionally remain separate from train.py.
    for key, value in vars(cli).items():
        setattr(training_args, key.replace("-", "_"), value)
    if not training_args.balanced_cache_path:
        training_args.balanced_cache_path = str(Path(training_args.data_dir) / "class_buckets.pkl")
    return training_args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    selected = args.configuration or list(CONFIGURATIONS)
    results = []
    for configuration in selected:
        print(f"Benchmarking {configuration} ...", flush=True)
        benchmark_fn = benchmark_configuration if args.no_isolate else benchmark_configuration_isolated
        results.append(benchmark_fn(args, configuration))
    _write_results(results, Path(args.output))
    _print_results(results)
    print(f"Wrote {args.output} and {Path(args.output).with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
