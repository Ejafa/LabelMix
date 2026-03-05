#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from typing import Dict, List

GROUP_RESOURCE_PREFIX = "GPU_GROUP"


def wait_ray(args: argparse.Namespace) -> int:
    try:
        import ray
    except Exception as exc:
        raise RuntimeError("Failed to import ray while waiting for Ray readiness.") from exc

    deadline = time.time() + max(1, int(args.timeout_seconds))
    last_err = None
    while time.time() < deadline:
        try:
            ray.init(address=args.address, ignore_reinit_error=True)
            print("Ray connected:", ray.cluster_resources())
            ray.shutdown()
            return 0
        except Exception as exc:
            last_err = exc
            time.sleep(2)

    raise RuntimeError(f"Ray not ready at {args.address}: {last_err}")


def wait_tcp(args: argparse.Namespace) -> int:
    deadline = time.time() + max(1, int(args.timeout_seconds))
    last_err = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect((args.host, int(args.port)))
            return 0
        except Exception as exc:
            last_err = exc
            time.sleep(2)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    raise RuntimeError(f"TCP endpoint not reachable: {args.host}:{args.port} ({last_err})")


def _query_gpu_memory_mb(mode: str) -> List[float]:
    if mode not in {"free", "total"}:
        raise ValueError("--mode must be one of: free, total")

    cmd = [
        "nvidia-smi",
        f"--query-gpu=memory.{mode}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        raise RuntimeError("Failed to query GPU memory with nvidia-smi.") from exc

    values_mb: List[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values_mb.append(float(line))
        except ValueError:
            continue

    if not values_mb:
        raise RuntimeError("No GPU memory values detected from nvidia-smi output.")
    return values_mb


def _query_gpu_count() -> int:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        raise RuntimeError("Failed to detect GPU count with nvidia-smi.") from exc

    count = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        count += 1

    if count <= 0:
        raise RuntimeError("No GPUs detected from nvidia-smi output.")
    return count


def detect_gpu_count(args: argparse.Namespace) -> int:
    del args
    print(_query_gpu_count())
    return 0


def detect_vram_gb(args: argparse.Namespace) -> int:
    mode = str(args.mode).strip().lower()
    values_mb = _query_gpu_memory_mb(mode)

    if args.aggregate == "sum":
        value_gb = sum(values_mb) / 1024.0
    elif args.aggregate == "min":
        value_gb = min(values_mb) / 1024.0
    else:
        raise ValueError("--aggregate must be one of: sum, min")

    value_gb = max(0.0, value_gb - float(args.reserve_gb))
    print(f"{value_gb:.3f}")
    return 0


def build_group_resources(args: argparse.Namespace) -> int:
    gpus_per_group = int(args.gpus_per_group)
    max_experiments_per_group = int(args.max_experiments_per_group)
    mode = str(args.mode).strip().lower()
    reserve_gb = float(args.reserve_gb)

    if gpus_per_group <= 0:
        raise ValueError("--gpus-per-group must be >= 1")
    if max_experiments_per_group <= 0:
        raise ValueError("--max-experiments-per-group must be >= 1")
    values_mb: List[float] = []
    try:
        values_mb = _query_gpu_memory_mb(mode)
    except Exception as exc:
        if args.require_vram:
            raise
        print(
            f"Warning: unable to query VRAM for group resources ({exc}); scheduling without VRAM caps.",
            file=sys.stderr,
        )
        gpus_per_node = _query_gpu_count()
    else:
        gpus_per_node = len(values_mb)

    if gpus_per_node <= 0:
        raise RuntimeError("Auto-detected GPU count is zero.")
    if gpus_per_node % gpus_per_group != 0:
        raise ValueError(
            f"Auto-detected GPU count ({gpus_per_node}) must be divisible by --gpus-per-group ({gpus_per_group})"
        )

    groups_per_node = gpus_per_node // gpus_per_group
    resources: Dict[str, int] = {}
    if values_mb:
        for idx in range(groups_per_node):
            start = idx * gpus_per_group
            end = start + gpus_per_group
            group_values = values_mb[start:end]
            if args.aggregate == "sum":
                value_gb = sum(group_values) / 1024.0
            elif args.aggregate == "min":
                value_gb = min(group_values) / 1024.0
            else:
                raise ValueError("--aggregate must be one of: sum, min")
            value_gb = max(0.0, value_gb - reserve_gb)
            value_units = int(value_gb)
            slot_units = max_experiments_per_group
            if value_units <= 0:
                slot_units = 0
                print(
                    f"Warning: group {idx} has insufficient VRAM budget ({value_gb:.3f}GB) and is disabled.",
                    file=sys.stderr,
                )
            elif value_units < max_experiments_per_group:
                slot_units = value_units
                print(
                    f"Warning: group {idx} VRAM budget ({value_gb:.3f}GB -> {value_units} units) limits "
                    f"slots from {max_experiments_per_group} to {slot_units}.",
                    file=sys.stderr,
                )

            if slot_units > 0:
                resources[f"{GROUP_RESOURCE_PREFIX}_{idx}_SLOTS"] = int(slot_units)
            if value_units > 0:
                resources[f"{GROUP_RESOURCE_PREFIX}_{idx}_VRAM_GB"] = value_units
    else:
        for idx in range(groups_per_node):
            resources[f"{GROUP_RESOURCE_PREFIX}_{idx}_SLOTS"] = max_experiments_per_group

    print(json.dumps(resources, sort_keys=True, separators=(",", ":")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helpers for deployment/start.sh")
    sub = parser.add_subparsers(dest="command", required=True)

    p_wait_ray = sub.add_parser("wait-ray", help="Wait until Ray cluster is connectable.")
    p_wait_ray.add_argument("--address", required=True)
    p_wait_ray.add_argument("--timeout-seconds", type=int, default=120)
    p_wait_ray.set_defaults(func=wait_ray)

    p_wait_tcp = sub.add_parser("wait-tcp", help="Wait until TCP endpoint is reachable.")
    p_wait_tcp.add_argument("--host", required=True)
    p_wait_tcp.add_argument("--port", type=int, required=True)
    p_wait_tcp.add_argument("--timeout-seconds", type=int, default=180)
    p_wait_tcp.set_defaults(func=wait_tcp)

    p_detect_gpu_count = sub.add_parser("detect-gpu-count", help="Detect number of GPUs on this node.")
    p_detect_gpu_count.set_defaults(func=detect_gpu_count)

    p_detect_vram = sub.add_parser("detect-vram-gb", help="Detect node VRAM budget in GB.")
    p_detect_vram.add_argument("--mode", default="free", choices=["free", "total"])
    p_detect_vram.add_argument("--aggregate", default="sum", choices=["sum", "min"])
    p_detect_vram.add_argument("--reserve-gb", type=float, default=0.0)
    p_detect_vram.set_defaults(func=detect_vram_gb)

    p_group_resources = sub.add_parser(
        "build-group-resources",
        help=f"Build Ray custom resources for fixed-size GPU groups ({GROUP_RESOURCE_PREFIX}_*).",
    )
    p_group_resources.add_argument("--gpus-per-group", type=int, required=True)
    p_group_resources.add_argument("--max-experiments-per-group", type=int, default=1)
    p_group_resources.add_argument("--mode", default="free", choices=["free", "total"])
    p_group_resources.add_argument("--aggregate", default="min", choices=["sum", "min"])
    p_group_resources.add_argument("--reserve-gb", type=float, default=0.0)
    p_group_resources.add_argument(
        "--require-vram",
        action="store_true",
        help="Fail if GPU memory cannot be queried.",
    )
    p_group_resources.set_defaults(func=build_group_resources)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
