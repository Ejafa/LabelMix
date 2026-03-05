#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from typing import Dict

GROUP_RESOURCE_PREFIX = "GPU_GROUP"


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


def build_group_resources(args: argparse.Namespace) -> int:
    gpus_per_group = int(args.gpus_per_group)
    max_experiments_per_group = int(args.max_experiments_per_group)

    if gpus_per_group <= 0:
        raise ValueError("--gpus-per-group must be >= 1")
    if max_experiments_per_group <= 0:
        raise ValueError("--max-experiments-per-group must be >= 1")
    gpus_per_node = _query_gpu_count()

    if gpus_per_node <= 0:
        raise RuntimeError("Auto-detected GPU count is zero.")
    if gpus_per_node % gpus_per_group != 0:
        raise ValueError(
            f"Auto-detected GPU count ({gpus_per_node}) must be divisible by --gpus-per-group ({gpus_per_group})"
        )

    groups_per_node = gpus_per_node // gpus_per_group
    resources: Dict[str, int] = {}
    for idx in range(groups_per_node):
        resources[f"{GROUP_RESOURCE_PREFIX}_{idx}_SLOTS"] = max_experiments_per_group

    print(json.dumps(resources, sort_keys=True, separators=(",", ":")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helpers for deployment/start.sh")
    sub = parser.add_subparsers(dest="command", required=True)

    p_wait_tcp = sub.add_parser("wait-tcp", help="Wait until TCP endpoint is reachable.")
    p_wait_tcp.add_argument("--host", required=True)
    p_wait_tcp.add_argument("--port", type=int, required=True)
    p_wait_tcp.add_argument("--timeout-seconds", type=int, default=180)
    p_wait_tcp.set_defaults(func=wait_tcp)

    p_detect_gpu_count = sub.add_parser("detect-gpu-count", help="Detect number of GPUs on this node.")
    p_detect_gpu_count.set_defaults(func=detect_gpu_count)

    p_group_resources = sub.add_parser(
        "build-group-resources",
        help=f"Build Ray custom resources for fixed-size GPU groups ({GROUP_RESOURCE_PREFIX}_*).",
    )
    p_group_resources.add_argument("--gpus-per-group", type=int, required=True)
    p_group_resources.add_argument("--max-experiments-per-group", type=int, default=1)
    p_group_resources.set_defaults(func=build_group_resources)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
