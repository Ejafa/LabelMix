#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import subprocess
import time


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


def detect_vram_gb(args: argparse.Namespace) -> int:
    mode = str(args.mode).strip().lower()
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

    values_mb = []
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

    if args.aggregate == "sum":
        value_gb = sum(values_mb) / 1024.0
    elif args.aggregate == "min":
        value_gb = min(values_mb) / 1024.0
    else:
        raise ValueError("--aggregate must be one of: sum, min")

    value_gb = max(0.0, value_gb - float(args.reserve_gb))
    print(f"{value_gb:.3f}")
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

    p_detect_vram = sub.add_parser("detect-vram-gb", help="Detect node VRAM budget in GB.")
    p_detect_vram.add_argument("--mode", default="free", choices=["free", "total"])
    p_detect_vram.add_argument("--aggregate", default="sum", choices=["sum", "min"])
    p_detect_vram.add_argument("--reserve-gb", type=float, default=0.0)
    p_detect_vram.set_defaults(func=detect_vram_gb)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
