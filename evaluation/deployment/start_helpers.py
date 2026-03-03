#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
