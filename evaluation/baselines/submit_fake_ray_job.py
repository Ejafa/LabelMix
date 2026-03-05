#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit simple fake jobs to an existing Ray cluster.")
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "auto"),
        help="Ray cluster address (e.g. 29.200.14.73:6379). Default: $RAY_ADDRESS or 'auto'.",
    )
    parser.add_argument("--namespace", default="labelmix-debug", help="Ray namespace for the debug run.")
    parser.add_argument("--num-jobs", type=int, default=1, help="How many fake jobs to submit.")
    parser.add_argument("--sleep-seconds", type=float, default=3.0, help="Sleep time per fake job.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Timeout for ray.get over all jobs.")
    parser.add_argument("--num-cpus", type=float, default=1.0, help="CPU requested by each fake job.")
    parser.add_argument("--num-gpus", type=float, default=0.0, help="GPU requested by each fake job.")
    parser.add_argument(
        "--resources-json",
        default="",
        help='Optional task resources JSON (example: \'{"GPU_GROUP_0_SLOTS": 1}\').',
    )
    parser.add_argument("--payload", default="hello-from-fake-job", help="Message payload echoed by each job.")
    return parser


def _parse_resources(raw: str) -> Dict[str, float]:
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--resources-json must be a JSON object.")
    normalized: Dict[str, float] = {}
    for key, value in parsed.items():
        normalized[str(key)] = float(value)
    return normalized


def main() -> None:
    args = build_parser().parse_args()

    if args.num_jobs <= 0:
        raise ValueError("--num-jobs must be >= 1")
    if args.num_cpus < 0:
        raise ValueError("--num-cpus must be >= 0")
    if args.num_gpus < 0:
        raise ValueError("--num-gpus must be >= 0")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")

    resources = _parse_resources(args.resources_json)

    try:
        import ray
    except Exception as exc:
        raise RuntimeError("Ray is not installed in this Python environment.") from exc

    print(f"Connecting to Ray: {args.ray_address} (namespace={args.namespace})")
    ray.init(address=args.ray_address, namespace=args.namespace, ignore_reinit_error=True)
    print("Connected.")
    print("Cluster resources:", ray.cluster_resources())
    print("Available resources:", ray.available_resources())

    @ray.remote
    def fake_job(job_idx: int, sleep_seconds: float, payload: str) -> Dict[str, Any]:
        start = time.time()
        time.sleep(max(0.0, sleep_seconds))
        end = time.time()
        return {
            "job_idx": job_idx,
            "payload": payload,
            "hostname": socket.gethostname(),
            "start_utc": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
            "duration_seconds": round(end - start, 3),
        }

    task = fake_job.options(num_cpus=args.num_cpus, num_gpus=args.num_gpus, resources=resources)
    refs: List[Any] = [task.remote(i, args.sleep_seconds, args.payload) for i in range(args.num_jobs)]
    print(
        f"Submitted {args.num_jobs} fake jobs "
        f"(num_cpus={args.num_cpus}, num_gpus={args.num_gpus}, resources={resources or {}})."
    )

    results = ray.get(refs, timeout=args.timeout_seconds)
    print("Results:")
    for item in results:
        print(json.dumps(item, sort_keys=True))

    ray.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
