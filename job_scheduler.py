#!/usr/bin/env python3
"""Split a jobs.yaml across multiple nodes based on available GPU memory.

Queries GPU free memory on each node (local or via SSH), then distributes
jobs proportionally so that nodes with more headroom get more jobs.

Outputs:
    node_0_jobs.yaml   — jobs for the first node
    node_1_jobs.yaml   — jobs for the second node
    ...

Usage::

    # Discover GPU memory automatically (SSH into remote nodes)
    python job_scheduler.py --input jobs.yaml \\
        --nodes 28.12.129.140 28.12.25.40 28.12.130.213

    # Override with manual free-memory values (MiB per node)
    python job_scheduler.py --input jobs.yaml \\
        --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\
        --free-memory 320000 320000 310000

    # Preview the split without writing files
    python job_scheduler.py --input jobs.yaml \\
        --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\
        --dry-run

    # Then on each node, start daemon and submit:
    #   python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7
    #   python jobdaemon.py submit node_0_jobs.yaml
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# GPU memory query
# ---------------------------------------------------------------------------

def get_local_gpu_free_memory() -> List[Tuple[int, int, int]]:
    """Query local GPU memory via pynvml or nvidia-smi.

    Returns list of (gpu_index, total_mib, free_mib).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        result = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total = mem.total // (1024 * 1024)
            free = mem.free // (1024 * 1024)
            result.append((i, total, free))
        pynvml.nvmlShutdown()
        return result
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return []
        result = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                result.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return result
    except Exception:
        return []


def get_remote_gpu_free_memory(host: str, timeout: int = 30) -> List[Tuple[int, int, int]]:
    """Query GPU memory on a remote node via SSH + nvidia-smi.

    Returns list of (gpu_index, total_mib, free_mib).
    """
    cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout={timeout} "
        f"root@{host} "
        f"\"nvidia-smi --query-gpu=index,memory.total,memory.free "
        f"--format=csv,noheader,nounits\""
    )
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout + 10,
        )
        if r.returncode != 0:
            print(f"  ⚠️  SSH to {host} failed: {r.stderr.strip()}")
            return []
        result = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                result.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return result
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  SSH to {host} timed out after {timeout}s")
        return []
    except Exception as e:
        print(f"  ⚠️  SSH to {host} error: {e}")
        return []


def is_local(host: str) -> bool:
    """Check if a hostname/IP refers to this machine."""
    local_hostname = socket.gethostname()
    try:
        local_ips = set()
        for info in socket.getaddrinfo(local_hostname, None):
            local_ips.add(info[4][0])
        # Also add common loopback
        local_ips.add("127.0.0.1")
        local_ips.add("::1")
        local_ips.add(local_hostname)

        # Resolve the target host
        target_ips = set()
        for info in socket.getaddrinfo(host, None):
            target_ips.add(info[4][0])
        target_ips.add(host)

        return bool(local_ips & target_ips)
    except Exception:
        return host in ("localhost", "127.0.0.1", local_hostname)


def query_node_memory(host: str) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """Query a node's GPU memory.

    Returns (num_gpus, total_free_mib, per_gpu_info).
    """
    if is_local(host):
        print(f"  📡 {host} (local) — querying GPU memory...")
        info = get_local_gpu_free_memory()
    else:
        print(f"  📡 {host} (remote) — querying GPU memory via SSH...")
        info = get_remote_gpu_free_memory(host)

    if not info:
        print(f"  ❌ {host}: could not query GPU memory")
        return 0, 0, []

    total_free = sum(free for _, _, free in info)
    return len(info), total_free, info


# ---------------------------------------------------------------------------
# Job distribution (Hamilton's largest-remainder method)
# ---------------------------------------------------------------------------

def distribute_jobs_proportionally(
    total_jobs: int,
    node_weights: List[float],
) -> List[int]:
    """Distribute total_jobs across nodes proportionally to their weights.

    Uses Hamilton's largest-remainder method for fair integer allocation:
    1. Compute ideal (fractional) allocation per node.
    2. Give each node floor(ideal) jobs.
    3. Distribute remaining jobs to nodes with the largest remainders.

    This ensures no jobs are dropped and the split closely follows the
    memory proportions.
    """
    if not node_weights or total_jobs == 0:
        return [0] * len(node_weights)

    total_weight = sum(node_weights)
    if total_weight == 0:
        # Equal split if all weights are zero
        base = total_jobs // len(node_weights)
        remainder = total_jobs % len(node_weights)
        result = [base] * len(node_weights)
        for i in range(remainder):
            result[i] += 1
        return result

    # Step 1: Compute fractional allocations
    fractions = [(w / total_weight) * total_jobs for w in node_weights]

    # Step 2: Floor allocations
    floors = [int(f) for f in fractions]
    remainders = [f - int(f) for f in fractions]

    # Step 3: Distribute leftovers to largest remainders
    leftover = total_jobs - sum(floors)
    # Get indices sorted by remainder (descending)
    sorted_indices = sorted(range(len(remainders)),
                            key=lambda i: remainders[i], reverse=True)
    for i in range(leftover):
        floors[sorted_indices[i]] += 1

    return floors


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_jobs_yaml(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a jobs.yaml and return (defaults_dict, jobs_list)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not data or "jobs" not in data:
        print(f"Error: {path} has no 'jobs' key")
        sys.exit(1)
    defaults = data.get("defaults", {})
    jobs = data["jobs"]
    return defaults, jobs


def write_node_yaml(
    path: str,
    defaults: Dict[str, Any],
    jobs: List[Dict[str, Any]],
) -> None:
    """Write a per-node jobs.yaml."""
    output = {"defaults": defaults, "jobs": jobs}
    with open(path, "w") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)


def schedule(
    input_path: str,
    nodes: List[str],
    output_dir: str = ".",
    output_prefix: str = "node",
    free_memory_override: Optional[List[int]] = None,
    dry_run: bool = False,
) -> None:
    """Split jobs across nodes based on GPU free memory.

    Parameters
    ----------
    input_path : str
        Path to the input jobs.yaml.
    nodes : list[str]
        List of node IPs/hostnames.
    output_dir : str
        Directory to write per-node YAML files.
    output_prefix : str
        Prefix for output files (e.g. "node" → node_0_jobs.yaml).
    free_memory_override : list[int] | None
        If provided, skip GPU queries and use these values (MiB per node).
    dry_run : bool
        If True, print the plan but don't write files.
    """
    # Load jobs
    defaults, jobs = load_jobs_yaml(input_path)
    total_jobs = len(jobs)
    print(f"\n📋 Loaded {total_jobs} job(s) from {input_path}")
    print(f"   Defaults: gpus={defaults.get('gpus', 1)}, "
          f"max_retries={defaults.get('max_retries', 3)}")

    # Query or use overridden memory
    print(f"\n🖥️  Querying {len(nodes)} node(s)...")
    node_info: List[Dict[str, Any]] = []
    for i, host in enumerate(nodes):
        if free_memory_override and i < len(free_memory_override):
            total_free = free_memory_override[i]
            node_info.append({
                "host": host,
                "num_gpus": 8,  # assume 8 GPUs when overriding
                "total_free_mib": total_free,
                "per_gpu": [],
            })
            print(f"  📡 {host} — using override: {total_free:,} MiB free")
        else:
            num_gpus, total_free, per_gpu = query_node_memory(host)
            node_info.append({
                "host": host,
                "num_gpus": num_gpus,
                "total_free_mib": total_free,
                "per_gpu": per_gpu,
            })
            if per_gpu:
                gpu_strs = [f"GPU {idx}: {free:,}/{total:,} MiB"
                            for idx, total, free in per_gpu]
                print(f"  ✅ {host}: {num_gpus} GPUs, "
                      f"{total_free:,} MiB total free")
                for gs in gpu_strs:
                    print(f"       {gs}")
            else:
                print(f"  ❌ {host}: no GPU info (will get 0 jobs)")

    # Compute weights (total free memory per node)
    weights = [n["total_free_mib"] for n in node_info]

    # Distribute
    allocation = distribute_jobs_proportionally(total_jobs, weights)

    # Print plan
    print(f"\n📊 Job Distribution Plan:")
    print(f"{'─' * 65}")
    print(f"{'Node':<25} {'Free Memory':>15} {'Jobs':>8} {'Weight':>10}")
    print(f"{'─' * 65}")
    total_weight = sum(weights) or 1
    for i, (info, count) in enumerate(zip(node_info, allocation)):
        pct = (info['total_free_mib'] / total_weight) * 100 if total_weight else 0
        print(f"  node_{i} ({info['host']:<15}) "
              f"{info['total_free_mib']:>10,} MiB "
              f"{count:>6} "
              f"{pct:>8.1f}%")
    print(f"{'─' * 65}")
    print(f"  {'TOTAL':<25} {sum(weights):>10,} MiB {sum(allocation):>6}")

    if sum(allocation) != total_jobs:
        print(f"\n⚠️  WARNING: allocation sum ({sum(allocation)}) != "
              f"total jobs ({total_jobs})")

    # Split jobs
    splits: List[List[Dict[str, Any]]] = []
    offset = 0
    for count in allocation:
        splits.append(jobs[offset:offset + count])
        offset += count

    # Write output files
    if dry_run:
        print(f"\n🔍 DRY RUN — no files written.")
        for i, (info, split) in enumerate(zip(node_info, splits)):
            print(f"\n  node_{i} ({info['host']}): {len(split)} job(s)")
            for j in split[:5]:
                print(f"    - {j['name']}")
            if len(split) > 5:
                print(f"    ... and {len(split) - 5} more")
        return

    os.makedirs(output_dir, exist_ok=True)
    written_files = []
    for i, (info, split) in enumerate(zip(node_info, splits)):
        fname = f"{output_prefix}_{i}_jobs.yaml"
        fpath = os.path.join(output_dir, fname)
        write_node_yaml(fpath, defaults, split)
        written_files.append((fpath, info["host"], len(split)))

    # Print summary and next steps
    print(f"\n✅ Written {len(written_files)} file(s):")
    for fpath, host, count in written_files:
        print(f"   {fpath}  ({count} jobs for {host})")

    print(f"\n🚀 Next steps — on each node, start the daemon and submit:")
    print(f"{'─' * 70}")
    for i, (fpath, host, count) in enumerate(written_files):
        if count == 0:
            print(f"\n  # Node {i} ({host}): 0 jobs — skip")
            continue
        fname = os.path.basename(fpath)
        if is_local(host):
            print(f"\n  # Node {i} ({host}) — LOCAL:")
            print(f"  tmux new -s daemon")
            print(f"  python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7")
            print(f"  # (in another terminal)")
            print(f"  python jobdaemon.py submit {fname}")
        else:
            print(f"\n  # Node {i} ({host}) — REMOTE:")
            print(f"  ssh root@{host}")
            print(f"  cd {os.getcwd()}")
            print(f"  tmux new -s daemon")
            print(f"  python jobdaemon.py start --gpus 0,1,2,3,4,5,6,7")
            print(f"  # (in another terminal)")
            print(f"  python jobdaemon.py submit {fname}")
    print(f"{'─' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split jobs.yaml across nodes based on GPU memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Auto-discover GPU memory on 3 nodes:\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213\n"
            "\n"
            "  # Manual memory override (MiB):\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\\n"
            "      --free-memory 320000 320000 310000\n"
            "\n"
            "  # Preview without writing:\n"
            "  python job_scheduler.py --input jobs.yaml \\\n"
            "      --nodes 28.12.129.140 28.12.25.40 28.12.130.213 \\\n"
            "      --dry-run\n"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input jobs.yaml",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        required=True,
        help="Node IPs or hostnames (e.g. 28.12.129.140 28.12.25.40 28.12.130.213)",
    )
    parser.add_argument(
        "--free-memory",
        nargs="+",
        type=int,
        default=None,
        help="Override free GPU memory (MiB) per node. "
             "Must match --nodes count. Skips SSH queries.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to write per-node YAML files (default: cwd)",
    )
    parser.add_argument(
        "--prefix",
        default="node",
        help="Filename prefix (default: 'node' → node_0_jobs.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing files",
    )
    args = parser.parse_args()

    # Validate
    if args.free_memory and len(args.free_memory) != len(args.nodes):
        print(f"Error: --free-memory count ({len(args.free_memory)}) "
              f"!= --nodes count ({len(args.nodes)})")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    schedule(
        input_path=args.input,
        nodes=args.nodes,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        free_memory_override=args.free_memory,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
