from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step-Based Experiments")
    parser.add_argument("--config", default="./config/cifar100/base.yaml",
                        help="Base config file")
    parser.add_argument("--nproc", type=int, default=2,
                        help="Number of GPUs per experiment (torchrun --nproc_per_node)")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Max number of experiments to run concurrently")
    parser.add_argument("--stagger-seconds", type=int, default=120,
                        help="Delay between launching experiments")
    parser.add_argument("--output-root", default="./output_runs",
                        help="Base output directory")
    parser.add_argument("--log-dir", default="./logs",
                        help="Directory for stdout/stderr logs")
    parser.add_argument("--master-port-base", type=int, default=29500,
                        help="Starting port to search for free torchrun master ports")
    parser.add_argument("--cuda-visible-devices", default="3,2,3,2",
                        help="Comma-separated GPU ids")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return parser


def _resolve_gpu_pool(raw: str) -> List[str]:
    if raw:
        return [gpu.strip() for gpu in raw.split(",") if gpu.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env:
        return [gpu.strip() for gpu in env.split(",") if gpu.strip()]
    try:
        import torch
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return ["0"]


def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    base_torchrun: List[str] = [
        "torchrun",
        f"--nproc_per_node={args.nproc}",
    ]

    base_train_common: List[str] = [
        "--amp-dtype", "bfloat16",
        "--batch-size", "128",
        "--num-steps", "20000",
        "--warmup-steps", "1000",
        "--patience-steps", "1000",
        "--warmup-prefix",
        "--sched-on-updates",
        "--num-logs", "1000",
        "--num-evals", "100",
        "--num-saves", "10",
        "--wandb-project", "labelmix",
        "--log-wandb",
    ]

    augmentation_args: List[str] = [
        "--aa", "rand-m8-inc1-mstd1.0",
        "--aug-repeats", "0",
        "--aug-splits", "0",
        "--train-interpolation", "random",
        "--scale", "0.08", "1.0",
        "--ratio", "0.75", "1.3333333333333333",
        "--hflip", "0.5",
        "--vflip", "0.0",
        "--color-jitter", "0.4",
        "--grayscale-prob", "0.1",
        "--gaussian-blur-prob", "0.05",
        "--reprob", "0.25",
        "--remode", "pixel",
        "--recount", "1",
        "--mixup", "0.0",
        "--cutmix", "0.0",
        "--mixup-prob", "1.0",
        "--mixup-switch-prob", "0.5",
        "--mixup-mode", "batch",
        "--smoothing", "0.1",
    ]

    labelmix_args: List[str] = [
        "--labelmix",
        "--labelmix-mix-k", "5",
        "--labelmix-alpha-min", "0.1",
        "--labelmix-alpha-max", "1.0",
        "--labelmix-schedule", "linear",
        "--labelmix-step-mode", "total",
        "--labelmix-sampling",
        "--labelmix-sampling-min-side-px", "8",
        "--labelmix-sampling-max-aspect", "10.0",
        "--labelmix-sampling-bins", "16",
        "--labelmix-sampling-pool-size", "256",
        "--labelmix-sampling-low-watermark", "64",
        "--labelmix-sampling-max-attempts", "200",
    ]
        

    models = [
        "mobilenetv4_conv_small",
        # "mobilenetv4_conv_medium",
        # "mobilenetv4_hybrid_medium",
        # "vit_wee_patch16_reg1_gap_256",
        # "vit_little_patch16_reg1_gap_256",
        # "vit_base_patch16_reg4_gap_256",
    ]
    

    experiments: List[Dict[str, Any]] = []
    for model in models:
        experiments.extend([
            {
                "name": f"baseline_{model}_noaug",
                "config": args.config,
                "extra": [
                    "--model", model,
                    "--no-aug",
                    "--pin-mem",
                ],
            },
            # {
            #     "name": f"baseline_{model}_aug",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #     ],
            # },
            # {
            #     "name": f"labelmix_{model}_noaug",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #         "--no-aug",
            #         *labelmix_args,
            #     ],
            # },
            # {
            #     "name": f"labelmix_{model}_aug",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #         *labelmix_args,
            #     ],
            # },
        ])

    gpu_pool = _resolve_gpu_pool(args.cuda_visible_devices)
    if args.nproc <= 0:
        raise ValueError("--nproc must be >= 1")
    
    # Chunk GPUs into groups for each experiment
    groups = [
        gpu_pool[i:i + args.nproc]
        for i in range(0, len(gpu_pool) - len(gpu_pool) % args.nproc, args.nproc)
    ]
    
    if not groups:
        raise ValueError(f"Not enough GPUs ({len(gpu_pool)}) for requested --nproc ({args.nproc}).")

    max_parallel = min(args.max_parallel if args.max_parallel else len(groups), len(groups))

    print(f"Detected GPU Pool: {gpu_pool}")
    print(f"Formed GPU Groups: {groups}")
    print(f"Max Parallel Jobs: {max_parallel}")

    exp_queue = deque(experiments)
    running: List[Dict[str, Any]] = []
    available_groups: Deque[List[str]] = deque(groups)
    used_ports: Set[int] = set()

    def _is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
            return True

    def _pick_port() -> int:
        port = int(args.master_port_base)
        for _ in range(1000):
            if port not in used_ports and _is_port_free(port):
                used_ports.add(port)
                return port
            port += 1
        raise RuntimeError("Unable to find free port for torchrun.")

    def _start_next() -> Optional[Dict[str, Any]]:
        if not exp_queue or not available_groups:
            return None
        
        exp = exp_queue.popleft()
        exp_name = exp["name"]
        
        output_dir = os.path.join(args.output_root, f"{exp_name}_{timestamp}")
        log_path = os.path.join(args.log_dir, f"{exp_name}_{timestamp}.txt")
        
        gpu_group = available_groups.popleft()
        master_port = _pick_port()
        exp_config = exp.get("config", args.config)
        
        # Construct Command
        cmd = list(base_torchrun)
        cmd.extend(["--master_port", str(master_port)])
        
        cmd.append("train.py")
        cmd.extend(["-c", exp_config])
        cmd.extend(base_train_common)
        cmd.extend(augmentation_args)
        
        
            
        cmd.extend(["--experiment", exp_name])
        cmd.extend(["--output", output_dir])
        
        # Add experiment specific overrides
        cmd.extend(exp["extra"])
        # Add any command line extras passed to this script
        cmd.extend(extra)

        if args.dry_run:
            print(f"\n=== Dry run [{exp_name}]:")
            print(" ".join(cmd))
            print(f"GPUs: {','.join(gpu_group)}")
            available_groups.append(gpu_group)
            used_ports.discard(master_port)
            return {}

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
        env["OMP_NUM_THREADS"] = "1" 

        print(f"\n=== Running [{exp_name}]:")
        print(f"Cmd: {' '.join(cmd)}")
        print(f"GPUs: {env['CUDA_VISIBLE_DEVICES']} (Port: {master_port})")
        print(f"Log: {log_path}")

        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        
        if args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)
            
        return {
            "proc": proc,
            "log_f": log_f,
            "gpu_group": gpu_group,
            "name": exp_name,
            "master_port": master_port,
        }

    try:
        while True:
            finished = []
            for info in running:
                if info.get("proc") is None: # handle dry run
                    finished.append(info)
                    continue
                    
                ret = info["proc"].poll()
                if ret is not None:
                    # Process finished
                    info["log_f"].close()
                    available_groups.append(info["gpu_group"])
                    used_ports.discard(info["master_port"])
                    
                    if ret != 0:
                        print(f"!!! Experiment {info['name']} FAILED with exit code {ret}")
                    else:
                        print(f"*** Experiment {info['name']} FINISHED successfully.")
                    
                    finished.append(info)
            
            for info in finished:
                running.remove(info)

            # Start new processes if resources available
            while len(running) < max_parallel:
                info = _start_next()
                if info is None:
                    break
                if info: 
                    running.append(info)

            if not running and not exp_queue:
                break
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating running processes...")
        for info in running:
            if info.get("proc"):
                info["proc"].terminate()
        print("Done.")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
