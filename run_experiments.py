from __future__ import annotations

import argparse
import os
import resource
import socket
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set

import yaml

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step-Based Experiments")
    parser.add_argument("--config", default="./config/imagenet1k/mnv4_small.yaml",
                        help="Base config file")
    parser.add_argument("--nproc", type=int, default=4,
                        help="Number of GPUs per experiment (torchrun --nproc_per_node)")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Max number of experiments to run concurrently")
    parser.add_argument("--stagger-seconds", type=int, default=0,
                        help="Delay between launching experiments")
    parser.add_argument("--output-root", default="./output_runs/imagenet1k",
                        help="Base output directory passed to train.py --output")
    parser.add_argument("--log-dir", default="./logs/imagenet1k",
                        help="Directory for stdout/stderr logs")
    parser.add_argument("--master-port-base", type=int, default=29500,
                        help="Starting port to search for free torchrun master ports")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,0,1,2,3",
                        help="Comma-separated GPU ids")
    parser.add_argument("--ulimit-nofile", type=int, default=8192,
                        help="Set soft RLIMIT_NOFILE (0 to skip)")
    parser.add_argument("--labelmix-k-reverse", action="store_true",
                        help="Reverse LabelMix K schedule (max->min).")
    parser.add_argument("--labelmix-k-warmup-epochs", type=int, default=0,
                        help="Warmup epochs for LabelMix K schedule.")
    parser.add_argument("--labelmix-k-total-epochs", type=int, default=None,
                        help="Total epochs for LabelMix K schedule.")
    parser.add_argument("--status-file", default="./experiment_status.yaml",
                        help="Status YAML used to skip finished experiments.")
    parser.add_argument("--disable-status-checkup", action="store_true",
                        help="Deprecated (ignored).")
    parser.add_argument("--disable-train-check-resume", action="store_true",
                        help="Do not pass --check-resume to train.py.")
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


def _set_nofile_limit(target: int) -> None:
    if target <= 0:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard)
        if new_soft != soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"RLIMIT_NOFILE: soft={new_soft}, hard={hard}")
        if new_soft < target:
            print(f"RLIMIT_NOFILE: requested {target} but hard limit is {hard}.")
    except Exception as exc:
        print(f"Warning: unable to set RLIMIT_NOFILE: {exc}")


def _upsert_flag(args_list: List[str], flag: str, value: Optional[str] = None) -> None:
    if flag in args_list:
        i = args_list.index(flag)
        if value is not None and i + 1 < len(args_list):
            args_list[i + 1] = value
        elif value is not None and i + 1 >= len(args_list):
            args_list.append(value)
        return
    args_list.append(flag)
    if value is not None:
        args_list.append(value)


def _load_finished_names(status_path: str) -> Set[str]:
    if not status_path or not os.path.exists(status_path):
        return set()
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return set()
    finished: Set[str] = set()
    if isinstance(data, dict):
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("finished") is True:
                name = str(entry.get("name") or "").strip()
                if name:
                    finished.add(name)
    return finished


def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    _set_nofile_limit(args.ulimit_nofile)

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    base_torchrun: List[str] = [
        "torchrun",
        f"--nproc_per_node={args.nproc}",
    ]

    balanced_buffer_steps = 5
    balanced_cache_threshold_steps = 4

    base_train_common: List[str] = [
        "--amp-dtype", "bfloat16",
        "--batch-size", "256",
        "--num-steps", "200000",
        "--warmup-steps", "10000",
        "--patience-steps", "10000",
        "--warmup-prefix",
        "--sched-on-updates",
        "--num-logs", "1000",
        "--num-evals", "100",
        "--num-saves", "10",
        "--wandb-project", "labelmix",
        "--log-wandb",
        "--workers", "4",
        "--loader-prefetch-factor", "4",
        "--balanced-mode", "1280",
        "--balanced-buffer-steps", str(balanced_buffer_steps),
        "--balanced-cache-threshold-steps", str(balanced_cache_threshold_steps),
    ]

    augmentation_args: List[str] = [
        "--img-size", "256",
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
        "--smoothing", "0.1",
    ]

    labelmix_common_args: List[str] = [
        "--labelmix",
        "--labelmix-mix-k", "5",
        "--labelmix-k-min", "1",
        "--labelmix-k-max", "5",
        "--labelmix-k-schedule", "linear",
        "--labelmix-alpha-min", "1.0",
        "--labelmix-alpha-max", "1.0",
        "--labelmix-schedule", "fixed",
        "--labelmix-step-mode", "total",
        "--labelmix-sampling",
        "--labelmix-sampling-min-side-px", "8",
        "--labelmix-sampling-max-aspect", "10.0",
        "--labelmix-sampling-bins", "16",
        "--labelmix-sampling-pool-size", "256",
        "--labelmix-sampling-low-watermark", "64",
        "--labelmix-sampling-max-attempts", "200",
    ]
    if args.labelmix_k_reverse:
        labelmix_common_args.append("--labelmix-k-reverse")
    if args.labelmix_k_warmup_epochs > 0:
        labelmix_common_args.extend(["--labelmix-k-warmup-epochs", str(args.labelmix_k_warmup_epochs)])
    if args.labelmix_k_total_epochs is not None:
        labelmix_common_args.extend(["--labelmix-k-total-epochs", str(args.labelmix_k_total_epochs)])

    models = [
        "vit_wee_patch16_reg1_gap_256",
    ]

    experiments: List[Dict[str, Any]] = []
    for model in models:
        model_kwargs_args: List[str] = []
        if model.startswith("vit_"):
            model_kwargs_args = ["--model-kwargs", "fix_init=True"]

        experiments.extend([
            
            {
                "name": f"labelmix_imagenet1k_{model}_aug_k_sched_pl_loss_ggez",
                "config": args.config,
                "extra": [
                    "--model", model,
                    "--pin-mem",
                    *model_kwargs_args,
                    "--labelmix-loss", "pl_loss",
                    *labelmix_common_args,
                ],
            },
            {
                "name": f"labelmix_imagenet1k_{model}_aug_k_sched_soft_ce_ggez",
                "config": args.config,
                "extra": [
                    "--model", model,
                    "--pin-mem",
                    *model_kwargs_args,
                    "--labelmix-loss", "soft_ce",
                    *labelmix_common_args,
                ],
            },
        ])

    gpu_pool = _resolve_gpu_pool(args.cuda_visible_devices)
    if args.nproc <= 0:
        raise ValueError("--nproc must be >= 1")

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

    status_file_path = str(args.status_file)
    finished_names = _load_finished_names(status_file_path)
    jobs: List[Dict[str, Any]] = []
    for exp in experiments:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = str(exp["name"])
        if exp_name in finished_names:
            print(f"Skipping finished experiment: {exp_name}")
            continue
        jobs.append({
            "name": exp_name,
            "config": str(exp.get("config", args.config)),
            "extra": list(exp.get("extra", [])),
            "runner_extra": list(extra),
            "log_path": os.path.join(args.log_dir, f"{exp_name}_{ts}.txt"),
        })

    exp_queue = deque(jobs)
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

        job = exp_queue.popleft()
        exp_name = str(job["name"])
        log_path = str(job["log_path"])
        exp_config = str(job["config"])
        exp_extra = list(job.get("extra", []))
        runner_extra = list(job.get("runner_extra", []))
        expected_output_dir = os.path.join(args.output_root, exp_name)

        if not args.disable_train_check_resume and not os.path.exists(status_file_path):
            _upsert_flag(exp_extra, "--check-resume")
            _upsert_flag(exp_extra, "--check-resume-log-dir", args.log_dir)

        gpu_group = available_groups.popleft()
        master_port = _pick_port()

        cmd = list(base_torchrun)
        cmd.extend(["--master_port", str(master_port)])
        cmd.append("train.py")
        cmd.extend(["-c", exp_config])
        cmd.extend(base_train_common)
        cmd.extend(augmentation_args)
        cmd.extend(["--experiment", exp_name])
        cmd.extend(["--output", args.output_root])
        cmd.extend(exp_extra)
        cmd.extend(runner_extra)

        if args.dry_run:
            print(f"\n=== Dry run [{exp_name}]:")
            print(" ".join(cmd))
            print(f"GPUs: {','.join(gpu_group)}")
            print(f"Expected output: {expected_output_dir}")
            available_groups.append(gpu_group)
            used_ports.discard(master_port)
            return {}

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
        env["OMP_NUM_THREADS"] = "1"

        print(f"\n=== Running [{exp_name}]:")
        print(f"Cmd: {' '.join(cmd)}")
        print(f"GPUs: {env['CUDA_VISIBLE_DEVICES']} (Port: {master_port})")
        print(f"Expected output: {expected_output_dir}")
        print(f"Log: {log_path}")

        os.makedirs(args.output_root, exist_ok=True)
        log_parent = os.path.dirname(log_path)
        if log_parent:
            os.makedirs(log_parent, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            log_f.write(f"RLIMIT_NOFILE: soft={soft}, hard={hard}\n")
            log_f.flush()
        except Exception as exc:
            log_f.write(f"RLIMIT_NOFILE: unavailable ({exc})\n")
            log_f.flush()

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
            finished: List[Dict[str, Any]] = []

            for info in running:
                if info.get("proc") is None:
                    finished.append(info)
                    continue

                ret = info["proc"].poll()
                if ret is not None:
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

            if len(running) < max_parallel:
                info = _start_next()
                if info:
                    running.append(info)

            if not running and not exp_queue:
                break

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nInterrupted! Terminating running processes...")
        for info in running:
            if info.get("proc") and info["proc"].poll() is None:
                info["proc"].terminate()
            log_f = info.get("log_f")
            if log_f and not log_f.closed:
                log_f.close()
        print("Done.")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
