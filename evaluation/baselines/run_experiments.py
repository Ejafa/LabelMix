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

STORAGE_ROOT = "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
IMAGENET1K_DATA_DIR = os.path.join(STORAGE_ROOT, "data", "imagenet-1k")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step-Based Experiments")
    parser.add_argument("--scheduler", choices=["local", "ray"], default="local",
                        help="Job scheduler backend.")
    parser.add_argument("--nproc", type=int, default=1,
                        help="Number of GPUs per experiment (torchrun --nproc_per_node)")
    parser.add_argument("--stagger-seconds", type=int, default=120,
                        help="Delay between launching experiments")
    parser.add_argument("--multi-node-stagger", action="store_true",
                        help="Apply a one-time node-index-based stagger before launching any jobs.")
    parser.add_argument("--output-root", default=os.path.join(STORAGE_ROOT, "output_runs", "imagenet1k"),
                        help="Base output directory passed to train.py --output")
    parser.add_argument("--log-dir", default=os.path.join(STORAGE_ROOT, "logs", "imagenet1k"),
                        help="Directory for stdout/stderr logs")
    parser.add_argument("--master-port-base", type=int, default=29500,
                        help="Starting port to search for free torchrun master ports")
    parser.add_argument("--cuda-visible-devices", default=None,
                        help="Comma-separated GPU ids")
    parser.add_argument("--gpu-per-node", type=int, required=True,
                        help="Number of GPUs per node (used to build default CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--gpu-nodes", type=int, default=1,
                        help="Total number of GPU nodes used to shard experiments.")
    parser.add_argument("--node-index", type=int, default=0,
                        help="Index of this node in [0, gpu-nodes - 1].")
    parser.add_argument("--ulimit-nofile", type=int, default=8192,
                        help="Set soft RLIMIT_NOFILE (0 to skip)")
    parser.add_argument("--labelmix-k-reverse", action="store_true",
                        help="Reverse LabelMix K schedule (max->min).")
    parser.add_argument("--labelmix-k-warmup-epochs", type=int, default=0,
                        help="Warmup epochs for LabelMix K schedule.")
    parser.add_argument("--labelmix-k-total-epochs", type=int, default=None,
                        help="Total epochs for LabelMix K schedule.")
    parser.add_argument("--model-configs-path", default="evaluation/baselines/configs",
                        help="YAML config file or directory of YAML configs to run.")
    parser.add_argument("--status-file", default=os.path.join(STORAGE_ROOT, "experiment_status.yaml"),
                        help="Status YAML used to skip finished experiments.")
    parser.add_argument("--disable-train-check-resume", action="store_true",
                        help="Do not pass --check-resume to train.py.")

    parser.add_argument("--ray-address", default="auto",
                        help="Ray cluster address. Use 'auto' for auto-discovery.")
    parser.add_argument("--ray-nodes-per-exp", type=int, default=1,
                        help="Compatibility flag. Ray experiments are single-node and this is forced to 1.")
    parser.add_argument("--ray-gpus-per-node", type=int, default=_env_int("HOST_GPU_NUM", 1),
                        help="GPUs allocated per Ray node for one experiment.")
    parser.add_argument("--ray-cpus-per-node", type=int, default=_env_int("RAY_CPUS_PER_NODE", 32),
                        help="CPUs allocated per Ray node for one experiment.")
    parser.add_argument("--ray-strategy", choices=["STRICT_SPREAD", "PACK"], default="STRICT_SPREAD",
                        help="Placement group strategy for Ray experiments.")
    parser.add_argument("--ray-max-parallel", type=int, default=0,
                        help="Max concurrent experiments in Ray mode (0 = auto from cluster GPU capacity).")
    parser.add_argument("--ray-pg-timeout-seconds", type=int, default=3600,
                        help="Timeout waiting for Ray placement group resources.")

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


def _load_model_configs(config_path: str) -> List[str]:
    if not config_path:
        return []
    path = os.path.expanduser(config_path)
    if os.path.isdir(path):
        configs = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith((".yaml", ".yml"))
        ]
        return sorted(configs)
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Model config path not found: {config_path}")


def _build_jobs(
    args: argparse.Namespace,
    experiments: List[Dict[str, Any]],
    runner_extra: List[str],
    status_file_path: str,
) -> List[Dict[str, Any]]:
    finished_names = _load_finished_names(status_file_path)
    jobs: List[Dict[str, Any]] = []
    for exp_idx, exp in enumerate(experiments):
        if args.scheduler == "local" and exp_idx % args.gpu_nodes != args.node_index:
            continue
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = str(exp["name"])
        if exp_name in finished_names:
            print(f"Skipping finished experiment: {exp_name}")
            continue
        jobs.append({
            "name": exp_name,
            "config": str(exp["config"]),
            "extra": list(exp.get("extra", [])),
            "runner_extra": list(runner_extra),
            "log_path": os.path.join(args.log_dir, f"{exp_name}_{ts}.txt"),
        })
    return jobs


def _run_local_jobs(
    args: argparse.Namespace,
    jobs: List[Dict[str, Any]],
    base_torchrun: List[str],
    base_train_common: List[str],
    status_file_path: str,
) -> None:
    print(f"Experiment sharding: index {args.node_index} of {args.gpu_nodes} nodes")

    print(args.cuda_visible_devices)
    gpu_pool = _resolve_gpu_pool(args.cuda_visible_devices)

    groups = [
        gpu_pool[i:i + args.nproc]
        for i in range(0, len(gpu_pool) - len(gpu_pool) % args.nproc, args.nproc)
    ]
    if not groups:
        raise ValueError(f"Not enough GPUs ({len(gpu_pool)}) for requested --nproc ({args.nproc}).")

    max_parallel = len(groups)
    print(f"Detected GPU Pool: {gpu_pool}")
    print(f"Formed GPU Groups: {groups}")
    print(f"Max Parallel Jobs: {max_parallel}")

    exp_queue = deque(jobs)
    running: List[Dict[str, Any]] = []
    available_groups: Deque[List[str]] = deque(groups)
    used_ports: Set[int] = set()

    if args.multi_node_stagger and args.stagger_seconds > 0 and args.node_index > 0:
        initial_delay = (args.stagger_seconds / 4.0) * args.node_index
        print(f"Multi-node stagger: sleeping {initial_delay:.1f}s before starting jobs.")
        time.sleep(initial_delay)

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
            if info.get("proc") and info["proc"].poll() is None:
                info["proc"].terminate()
            log_f = info.get("log_f")
            if log_f and not log_f.closed:
                log_f.close()
        print("Done.")


def _run_ray_jobs(
    args: argparse.Namespace,
    jobs: List[Dict[str, Any]],
    base_train_common: List[str],
    status_file_path: str,
) -> None:
    try:
        import ray
        from ray.util.placement_group import placement_group, remove_placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    except Exception as exc:
        raise RuntimeError(
            "Ray mode selected, but Ray is not installed. Install with `pip install ray`."
        ) from exc

    if args.ray_nodes_per_exp != 1:
        print(
            f"Ray single-node mode: overriding --ray-nodes-per-exp={args.ray_nodes_per_exp} to 1."
        )
    if args.ray_gpus_per_node <= 0:
        raise ValueError("--ray-gpus-per-node must be >= 1")
    if args.ray_cpus_per_node <= 0:
        raise ValueError("--ray-cpus-per-node must be >= 1")

    ray_nodes_per_exp = 1

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    cluster = ray.cluster_resources()
    print(f"Connected to Ray cluster. Resources: {cluster}")

    per_exp_gpus = args.ray_gpus_per_node
    total_cluster_gpus = int(cluster.get("GPU", 0))
    if args.ray_max_parallel > 0:
        max_parallel = args.ray_max_parallel
    else:
        max_parallel = max(1, total_cluster_gpus // per_exp_gpus) if per_exp_gpus > 0 else 1
    print(
        "Ray scheduling config: "
        f"nodes/exp={ray_nodes_per_exp}, gpus/node={args.ray_gpus_per_node}, "
        f"cpus/node={args.ray_cpus_per_node}, strategy={args.ray_strategy}, "
        f"max_parallel={max_parallel}"
    )

    @ray.remote(max_retries=0)
    def _run_torchrun(cmd: List[str], env_updates: Dict[str, str], log_path: str) -> int:
        env = os.environ.copy()
        env.update(env_updates)

        log_parent = os.path.dirname(log_path)
        if log_parent:
            os.makedirs(log_parent, exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as log_f:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                log_f.write(f"RLIMIT_NOFILE: soft={soft}, hard={hard}\n")
                log_f.flush()
            except Exception as exc:
                log_f.write(f"RLIMIT_NOFILE: unavailable ({exc})\n")
                log_f.flush()

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return int(proc.wait())

    exp_queue = deque(jobs)
    running: Dict[Any, Dict[str, Any]] = {}
    failures: List[str] = []

    def _schedule_one(job: Dict[str, Any]) -> bool:
        exp_name = str(job["name"])
        exp_config = str(job["config"])
        exp_extra = list(job.get("extra", []))
        runner_extra = list(job.get("runner_extra", []))
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        expected_output_dir = os.path.join(args.output_root, exp_name)
        log_path = os.path.join(args.log_dir, f"{exp_name}_{ts}.txt")

        if not args.disable_train_check_resume and not os.path.exists(status_file_path):
            _upsert_flag(exp_extra, "--check-resume")
            _upsert_flag(exp_extra, "--check-resume-log-dir", args.log_dir)

        cmd = [
            "torchrun",
            "--nnodes=1",
            f"--nproc_per_node={args.ray_gpus_per_node}",
            "train.py",
            "-c", exp_config,
            *base_train_common,
            "--experiment", exp_name,
            "--output", args.output_root,
            *exp_extra,
            *runner_extra,
        ]

        if args.dry_run:
            print(f"\n=== Dry run [{exp_name}] (Ray single-node):")
            print(" ".join(cmd))
            print(f"Expected output: {expected_output_dir}")
            print(f"Log: {log_path}")
            return True

        pg = placement_group(
            bundles=[{"GPU": float(args.ray_gpus_per_node), "CPU": float(args.ray_cpus_per_node)}],
            strategy=args.ray_strategy,
        )
        try:
            ray.get(pg.ready(), timeout=args.ray_pg_timeout_seconds)
        except Exception as exc:
            print(f"!!! Experiment {exp_name} FAILED: unable to reserve placement group: {exc}")
            remove_placement_group(pg)
            failures.append(exp_name)
            return False

        scheduling = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        env_updates = {"OMP_NUM_THREADS": "1"}
        ref = _run_torchrun.options(
            num_gpus=args.ray_gpus_per_node,
            num_cpus=args.ray_cpus_per_node,
            scheduling_strategy=scheduling,
        ).remote(cmd, env_updates, log_path)
        running[ref] = {"name": exp_name, "pg": pg}

        print(f"\n=== Running [{exp_name}] with Ray (single-node):")
        print(f"GPUs per experiment: {args.ray_gpus_per_node}")
        print(f"Expected output: {expected_output_dir}")
        print(f"Log: {log_path}")
        return True

    try:
        while exp_queue or running:
            while exp_queue and len(running) < max_parallel:
                job = exp_queue.popleft()
                _schedule_one(job)
                if args.stagger_seconds > 0:
                    time.sleep(args.stagger_seconds)

            if not running:
                continue

            ready, _ = ray.wait(list(running.keys()), num_returns=1, timeout=2)
            if not ready:
                continue

            ref = ready[0]
            info = running.pop(ref)
            exp_name = str(info["name"])
            pg = info["pg"]
            try:
                ret = int(ray.get(ref))
                if ret == 0:
                    print(f"*** Experiment {exp_name} FINISHED successfully.")
                else:
                    print(f"!!! Experiment {exp_name} FAILED with exit code {ret}")
                    failures.append(exp_name)
            except Exception as exc:
                print(f"!!! Experiment {exp_name} FAILED: {exc}")
                failures.append(exp_name)
            finally:
                remove_placement_group(pg)

    except KeyboardInterrupt:
        print("\nInterrupted! Cancelling running Ray jobs...")
        for ref in list(running.keys()):
            ray.cancel(ref, force=True)
        for ref, info in list(running.items()):
            try:
                ray.get(ref, timeout=1)
            except Exception:
                pass
            remove_placement_group(info["pg"])
        running.clear()
        print("Done.")
    finally:
        ray.shutdown()

    if failures:
        print(f"\nRay mode completed with failures: {sorted(set(failures))}")


def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    if args.nproc <= 0:
        raise ValueError("--nproc must be >= 1")

    if args.scheduler == "local":
        if args.gpu_nodes <= 0:
            raise ValueError("--gpu-nodes must be >= 1")
        if args.node_index < 0 or args.node_index >= args.gpu_nodes:
            raise ValueError("--node-index must be in [0, gpu-nodes - 1]")
        if args.gpu_per_node <= 0:
            raise ValueError("--gpu-per-node must be >= 1")
        if not args.cuda_visible_devices:
            args.cuda_visible_devices = ",".join(str(i) for i in range(args.gpu_per_node))
            print(f"Auto-configured --cuda-visible-devices: {args.cuda_visible_devices}")

    _set_nofile_limit(args.ulimit_nofile)

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    base_torchrun: List[str] = [
        "torchrun",
        f"--nproc_per_node={args.nproc}",
    ]

    base_batch_size = 1024
    if base_batch_size % args.nproc != 0:
        raise ValueError(
            f"Base batch size {base_batch_size} must be divisible by --nproc ({args.nproc})."
        )
    per_gpu_batch_size = base_batch_size // args.nproc

    base_train_common: List[str] = [
        "--amp-dtype", "bfloat16",
        "--batch-size", str(per_gpu_batch_size),  # 1024 / nproc
        "--warmup-prefix",
        "--aug-repeats", "0",
        "--img-size", "256",

        "--sched-on-updates",
        "--num-logs", "1000",
        "--num-evals", "100",
        "--num-saves", "10",
        "--wandb-project", "labelmix",
        "--log-wandb",
        "--workers", "8",
        "--loader-prefetch-factor", "4",
        "--balanced-buffer-steps", "1",
        "--balanced-cache-threshold-steps", "1",
    ]

    imagenet_args: List[str] = [
        "--dataset", "hfds/ILSVRC/imagenet-1k",
        "--data-dir", IMAGENET1K_DATA_DIR,
        "--train-split", "train",
        "--val-split", "validation",
        "--input-key", "image",
        "--target-key", "label",
        "--balanced-mode", "1280",
        "--num-classes", "1000",
        "--num-steps", "362500",  # 290 epochs. 1280000/1024 = 1250 steps/epoch
        "--warmup-steps", "12500",  # 10 epochs
    ]

    model_configs = _load_model_configs(args.model_configs_path)

    experiments: List[Dict[str, Any]] = []
    learning_rates = [0.0005, 0.001, 0.0015]
    for model_config in model_configs:
        for lr in learning_rates:
            model = os.path.splitext(os.path.basename(model_config))[0]
            model_kwargs_args: List[str] = []
            if model.startswith("vit_"):
                model_kwargs_args = ["--model-kwargs", "fix_init=True", "img_size=256"]
            exp_name = f"baseline_imagenet1k_{model}_lr{lr}"
            experiments.append({
                "name": exp_name,
                "config": model_config,
                "extra": [
                    "--pin-mem",
                    *model_kwargs_args,
                    *imagenet_args,
                    "--lr", str(lr),
                ],
            })

    status_file_path = str(args.status_file)
    jobs = _build_jobs(
        args=args,
        experiments=experiments,
        runner_extra=list(extra),
        status_file_path=status_file_path,
    )

    if args.scheduler == "ray":
        _run_ray_jobs(
            args=args,
            jobs=jobs,
            base_train_common=base_train_common,
            status_file_path=status_file_path,
        )
    else:
        _run_local_jobs(
            args=args,
            jobs=jobs,
            base_torchrun=base_torchrun,
            base_train_common=base_train_common,
            status_file_path=status_file_path,
        )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
