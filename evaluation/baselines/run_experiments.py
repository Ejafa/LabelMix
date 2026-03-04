from __future__ import annotations

import argparse
import os
import resource
from datetime import datetime
from typing import Any, Dict, List, Set

import yaml
try:
    from .ray_execution import run_ray_jobs
except ImportError:
    from ray_execution import run_ray_jobs
try:
    from .vram_estimation import estimate_job_vram_gb
except ImportError:
    from vram_estimation import estimate_job_vram_gb

DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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
    parser.add_argument(
        "--project-root",
        "--storage-root",
        dest="project_root",
        default=DEFAULT_PROJECT_ROOT,
        help="Base path used for default output/log/status/data paths.",
    )
    parser.add_argument("--stagger-seconds", type=int, default=120,
                        help="Delay between launching experiments")
    parser.add_argument("--output-root", default=None,
                        help="Base output directory passed to train.py --output. "
                             "Default: <project-root>/output_runs/imagenet1k")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for stdout/stderr logs. "
                             "Default: <project-root>/logs/imagenet1k")
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
    parser.add_argument("--status-file", default=None,
                        help="Status YAML used to skip finished experiments. "
                             "Default: <project-root>/experiment_status.yaml")
    parser.add_argument("--disable-train-check-resume", action="store_true",
                        help="Do not pass --check-resume to train.py.")

    parser.add_argument("--ray-address", default="auto",
                        help="Ray cluster address. Use 'auto' for auto-discovery.")
    parser.add_argument("--ray-nodes-per-exp", type=int, default=1,
                        help="Compatibility flag. Ray experiments are single-node and this is forced to 1.")
    parser.add_argument("--ray-gpus-per-node", type=int, default=_env_int("HOST_GPU_NUM", 1),
                        help="GPUs allocated per Ray node for one experiment.")
    parser.add_argument(
        "--cpu-per-experiment",
        "--ray-cpus-per-node",
        dest="cpu_per_experiment",
        type=int,
        default=_env_int(
            "CPU_PER_EXPERIMENT",
            _env_int("RAY_CPUS_PER_NODE", 32),
        ),
        help="CPUs allocated per experiment in Ray mode.",
    )
    parser.add_argument("--ray-strategy", choices=["STRICT_SPREAD", "PACK"], default="STRICT_SPREAD",
                        help="Placement group strategy for Ray experiments.")
    parser.add_argument("--ray-pg-timeout-seconds", type=int, default=3600,
                        help="Timeout waiting for Ray placement group resources.")

    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return parser

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


def _is_finished_in_status_file(status_path: str, exp_name: str) -> bool:
    if not status_path or not os.path.exists(status_path):
        return False
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    exp_name = str(exp_name or "").strip()
    if not exp_name:
        return False
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name") or "").strip() != exp_name:
            continue
        if entry.get("finished") is True:
            return True
    return False


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


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _build_jobs(
    args: argparse.Namespace,
    experiments: List[Dict[str, Any]],
    base_train_common: List[str],
    runner_extra: List[str],
    status_file_path: str,
) -> List[Dict[str, Any]]:
    finished_names = _load_finished_names(status_file_path)
    config_cache: Dict[str, Dict[str, Any]] = {}
    jobs: List[Dict[str, Any]] = []
    for exp in experiments:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = str(exp["name"])
        exp_config = str(exp["config"])
        exp_extra = list(exp.get("extra", []))

        exp_status_file = os.path.join(args.output_root, exp_name, "run_status.yaml")
        if exp_name in finished_names or _is_finished_in_status_file(exp_status_file, exp_name):
            print(f"Skipping finished experiment: {exp_name}")
            continue

        if exp_config not in config_cache:
            config_cache[exp_config] = _load_yaml_dict(exp_config)
        estimate = estimate_job_vram_gb(
            ray_gpus_per_node=max(1, int(args.ray_gpus_per_node)),
            config_data=config_cache[exp_config],
            base_train_common=base_train_common,
            exp_extra=exp_extra,
            runner_extra=runner_extra,
        )

        jobs.append({
            "name": exp_name,
            "config": exp_config,
            "extra": exp_extra,
            "runner_extra": list(runner_extra),
            "log_path": os.path.join(args.log_dir, f"{exp_name}_{ts}.txt"),
            "estimated_vram_gb": float(estimate["estimated_vram_gb"]),
            "vram_factors": dict(estimate["vram_factors"]),
            "status_file": exp_status_file,
        })
    return jobs

def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    project_root = os.path.abspath(os.path.expanduser(str(args.project_root)))
    if not args.output_root:
        args.output_root = os.path.join(project_root, "output_runs", "imagenet1k")
    if not args.log_dir:
        args.log_dir = os.path.join(project_root, "logs", "imagenet1k")
    if not args.status_file:
        args.status_file = os.path.join(project_root, "experiment_status.yaml")
    imagenet1k_data_dir = os.path.join(project_root, "data", "imagenet-1k")

    _set_nofile_limit(args.ulimit_nofile)

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    nproc_per_experiment = int(args.ray_gpus_per_node)
    if nproc_per_experiment <= 0:
        raise ValueError("Per-experiment GPU count must be >= 1.")

    base_batch_size = 1024
    if base_batch_size % nproc_per_experiment != 0:
        raise ValueError(
            f"Base batch size {base_batch_size} must be divisible by per-experiment GPU count "
            f"({nproc_per_experiment})."
        )
    per_gpu_batch_size = base_batch_size // nproc_per_experiment

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
        "--data-dir", imagenet1k_data_dir,
        "--train-split", "train",
        "--val-split", "validation",
        "--input-key", "image",
        "--target-key", "label",
        "--balanced-mode", "1280",
        "--num-classes", "1000",
        "--num-steps", "362500",  # 290 epochs. 1280000/1024 = 1250 steps/epoch
        "--warmup-steps", "12500",  # 10 epochs
    ]

    # TODO: For testing purposes'
    cifar100_args: List[str] = [
        "--dataset", "hfds/uoft-cs/cifar100",
        "--data-dir", os.path.join(project_root, "data", "cifar100"),
        "--train-split", "train",
        "--val-split", "test",
        "--input-key", "img",
        "--target-key", "fine_label",
        "--balanced-mode", "500",
        "--num-classes", "100",
        "--num-steps", "10000", 
        "--warmup-steps", "500",
        "--img-size", "32",
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
                    # *imagenet_args,
                    *cifar100_args,
                    "--lr", str(lr),
                ],
            })

    status_file_path = str(args.status_file)
    jobs = _build_jobs(
        args=args,
        experiments=experiments,
        base_train_common=base_train_common,
        runner_extra=list(extra),
        status_file_path=status_file_path,
    )

    run_ray_jobs(
        args=args,
        jobs=jobs,
        base_train_common=base_train_common,
        status_file_path=status_file_path,
    )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
