from __future__ import annotations

import argparse
import os
import resource
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import yaml
try:
    from .ray_execution import run_ray_jobs
except ImportError:
    from ray_execution import run_ray_jobs

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
    parser.add_argument(
        "--ray-init-timeout-seconds",
        type=float,
        default=float(os.environ.get("RAY_INIT_TIMEOUT_SECONDS", "90")),
        help="Timeout for one ray.init() attempt.",
    )
    parser.add_argument(
        "--ray-init-retries",
        type=int,
        default=int(os.environ.get("RAY_INIT_RETRIES", "3")),
        help="How many times to retry ray.init() before failing.",
    )
    parser.add_argument("--ray-nodes-per-exp", type=int, default=1,
                        help="Compatibility flag. Ray experiments are single-node and this is forced to 1.")
    parser.add_argument("--ray-gpus-per-node", type=int, default=_env_int("HOST_GPU_NUM", 1),
                        help="Total GPUs on each Ray node.")
    parser.add_argument(
        "--ray-gpus-per-group",
        type=int,
        default=_env_int("RAY_GPUS_PER_GROUP", _env_int("HOST_GPU_NUM", 1)),
        help="GPUs used by one experiment (group size). Must divide --ray-gpus-per-node.",
    )
    parser.add_argument(
        "--max-experiments-per-group",
        type=int,
        default=_env_int("MAX_EXPERIMENTS_PER_GROUP", 1),
        help="Maximum concurrent experiments allowed on each GPU group.",
    )
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


def _resolve_config_img_size(config_data: Dict[str, Any]) -> Optional[Any]:
    config_model_kwargs = config_data.get("model_kwargs")
    if isinstance(config_model_kwargs, dict) and config_model_kwargs.get("img_size") is not None:
        return config_model_kwargs.get("img_size")
    if config_data.get("img_size") is not None:
        return config_data.get("img_size")
    return None


def _find_img_size_in_args(total_args: List[str]) -> Optional[str]:
    # "Last write wins": later args in total_args override earlier ones.
    img_size_value: Optional[str] = None
    idx = 0
    while idx < len(total_args):
        token = total_args[idx]
        if token == "--img-size" and idx + 1 < len(total_args):
            img_size_value = str(total_args[idx + 1])
            idx += 2
            continue
        idx += 1
    return img_size_value


def _resolve_final_img_size(total_args: List[str], config_img_size: Optional[Any]) -> Optional[Any]:
    arg_img_size = _find_img_size_in_args(total_args)
    return arg_img_size if arg_img_size is not None else config_img_size


def _find_model_in_args(total_args: List[str]) -> Optional[str]:
    # "Last write wins": later args in total_args override earlier ones.
    model_value: Optional[str] = None
    idx = 0
    while idx < len(total_args):
        token = total_args[idx]
        if token == "--model" and idx + 1 < len(total_args):
            model_value = str(total_args[idx + 1])
            idx += 2
            continue
        if token.startswith("--model="):
            model_value = token.split("=", 1)[1]
        idx += 1
    return model_value


def _is_vit_model(total_args: List[str], config_model_name: str, fallback_model_name: str) -> bool:
    arg_model_name = _find_model_in_args(total_args)
    final_model_name = str(arg_model_name or config_model_name or fallback_model_name).strip().lower()
    return final_model_name.startswith("vit")


def _build_jobs(
    args: argparse.Namespace,
    experiments: List[Dict[str, Any]],
    runner_extra: List[str],
    status_file_path: str,
) -> List[Dict[str, Any]]:
    finished_names = _load_finished_names(status_file_path)
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

        jobs.append({
            "name": exp_name,
            "config": exp_config,
            "extra": exp_extra,
            "runner_extra": list(runner_extra),
            "log_path": os.path.join(args.log_dir, f"{exp_name}_{ts}.txt"),
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

    total_gpus_per_node = int(args.ray_gpus_per_node)
    nproc_per_experiment = int(args.ray_gpus_per_group)
    max_experiments_per_group = int(args.max_experiments_per_group)

    if total_gpus_per_node <= 0:
        raise ValueError("--ray-gpus-per-node must be >= 1.")
    if nproc_per_experiment <= 0:
        raise ValueError("--ray-gpus-per-group must be >= 1.")
    if total_gpus_per_node % nproc_per_experiment != 0:
        raise ValueError(
            f"--ray-gpus-per-node ({total_gpus_per_node}) must be divisible by "
            f"--ray-gpus-per-group ({nproc_per_experiment})."
        )
    if max_experiments_per_group <= 0:
        raise ValueError("--max-experiments-per-group must be >= 1.")

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
        "--img-size", "256",
    ]

    labelmix_common_args: List[str] = [
        "--labelmix",
        # "--labelmix-mix-k", "4",
        # "--labelmix-k-min", "1",
        # "--labelmix-k-max", "5",
        # "--labelmix-k-schedule", "linear",
        # "--labelmix-alpha-min", "1.0",
        # "--labelmix-alpha-max", "1.0",
        "--labelmix-schedule", "fixed",
        "--labelmix-step-mode", "total",
        # "--labelmix-sampling",
        # "--labelmix-sampling-min-side-px", "8",
        # "--labelmix-sampling-max-aspect", "10.0",
        # "--labelmix-sampling-bins", "16",
        # "--labelmix-sampling-pool-size", "256",
        # "--labelmix-sampling-low-watermark", "64",
        # "--labelmix-sampling-max-attempts", "200",

        "--mixup", "0",
        "--cutmix", "0",
        "--mixup-prob", "0.0",
    ]

    # model_configs = _load_model_configs(args.model_configs_path)
    model_configs = [
        "evaluation/baselines/configs/mnv4-conv-medium.yaml",
        "evaluation/baselines/configs/vit-wee.yaml",
    ]

    experiments: List[Dict[str, Any]] = []
    ks = [3, 4, 5, 6]
    alpha = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    loss = ["pl_loss", "soft_ce"]
    for model_config in model_configs:
        config_data = _load_yaml_dict(model_config)
        config_img_size = _resolve_config_img_size(config_data)
        for k in ks:
            for a in alpha:
                for l in loss:
                    total_args = [*imagenet_args, *extra, *labelmix_common_args,]
                    model = os.path.splitext(os.path.basename(model_config))[0]
                    model_kwargs_args: List[str] = []
                    config_model_name = str(config_data.get("model") or "")
                    is_vit_model = _is_vit_model(total_args, config_model_name, model)
                    if is_vit_model:
                        model_img_size = _resolve_final_img_size(total_args, config_img_size)
                        if model_img_size is None:
                            raise ValueError(
                                f"Config '{model_config}' is missing 'img_size' required for ViT model kwargs."
                            )
                        model_kwargs_args = [
                            "--model-kwargs",
                            f"img_size={model_img_size}",
                        ]
                        if not config_model_name.lower().startswith("vit_base"):
                            model_kwargs_args.append("fix_init=True")
                        total_args = total_args + model_kwargs_args

                    exp_name = f"baseline_imagenet1k_{model}_labelmix_k{k}_a{str(a).replace('.', 'p')}_loss{l}"
                    experiments.append({
                        "name": exp_name,
                        "config": model_config,
                        "extra": [
                            "--pin-mem",
                            *total_args,
                            "--labelmix-mix-k", str(k),
                            "--labelmix-k-min", str(k),
                            "--labelmix-k-max", str(k),
                            "--labelmix-alpha-min", str(a),
                            "--labelmix-alpha-max", str(a),
                            "--labelmix-loss", l,
                        ],
                    })

    status_file_path = str(args.status_file)
    jobs = _build_jobs(
        args=args,
        experiments=experiments,
        runner_extra=list(extra),
        status_file_path=status_file_path,
    )

    run_ray_jobs(
        args=args,
        jobs=jobs,
        base_train_common=base_train_common,
    )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
