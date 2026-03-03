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

STORAGE_ROOT = "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
IMAGENET1K_DATA_DIR = os.path.join(STORAGE_ROOT, "data", "imagenet-1k")

# Per-model parameter counts (in millions). Update these values with your
# manually verified counts for accurate VRAM scheduling.
MODEL_PARAMS_MILLIONS: Dict[str, float] = {
    "mobilenetv4_conv_large": 32.59,
    "mobilenetv4_conv_medium": 9.72,
    "mobilenetv4_hybrid_large": 37.76,
    "mobilenetv4_hybrid_medium": 11.07,
    "resnetv2_101": 44.54,
    "resnetv2_50": 25.55,
    "vit_base_patch16_rope_reg1_gap_256": 86.43,
    "vit_little_patch16_reg4_gap_256": 22.52,
    "vit_medium_patch16_reg1_gap_256": 38.88,
    "vit_wee_patch16_reg1_gap_256": 13.42,
}
DEFAULT_MODEL_PARAMS_MILLIONS = 30.0
_WARNED_UNKNOWN_MODELS: Set[str] = set()

# Approximate base VRAM from parameter count:
# base_model_gb ~= offset + slope * params_million
MODEL_VRAM_BASE_OFFSET_GB = 0.9
MODEL_VRAM_BASE_PER_MPARAM_GB = 0.065

OPTIMIZER_VRAM_MULTIPLIER: Dict[str, float] = {
    "sgd": 1.00,
    "momentum": 1.00,
    "nesterov": 1.00,
    "lars": 1.00,
    "adam": 1.30,
    "adamw": 1.30,
    "lamb": 1.30,
    "adafactor": 1.15,
    "adagrad": 1.15,
    "rmsprop": 1.15,
    "__default__": 1.20,
}

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
    parser.add_argument("--stagger-seconds", type=int, default=120,
                        help="Delay between launching experiments")
    parser.add_argument("--output-root", default=os.path.join(STORAGE_ROOT, "output_runs", "imagenet1k"),
                        help="Base output directory passed to train.py --output")
    parser.add_argument("--log-dir", default=os.path.join(STORAGE_ROOT, "logs", "imagenet1k"),
                        help="Directory for stdout/stderr logs")
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


def _get_flag_value(args_list: List[str], flag: str) -> Optional[str]:
    for i in range(len(args_list) - 1, -1, -1):
        token = str(args_list[i])
        if token == flag:
            if i + 1 < len(args_list):
                return str(args_list[i + 1])
            return None
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _resolve_effective_flag(flag: str, layers: List[List[str]]) -> Optional[str]:
    for args_list in reversed(layers):
        value = _get_flag_value(args_list, flag)
        if value is not None:
            return value
    return None


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _model_params_millions(model_name: str) -> float:
    key = str(model_name or "").strip().lower()
    if key in MODEL_PARAMS_MILLIONS:
        return float(MODEL_PARAMS_MILLIONS[key])
    if key and key not in _WARNED_UNKNOWN_MODELS:
        _WARNED_UNKNOWN_MODELS.add(key)
        print(
            "Warning: model not found in MODEL_PARAMS_MILLIONS: "
            f"'{key}', using default={DEFAULT_MODEL_PARAMS_MILLIONS}M."
        )
    return float(DEFAULT_MODEL_PARAMS_MILLIONS)


def _optimizer_factor(opt_name: str) -> float:
    opt = str(opt_name or "").strip().lower()
    return float(OPTIMIZER_VRAM_MULTIPLIER.get(opt, OPTIMIZER_VRAM_MULTIPLIER["__default__"]))


def _mode_factor(mode_value: Any) -> float:
    _ = mode_value
    return 1.1


def _estimate_job_vram_gb(
    ray_gpus_per_node: int,
    config_data: Dict[str, Any],
    base_train_common: List[str],
    exp_extra: List[str],
    runner_extra: List[str],
) -> Dict[str, Any]:
    layers = [base_train_common, exp_extra, runner_extra]

    model = str(config_data.get("model") or "")
    optimizer = str(
        _resolve_effective_flag("--opt", layers)
        or config_data.get("opt")
        or "adamw"
    ).strip()
    batch_size = _to_int(
        _resolve_effective_flag("--batch-size", layers)
        or config_data.get("batch_size")
        or 32,
        default=128,
    )
    img_size = _to_int(
        _resolve_effective_flag("--img-size", layers)
        or config_data.get("img_size")
        or 256,
        default=256,
    )
    if img_size <= 0:
        img_size = 256

    mode = (
        _resolve_effective_flag("--balanced-mode", layers)
        or config_data.get("balanced_mode")
    )

    model_params_m = _model_params_millions(model)
    base_model_gb = MODEL_VRAM_BASE_OFFSET_GB + (MODEL_VRAM_BASE_PER_MPARAM_GB * model_params_m)
    image_scale = (float(img_size) / 224.0) ** 2
    activation_gb = 0.010 * float(batch_size) * image_scale
    per_gpu_vram_gb = (base_model_gb + activation_gb) * _optimizer_factor(optimizer) * _mode_factor(mode)
    per_gpu_vram_gb = max(0.5, per_gpu_vram_gb)

    total_vram_gb = per_gpu_vram_gb * float(max(1, int(ray_gpus_per_node)))
    return {
        "estimated_vram_gb": round(total_vram_gb, 3),
        "vram_factors": {
            "model": model or "unknown",
            "model_params_m": round(model_params_m, 3),
            "optimizer": optimizer or "unknown",
            "batch_size": batch_size,
            "img_size": img_size,
            "mode": mode,
            "base_model_gb": round(base_model_gb, 3),
            "per_gpu_vram_gb": round(per_gpu_vram_gb, 3),
            "gpus_per_exp": max(1, int(ray_gpus_per_node)),
        },
    }


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
        if exp_name in finished_names:
            print(f"Skipping finished experiment: {exp_name}")
            continue

        if exp_config not in config_cache:
            config_cache[exp_config] = _load_yaml_dict(exp_config)
        estimate = _estimate_job_vram_gb(
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
        })
    return jobs

def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

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
