from __future__ import annotations

import argparse
import os
import re
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
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Max number of experiments to run concurrently")
    parser.add_argument("--stagger-seconds", type=int, default=120,
                        help="Delay between launching experiments")
    parser.add_argument("--output-root", default="./output_runs/imagenet1k",
                        help="Base output directory")
    parser.add_argument("--log-dir", default="./logs/imagenet1k",
                        help="Directory for stdout/stderr logs")
    parser.add_argument("--master-port-base", type=int, default=29500,
                        help="Starting port to search for free torchrun master ports")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,0,1,2,3",
                        help="Comma-separated GPU ids")
    parser.add_argument("--ulimit-nofile", type=int, default=8192,
                        help="Set soft RLIMIT_NOFILE (0 to skip)")
    parser.add_argument("--labelmix-loss", default="soft_ce", choices=["soft_ce", "pl_loss"],
                        help="LabelMix loss to pass to train.py.")
    parser.add_argument("--labelmix-mix-k", type=int, default=5,
                        help="Base LabelMix K.")
    parser.add_argument("--labelmix-k-min", type=int, default=1,
                        help="Minimum K for LabelMix K scheduling.")
    parser.add_argument("--labelmix-k-max", type=int, default=5,
                        help="Maximum K for LabelMix K scheduling.")
    parser.add_argument("--labelmix-k-schedule", default="linear", choices=["fixed", "linear", "cosine"],
                        help="LabelMix K schedule.")
    parser.add_argument("--labelmix-k-reverse", action="store_true",
                        help="Reverse LabelMix K schedule (max->min).")
    parser.add_argument("--labelmix-k-warmup-epochs", type=int, default=0,
                        help="Warmup epochs for LabelMix K schedule.")
    parser.add_argument("--labelmix-k-total-epochs", type=int, default=None,
                        help="Total epochs for LabelMix K schedule.")
    parser.add_argument("--labelmix-alpha", type=float, default=1.0,
                        help="Fixed alpha value for LabelMix (used for both alpha min/max).")
    parser.add_argument("--status-file", default="./experiment_status.yaml",
                        help="YAML file tracking experiment status for resume/checkup.")
    parser.add_argument("--disable-status-checkup", action="store_true",
                        help="Ignore existing unfinished status entries and schedule fresh experiments.")
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


_WANDB_RUN_RE = re.compile(r"wandb:\s+setting up run\s+([A-Za-z0-9]+)")


def _load_status_file(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
    except Exception as exc:
        print(f"Warning: failed to read status file '{path}': {exc}")
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def _save_status_file(path: str, status: Dict[str, Dict[str, Any]]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(status, f, sort_keys=True, default_flow_style=False)
    os.replace(tmp_path, path)


def _sanitize_key(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    out = out.strip("_")
    return out or "exp"


def _new_status_key(name: str, created_at: str, status: Dict[str, Dict[str, Any]]) -> str:
    base = f"experiment_{created_at}_{_sanitize_key(name)}"
    key = base
    idx = 2
    while key in status:
        key = f"{base}_{idx}"
        idx += 1
    return key


def _upsert_flag(args_list: List[str], flag: str, value: str) -> None:
    if flag in args_list:
        i = args_list.index(flag)
        if i + 1 < len(args_list):
            args_list[i + 1] = value
        else:
            args_list.append(value)
    else:
        args_list.extend([flag, value])


def _extract_wandb_id_from_log(log_path: str) -> Optional[str]:
    if not log_path or not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(512 * 1024)
    except Exception:
        return None
    m = _WANDB_RUN_RE.search(head)
    return m.group(1) if m else None


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

    

    # Balanced dataset options
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

    cutmix_mixup_args: List[str] = [
        "--mixup", "0.8",
        "--cutmix", "1.0",
        "--mixup-prob", "1.0",
        "--mixup-switch-prob", "0.5",
        "--mixup-mode", "batch",
    ]

    labelmix_mix_k = "5"
    labelmix_k_min = "1"
    labelmix_k_max = "5"
    labelmix_k_schedule = "linear"
    labelmix_alpha_min = "1.0"
    labelmix_alpha_max = "1.0"

    labelmix_common_args: List[str] = [
        "--labelmix",
        "--labelmix-mix-k", labelmix_mix_k,
        "--labelmix-k-min", labelmix_k_min,
        "--labelmix-k-max", labelmix_k_max,
        "--labelmix-k-schedule", labelmix_k_schedule,
        "--labelmix-alpha-min", labelmix_alpha_min,
        "--labelmix-alpha-max", labelmix_alpha_max,
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
        # "mobilenetv4_conv_small",
        # "mobilenetv4_conv_medium",
        # "mobilenetv4_hybrid_medium",
         "vit_wee_patch16_reg1_gap_256",
        # "vit_little_patch16_reg1_gap_256",
        # "vit_base_patch16_reg4_gap_256",
    ]
    

    experiments: List[Dict[str, Any]] = []
    for model in models:
        model_kwargs_args: List[str] = []
        if model.startswith("vit_"):
            model_kwargs_args = ["--model-kwargs", "fix_init=True"]

        experiments.extend([
            # {
            #     "name": f"baseline_imagenet1k_{model}_noaug",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--no-aug",
            #         "--pin-mem",
            #     ],
            # },
            # {
            #     "name": f"baseline_imagenet1k_{model}_aug_cutmix_mixup",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #         *model_kwargs_args,
            #         *cutmix_mixup_args,
            #     ],
            # },
            # {
            #     "name": f"labelmix_imagenet1k_{model}_aug",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #         *labelmix_common_args,
            #         "--labelmix-loss", "soft_ce",
            #     ],
            # },
            # {
            #     "name": f"labelmix_imagenet1k_{model}_noaug",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #         "--no-aug",
            #         *labelmix_common_args,
            #         "--labelmix-loss", "soft_ce",
            #     ],
            # },
            {
                "name": f"labelmix_imagenet1k_{model}_aug_k_sched_soft_ce",
                "config": args.config,
                "extra": [
                    "--model", model,
                    "--pin-mem",
                    *model_kwargs_args,
                    "--labelmix-loss", "soft_ce",
                    *labelmix_common_args,
                ],
            },
            {
                "name": f"labelmix_imagenet1k_{model}_aug_k_sched_pl_loss",
                "config": args.config,
                "extra": [
                    "--model", model,
                    "--pin-mem",
                    *model_kwargs_args,
                    "--labelmix-loss", "pl_loss",
                    *labelmix_common_args,
                ],
            },
            # {
            #     "name": f"labelmix_imagenet1k_{model}_noaug_pl_loss",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         "--pin-mem",
            #         "--labelmix-loss", "pl_loss",
            #         "--no-aug",
            #         *labelmix_common_args,
            #     ],
            # },

            # {
            #     "name": f"kd_pld_imagenet1k_{model}_teacher_vit_base",
            #     "config": args.config,
            #     "extra": [
            #         "--model", model,
            #         *model_kwargs_args,
            #         "--pin-mem",
            #         "--kd-model-name", "timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
            #         "--kd-distill-type", "logit",
            #         "--kd-loss-type", "plackett_luce",
            #         "--kd-temperature", "1.0",
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

    status_file = args.status_file
    status_data = _load_status_file(status_file)
    if status_data:
        print(f"Loaded {len(status_data)} status entries from {status_file}")

    def _create_status_entry(exp: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        created_at = datetime.now().strftime("%Y%m%d-%H%M%S")
        status_key = _new_status_key(exp["name"], created_at, status_data)
        output_dir = os.path.join(args.output_root, f"{exp['name']}_{created_at}")
        log_path = os.path.join(args.log_dir, f"{exp['name']}_{created_at}.txt")
        entry: Dict[str, Any] = {
            "name": exp["name"],
            "created_at": created_at,
            "updated_at": created_at,
            "wandb_id": None,
            "log": log_path,
            "output": output_dir,
            "finished": False,
            "status": "pending",
            "config": exp.get("config", args.config),
            "extra": list(exp.get("extra", [])),
            "runner_extra": list(extra),
            "launch_count": 0,
            "last_exit_code": None,
        }
        status_data[status_key] = entry
        return status_key, entry

    jobs: List[Dict[str, Any]] = []
    status_dirty = False
    experiments_by_name = {exp["name"]: exp for exp in experiments}

    if args.disable_status_checkup:
        for exp in experiments:
            status_key, entry = _create_status_entry(exp)
            jobs.append({"status_key": status_key, "entry": entry})
        status_dirty = True
        print("Status checkup disabled: scheduling fresh experiments.")
    else:
        unfinished_entries: List[tuple[str, Dict[str, Any]]] = []
        finished_names: Set[str] = set()

        for status_key, entry in status_data.items():
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue

            if bool(entry.get("finished", False)):
                finished_names.add(name)
                continue

            extra_list = entry.get("extra", [])
            if not isinstance(extra_list, list):
                extra_list = list(extra_list) if isinstance(extra_list, tuple) else []
            entry["extra"] = [str(x) for x in extra_list]

            runner_extra = entry.get("runner_extra", [])
            if "runner_extra" not in entry:
                runner_extra = list(extra)
            elif not isinstance(runner_extra, list):
                runner_extra = list(runner_extra) if isinstance(runner_extra, tuple) else []
            entry["runner_extra"] = [str(x) for x in runner_extra]

            if not entry.get("config"):
                exp = experiments_by_name.get(name)
                if exp is not None:
                    entry["config"] = exp.get("config", args.config)
                else:
                    print(f"Skipping unfinished status entry '{status_key}' (missing config for unknown experiment).")
                    continue

            created_at = str(entry.get("created_at", "")).strip()
            if not created_at:
                created_at = datetime.now().strftime("%Y%m%d-%H%M%S")
                entry["created_at"] = created_at

            if not entry.get("output"):
                entry["output"] = os.path.join(args.output_root, f"{name}_{created_at}")
            if not entry.get("log"):
                entry["log"] = os.path.join(args.log_dir, f"{name}_{created_at}.txt")
            entry.setdefault("status", "pending")
            entry.setdefault("launch_count", 0)
            entry.setdefault("last_exit_code", None)
            entry.setdefault("updated_at", datetime.now().strftime("%Y%m%d-%H%M%S"))
            unfinished_entries.append((status_key, entry))
            status_dirty = True

        unfinished_entries.sort(key=lambda x: str(x[1].get("created_at", "")))
        if unfinished_entries:
            print(f"Found {len(unfinished_entries)} unfinished experiments in status file.")
        queued_names: Set[str] = set()
        for status_key, entry in unfinished_entries:
            jobs.append({"status_key": status_key, "entry": entry})
            queued_names.add(str(entry.get("name", "")))

        for exp in experiments:
            name = exp["name"]
            if name in queued_names:
                continue
            if name in finished_names:
                print(f"Skipping finished experiment from status file: {name}")
                continue
            status_key, entry = _create_status_entry(exp)
            jobs.append({"status_key": status_key, "entry": entry})
            status_dirty = True

    if status_dirty:
        _save_status_file(status_file, status_data)

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
        status_key = str(job["status_key"])
        entry = job["entry"]
        exp_name = str(entry["name"])
        output_dir = str(entry["output"])
        log_path = str(entry["log"])
        exp_config = str(entry.get("config", args.config))
        exp_extra = list(entry.get("extra", []))
        runner_extra = list(entry.get("runner_extra", extra))

        launch_count = int(entry.get("launch_count", 0))
        prior_status = str(entry.get("status", "pending")).lower()
        needs_resume = (not args.disable_status_checkup) and (
            launch_count > 0 or prior_status in ("running", "failed", "interrupted")
        )
        if needs_resume:
            resume_ckpt = os.path.join(output_dir, exp_name, "last.pth.tar")
            if os.path.exists(resume_ckpt):
                _upsert_flag(exp_extra, "--resume", resume_ckpt)
            wandb_id = entry.get("wandb_id")
            if wandb_id:
                _upsert_flag(exp_extra, "--wandb-resume-id", str(wandb_id))

        gpu_group = available_groups.popleft()
        master_port = _pick_port()

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
        cmd.extend(exp_extra)
        # Add launcher-level extra args (kept per status entry for reproducibility)
        cmd.extend(runner_extra)

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

        os.makedirs(output_dir, exist_ok=True)
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

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        entry["status"] = "running"
        entry["finished"] = False
        entry["updated_at"] = now
        entry["last_started_at"] = now
        entry["launch_count"] = launch_count + 1
        entry["last_exit_code"] = None
        entry["cmd"] = cmd
        status_data[status_key] = entry
        _save_status_file(status_file, status_data)

        if args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

        return {
            "proc": proc,
            "log_f": log_f,
            "gpu_group": gpu_group,
            "name": exp_name,
            "master_port": master_port,
            "status_key": status_key,
            "log_path": log_path,
        }

    try:
        while True:
            finished = []
            status_changed = False
            for info in running:
                status_key = str(info.get("status_key", ""))
                if status_key and status_key in status_data:
                    entry = status_data[status_key]
                    if not entry.get("wandb_id"):
                        wandb_id = _extract_wandb_id_from_log(str(info.get("log_path", "")))
                        if wandb_id:
                            entry["wandb_id"] = wandb_id
                            entry["updated_at"] = datetime.now().strftime("%Y%m%d-%H%M%S")
                            status_data[status_key] = entry
                            status_changed = True

                if info.get("proc") is None:  # handle dry run
                    finished.append(info)
                    continue

                ret = info["proc"].poll()
                if ret is not None:
                    # Process finished
                    info["log_f"].close()
                    available_groups.append(info["gpu_group"])
                    used_ports.discard(info["master_port"])

                    if status_key and status_key in status_data:
                        entry = status_data[status_key]
                        wandb_id = entry.get("wandb_id") or _extract_wandb_id_from_log(str(info.get("log_path", "")))
                        if wandb_id:
                            entry["wandb_id"] = wandb_id
                        now = datetime.now().strftime("%Y%m%d-%H%M%S")
                        entry["updated_at"] = now
                        entry["last_finished_at"] = now
                        entry["last_exit_code"] = int(ret)
                        if ret == 0:
                            entry["finished"] = True
                            entry["status"] = "finished"
                        else:
                            entry["finished"] = False
                            entry["status"] = "failed"
                        status_data[status_key] = entry
                        status_changed = True

                    if ret != 0:
                        print(f"!!! Experiment {info['name']} FAILED with exit code {ret}")
                    else:
                        print(f"*** Experiment {info['name']} FINISHED successfully.")

                    finished.append(info)

            for info in finished:
                running.remove(info)

            if status_changed:
                _save_status_file(status_file, status_data)

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
            status_key = str(info.get("status_key", ""))
            if status_key and status_key in status_data:
                entry = status_data[status_key]
                entry["finished"] = False
                entry["status"] = "interrupted"
                entry["updated_at"] = datetime.now().strftime("%Y%m%d-%H%M%S")
                status_data[status_key] = entry
        _save_status_file(status_file, status_data)
        print("Done.")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
