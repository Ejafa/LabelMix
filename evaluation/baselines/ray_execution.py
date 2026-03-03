from __future__ import annotations

import argparse
import math
import os
import resource
import socket
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
try:
    from .ray_scheduler_logging import RaySchedulerLogger
except ImportError:
    from ray_scheduler_logging import RaySchedulerLogger


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


def _vram_resource_units(estimated_vram_gb: float) -> int:
    value = float(estimated_vram_gb)
    if value <= 0:
        return 0
    # This Ray build requires whole-number custom resources.
    return int(math.ceil(value))


def run_ray_jobs(
    args: argparse.Namespace,
    jobs: List[Dict[str, Any]],
    base_train_common: List[str],
    status_file_path: str,
) -> None:
    if args.ray_gpus_per_node <= 0:
        raise ValueError("--ray-gpus-per-node must be >= 1")
    if args.cpu_per_experiment <= 0:
        raise ValueError("--cpu-per-experiment must be >= 1")

    logger = RaySchedulerLogger(args.log_dir)

    try:
        try:
            import ray
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        except Exception as exc:
            raise RuntimeError(
                "Ray mode selected, but Ray is not installed. Install with `pip install ray`."
            ) from exc
        try:
            from ray.exceptions import GetTimeoutError as RayGetTimeoutError
        except Exception:
            RayGetTimeoutError = TimeoutError

        if args.ray_nodes_per_exp != 1:
            logger.info(
                f"Ray single-node mode: overriding --ray-nodes-per-exp={args.ray_nodes_per_exp} to 1."
            )

        ray_nodes_per_exp = 1

        ray.init(address=args.ray_address, ignore_reinit_error=True)
        cluster = ray.cluster_resources()
        logger.info(f"Connected to Ray cluster. Resources: {cluster}")

        vram_resource_name = "VRAM_GB"
        total_cluster_vram_units = int(float(cluster.get(vram_resource_name, 0.0)))
        all_requested_vram_units = [_vram_resource_units(float(job.get("estimated_vram_gb", 0.0))) for job in jobs]
        all_requested_vram_units = [v for v in all_requested_vram_units if v > 0]
        if all_requested_vram_units and total_cluster_vram_units <= 0:
            raise ValueError(
                "VRAM scheduling enabled by per-experiment estimation, "
                f"but cluster resource '{vram_resource_name}' is unavailable."
            )

        logger.info(
            "Ray scheduling config: "
            f"nodes/exp={ray_nodes_per_exp}, gpus/node={args.ray_gpus_per_node}, "
            f"cpu/exp={args.cpu_per_experiment}, strategy={args.ray_strategy}, "
            "parallelism=dynamic"
        )
        if all_requested_vram_units:
            logger.info(
                "Ray VRAM scheduling: "
                f"resource={vram_resource_name}, total={total_cluster_vram_units}, "
                f"min_per_exp={min(all_requested_vram_units)}, max_per_exp={max(all_requested_vram_units)} "
                "(integer units)"
            )

        @ray.remote(max_retries=0)
        def _run_torchrun(cmd: List[str], env_updates: Dict[str, str], log_path: str) -> int:
            def _pick_free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])

            env = os.environ.copy()
            env.update(env_updates)
            cmd_local = list(cmd)
            if "--master_port" not in cmd_local:
                try:
                    train_idx = cmd_local.index("train.py")
                except ValueError:
                    train_idx = 1
                cmd_local[train_idx:train_idx] = ["--master_port", str(_pick_free_port())]

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
                    cmd_local,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                return int(proc.wait())

        jobs_sorted = sorted(
            jobs,
            key=lambda job: float(job.get("estimated_vram_gb", 0.0)),
            reverse=True,
        )
        exp_queue = deque(jobs_sorted)
        running: Dict[Any, Dict[str, Any]] = {}
        failures: List[str] = []
        pg_probe_timeout_seconds = max(0.1, min(float(args.ray_pg_timeout_seconds), 5.0))

        def _schedule_one(job: Dict[str, Any]) -> str:
            exp_name = str(job["name"])
            exp_config = str(job["config"])
            exp_extra = list(job.get("extra", []))
            runner_extra = list(job.get("runner_extra", []))
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            expected_output_dir = os.path.join(args.output_root, exp_name)
            log_path = os.path.join(args.log_dir, f"{exp_name}_{ts}.txt")
            estimated_vram_gb = float(job.get("estimated_vram_gb", 0.0))
            requested_vram_units = _vram_resource_units(estimated_vram_gb)
            vram_factors = dict(job.get("vram_factors", {}))

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
            cmd_str = " ".join(cmd)

            if args.dry_run:
                logger.info(f"=== Dry run [{exp_name}] (Ray single-node):")
                logger.info(cmd_str)
                if requested_vram_units > 0:
                    logger.info(
                        "VRAM budget per experiment: "
                        f"estimated={estimated_vram_gb:.3f}GB, requested={requested_vram_units} "
                        f"({vram_resource_name})"
                    )
                    logger.info(f"VRAM factors: {vram_factors}")
                logger.info(f"Expected output: {expected_output_dir}")
                logger.info(f"Log: {log_path}")
                return "scheduled"

            bundle: Dict[str, float] = {"CPU": float(args.cpu_per_experiment)}
            if requested_vram_units > 0:
                bundle[vram_resource_name] = requested_vram_units

            pg = placement_group(bundles=[bundle], strategy=args.ray_strategy)
            try:
                ray.get(pg.ready(), timeout=pg_probe_timeout_seconds)
            except RayGetTimeoutError:
                remove_placement_group(pg)
                return "pending_capacity"
            except Exception as exc:
                logger.info(f"!!! Experiment {exp_name} FAILED: unable to reserve placement group: {exc}")
                remove_placement_group(pg)
                failures.append(exp_name)
                return "failed"

            scheduling = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            env_updates = {
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(args.ray_gpus_per_node)),
            }
            task_options: Dict[str, Any] = dict(
                num_gpus=0,
                num_cpus=args.cpu_per_experiment,
                scheduling_strategy=scheduling,
            )
            if requested_vram_units > 0:
                task_options["resources"] = {vram_resource_name: requested_vram_units}

            ref = _run_torchrun.options(**task_options).remote(cmd, env_updates, log_path)
            running[ref] = {
                "name": exp_name,
                "pg": pg,
                "requested_vram_units": int(requested_vram_units),
                "estimated_vram_gb": float(estimated_vram_gb),
                "start_time": time.time(),
                "log_path": log_path,
                "output_dir": expected_output_dir,
                "config": exp_config,
                "task_ref": str(ref),
                "command": cmd_str,
            }

            logger.event(
                "RAY_LAUNCH",
                {
                    "experiment": exp_name,
                    "task_ref": str(ref),
                    "config": exp_config,
                    "requested_vram_units": int(requested_vram_units),
                    "estimated_vram_gb": round(float(estimated_vram_gb), 3),
                    "command": cmd_str,
                    "log_path": log_path,
                    "output_dir": expected_output_dir,
                },
            )
            return "scheduled"

        try:
            while exp_queue or running:
                scheduled_any = False
                if exp_queue:
                    pending_count = len(exp_queue)
                    for _ in range(pending_count):
                        job = exp_queue.popleft()
                        status = _schedule_one(job)
                        if status == "scheduled":
                            scheduled_any = True
                            if args.stagger_seconds > 0:
                                time.sleep(args.stagger_seconds)
                        elif status == "pending_capacity":
                            exp_queue.append(job)

                if exp_queue and not running and not scheduled_any:
                    pending_names = [str(job.get("name", "<unknown>")) for job in exp_queue]
                    raise RuntimeError(
                        "No pending experiment can be scheduled with current Ray resources. "
                        f"Unschedulable jobs: {pending_names}"
                    )

                if not running:
                    continue

                ready, _ = ray.wait(list(running.keys()), num_returns=1, timeout=2)
                if not ready:
                    continue

                ref = ready[0]
                info = running.pop(ref)
                exp_name = str(info["name"])
                pg = info["pg"]
                start_time = float(info.get("start_time", time.time()))
                duration_seconds = max(0.0, time.time() - start_time)
                term_status = "unknown"
                exit_code: Optional[int] = None
                error_text = ""
                try:
                    ret = int(ray.get(ref))
                    exit_code = int(ret)
                    if ret == 0:
                        term_status = "success"
                    else:
                        term_status = "failed_exit_code"
                        failures.append(exp_name)
                except Exception as exc:
                    term_status = "failed_exception"
                    error_text = str(exc)
                    failures.append(exp_name)
                finally:
                    remove_placement_group(pg)
                    logger.event(
                        "RAY_TERMINATE",
                        {
                            "experiment": exp_name,
                            "task_ref": str(info.get("task_ref", ref)),
                            "status": term_status,
                            "exit_code": exit_code,
                            "error": error_text,
                            "duration_seconds": round(duration_seconds, 2),
                            "requested_vram_units": int(info.get("requested_vram_units", 0)),
                            "estimated_vram_gb": round(float(info.get("estimated_vram_gb", 0.0)), 3),
                            "command": str(info.get("command", "")),
                            "log_path": str(info.get("log_path", "")),
                            "output_dir": str(info.get("output_dir", "")),
                        },
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted! Cancelling running Ray jobs...")
            for ref in list(running.keys()):
                ray.cancel(ref, force=True)
            for ref, info in list(running.items()):
                exp_name = str(info.get("name", "<unknown>"))
                try:
                    ray.get(ref, timeout=1)
                except Exception:
                    pass
                remove_placement_group(info["pg"])
                logger.event(
                    "RAY_TERMINATE",
                    {
                        "experiment": exp_name,
                        "task_ref": str(info.get("task_ref", ref)),
                        "status": "cancelled",
                        "exit_code": None,
                        "error": "cancelled by KeyboardInterrupt",
                        "duration_seconds": round(max(0.0, time.time() - float(info.get("start_time", time.time()))), 2),
                        "requested_vram_units": int(info.get("requested_vram_units", 0)),
                        "estimated_vram_gb": round(float(info.get("estimated_vram_gb", 0.0)), 3),
                        "command": str(info.get("command", "")),
                        "log_path": str(info.get("log_path", "")),
                        "output_dir": str(info.get("output_dir", "")),
                    },
                )
            running.clear()
            logger.info("Done.")
        finally:
            ray.shutdown()
            logger.info("Ray shutdown complete.")

        if failures:
            logger.info(f"Ray mode completed with failures: {sorted(set(failures))}")
        else:
            logger.info("Ray mode completed successfully with no failures.")
    finally:
        logger.close()
