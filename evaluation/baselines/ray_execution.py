from __future__ import annotations

import argparse
import os
import resource
import socket
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional
try:
    from .ray_scheduler_logging import RaySchedulerLogger
except ImportError:
    from ray_scheduler_logging import RaySchedulerLogger

GROUP_RESOURCE_PREFIX = "GPU_GROUP"


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


def _discover_gpu_group_resources(cluster_resources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract group slot resources from Ray cluster resources.

    Simple mapping:
        {"GPU_GROUP_0_SLOTS": 2.0}
        -> [{"index": 0, "slots_resource": "GPU_GROUP_0_SLOTS", "slots_total": 2.0}]

    Example input:
        cluster_resources = {
            "CPU": 64.0,
            "GPU_GROUP_0_SLOTS": 1.0,
            "GPU_GROUP_1_SLOTS": 1.0,
        }

    Example output:
        [
            {
                "index": 0,
                "slots_resource": "GPU_GROUP_0_SLOTS",
                "slots_total": 1.0,
            },
            {
                "index": 1,
                "slots_resource": "GPU_GROUP_1_SLOTS",
                "slots_total": 1.0,
            },
        ]
    """
    prefix = f"{GROUP_RESOURCE_PREFIX}_"
    slot_suffix = "_SLOTS"
    grouped: Dict[int, Dict[str, Any]] = {}

    for resource_name, raw_capacity in cluster_resources.items():
        if not resource_name.startswith(prefix):
            continue
        try:
            capacity = float(raw_capacity)
        except (TypeError, ValueError):
            continue
        if capacity <= 0:
            continue

        if resource_name.endswith(slot_suffix):
            idx_text = resource_name[len(prefix):-len(slot_suffix)]
            if idx_text.isdigit():
                idx = int(idx_text)
                item = grouped.setdefault(idx, {"index": idx})
                item["slots_resource"] = resource_name
                item["slots_total"] = capacity

    discovered: List[Dict[str, Any]] = []
    for idx in sorted(grouped):
        item = grouped[idx]
        if "slots_resource" not in item:
            continue
        discovered.append(item)
    return discovered


def _group_gpu_ids(group_index: int, gpus_per_group: int) -> List[str]:
    start = int(group_index) * int(gpus_per_group)
    return [str(start + offset) for offset in range(int(gpus_per_group))]


def run_ray_jobs(
    args: argparse.Namespace,
    jobs: List[Dict[str, Any]],
    base_train_common: List[str],
) -> None:
    ray_gpus_per_node = int(args.ray_gpus_per_node)
    ray_gpus_per_group = int(getattr(args, "ray_gpus_per_group", ray_gpus_per_node))
    max_experiments_per_group = int(getattr(args, "max_experiments_per_group", 1))
    group_resource_prefix = GROUP_RESOURCE_PREFIX

    if ray_gpus_per_node <= 0:
        raise ValueError("--ray-gpus-per-node must be >= 1")
    if ray_gpus_per_group <= 0:
        raise ValueError("--ray-gpus-per-group must be >= 1")
    if ray_gpus_per_node % ray_gpus_per_group != 0:
        raise ValueError(
            f"--ray-gpus-per-node ({ray_gpus_per_node}) must be divisible by "
            f"--ray-gpus-per-group ({ray_gpus_per_group})"
        )
    if max_experiments_per_group <= 0:
        raise ValueError("--max-experiments-per-group must be >= 1")
    if args.cpu_per_experiment <= 0:
        raise ValueError("--cpu-per-experiment must be >= 1")
    ray_job_max_retries = int(getattr(args, "ray_job_max_retries", 0))
    ray_retry_backoff_seconds = int(getattr(args, "ray_retry_backoff_seconds", 0))
    if ray_job_max_retries < 0:
        raise ValueError("--ray-job-max-retries must be >= 0")
    if ray_retry_backoff_seconds < 0:
        raise ValueError("--ray-retry-backoff-seconds must be >= 0")

    logger = RaySchedulerLogger(args.log_dir)

    try:
        try:
            import ray
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util import placement_group_table
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
        logger.info(f"Initializing Ray (address={args.ray_address})")
        try:
            ray.init(address=args.ray_address)
        except Exception as init_exc:
            raise RuntimeError(
                "Unable to connect to Ray cluster via ray.init() "
                f"to address={args.ray_address}."
            ) from init_exc

        cluster = ray.cluster_resources()
        logger.info(f"Connected to Ray cluster. Resources: {cluster}")

        gpu_group_resources = _discover_gpu_group_resources(cluster)
        gpu_group_cycle: Deque[Dict[str, Any]] = deque(gpu_group_resources)

        if not gpu_group_resources:
            raise ValueError(
                "Group-aware scheduling requested, but no group resources were discovered. "
                f"Expected resources named like '{group_resource_prefix}_<group_idx>_SLOTS'."
            )

        logger.info(
            "Ray scheduling config: "
            f"nodes/exp={ray_nodes_per_exp}, gpus/node={ray_gpus_per_node}, "
            f"gpus/group={ray_gpus_per_group}, max_exp/group={max_experiments_per_group}, "
            f"cpu/exp={args.cpu_per_experiment}, strategy={args.ray_strategy}, "
            f"max_retries={ray_job_max_retries}, retry_backoff={ray_retry_backoff_seconds}s, "
            "parallelism=dynamic"
        )
        logger.info(
            f"Ray group scheduling enabled: prefix={group_resource_prefix}, discovered_groups={len(gpu_group_resources)}"
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

        exp_queue: Deque[Dict[str, Any]] = deque()
        for job in jobs:
            first_attempt_job = dict(job)
            first_attempt_job["attempt"] = 1
            first_attempt_job["not_before"] = 0.0
            exp_queue.append(first_attempt_job)
        running: Dict[Any, Dict[str, Any]] = {}
        failures: List[str] = []
        pg_probe_timeout_seconds = max(0.1, min(float(args.ray_pg_timeout_seconds), 5.0))
        pg_remove_wait_seconds = max(0.1, min(30.0, float(args.ray_pg_timeout_seconds)))

        def _max_attempts() -> int:
            return ray_job_max_retries + 1

        def _wait_pg_removed(pg: Any, timeout_seconds: float) -> bool:
            deadline = time.time() + max(0.1, float(timeout_seconds))
            while time.time() < deadline:
                try:
                    table = placement_group_table(pg)
                except Exception:
                    table = {}
                if not table:
                    return True
                state = str(table.get("state") or "").strip().upper()
                if state == "REMOVED":
                    return True
                time.sleep(5.0)
            return False

        def _requeue_failed_job(
            job: Dict[str, Any],
            *,
            failure_status: str,
            exit_code: Optional[int] = None,
            error_text: str = "",
            log_path: str = "",
        ) -> bool:
            exp_name = str(job.get("name", "<unknown>"))
            attempt = int(job.get("attempt", 1))
            max_attempts = _max_attempts()

            if attempt < max_attempts:
                retry_job = dict(job)
                retry_job["attempt"] = attempt + 1
                if ray_retry_backoff_seconds > 0:
                    retry_job["not_before"] = time.time() + float(ray_retry_backoff_seconds)
                else:
                    retry_job["not_before"] = 0.0
                exp_queue.append(retry_job)
                logger.event(
                    "RAY_RETRY",
                    {
                        "experiment": exp_name,
                        "attempt": attempt,
                        "next_attempt": int(retry_job["attempt"]),
                        "max_attempts": max_attempts,
                        "failure_status": failure_status,
                        "exit_code": exit_code,
                        "error": error_text,
                        "log_path": log_path,
                        "backoff_seconds": ray_retry_backoff_seconds,
                    },
                )
                return True

            failures.append(exp_name)
            logger.event(
                "RAY_RETRY_EXHAUSTED",
                {
                    "experiment": exp_name,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "failure_status": failure_status,
                    "exit_code": exit_code,
                    "error": error_text,
                    "log_path": log_path,
                },
            )
            return False

        def _schedule_one(job: Dict[str, Any]) -> str:
            exp_name = str(job["name"])
            exp_config = str(job["config"])
            exp_extra = list(job.get("extra", []))
            runner_extra = list(job.get("runner_extra", []))
            attempt = int(job.get("attempt", 1))
            max_attempts = _max_attempts()
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            expected_output_dir = os.path.join(args.output_root, exp_name)
            log_path = os.path.join(args.log_dir, f"{exp_name}_{ts}.txt")
            exp_status_file = str(job.get("status_file") or os.path.join(expected_output_dir, "run_status.yaml"))

            if not args.disable_train_check_resume:
                _upsert_flag(exp_extra, "--check-resume")
                _upsert_flag(exp_extra, "--check-resume-log-dir", args.log_dir)
                _upsert_flag(exp_extra, "--check-resume-status-file", exp_status_file)
            cmd = [
                "torchrun",
                "--nnodes=1",
                f"--nproc_per_node={ray_gpus_per_group}",
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
                logger.info(f"=== Dry run [{exp_name}] attempt={attempt}/{max_attempts} (Ray single-node):")
                logger.info(cmd_str)
                logger.info(
                    f"GPU group placement: prefix={group_resource_prefix}, "
                    f"group_size={ray_gpus_per_group}, candidate_groups={len(gpu_group_resources)}"
                )
                logger.info(f"Expected output: {expected_output_dir}")
                logger.info(f"Log: {log_path}")
                return "scheduled"

            selected_pg = None
            selected_group_index: Optional[int] = None
            selected_group_slots_resource = ""
            selected_group_gpu_ids = [str(i) for i in range(ray_gpus_per_group)]
            task_resources: Dict[str, float] = {}

            if not gpu_group_cycle:
                logger.info(
                    f"!!! Experiment {exp_name} FAILED: no GPU group resources found for prefix '{group_resource_prefix}'."
                )
                return "failed"

            attempts = len(gpu_group_cycle)
            for _ in range(attempts):
                group = gpu_group_cycle[0]
                gpu_group_cycle.rotate(-1)

                group_index = int(group.get("index", -1))
                slot_resource = str(group["slots_resource"])

                bundle: Dict[str, float] = {
                    "CPU": float(args.cpu_per_experiment),
                    slot_resource: 1.0,
                }
                candidate_task_resources: Dict[str, float] = {slot_resource: 1.0}

                pg = placement_group(bundles=[bundle], strategy=args.ray_strategy)
                try:
                    ray.get(pg.ready(), timeout=pg_probe_timeout_seconds)
                except RayGetTimeoutError:
                    remove_placement_group(pg)
                    continue
                except Exception as exc:
                    logger.info(
                        f"!!! Experiment {exp_name} FAILED: unable to reserve placement group for "
                        f"group {group_index}: {exc}"
                    )
                    remove_placement_group(pg)
                    return "failed"

                selected_pg = pg
                selected_group_index = group_index
                selected_group_slots_resource = slot_resource
                selected_group_gpu_ids = _group_gpu_ids(group_index, ray_gpus_per_group)
                task_resources = candidate_task_resources
                break

            if selected_pg is None:
                return "pending_capacity"

            scheduling = PlacementGroupSchedulingStrategy(
                placement_group=selected_pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            env_updates = {
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": ",".join(selected_group_gpu_ids),
            }
            task_options: Dict[str, Any] = dict(
                num_gpus=0,
                num_cpus=args.cpu_per_experiment,
                scheduling_strategy=scheduling,
            )
            if task_resources:
                task_options["resources"] = dict(task_resources)

            ref = _run_torchrun.options(**task_options).remote(cmd, env_updates, log_path)
            running[ref] = {
                "name": exp_name,
                "pg": selected_pg,
                "start_time": time.time(),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "log_path": log_path,
                "output_dir": expected_output_dir,
                "config": exp_config,
                "task_ref": str(ref),
                "command": cmd_str,
                "group_index": selected_group_index,
                "group_slots_resource": selected_group_slots_resource,
                "cuda_visible_devices": env_updates["CUDA_VISIBLE_DEVICES"],
                "job": dict(job),
            }

            logger.event(
                "RAY_LAUNCH",
                {
                    "experiment": exp_name,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "task_ref": str(ref),
                    "config": exp_config,
                    "command": cmd_str,
                    "log_path": log_path,
                    "output_dir": expected_output_dir,
                    "group_index": selected_group_index,
                    "group_slots_resource": selected_group_slots_resource,
                    "cuda_visible_devices": env_updates["CUDA_VISIBLE_DEVICES"],
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
                        not_before = float(job.get("not_before", 0.0))
                        if not_before > time.time():
                            exp_queue.append(job)
                            continue
                        status = _schedule_one(job)
                        if status == "scheduled":
                            scheduled_any = True
                            if args.stagger_seconds > 0:
                                time.sleep(args.stagger_seconds)
                        elif status == "pending_capacity":
                            exp_queue.append(job)
                        elif status == "failed":
                            _requeue_failed_job(job, failure_status="scheduling_failed")
                        else:
                            raise RuntimeError(f"Unexpected scheduler status '{status}' for job {job.get('name')}")

                if exp_queue and not running and not scheduled_any:
                    earliest_not_before = min(float(job.get("not_before", 0.0)) for job in exp_queue)
                    if earliest_not_before > time.time():
                        sleep_for = max(0.0, min(1.0, earliest_not_before - time.time()))
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        continue
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
                will_retry = False
                try:
                    ret = int(ray.get(ref))
                    exit_code = int(ret)
                    if ret == 0:
                        term_status = "success"
                    else:
                        term_status = "failed_exit_code"
                        will_retry = _requeue_failed_job(
                            info.get("job", {"name": exp_name, "attempt": info.get("attempt", 1)}),
                            failure_status=term_status,
                            exit_code=exit_code,
                            log_path=str(info.get("log_path", "")),
                        )
                except Exception as exc:
                    term_status = "failed_exception"
                    error_text = str(exc)
                    will_retry = _requeue_failed_job(
                        info.get("job", {"name": exp_name, "attempt": info.get("attempt", 1)}),
                        failure_status=term_status,
                        exit_code=exit_code,
                        error_text=error_text,
                        log_path=str(info.get("log_path", "")),
                    )
                finally:
                    remove_placement_group(pg)
                    terminate_payload = {
                        "experiment": exp_name,
                        "attempt": int(info.get("attempt", 1)),
                        "max_attempts": int(info.get("max_attempts", _max_attempts())),
                        "task_ref": str(info.get("task_ref", ref)),
                        "status": term_status,
                        "exit_code": exit_code,
                        "error": error_text,
                        "will_retry": will_retry,
                        "duration_seconds": round(duration_seconds, 2),
                        "command": str(info.get("command", "")),
                        "log_path": str(info.get("log_path", "")),
                        "output_dir": str(info.get("output_dir", "")),
                        "group_index": info.get("group_index"),
                        "group_slots_resource": str(info.get("group_slots_resource", "")),
                        "cuda_visible_devices": str(info.get("cuda_visible_devices", "")),
                    }
                    if _wait_pg_removed(pg, pg_remove_wait_seconds):
                        logger.event("RAY_TERMINATE", terminate_payload)
                    else:
                        pending_payload = dict(terminate_payload)
                        pending_payload["pg_remove_wait_seconds"] = pg_remove_wait_seconds
                        logger.event("RAY_TERMINATE_PENDING", pending_payload)

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
                terminate_payload = {
                    "experiment": exp_name,
                    "task_ref": str(info.get("task_ref", ref)),
                    "status": "cancelled",
                    "exit_code": None,
                    "error": "cancelled by KeyboardInterrupt",
                    "duration_seconds": round(max(0.0, time.time() - float(info.get("start_time", time.time()))), 2),
                    "command": str(info.get("command", "")),
                    "log_path": str(info.get("log_path", "")),
                    "output_dir": str(info.get("output_dir", "")),
                    "group_index": info.get("group_index"),
                    "group_slots_resource": str(info.get("group_slots_resource", "")),
                    "cuda_visible_devices": str(info.get("cuda_visible_devices", "")),
                }
                if _wait_pg_removed(info["pg"], pg_remove_wait_seconds):
                    logger.event("RAY_TERMINATE", terminate_payload)
                else:
                    pending_payload = dict(terminate_payload)
                    pending_payload["pg_remove_wait_seconds"] = pg_remove_wait_seconds
                    logger.event("RAY_TERMINATE_PENDING", pending_payload)
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
