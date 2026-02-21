#!/usr/bin/env python3
"""
Batch runner for run.py using a job config file.

Example:
  python run_batch.py --jobs jobs.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_jobs(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML job files") from exc
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    raise ValueError("Job file must be .json or .yaml/.yml")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _join_list(values: List[str]) -> str:
    return ",".join(v for v in values if v)


def _build_command(job: Dict[str, Any], shared: Dict[str, Any], run_py: str) -> List[str]:
    merged = dict(shared)
    merged.update(job)

    job_name = str(merged.get("name", "")).strip()
    configs = _join_list(_as_list(merged.get("configs")))
    if not configs:
        raise ValueError(f"Job '{job_name}' is missing configs")

    timm_model = merged.get("timm_model")
    if not timm_model:
        raise ValueError(f"Job '{job_name}' is missing timm_model")

    work_dir = merged.get("work_dir", "output_test/mmdet_eval")
    if job_name:
        work_dir = os.path.join(work_dir, job_name)

    results_file = merged.get("results_file")
    if not results_file:
        results_file = os.path.join(work_dir, "results.jsonl")

    cmd = [
        sys.executable,
        run_py,
        "--configs",
        configs,
        "--timm-model",
        str(timm_model),
        "--work-dir",
        work_dir,
        "--results-file",
        results_file,
    ]

    checkpoints = _join_list(_as_list(merged.get("backbone_checkpoints")))
    if checkpoints:
        cmd.extend(["--backbone-checkpoints", checkpoints])

    det_ckpt = merged.get("det_checkpoint")
    if det_ckpt:
        cmd.extend(["--det-checkpoint", str(det_ckpt)])

    if merged.get("timm_pretrained"):
        cmd.append("--timm-pretrained")

    out_indices = merged.get("out_indices")
    if out_indices is not None and str(out_indices).strip() != "":
        cmd.extend(["--out-indices", str(out_indices)])

    for key, flag in (
        ("data_root", "--data-root"),
        ("device", "--device"),
        ("run", "--run"),
        ("dump_config", "--dump-config"),
    ):
        val = merged.get(key)
        if val:
            cmd.extend([flag, str(val)])

    for key, flag in (
        ("batch_size", "--batch-size"),
        ("num_workers", "--num-workers"),
    ):
        val = merged.get(key)
        if val is not None:
            cmd.extend([flag, str(val)])

    if merged.get("show_config"):
        cmd.append("--show-config")
    if merged.get("fail_fast"):
        cmd.append("--fail-fast")

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch runner for object detection jobs")
    parser.add_argument("--jobs", required=True, type=str, help="Path to jobs.yaml or jobs.json")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    config = _load_jobs(args.jobs)
    shared = config.get("shared", {})
    jobs = config.get("jobs", [])
    if not jobs:
        raise ValueError("No jobs found in config file")

    run_py = str(Path(__file__).parent / "run.py")
    for job in jobs:
        cmd = _build_command(job, shared, run_py)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.check_call(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
