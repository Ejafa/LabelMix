#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Required environment variables (must be provided by Taiji runtime or caller):
# NODE_IP_LIST, NODE_IP, TAIJI_HOST_NUM, HOST_GPU_NUM
#
# User-configurable knobs (optional overrides; defaults are provided):
NPROC="${NPROC:-}"
INITIAL_NODE_STAGGER_SECONDS="${INITIAL_NODE_STAGGER_SECONDS:-30}"
EXPERIMENT_STAGGER_SECONDS="${EXPERIMENT_STAGGER_SECONDS:-0}"
ENABLE_MULTI_NODE_STAGGER="${ENABLE_MULTI_NODE_STAGGER:-0}"

# Scheduler/Ray knobs (optional overrides):
SCHEDULER_MODE="${SCHEDULER_MODE:-ray}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_HEAD_IP="${RAY_HEAD_IP:-${CHIEF_IP:-}}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(nproc)}"
RAY_GPUS_PER_NODE="${RAY_GPUS_PER_NODE:-}"
RAY_CPUS_PER_NODE="${RAY_CPUS_PER_NODE:-32}"
RAY_STRATEGY="${RAY_STRATEGY:-STRICT_SPREAD}"
RAY_MAX_PARALLEL="${RAY_MAX_PARALLEL:-0}"

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "ERROR: NODE_IP_LIST is not set" >&2
  exit 1
fi
if [[ -z "${NODE_IP:-}" ]]; then
  echo "ERROR: NODE_IP is not set" >&2
  exit 1
fi

IFS=',' read -ra ITEMS <<< "$NODE_IP_LIST"

NODE_RANK=""
for i in "${!ITEMS[@]}"; do
  ip="${ITEMS[$i]%%:*}" # strip ":8"
  if [[ "$ip" == "$NODE_IP" ]]; then
    NODE_RANK="$i"
    break
  fi
done

if [[ -z "$NODE_RANK" ]]; then
  echo "ERROR: NODE_IP=$NODE_IP not found in NODE_IP_LIST=$NODE_IP_LIST" >&2
  exit 1
fi

if [[ -z "${TAIJI_HOST_NUM:-}" ]]; then
  echo "ERROR: TAIJI_HOST_NUM is not set" >&2
  exit 1
fi
if [[ -z "${HOST_GPU_NUM:-}" ]]; then
  echo "ERROR: HOST_GPU_NUM is not set" >&2
  exit 1
fi

if [[ -z "${NPROC}" ]]; then
  NPROC="${HOST_GPU_NUM}"
fi
if [[ -z "${RAY_GPUS_PER_NODE}" ]]; then
  RAY_GPUS_PER_NODE="${HOST_GPU_NUM}"
fi
if [[ -z "${RAY_HEAD_IP}" ]]; then
  RAY_HEAD_IP="${ITEMS[0]%%:*}"
fi
if [[ -z "${RAY_ADDRESS}" ]]; then
  RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: neither python nor python3 is available in PATH." >&2
  exit 1
fi

HELPER_PY="${SCRIPT_DIR}/start_helpers.py"
if [[ ! -f "${HELPER_PY}" ]]; then
  echo "ERROR: helper file not found: ${HELPER_PY}" >&2
  exit 1
fi

if [[ "$INITIAL_NODE_STAGGER_SECONDS" -gt 0 && "$NODE_RANK" -gt 0 ]]; then
  delay=$((INITIAL_NODE_STAGGER_SECONDS * NODE_RANK))
  echo "Node rank ${NODE_RANK}: initial one-time sleep ${delay}s to smooth shared cache access."
  sleep "$delay"
fi

if [[ "$SCHEDULER_MODE" == "local" ]]; then
  cmd=(
    "${PYTHON_BIN}" evaluation/baselines/run_experiments.py
    --scheduler local
    --gpu-nodes "${TAIJI_HOST_NUM}"
    --gpu-per-node "${HOST_GPU_NUM}"
    --nproc "${NPROC}"
    --node-index "${NODE_RANK}"
    --stagger-seconds "${EXPERIMENT_STAGGER_SECONDS}"
  )

  if [[ "$ENABLE_MULTI_NODE_STAGGER" == "1" ]]; then
    cmd+=(--multi-node-stagger)
  fi

  cmd+=("$@")
  "${cmd[@]}"
  exit 0
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "ERROR: ray command not found. Install Ray in the environment first." >&2
  exit 1
fi

echo "Node rank ${NODE_RANK}: restarting Ray runtime."
ray stop --force >/dev/null 2>&1 || true

if [[ "$NODE_RANK" == "0" ]]; then
  echo "Node rank 0: starting Ray head at ${NODE_IP}:${RAY_PORT}"
  ray start --head \
    --node-ip-address "${NODE_IP}" \
    --port "${RAY_PORT}" \
    --num-gpus "${HOST_GPU_NUM}" \
    --num-cpus "${RAY_NUM_CPUS}"

  echo "Node rank 0: waiting for Ray cluster readiness at ${RAY_ADDRESS}"
  "${PYTHON_BIN}" "${HELPER_PY}" wait-ray \
    --address "${RAY_ADDRESS}" \
    --timeout-seconds 120

  cmd=(
    "${PYTHON_BIN}" evaluation/baselines/run_experiments.py
    --scheduler ray
    --ray-address "${RAY_ADDRESS}"
    --ray-gpus-per-node "${RAY_GPUS_PER_NODE}"
    --ray-cpus-per-node "${RAY_CPUS_PER_NODE}"
    --ray-strategy "${RAY_STRATEGY}"
    --ray-max-parallel "${RAY_MAX_PARALLEL}"
    --gpu-nodes "${TAIJI_HOST_NUM}"
    --gpu-per-node "${HOST_GPU_NUM}"
    --nproc "${RAY_GPUS_PER_NODE}"
    --stagger-seconds "${EXPERIMENT_STAGGER_SECONDS}"
  )

  cmd+=("$@")
  "${cmd[@]}"
else
  echo "Node rank ${NODE_RANK}: waiting for Ray head ${RAY_HEAD_IP}:${RAY_PORT}."
  "${PYTHON_BIN}" "${HELPER_PY}" wait-tcp \
    --host "${RAY_HEAD_IP}" \
    --port "${RAY_PORT}" \
    --timeout-seconds 180

  echo "Node rank ${NODE_RANK}: starting Ray worker to ${RAY_ADDRESS} (blocking)."
  ray start \
    --address "${RAY_ADDRESS}" \
    --node-ip-address "${NODE_IP}" \
    --num-gpus "${HOST_GPU_NUM}" \
    --num-cpus "${RAY_NUM_CPUS}" \
    --block
fi
