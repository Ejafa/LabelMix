#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Required environment variables (must be provided by Taiji runtime or caller):
# NODE_IP_LIST, NODE_IP, TAIJI_HOST_NUM, HOST_GPU_NUM
#
# User-configurable knobs (optional overrides; defaults are provided):
INITIAL_NODE_STAGGER_SECONDS="${INITIAL_NODE_STAGGER_SECONDS:-30}"
EXPERIMENT_STAGGER_SECONDS="${EXPERIMENT_STAGGER_SECONDS:-15}"

# Auto-detected capacity knobs (leave empty/default to auto-detect):
# - NODE_CPU_COUNT: local logical CPU count
# - RAY_VRAM_GB_PER_NODE: auto-detected from nvidia-smi
NODE_CPU_COUNT="$(nproc)"
CPU_PER_EXPERIMENT="${CPU_PER_EXPERIMENT:-${RAY_CPUS_PER_NODE:-8}}"
RAY_VRAM_RESERVE_GB="${RAY_VRAM_RESERVE_GB:-3}"

# Ray execution knobs (manual defaults):
RAY_PORT="${RAY_PORT:-6379}"
RAY_STRATEGY="${RAY_STRATEGY:-STRICT_SPREAD}"

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

RAY_HEAD_IP="${CHIEF_IP:-${ITEMS[0]%%:*}}"
RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"

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

RAY_VRAM_GB_PER_NODE=""
if detected_vram="$("${PYTHON_BIN}" "${HELPER_PY}" detect-vram-gb \
  --mode "free" \
  --aggregate "min" \
  --reserve-gb "${RAY_VRAM_RESERVE_GB}" 2>/dev/null)"; then
  RAY_VRAM_GB_PER_NODE="${detected_vram}"
else
  echo "Warning: could not auto-detect node VRAM budget; VRAM resource scheduling disabled."
fi

RAY_RESOURCES_JSON=""
if [[ -n "${RAY_VRAM_GB_PER_NODE}" ]]; then
  if awk "BEGIN {exit !(${RAY_VRAM_GB_PER_NODE} > 0)}"; then
    RAY_RESOURCES_JSON="{\"VRAM_GB\":${RAY_VRAM_GB_PER_NODE}}"
    echo "Ray custom resource budget: ${RAY_RESOURCES_JSON}"
  else
    echo "Warning: invalid RAY_VRAM_GB_PER_NODE='${RAY_VRAM_GB_PER_NODE}', ignoring VRAM resource."
    RAY_VRAM_GB_PER_NODE=""
  fi
fi

if [[ "$INITIAL_NODE_STAGGER_SECONDS" -gt 0 && "$NODE_RANK" -gt 0 ]]; then
  delay=$((INITIAL_NODE_STAGGER_SECONDS * NODE_RANK))
  echo "Node rank ${NODE_RANK}: initial one-time sleep ${delay}s to smooth shared cache access."
  sleep "$delay"
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "ERROR: ray command not found. Install Ray in the environment first." >&2
  exit 1
fi

echo "Node rank ${NODE_RANK}: restarting Ray runtime."
ray stop --force >/dev/null 2>&1 || true

if [[ "$NODE_RANK" == "0" ]]; then
  echo "Node rank 0: starting Ray head at ${NODE_IP}:${RAY_PORT}"
  ray_head_cmd=(ray start --head \
    --node-ip-address "${NODE_IP}" \
    --port "${RAY_PORT}" \
    --num-gpus "${HOST_GPU_NUM}" \
    --num-cpus "${NODE_CPU_COUNT}")
  if [[ -n "${RAY_RESOURCES_JSON}" ]]; then
    ray_head_cmd+=(--resources "${RAY_RESOURCES_JSON}")
  fi
  "${ray_head_cmd[@]}"

  echo "Node rank 0: waiting for Ray cluster readiness at ${RAY_ADDRESS}"
  "${PYTHON_BIN}" "${HELPER_PY}" wait-ray \
    --address "${RAY_ADDRESS}" \
    --timeout-seconds 120

  cmd=(
    "${PYTHON_BIN}" evaluation/baselines/run_experiments.py
    --ray-address "${RAY_ADDRESS}"
    --ray-gpus-per-node "${HOST_GPU_NUM}"
    --cpu-per-experiment "${CPU_PER_EXPERIMENT}"
    --ray-strategy "${RAY_STRATEGY}"
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
  ray_worker_cmd=(ray start \
    --address "${RAY_ADDRESS}" \
    --node-ip-address "${NODE_IP}" \
    --num-gpus "${HOST_GPU_NUM}" \
    --num-cpus "${NODE_CPU_COUNT}" \
    --block)
  if [[ -n "${RAY_RESOURCES_JSON}" ]]; then
    ray_worker_cmd+=(--resources "${RAY_RESOURCES_JSON}")
  fi
  "${ray_worker_cmd[@]}"
fi
