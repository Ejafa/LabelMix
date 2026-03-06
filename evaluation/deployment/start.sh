#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Prefer env-local C++ runtime symbols (GLIBCXX) over base image libs.
SHARED_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEPLOY_ENV_PATH="${DEPLOY_ENV_PATH:-${SHARED_ROOT}/labelmix_env}"
DEPLOY_ENV_BIN="${DEPLOY_ENV_PATH}/bin"
if [[ -d "${DEPLOY_ENV_BIN}" ]]; then
  export PATH="${DEPLOY_ENV_BIN}:${PATH}"
fi
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -d "${DEPLOY_ENV_PATH}/lib" ]]; then
  export LD_LIBRARY_PATH="${DEPLOY_ENV_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Required environment variables (typically provided by Taiji runtime):
# NODE_IP_LIST, NODE_IP
# If NODE_IP_LIST or NODE_IP is missing, we fall back to single-node mode.
#
# User-configurable knobs (optional overrides; defaults are provided):
INITIAL_NODE_STAGGER_SECONDS="${INITIAL_NODE_STAGGER_SECONDS:-30}"
EXPERIMENT_STAGGER_SECONDS="${EXPERIMENT_STAGGER_SECONDS:-15}"

# Auto-detected capacity knobs (leave empty/default to auto-detect):
# - NODE_CPU_COUNT: local logical CPU count
NODE_CPU_COUNT="$(nproc)"
CPU_PER_EXPERIMENT="${CPU_PER_EXPERIMENT:-6}"
# HOST_GPU_NUM=""
RAY_GPUS_PER_GROUP="${RAY_GPUS_PER_GROUP:-2}"
MAX_EXPERIMENTS_PER_GROUP="${MAX_EXPERIMENTS_PER_GROUP:-2}"

# Ray execution knobs (manual defaults):
RAY_PORT="${RAY_PORT:-6379}"
RAY_STRATEGY="${RAY_STRATEGY:-STRICT_SPREAD}"

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  single_node_ip="${CHIEF_IP:-${NODE_IP:-127.0.0.1}}"
  NODE_IP="${single_node_ip}"
  NODE_IP_LIST="${single_node_ip}"
  echo "NODE_IP_LIST missing; assuming single-node deployment: NODE_IP=${NODE_IP}, NODE_IP_LIST=${NODE_IP_LIST}"
fi

IFS=',' read -ra ITEMS <<< "$NODE_IP_LIST"
if [[ "${#ITEMS[@]}" -eq 0 ]]; then
  echo "ERROR: NODE_IP_LIST is empty." >&2
  exit 1
fi

rank_for_ip() {
  local target_ip="${1%%:*}"
  local i ip
  for i in "${!ITEMS[@]}"; do
    ip="${ITEMS[$i]%%:*}" # strip ":<slots>"
    if [[ "$ip" == "$target_ip" ]]; then
      printf '%s\n' "$i"
      return 0
    fi
  done
  return 1
}

RAY_HEAD_IP="${CHIEF_IP:-${ITEMS[0]%%:*}}"
RAY_HEAD_IP="${RAY_HEAD_IP%%:*}"
RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
if ! rank_for_ip "${RAY_HEAD_IP}" >/dev/null; then
  echo "ERROR: head IP ${RAY_HEAD_IP} is not present in NODE_IP_LIST=${NODE_IP_LIST}" >&2
  exit 1
fi

NODE_RANK="0"
if [[ "${INDEX:-}" =~ ^[0-9]+$ ]] && (( INDEX < ${#ITEMS[@]} )); then
  NODE_RANK="${INDEX}"
fi

if [[ -z "${NODE_RANK}" ]]; then
  for candidate_ip in "${LOCAL_IP:-}" "${NODE_IP:-}"; do
    [[ -z "${candidate_ip}" ]] && continue
    if NODE_RANK="$(rank_for_ip "${candidate_ip}")"; then
      break
    fi
  done
fi

if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: could not resolve this node rank from INDEX/LOCAL_IP/NODE_IP against NODE_IP_LIST=${NODE_IP_LIST}" >&2
  exit 1
fi

# Always use the canonical IP from NODE_IP_LIST for this node.
NODE_IP="${ITEMS[$NODE_RANK]%%:*}"

RUN_PROJECT_ROOT="${RUN_PROJECT_ROOT:-${RUN_STORAGE_ROOT:-${PROJECT_ROOT}}}"

if [[ -x "${DEPLOY_ENV_BIN}/python" ]]; then
  PYTHON_BIN="${DEPLOY_ENV_BIN}/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: neither python nor python3 is available in PATH." >&2
  exit 1
fi

if [[ -x "${DEPLOY_ENV_BIN}/ray" ]]; then
  RAY_BIN="${DEPLOY_ENV_BIN}/ray"
elif command -v ray >/dev/null 2>&1; then
  RAY_BIN="$(command -v ray)"
else
  echo "ERROR: ray command not found. Install Ray in the environment first." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -V 2>&1 | head -n1 || true)"
RAY_VERSION="$("${RAY_BIN}" --version 2>&1 | head -n1 || true)"
echo "Runtime binaries: python=${PYTHON_BIN} (${PYTHON_VERSION})"
echo "Runtime binaries: ray=${RAY_BIN} (${RAY_VERSION})"

HELPER_PY="${SCRIPT_DIR}/start_helpers.py"
if [[ ! -f "${HELPER_PY}" ]]; then
  echo "ERROR: helper file not found: ${HELPER_PY}" >&2
  exit 1
fi

if detected_gpu_num="$("${PYTHON_BIN}" "${HELPER_PY}" detect-gpu-count)"; then
  HOST_GPU_NUM="${detected_gpu_num}"
else
  echo "ERROR: failed to detect GPU count for this node." >&2
  exit 1
fi
if (( HOST_GPU_NUM % RAY_GPUS_PER_GROUP != 0 )); then
  echo "ERROR: detected GPU count (${HOST_GPU_NUM}) must be divisible by RAY_GPUS_PER_GROUP (${RAY_GPUS_PER_GROUP})" >&2
  exit 1
fi
if [[ "${MAX_EXPERIMENTS_PER_GROUP}" -le 0 ]]; then
  echo "ERROR: MAX_EXPERIMENTS_PER_GROUP must be >= 1 (got ${MAX_EXPERIMENTS_PER_GROUP})" >&2
  exit 1
fi
GROUPS_PER_NODE=$((HOST_GPU_NUM / RAY_GPUS_PER_GROUP))
echo "Detected GPUs/node: ${HOST_GPU_NUM}"
echo "GPU group config: ${GROUPS_PER_NODE} groups/node, ${RAY_GPUS_PER_GROUP} GPUs/group, max ${MAX_EXPERIMENTS_PER_GROUP} experiments/group"

RAY_RESOURCES_JSON=""
if group_resources="$("${PYTHON_BIN}" "${HELPER_PY}" build-group-resources \
  --gpus-per-group "${RAY_GPUS_PER_GROUP}" \
  --max-experiments-per-group "${MAX_EXPERIMENTS_PER_GROUP}")"; then
  RAY_RESOURCES_JSON="${group_resources}"
  echo "Ray custom group resources: ${RAY_RESOURCES_JSON}"
else
  echo "ERROR: failed to build Ray custom group resources." >&2
  exit 1
fi

if [[ "$INITIAL_NODE_STAGGER_SECONDS" -gt 0 && "$NODE_RANK" -gt 0 ]]; then
  delay=$((INITIAL_NODE_STAGGER_SECONDS * NODE_RANK))
  echo "Node rank ${NODE_RANK}: initial one-time sleep ${delay}s to smooth shared cache access."
  sleep "$delay"
fi

echo "Node rank ${NODE_RANK}: restarting Ray runtime."
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true

if [[ "$NODE_IP" == "$RAY_HEAD_IP" ]]; then
  echo "Head node (rank ${NODE_RANK}): starting Ray head at ${NODE_IP}:${RAY_PORT}"
  ray_head_cmd=("${RAY_BIN}" start --head \
    --node-ip-address "${NODE_IP}" \
    --port "${RAY_PORT}" \
    --num-gpus "${HOST_GPU_NUM}" \
    --num-cpus "${NODE_CPU_COUNT}")
  if [[ -n "${RAY_RESOURCES_JSON}" ]]; then
    ray_head_cmd+=(--resources "${RAY_RESOURCES_JSON}")
  fi
  "${ray_head_cmd[@]}"

  for arg in "$@"; do
    if [[ "${arg}" == "--ray-gpus-per-node" || "${arg}" == --ray-gpus-per-node=* ]]; then
      echo "ERROR: --ray-gpus-per-node cannot be passed explicitly; it is auto-detected from node GPUs." >&2
      exit 1
    fi
  done

  cmd=(
    "${PYTHON_BIN}" evaluation/baselines/run_experiments.py
    --project-root "${RUN_PROJECT_ROOT}"
    --ray-address "${RAY_ADDRESS}"
    --ray-gpus-per-node "${HOST_GPU_NUM}"
    --ray-gpus-per-group "${RAY_GPUS_PER_GROUP}"
    --max-experiments-per-group "${MAX_EXPERIMENTS_PER_GROUP}"
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
  ray_worker_cmd=("${RAY_BIN}" start \
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
