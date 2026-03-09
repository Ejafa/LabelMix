#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  evaluation/deployment/start_all_nodes.sh [-- <start.sh args...>]

Required env:
  NODE_IP_LIST=ip1,ip2,ip3

Optional env:
  CHIEF_IP=<head ip>          default: first IP in NODE_IP_LIST
  SSH_USER=<user>             default: current user
  SSH_PORT=<port>             default: 22
  WORKER_LOG_DIR=<dir>        default: /tmp/labelmix-start
EOF
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

shell_join_quoted() {
  local out=()
  local arg
  for arg in "$@"; do
    out+=("$(printf '%q' "$arg")")
  done
  printf '%s' "${out[*]-}"
}

NODE_IP_LIST="${NODE_IP_LIST:-}"
CHIEF_IP="${CHIEF_IP:-}"
SSH_USER="${SSH_USER:-}"
SSH_PORT="${SSH_PORT:-22}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-/tmp/labelmix-start}"
START_ARGS=()

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      START_ARGS+=("$@")
      break
      ;;
    *)
      START_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${NODE_IP_LIST}" ]]; then
  echo "ERROR: NODE_IP_LIST is required." >&2
  exit 1
fi

IFS=',' read -r -a RAW_ITEMS <<< "${NODE_IP_LIST}"
NODES=()
for raw in "${RAW_ITEMS[@]}"; do
  item="$(trim "${raw}")"
  [[ -z "${item}" ]] && continue
  ip="${item%%:*}"
  ip="$(trim "${ip}")"
  [[ -z "${ip}" ]] && continue
  NODES+=("${ip}")
done

if [[ "${#NODES[@]}" -eq 0 ]]; then
  echo "ERROR: no valid nodes found in NODE_IP_LIST=${NODE_IP_LIST}" >&2
  exit 1
fi

if [[ -z "${CHIEF_IP}" ]]; then
  CHIEF_IP="${NODES[0]}"
fi
CHIEF_IP="${CHIEF_IP%%:*}"

HEAD_RANK="-1"
for i in "${!NODES[@]}"; do
  if [[ "${NODES[$i]}" == "${CHIEF_IP}" ]]; then
    HEAD_RANK="${i}"
    break
  fi
done

if [[ "${HEAD_RANK}" == "-1" ]]; then
  echo "ERROR: CHIEF_IP=${CHIEF_IP} is not in NODE_IP_LIST=${NODE_IP_LIST}" >&2
  exit 1
fi

START_ARGS_QUOTED="$(shell_join_quoted "${START_ARGS[@]}")"

build_start_cmd() {
  local cmd
  cmd="$(printf "cd %q && bash ./evaluation/deployment/start.sh" "${PROJECT_ROOT}")"
  if [[ -n "${START_ARGS_QUOTED}" ]]; then
    cmd+=" ${START_ARGS_QUOTED}"
  fi
  printf '%s\n' "${cmd}"
}

echo "Node list (${#NODES[@]}): ${NODES[*]}"
echo "Head node: ${CHIEF_IP} (rank=${HEAD_RANK})"

for i in "${!NODES[@]}"; do
  node_ip="${NODES[$i]}"
  if [[ "${i}" == "${HEAD_RANK}" ]]; then
    continue
  fi

  start_cmd="$(build_start_cmd)"
  worker_log="${WORKER_LOG_DIR}/worker_rank${i}_${node_ip}.log"
  worker_cmd="$(printf 'mkdir -p %q && nohup bash -lc %q > %q 2>&1 < /dev/null &' \
    "${WORKER_LOG_DIR}" \
    "${start_cmd}" \
    "${worker_log}")"

  target="${node_ip}"
  if [[ -n "${SSH_USER}" ]]; then
    target="${SSH_USER}@${node_ip}"
  fi

  echo "Starting worker rank=${i} on ${target}, log=${worker_log}"
  ssh -p "${SSH_PORT}" "${target}" "${worker_cmd}"
done

echo "Starting head locally (rank=${HEAD_RANK})."
bash ./evaluation/deployment/start.sh "${START_ARGS[@]}"
