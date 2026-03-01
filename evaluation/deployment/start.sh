#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "ERROR: NODE_IP_LIST is not set" >&2
  exit 1
fi
if [[ -z "${NODE_IP:-}" ]]; then
  echo "ERROR: NODE_IP is not set" >&2
  exit 1
fi

IFS=',' read -ra ITEMS <<< "$NODE_IP_LIST"

NODE_RANK=0
for i in "${!ITEMS[@]}"; do
  ip="${ITEMS[$i]%%:*}"     # strip ":8"
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

NPROC="${NPROC:-$HOST_GPU_NUM}"

python evaluation/baselines/run_experiments.py \
  --gpu-nodes "${TAIJI_HOST_NUM}" \
  --gpu-per-node "${HOST_GPU_NUM}" \
  --nproc "${NPROC}" \
  --node-index "${NODE_RANK}" \
  --multi-node-stagger