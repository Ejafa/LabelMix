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

NODE_RANK=""
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

python evaluation/baselines/run_experiments.py \
  --gpu-nodes "${TAIJI_HOST_NUM}" \
  --experiments-per-gpu 2 \
  --gpu-per-node "${HOST_GPU_NUM}" \
  --nproc 1 \
  --node-index "${NODE_RANK}"
