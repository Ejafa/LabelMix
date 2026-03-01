#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
cd "$PROJECT_ROOT"

if [[ -x "./evaluation/deployment/setup_proxy.sh" ]]; then
  ./evaluation/deployment/setup_proxy.sh
else
  echo "Warning: ./evaluation/deployment/setup_proxy.sh not found or not executable." >&2
fi

# This will overwrite the bash script - if possible just create a new bash file
if [[ -n "${CONFIG_FILE:-}" ]]; then
  source ~/"$(basename "$CONFIG_FILE")"
fi

if command -v start_proxy >/dev/null 2>&1; then
  start_proxy on
else
  echo "Warning: start_proxy not found in PATH." >&2
fi

conda create -f env.yaml -n labelmix
conda activate labelmix
