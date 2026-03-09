#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"

PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONDA_ROOT="${CONDA_ROOT:-${PROJECT_ROOT}/miniconda3}"
CONDA_BIN="${CONDA_BIN:-${CONDA_ROOT}/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-labelmix}"
PROXY_URL="${PROXY_URL:-http://star-proxy.oa.com:3128}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "ERROR: conda binary not found at ${CONDA_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"

eval "$("$CONDA_BIN" shell.bash hook)"

if ! conda activate "${CONDA_ENV_NAME}"; then
  echo "ERROR: failed to activate conda env '${CONDA_ENV_NAME}'." >&2
  echo "Hint: set CONDA_ENV_NAME or check environments under ${CONDA_ROOT}/envs." >&2
  exit 1
fi

python_bin="$(command -v python || true)"
ray_bin="$(command -v ray || true)"
echo "Activated env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "Python: ${python_bin:-<not found>}"
echo "Ray: ${ray_bin:-<not found>}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Note: init.sh was executed (not sourced), so activation only applied inside this script process."
  echo "Use: source evaluation/deployment/init.sh"
fi
