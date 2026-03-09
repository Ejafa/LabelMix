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
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
CONDA_ACTIVATE_SCRIPT="${CONDA_ACTIVATE_SCRIPT:-${CONDA_ROOT}/bin/activate}"
CONDA_INIT_SHELL="${CONDA_INIT_SHELL:-bash}"
RUN_START_AFTER_INIT="${RUN_START_AFTER_INIT:-}"
PROXY_URL="${PROXY_URL:-http://star-proxy.oa.com:3128}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "ERROR: conda binary not found at ${CONDA_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"

# Make conda available for future bash shells (idempotent).
"${CONDA_BIN}" init "${CONDA_INIT_SHELL}" >/dev/null 2>&1 || "${CONDA_BIN}" init "${CONDA_INIT_SHELL}"

if [[ ! -f "${CONDA_ACTIVATE_SCRIPT}" ]]; then
  echo "ERROR: conda activate script not found at ${CONDA_ACTIVATE_SCRIPT}" >&2
  exit 1
fi

source_conda_activate() {
  local target="$1"
  local had_nounset="0"
  if [[ "$-" == *u* ]]; then
    had_nounset="1"
    set +u
  fi
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE_SCRIPT}" "${target}"
  local rc=$?
  if [[ "${had_nounset}" == "1" ]]; then
    set -u
  fi
  return "${rc}"
}

activate_target="${CONDA_ENV_NAME}"
if [[ -n "${CONDA_ENV_PATH}" ]]; then
  activate_target="${CONDA_ENV_PATH}"
fi
if ! source_conda_activate "${activate_target}"; then
  if [[ -z "${CONDA_ENV_PATH}" && -d "${PROJECT_ROOT}/env/${CONDA_ENV_NAME}" ]] && source_conda_activate "${PROJECT_ROOT}/env/${CONDA_ENV_NAME}"; then
    activate_target="${PROJECT_ROOT}/env/${CONDA_ENV_NAME}"
  elif [[ -z "${CONDA_ENV_PATH}" && -d "${PROJECT_ROOT}/env/labemix" ]] && source_conda_activate "${PROJECT_ROOT}/env/labemix"; then
    activate_target="${PROJECT_ROOT}/env/labemix"
  elif [[ -z "${CONDA_ENV_PATH}" && -d "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}" ]] && source_conda_activate "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"; then
    activate_target="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
  else
    echo "ERROR: failed to activate conda env target '${activate_target}'." >&2
    echo "Hint: set CONDA_ENV_NAME=<env-name> or CONDA_ENV_PATH=<full-env-path>." >&2
    exit 1
  fi
fi

python_bin="$(command -v python || true)"
ray_bin="$(command -v ray || true)"
echo "Activated env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "Activation target: ${activate_target}"
echo "Python: ${python_bin:-<not found>}"
echo "Ray: ${ray_bin:-<not found>}"

if [[ -z "${RUN_START_AFTER_INIT}" ]]; then
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    RUN_START_AFTER_INIT="1"
  else
    RUN_START_AFTER_INIT="0"
  fi
fi

if [[ "${RUN_START_AFTER_INIT}" == "1" ]]; then
  START_SCRIPT="${SCRIPT_DIR}/start.sh"
  if [[ ! -x "${START_SCRIPT}" ]]; then
    echo "ERROR: start script not executable: ${START_SCRIPT}" >&2
    exit 1
  fi
  echo "init.sh: launching ${START_SCRIPT} in activated env."
  bash "${START_SCRIPT}" "$@"
  exit $?
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "init.sh completed without auto-start (RUN_START_AFTER_INIT=${RUN_START_AFTER_INIT})."
fi
