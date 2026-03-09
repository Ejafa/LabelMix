#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_RUNTIME_SCRIPT="${SCRIPT_DIR}/start_runtime.sh"
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
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

if [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "ERROR: conda binary not found at ${CONDA_BIN}" >&2
    echo "Hint: set CONDA_ROOT/CONDA_BIN, or run with SKIP_CONDA_ACTIVATE=1 if env is already active." >&2
    exit 1
  fi

  if [[ ! -f "${CONDA_ACTIVATE_SCRIPT}" ]]; then
    echo "ERROR: conda activate script not found at ${CONDA_ACTIVATE_SCRIPT}" >&2
    exit 1
  fi

  activate_target="${CONDA_ENV_NAME}"
  if [[ -n "${CONDA_ENV_PATH}" ]]; then
    activate_target="${CONDA_ENV_PATH}"
  fi

  if ! source "${CONDA_ACTIVATE_SCRIPT}" "${activate_target}"; then
    if [[ -z "${CONDA_ENV_PATH}" && -d "${PROJECT_ROOT}/env/${CONDA_ENV_NAME}" ]] && source "${CONDA_ACTIVATE_SCRIPT}" "${PROJECT_ROOT}/env/${CONDA_ENV_NAME}"; then
      activate_target="${PROJECT_ROOT}/env/${CONDA_ENV_NAME}"
    elif [[ -z "${CONDA_ENV_PATH}" && -d "${PROJECT_ROOT}/env/labemix" ]] && source "${CONDA_ACTIVATE_SCRIPT}" "${PROJECT_ROOT}/env/labemix"; then
      activate_target="${PROJECT_ROOT}/env/labemix"
    elif [[ -z "${CONDA_ENV_PATH}" && -d "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}" ]] && source "${CONDA_ACTIVATE_SCRIPT}" "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"; then
      activate_target="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
    else
      echo "ERROR: failed to activate conda env target '${activate_target}'." >&2
      echo "Hint: set CONDA_ENV_NAME=<env-name> or CONDA_ENV_PATH=<full-env-path>." >&2
      exit 1
    fi
  fi

  echo "start.sh: activated conda env target '${activate_target}'."
fi

if [[ ! -x "${START_RUNTIME_SCRIPT}" ]]; then
  echo "ERROR: runtime start script is missing or not executable: ${START_RUNTIME_SCRIPT}" >&2
  exit 1
fi

# Supervisor behavior:
# - Keeps the entrypoint process alive on successful child exit by default.
# - Optionally restarts on failures.
SUPERVISOR_KEEPALIVE_ON_SUCCESS="${SUPERVISOR_KEEPALIVE_ON_SUCCESS:-1}"
SUPERVISOR_RESTART_ON_FAILURE="${SUPERVISOR_RESTART_ON_FAILURE:-1}"
SUPERVISOR_MAX_FAILURE_RESTARTS="${SUPERVISOR_MAX_FAILURE_RESTARTS:-1}"  # 0 means unlimited
SUPERVISOR_RESTART_DELAY_SECONDS="${SUPERVISOR_RESTART_DELAY_SECONDS:-20}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if ! [[ "${SUPERVISOR_MAX_FAILURE_RESTARTS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SUPERVISOR_MAX_FAILURE_RESTARTS must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${SUPERVISOR_RESTART_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SUPERVISOR_RESTART_DELAY_SECONDS must be a non-negative integer." >&2
  exit 1
fi

failure_restarts=0
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "Supervisor: launching ${START_RUNTIME_SCRIPT} (attempt=${attempt})."

  set +e
  bash "${START_RUNTIME_SCRIPT}" "$@"
  child_rc=$?
  set -e

  if [[ "${child_rc}" -eq 0 ]]; then
    echo "Supervisor: child exited successfully (code=0)."
    if is_true "${SUPERVISOR_KEEPALIVE_ON_SUCCESS}"; then
      echo "Supervisor: keeping entrypoint alive (SUPERVISOR_KEEPALIVE_ON_SUCCESS=${SUPERVISOR_KEEPALIVE_ON_SUCCESS})."
      while true; do
        sleep 3600
      done
    fi
    exit 0
  fi

  echo "Supervisor: child exited with code ${child_rc}."
  if ! is_true "${SUPERVISOR_RESTART_ON_FAILURE}"; then
    exit "${child_rc}"
  fi

  failure_restarts=$((failure_restarts + 1))
  if (( SUPERVISOR_MAX_FAILURE_RESTARTS > 0 && failure_restarts > SUPERVISOR_MAX_FAILURE_RESTARTS )); then
    echo "Supervisor: max failure restarts reached (${SUPERVISOR_MAX_FAILURE_RESTARTS}). Exiting." >&2
    exit "${child_rc}"
  fi

  if (( SUPERVISOR_RESTART_DELAY_SECONDS > 0 )); then
    echo "Supervisor: restarting in ${SUPERVISOR_RESTART_DELAY_SECONDS}s."
    sleep "${SUPERVISOR_RESTART_DELAY_SECONDS}"
  fi
done
