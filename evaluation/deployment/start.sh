#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_RUNTIME_SCRIPT="${SCRIPT_DIR}/start_runtime.sh"

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
