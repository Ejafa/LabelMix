#!/usr/bin/env bash
# ============================================================================
# launch_all.sh — Launch / stop / check jobdaemon tmux sessions across nodes
#
# Usage:
#   ./launch_all.sh                   # launch daemons on all nodes
#   ./launch_all.sh --status          # check if tmux sessions are alive
#   ./launch_all.sh --stop            # kill tmux sessions on all nodes
#   ./launch_all.sh --logs <node_idx> # tail the daemon tmux pane for a node
#
# Prerequisites:
#   1. Run generate_jobs.py → jobs.yaml
#   2. Run job_scheduler.py --schedule-name <SCHEDULE_NAME>
#      → <SCHEDULE_NAME>_node_0_jobs.yaml, <SCHEDULE_NAME>_node_1_jobs.yaml, ...
#   3. Passwordless SSH to all remote nodes
# ============================================================================
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
# Edit these to match your cluster setup.

PROJECT_DIR="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix"
SSH_USER="root"
SSH_PORT="${JIZHI_SSH_PORT:-36000}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -p ${SSH_PORT}"
TMUX_SESSION="daemon"              # tmux session name on each node
SCHEDULE_NAME="ablation_sweep"     # isolates state/inbox/logs under schedules/<name>/
CONDA_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/miniconda3"
CONDA_ENV="labelmix"               # conda environment to activate before running daemon

# ─── Helpers ────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
BLU='\033[0;34m'
RST='\033[0m'

log()  { echo -e "${BLU}[launch]${RST} $*"; }
ok()   { echo -e "${GRN}  ✅${RST} $*"; }
warn() { echo -e "${YLW}  ⚠️${RST}  $*"; }
err()  { echo -e "${RED}  ❌${RST} $*"; }

remote_cmd() {
    local host="$1"; shift
    ssh ${SSH_OPTS} "${SSH_USER}@${host}" "$@"
}

# Check if a tmux session exists locally
local_tmux_exists() {
    tmux has-session -t "${TMUX_SESSION}" 2>/dev/null
}

# Check if a tmux session exists on a remote node
remote_tmux_exists() {
    local host="$1"
    remote_cmd "${host}" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null" 2>/dev/null
}

# ─── Parse NODE_IP_LIST ─────────────────────────────────────────────────────
# Format: "IP1:GPU_COUNT,IP2:GPU_COUNT,IP3:GPU_COUNT"
# Example: "28.12.129.140:8,28.12.25.40:8,28.12.130.213:8"
# First entry is LOCAL (master); remaining are REMOTE (accessed via SSH).

if [[ -z "${NODE_IP_LIST:-}" ]]; then
    echo -e "\033[0;31m  ❌\033[0m NODE_IP_LIST env variable is not set."
    echo "  Expected format: IP1:GPU_COUNT,IP2:GPU_COUNT,..."
    echo "  Example: export NODE_IP_LIST='28.12.129.140:8,28.12.25.40:8,28.12.130.213:8'"
    exit 1
fi

NODES=()
NODE_GPUS=()  # per-node GPU string, e.g. "0,1,2,3,4,5,6,7"
IFS=',' read -ra _NODE_ENTRIES <<< "${NODE_IP_LIST}"
for _entry in "${_NODE_ENTRIES[@]}"; do
    _ip="${_entry%%:*}"
    _gpu_count="${_entry##*:}"
    # Build comma-separated GPU ID list: "0,1,...,N-1"
    _gpu_list=""
    for (( _g=0; _g<_gpu_count; _g++ )); do
        [[ -n "$_gpu_list" ]] && _gpu_list+=","
        _gpu_list+="${_g}"
    done
    NODES+=("${_ip}")
    NODE_GPUS+=("${_gpu_list}")
done

log "Parsed NODE_IP_LIST: ${#NODES[@]} node(s)"
for _i in "${!NODES[@]}"; do
    log "  node_${_i}: ${NODES[$_i]}  GPUs: ${NODE_GPUS[$_i]}"
done
echo ""

# Per-node YAML files (must exist in PROJECT_DIR)
# Auto-generated from NODES array: ${SCHEDULE_NAME}_node_0_jobs.yaml, ${SCHEDULE_NAME}_node_1_jobs.yaml, ...
node_yaml() { echo "${SCHEDULE_NAME}_node_${1}_jobs.yaml"; }

# ─── LAUNCH ─────────────────────────────────────────────────────────────────

do_launch() {
    log "Launching daemons on ${#NODES[@]} node(s)..."
    echo ""

    for i in "${!NODES[@]}"; do
        local host="${NODES[$i]}"
        local yaml
        yaml=$(node_yaml "$i")
        local yaml_path="${PROJECT_DIR}/${yaml}"

        # Verify YAML file exists locally (they should have been generated already)
        if [[ ! -f "${yaml_path}" ]]; then
            err "node_${i} (${host}): ${yaml} not found in ${PROJECT_DIR}"
            err "  Run job_scheduler.py first to generate per-node YAML files."
            continue
        fi

        local job_count
        job_count=$(grep -c "^  - name:" "${yaml_path}" 2>/dev/null || echo "?")

        if [[ "$i" -eq 0 ]]; then
            # ── LOCAL node ──
            log "node_${i} (${host}) — LOCAL, ${job_count} jobs"

            if local_tmux_exists; then
                warn "tmux session '${TMUX_SESSION}' already exists locally. Skipping."
                warn "  Use --stop first, or attach with: tmux attach -t ${TMUX_SESSION}"
                continue
            fi

            # Create detached tmux session running the daemon
            local gpus="${NODE_GPUS[$i]}"
            tmux new-session -d -s "${TMUX_SESSION}" \
                "source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} && python jobdaemon.py -s ${SCHEDULE_NAME} --node-index ${i} start --gpus ${gpus}; exec bash"
            sleep 2

            # Submit jobs in the same tmux session (new window)
            tmux new-window -t "${TMUX_SESSION}" \
                "source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} && sleep 5 && python jobdaemon.py -s ${SCHEDULE_NAME} --node-index ${i} submit ${yaml}; exec bash"

            ok "node_${i} (${host}): daemon started, ${yaml} submitted"

        else
            # ── REMOTE node ──
            log "node_${i} (${host}) — REMOTE, ${job_count} jobs"

            if remote_tmux_exists "${host}"; then
                warn "tmux session '${TMUX_SESSION}' already exists on ${host}. Skipping."
                warn "  Use --stop first, or: ssh ${SSH_USER}@${host} 'tmux attach -t ${TMUX_SESSION}'"
                continue
            fi

            # Start daemon in a remote tmux session
            local gpus="${NODE_GPUS[$i]}"
            remote_cmd "${host}" "
                tmux new-session -d -s ${TMUX_SESSION} \
                    'source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} && python jobdaemon.py -s ${SCHEDULE_NAME} --node-index ${i} start --gpus ${gpus}; exec bash'
            " 2>/dev/null

            if ! remote_tmux_exists "${host}"; then
                err "node_${i} (${host}): failed to create tmux session"
                continue
            fi

            # Submit jobs in a second tmux window (after a short delay)
            remote_cmd "${host}" "
                tmux new-window -t ${TMUX_SESSION} \
                    'source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} && sleep 5 && python jobdaemon.py -s ${SCHEDULE_NAME} --node-index ${i} submit ${yaml}; exec bash'
            " 2>/dev/null

            ok "node_${i} (${host}): daemon started, ${yaml} submitted"
        fi
    done

    echo ""
    log "Done. Use --status to check, --stop to tear down."
    log "Attach locally:  tmux attach -t ${TMUX_SESSION}"
    log "Attach remotely: ssh ${SSH_USER}@<host> -t 'tmux attach -t ${TMUX_SESSION}'"
}

# ─── STATUS ─────────────────────────────────────────────────────────────────

do_status() {
    log "Checking daemon status on ${#NODES[@]} node(s)..."
    echo ""

    printf "  %-8s %-18s %-10s %s\n" "NODE" "HOST" "TMUX" "DAEMON"
    printf "  %-8s %-18s %-10s %s\n" "────────" "──────────────────" "──────────" "──────────────"

    for i in "${!NODES[@]}"; do
        local host="${NODES[$i]}"
        local tmux_status daemon_status

        if [[ "$i" -eq 0 ]]; then
            # Local
            if local_tmux_exists; then
                tmux_status="${GRN}alive${RST}"
                # Check if jobdaemon process is running
                if pgrep -f "jobdaemon.py start" >/dev/null 2>&1; then
                    daemon_status="${GRN}running${RST}"
                else
                    daemon_status="${YLW}tmux up, daemon gone${RST}"
                fi
            else
                tmux_status="${RED}none${RST}"
                daemon_status="${RED}not running${RST}"
            fi
        else
            # Remote
            if remote_tmux_exists "${host}" 2>/dev/null; then
                tmux_status="${GRN}alive${RST}"
                if remote_cmd "${host}" "pgrep -f 'jobdaemon.py start'" >/dev/null 2>&1; then
                    daemon_status="${GRN}running${RST}"
                else
                    daem/on_status="${YLW}tmux up, daemon gone${RST}"
                fi
            else
                tmux_status="${RED}none${RST}"
                daemon_status="${RED}not running${RST}"
            fi
        fi

        printf "  %-8s %-18s " "node_${i}" "${host}"
        echo -e "${tmux_status}      ${daemon_status}"
    done
    echo ""
}

# ─── STOP ───────────────────────────────────────────────────────────────────

do_stop() {
    log "Stopping daemons on ${#NODES[@]} node(s)..."
    echo ""

    for i in "${!NODES[@]}"; do
        local host="${NODES[$i]}"

        if [[ "$i" -eq 0 ]]; then
            # Local
            if local_tmux_exists; then
                tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
                ok "node_${i} (${host}): tmux session killed"
            else
                warn "node_${i} (${host}): no tmux session found"
            fi
            # Belt-and-suspenders: kill any stray daemon processes
            pkill -f "jobdaemon.py start" 2>/dev/null || true
        else
            # Remote
            if remote_tmux_exists "${host}" 2>/dev/null; then
                remote_cmd "${host}" "tmux kill-session -t ${TMUX_SESSION}" 2>/dev/null || true
                ok "node_${i} (${host}): tmux session killed"
            else
                warn "node_${i} (${host}): no tmux session found"
            fi
            remote_cmd "${host}" "pkill -f 'jobdaemon.py start'" 2>/dev/null || true
        fi
    done

    echo ""
    log "All daemons stopped."
}

# ─── LOGS ───────────────────────────────────────────────────────────────────

do_logs() {
    local idx="$1"

    if [[ "$idx" -ge "${#NODES[@]}" ]]; then
        err "Invalid node index: ${idx} (only ${#NODES[@]} nodes configured)"
        exit 1
    fi

    local host="${NODES[$idx]}"

    if [[ "$idx" -eq 0 ]]; then
        log "Attaching to local tmux session..."
        tmux attach -t "${TMUX_SESSION}"
    else
        log "Attaching to remote tmux session on ${host}..."
        ssh ${SSH_OPTS} -t "${SSH_USER}@${host}" "tmux attach -t ${TMUX_SESSION}"
    fi
}

# ─── JOB STATUS ─────────────────────────────────────────────────────────────

do_job_status() {
    log "Querying job status on ${#NODES[@]} node(s)..."
    echo ""

    for i in "${!NODES[@]}"; do
        local host="${NODES[$i]}"

        echo -e "${BLU}━━━ node_${i} (${host}) ━━━${RST}"

        if [[ "$i" -eq 0 ]]; then
            (cd "${PROJECT_DIR}" && python jobdaemon.py -s ${SCHEDULE_NAME} --node-index ${i} status 2>/dev/null) || warn "Could not query status"
        else
            remote_cmd "${host}" "source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} && python jobdaemon.py -s ${SCHEDULE_NAME} --node-index ${i} status" 2>/dev/null \
                || warn "Could not query status on ${host}"
        fi
        echo ""
    done
}

# ─── MAIN ───────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTION]

Options:
  (no flag)       Launch daemons and submit jobs on all nodes
  --status        Check if tmux sessions and daemons are alive
  --jobs          Query job status (pending/running/done) on all nodes
  --stop          Kill tmux sessions and daemons on all nodes
  --logs <N>      Attach to tmux session on node N (0-indexed)
  --help          Show this help message

Workflow:
  1. python generate_jobs.py -o jobs.yaml
  2. python job_scheduler.py --input jobs.yaml --nodes ${NODES[*]}
  3. $0                 # launch everything
  4. $0 --status        # check health
  5. $0 --jobs          # check job progress
  6. $0 --stop          # tear down when done
EOF
}

case "${1:-}" in
    --status)
        do_status
        ;;
    --jobs)
        do_job_status
        ;;
    --stop)
        do_stop
        ;;
    --logs)
        if [[ -z "${2:-}" ]]; then
            err "Usage: $0 --logs <node_index>"
            exit 1
        fi
        do_logs "$2"
        ;;
    --help|-h)
        usage
        ;;
    "")
        do_launch
        ;;
    *)
        err "Unknown option: $1"
        usage
        exit 1
        ;;
esac
