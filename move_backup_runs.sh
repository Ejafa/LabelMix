#!/usr/bin/env bash
# move_backup_runs.sh
#
# Move runs previously backed up by evaluation/scripts/backup_runs.py into
# the same destination used by sync_checkpoints.sh.
#
# backup_runs.py writes W&B/timm backups as:
#   <BACKUP_ROOT>/<group>/<experiment>/...
#
# This script moves them to:
#   <DST_BASE>/<group>/<experiment>/...
#
# A few source groups were already copied by sync_checkpoints.sh; those are
# skipped here to avoid repeating work.
#
# Usage:
#   bash move_backup_runs.sh [--dry-run]
#
# Optional environment overrides:
#   BACKUP_ROOT=/path/to/backup DST_BASE=/path/to/output_runs bash move_backup_runs.sh

set -euo pipefail

BACKUP_ROOT="${BACKUP_ROOT:-/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/backup}"
DST_BASE="${DST_BASE:-/apdcephfs/private_ethangeng/konstantin-garbers/labelmix/output_runs}"

# Top-level source directories that were already handled by sync_checkpoints.sh.
# We skip matching backup groups here because sync_checkpoints.sh already placed
# them under $DST_BASE/<group>/...
ALREADY_MOVED_GROUPS=(
    "sampling"
    "scheduling"
    "ablation"
    "reverse_k"
    "alpha_schedule"
    "alpha_reverse_schedule"
    "sampling_2"
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] No files will actually be moved."
elif [[ $# -gt 0 ]]; then
    echo "Usage: bash move_backup_runs.sh [--dry-run]" >&2
    exit 2
fi

if [[ ! -d "$BACKUP_ROOT" ]]; then
    echo "ERROR: backup root not found: $BACKUP_ROOT" >&2
    exit 1
fi

is_already_moved_group() {
    local group="$1"
    local moved_group
    for moved_group in "${ALREADY_MOVED_GROUPS[@]}"; do
        if [[ "$group" == "$moved_group" ]]; then
            return 0
        fi
    done
    return 1
}

move_count=0
skip_count=0

shopt -s nullglob
for group_dir in "$BACKUP_ROOT"/*; do
    [[ -d "$group_dir" ]] || continue

    group="$(basename "$group_dir")"

    if is_already_moved_group "$group"; then
        echo "Skipping already-moved group: $group_dir"
        (( skip_count++ )) || true
        continue
    fi

    dst_dir="$DST_BASE/$group"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] would move: $group_dir -> $dst_dir"
        (( move_count++ )) || true
        continue
    fi

    mkdir -p "$DST_BASE"

    # Use rsync instead of mv so existing destination files are merged safely.
    # --remove-source-files deletes files only after successful transfer; the
    # follow-up find removes empty source directories left behind by rsync.
    rsync -ah --checksum --remove-source-files "$group_dir/" "$dst_dir/"
    find "$group_dir" -depth -type d -empty -delete

    if [[ -d "$group_dir" ]]; then
        echo "Moved files but kept non-empty source directory: $group_dir -> $dst_dir"
    else
        echo "Moved: $group_dir -> $dst_dir"
    fi
    (( move_count++ )) || true
done
shopt -u nullglob

echo ""
echo "Done. Moved group(s): $move_count, skipped group(s): $skip_count."
