#!/usr/bin/env bash
# sync_checkpoints.sh
#
# Copies only args.yaml, model_best.pth.tar, and run_status.yaml from
# the source directories to the destination, preserving the relative
# directory structure (equivalent to rsync but file-type filtered).
#
# Usage:
#   bash sync_checkpoints.sh [--dry-run]

set -euo pipefail

SRC_DIRS=(
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/output_runs/sampling"
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/output_runs/scheduling"
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/important_data/ablation-study_1/ablation"
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix/output_runs/reverse_k"
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/output_runs/alpha_schedule"
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/output_runs/alpha_reverse_schedule"    
#    "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/output_runs/sampling_2"    
)
DST_BASE="/apdcephfs/private_ethangeng/konstantin-garbers/labelmix/output_runs"

# Files to copy (exact filename match)
TARGET_FILES=("args.yaml" "model_best.pth.tar" "run_status.yaml")

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] No files will actually be copied."
fi

copy_count=0
skip_count=0

for src_dir in "${SRC_DIRS[@]}"; do
    if [[ ! -d "$src_dir" ]]; then
        echo "WARNING: source directory not found, skipping: $src_dir"
        continue
    fi

    # Strip any trailing slash so dirname works correctly and the last
    # component (e.g. "sampling", "scheduling", "ablation") is preserved.
    src_dir="${src_dir%/}"

    # src_parent is the directory *containing* src_dir.
    # Stripping it from each file path keeps the last component in rel_path,
    # so e.g. ".../output_runs/sampling/exp1/args.yaml" becomes
    # "sampling/exp1/args.yaml" and lands at "$DST_BASE/sampling/exp1/args.yaml".
    src_parent="$(dirname "$src_dir")"

    for target_file in "${TARGET_FILES[@]}"; do
        # Find all matching files anywhere under src_dir
        while IFS= read -r -d '' src_file; do
            rel_path="${src_file#"${src_parent}/"}"
            dst_file="$DST_BASE/$rel_path"
            dst_dir="$(dirname "$dst_file")"

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[dry-run] would copy: $src_file -> $dst_file"
                (( copy_count++ )) || true
                continue
            fi

            mkdir -p "$dst_dir"

            # rsync: delta-transfer for large files, skip if destination is up-to-date
            # --checksum: compare by checksum, not just size+mtime
            # --out-format: print a line only when a file is actually transferred
            transferred=$(rsync -ah --checksum --update --out-format="%n" "$src_file" "$dst_file")
            if [[ -n "$transferred" ]]; then
                echo "Copying: $src_file -> $dst_file"
                (( copy_count++ )) || true
            else
                (( skip_count++ )) || true
            fi
        done < <(find "$src_dir" -type f -name "$target_file" -print0)
    done
done

echo ""
echo "Done. Copied: $copy_count file(s), skipped (up-to-date): $skip_count file(s)."
