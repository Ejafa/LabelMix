#!/usr/bin/env bash
# Download the COCO 2017 dataset in the layout expected by detectron2.
#
# Final layout:
#   ${COCO_ROOT}/
#       annotations/
#           instances_train2017.json
#           instances_val2017.json
#           ...
#       train2017/
#           <image files>
#       val2017/
#           <image files>
#
# Usage:
#   bash download_coco.sh [TARGET_DIR]
#
# If TARGET_DIR is omitted the script uses ${COCO_ROOT} (if set) or
# ./datasets/coco next to this script.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-${COCO_ROOT:-${HERE}/datasets/coco}}"
mkdir -p "${TARGET}"
cd "${TARGET}"

echo "=== COCO dataset will be installed in: ${TARGET}"

# -----------------------------------------------------------------------------
# Files to fetch. You can comment out test2017 to save ~6 GB if you only need
# train/val for the standard VitDet recipe.
# -----------------------------------------------------------------------------
FILES=(
    "http://images.cocodataset.org/zips/train2017.zip"
    "http://images.cocodataset.org/zips/val2017.zip"
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

for URL in "${FILES[@]}"; do
    FNAME=$(basename "${URL}")
    if [[ -f "${FNAME}.done" ]]; then
        echo "[skip] ${FNAME} already downloaded & extracted."
        continue
    fi
    echo ""
    echo "--- downloading ${FNAME} ---"
    # -c enables resume; --retry on transient errors.
    wget --continue --tries=10 --waitretry=10 "${URL}" -O "${FNAME}"

    echo "--- extracting ${FNAME} ---"
    unzip -q -o "${FNAME}"
    touch "${FNAME}.done"
    # Remove the zip to save disk.
    rm -f "${FNAME}"
done

echo ""
echo "=== Done. Directory listing: ==="
ls -l "${TARGET}"

# Detectron2 uses the environment variable DETECTRON2_DATASETS to locate
# datasets. Export this when training:
echo ""
echo "Remember to export:"
echo "  export DETECTRON2_DATASETS=\"$(dirname "${TARGET}")\""
echo "(detectron2 will then look for \$DETECTRON2_DATASETS/coco)"
