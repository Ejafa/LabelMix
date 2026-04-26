#!/usr/bin/env bash
# Install detectron2 and fetch the official ViTDet project code.
#
# Usage:
#   bash install.sh [--dev]
#
# The --dev flag builds detectron2 from source (required if the pre-built
# wheels do not match the local CUDA/Torch combination).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

DEV=0
for arg in "$@"; do
    case "${arg}" in
        --dev) DEV=1 ;;
        *) echo "unknown argument: ${arg}" && exit 1 ;;
    esac
done

echo "=== [1/4] Install Python dependencies ==="
python -m pip install --upgrade pip
python -m pip install \
    "opencv-python>=4.5" \
    "pycocotools>=2.0.6" \
    "fvcore>=0.1.5.post20221221" \
    "iopath>=0.1.9,<0.1.10" \
    "omegaconf>=2.1" \
    "hydra-core>=1.1" \
    "black==21.4b2" \
    "yacs>=0.1.8" \
    "termcolor>=1.1" \
    "cloudpickle" \
    "tabulate" \
    "matplotlib" \
    "Pillow" \
    "timm>=0.9"

echo "=== [2/4] Install detectron2 ==="
if [[ "${DEV}" == "1" ]]; then
    # Build from source. This is the safest option when the user has a custom
    # PyTorch / CUDA build.
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
else
    # Try a pre-built wheel first (fast path).
    TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))" 2>/dev/null || echo "2.0")
    CUDA_VER=$(python -c "import torch; v=torch.version.cuda; print('cu'+v.replace('.','')) if v else print('cpu')" 2>/dev/null || echo "cpu")
    WHEEL_URL="https://dl.fbaipublicfiles.com/detectron2/wheels/${CUDA_VER}/torch${TORCH_VER}/index.html"
    echo "Trying pre-built wheel: ${WHEEL_URL}"
    if ! python -m pip install detectron2 -f "${WHEEL_URL}"; then
        echo "Pre-built wheel not available, falling back to source build."
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    fi
fi

echo "=== [3/4] Fetch the official ViTDet project tree ==="
VITDET_DIR="${HERE}/upstream_vitdet"
if [[ -d "${VITDET_DIR}/.git" ]]; then
    echo "Already cloned -> git pull"
    git -C "${VITDET_DIR}" pull --ff-only
else
    TMP_CLONE="${HERE}/_detectron2_sparse"
    rm -rf "${TMP_CLONE}"
    # Sparse-checkout only the ViTDet project directory to save bandwidth.
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/facebookresearch/detectron2.git "${TMP_CLONE}"
    git -C "${TMP_CLONE}" sparse-checkout set projects/ViTDet configs/common tools
    mkdir -p "${VITDET_DIR}"
    cp -r "${TMP_CLONE}/projects/ViTDet/"* "${VITDET_DIR}/"
    cp -r "${TMP_CLONE}/tools" "${VITDET_DIR}/tools"
    rm -rf "${TMP_CLONE}"
fi

echo "=== [4/4] Sanity check ==="
python -c "import detectron2, torch; print('detectron2', detectron2.__version__); print('torch', torch.__version__)"

echo ""
echo "Done. Upstream ViTDet code is in: ${VITDET_DIR}"
echo "Training entry point: ${VITDET_DIR}/tools/lazyconfig_train_net.py"
