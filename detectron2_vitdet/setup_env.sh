#!/usr/bin/env bash
# Create a dedicated conda environment for detectron2 + ViTDet.
#
# This installs a KNOWN-GOOD combination:
#   - Python 3.10
#   - PyTorch 2.1.2 + CUDA 12.1
#   - torchvision 0.16.2
#   - detectron2 (pre-built wheel for cu121/torch2.1, falls back to source)
#   - all python deps required by the ViTDet configs in this repo
#
# Why a dedicated env?
#   The project's main `labelmix` env is on Python 3.14 / torch 2.10, which is
#   too new for detectron2's custom CUDA extensions. Instead of patching
#   detectron2's C++ source, we just use a compatible stack in isolation.
#
# Usage:
#   bash detectron2_vitdet/setup_env.sh                 # default name: det2
#   ENV_NAME=mydet2 bash detectron2_vitdet/setup_env.sh
#   CUDA_VER=cu118  bash detectron2_vitdet/setup_env.sh # if your driver is older
#
# After it finishes:
#   conda activate /apdcephfs_fsgm/.../ggez/envs/det2
#   python detectron2_vitdet/upstream_vitdet/tools/lazyconfig_train_net.py --help
#
# Persistence:
#   The conda env, conda package cache, and pip cache are all placed on the
#   shared persistent storage next to `miniconda3/`. Running this script on a
#   freshly-spawned node will therefore re-use everything and finish in
#   seconds instead of re-downloading gigabytes.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable knobs
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-det2}"
PY_VER="${PY_VER:-3.10}"
TORCH_VER="${TORCH_VER:-2.1.2}"
TORCHVISION_VER="${TORCHVISION_VER:-0.16.2}"
# cu121 is the most reliable pre-built wheel target for detectron2.
# Use cu118 if your driver does not support CUDA 12.
CUDA_VER="${CUDA_VER:-cu121}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LabelMix lives at  <PERSIST_ROOT>/LabelMix/ ; miniconda3/ is its sibling.
PERSIST_ROOT="$(cd "${HERE}/../.." && pwd)"

# Put the env + all caches on persistent storage so that re-creating the
# environment on a new node is (nearly) free.
ENV_ROOT="${ENV_ROOT:-${PERSIST_ROOT}/envs}"
ENV_PREFIX="${ENV_ROOT}/${ENV_NAME}"
CONDA_PKGS_DIR="${PERSIST_ROOT}/conda_pkgs"
PIP_CACHE_DIR_PERSIST="${PERSIST_ROOT}/pip_cache"
mkdir -p "${ENV_ROOT}" "${CONDA_PKGS_DIR}" "${PIP_CACHE_DIR_PERSIST}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_PERSIST}"

echo "=== [0/5] Locate conda ==="
# Prefer a miniconda that is itself on persistent storage (so it survives a
# fresh node too). Fall back to whatever conda is on PATH.
if [[ -x "${PERSIST_ROOT}/miniconda3/bin/conda" ]]; then
    CONDA_EXE="${PERSIST_ROOT}/miniconda3/bin/conda"
elif [[ -z "${CONDA_EXE:-}" ]]; then
    CONDA_EXE="$(command -v conda || true)"
fi
if [[ -z "${CONDA_EXE}" ]]; then
    echo "ERROR: 'conda' not found. Install miniconda on persistent storage first." >&2
    exit 1
fi
CONDA_BASE="$(${CONDA_EXE} info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "  conda           : ${CONDA_EXE}"
echo "  env prefix      : ${ENV_PREFIX}"
echo "  conda pkg cache : ${CONDA_PKGS_DIR}"
echo "  pip cache       : ${PIP_CACHE_DIR_PERSIST}"

# ---------------------------------------------------------------------------
# Fast path: if the env already has a working detectron2, we're done.
# This makes `bash setup_env.sh` a no-op after the first successful run,
# even on a brand-new node.
# ---------------------------------------------------------------------------
if [[ "${FORCE:-0}" != "1" ]] && [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    if "${ENV_PREFIX}/bin/python" -c \
        'import detectron2, torch; from detectron2 import _C' >/dev/null 2>&1; then
        echo "=== Env '${ENV_PREFIX}' is already fully set up -> skipping. ==="
        echo "    (run with FORCE=1 to reinstall)"
        echo "    activate with: conda activate ${ENV_PREFIX}"
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# [1/5] Create env (idempotent)
# ---------------------------------------------------------------------------
echo "=== [1/5] Create conda env '${ENV_NAME}' (python=${PY_VER}) ==="

# Recent conda builds refuse to use the Anaconda default channels
# (`pkgs/main`, `pkgs/r`) until their Terms of Service have been accepted.
# Try to accept them silently; if that isn't supported or fails, fall back
# to conda-forge only (which has no ToS gate).
accept_tos_or_use_conda_forge() {
    local ok=1
    if conda tos --help >/dev/null 2>&1; then
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || ok=0
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || ok=0
    else
        ok=0
    fi
    if [[ "${ok}" != "1" ]]; then
        echo "Could not auto-accept Anaconda ToS -> using conda-forge channel only."
        CREATE_CHANNEL_ARGS=(-c conda-forge --override-channels)
    else
        CREATE_CHANNEL_ARGS=()
    fi
}

# If FORCE=1 the user wants a true clean reinstall -> remove the env dir.
# (The fast-path check above is also bypassed via FORCE=1, so we only reach
# this branch when the user has explicitly asked to rebuild.)
if [[ "${FORCE:-0}" == "1" ]] && [[ -d "${ENV_PREFIX}" ]]; then
    echo "FORCE=1 -> removing existing env at ${ENV_PREFIX}"
    # `conda env remove` is the polite way, but it sometimes fails on prefix
    # envs; fall back to plain rm -rf which always works.
    conda env remove -y -p "${ENV_PREFIX}" >/dev/null 2>&1 || true
    rm -rf "${ENV_PREFIX}"
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "Env '${ENV_PREFIX}' already exists -> reusing."
else
    accept_tos_or_use_conda_forge
    if ! conda create -y -p "${ENV_PREFIX}" "${CREATE_CHANNEL_ARGS[@]}" "python=${PY_VER}"; then
        echo "First attempt failed -> retrying with conda-forge only."
        conda create -y -p "${ENV_PREFIX}" -c conda-forge --override-channels "python=${PY_VER}"
    fi
fi
# conda's activation hooks (esp. those shipped by gcc_linux-64 /
# gxx_linux-64 / cuda-*) reference variables unconditionally and break
# under `set -u`. Relax strict mode around `conda activate`.
set +u
conda activate "${ENV_PREFIX}"
set -u

# NOTE: pin setuptools<81. torch 2.1's `torch.utils.cpp_extension` does
#   `from pkg_resources import packaging`
# which blows up with setuptools>=81 (that module was removed). detectron2's
# setup.py imports cpp_extension at top level, so an un-pinned setuptools
# makes the source build fail before it even starts.
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81" "packaging" "ninja"

# ---------------------------------------------------------------------------
# [2/5] Install PyTorch from the official index
# ---------------------------------------------------------------------------
echo "=== [2/5] Install torch==${TORCH_VER} + torchvision==${TORCHVISION_VER} (${CUDA_VER}) ==="
python -m pip install \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VER}"

# Sanity-check torch before anything else tries to build against it.
python - <<'PY'
import torch, torchvision
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("cuda build  :", torch.version.cuda)
print("cuda avail  :", torch.cuda.is_available())
PY

# ---------------------------------------------------------------------------
# [3/5] Install Python deps required by the ViTDet configs
# ---------------------------------------------------------------------------
echo "=== [3/5] Install project Python dependencies ==="
# Use opencv-python-headless instead of opencv-python: the compute nodes
# have no libGL.so.1 (no display server), and the headless build doesn't
# link against it. Remove any prior non-headless install first so both
# variants don't fight over the `cv2` namespace.
python -m pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
python -m pip install \
    "opencv-python-headless>=4.5" \
    "pycocotools>=2.0.6" \
    "fvcore>=0.1.5.post20221221" \
    "iopath>=0.1.9,<0.1.10" \
    "omegaconf>=2.1" \
    "hydra-core>=1.1" \
    "yacs>=0.1.8" \
    "termcolor>=1.1" \
    "cloudpickle" \
    "tabulate" \
    "matplotlib" \
    "Pillow" \
    "timm>=0.9" \
    "numpy<2"   # detectron2 has a few spots that still assume numpy 1.x

# Re-pin numpy AFTER everything else, because opencv/headless/etc. routinely
# pull in numpy 2.x as a newer candidate and silently upgrade it. torch 2.1.2
# and pycocotools were built against numpy 1.x, so a numpy-2 env will emit
# `_ARRAY_API not found` warnings and eventually crash at runtime.
python -m pip install --force-reinstall --no-deps "numpy<2"

# ---------------------------------------------------------------------------
# [4/5] Install detectron2
# ---------------------------------------------------------------------------
echo "=== [4/5] Install detectron2 ==="
TORCH_MAJMIN="$(python -c 'import torch,sys; v=torch.__version__.split("+")[0]; print(".".join(v.split(".")[:2]))')"
WHEEL_URL="https://dl.fbaipublicfiles.com/detectron2/wheels/${CUDA_VER}/torch${TORCH_MAJMIN}/index.html"
echo "Trying pre-built wheel index: ${WHEEL_URL}"
if python -m pip install detectron2 -f "${WHEEL_URL}"; then
    echo "Installed detectron2 from pre-built wheel."
else
    echo "Pre-built wheel not available, building from source (this takes a few minutes)..."

    # -----------------------------------------------------------------------
    # Ensure an nvcc that MATCHES torch's CUDA build is available.
    #
    # torch 2.1.2 + cu121 refuses to build any C++/CUDA extension unless the
    # nvcc it finds is from CUDA 12.1 (major version must match exactly).
    # The system image here ships CUDA 13.0, which trips `_check_cuda_version`.
    #
    # We therefore install a 12.1 toolkit INTO THE CONDA ENV (no root needed,
    # cached on persistent storage via CONDA_PKGS_DIRS, nothing touches the
    # host). Then we point CUDA_HOME + PATH at it for the build.
    # -----------------------------------------------------------------------
    TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
    echo "torch was built against CUDA: ${TORCH_CUDA:-<none>}"

    if [[ -n "${TORCH_CUDA}" ]] && [[ ! -x "${ENV_PREFIX}/bin/nvcc" ]]; then
        echo "Installing matching CUDA ${TORCH_CUDA} toolkit into the env..."
        # Conda (de)activation hooks shipped by cuda/gxx packages reference
        # variables unconditionally and break under `set -u`. Relax just here.
        set +u
        # `nvidia/label/cuda-<ver>` has exact-version sub-channels.
        if ! conda install -y -p "${ENV_PREFIX}" \
                -c "nvidia/label/cuda-${TORCH_CUDA}.0" --override-channels \
                cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl ; then
            echo "Exact-version channel failed, trying generic nvidia channel..."
            conda install -y -p "${ENV_PREFIX}" -c nvidia --override-channels \
                "cuda-nvcc=${TORCH_CUDA}.*" \
                "cuda-cudart-dev=${TORCH_CUDA}.*" \
                "cuda-libraries-dev=${TORCH_CUDA}.*"
        fi
        set -u
    fi

    # -----------------------------------------------------------------------
    # Ensure a host compiler that CUDA 12.1's nvcc will accept.
    #
    # nvcc <= 12.1 rejects GCC >= 13 via host_config.h:
    #   "#error -- unsupported GNU version! gcc versions later than 12 ..."
    # Modern base images (Ubuntu 24.04 / Debian trixie) ship GCC 13+ as the
    # system default, so compiling detectron2's .cu files against the system
    # compiler is impossible.
    #
    # Fix: install a conda-forge gxx_linux-64 <= 12 into the env and point
    # nvcc at it via NVCC_CCBIN (and CC/CXX for the C++-only files).
    # Nothing leaks out of the env.
    # -----------------------------------------------------------------------
    if [[ ! -x "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]]; then
        echo "Installing conda-forge gcc/gxx 12 into the env (nvcc host compiler)..."
        # `conda install` runs the env's (de)activation hooks, some of which
        # reference variables unconditionally and blow up under `set -u`.
        # Temporarily relax strict mode just for this call.
        set +u
        conda install -y -p "${ENV_PREFIX}" -c conda-forge --override-channels \
            "gcc_linux-64=12.*" "gxx_linux-64=12.*" "sysroot_linux-64>=2.17"
        set -u
    fi
    CONDA_GCC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    CONDA_GXX="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
    if [[ -x "${CONDA_GXX}" ]]; then
        # Force our compilers; the gxx_linux-64 activation hook exports its
        # own CC/CXX, but we want to pin them explicitly for pip's build.
        export CC="${CONDA_GCC}"
        export CXX="${CONDA_GXX}"
        # nvcc honours -ccbin / NVCC_CCBIN (and PyTorch's cpp_extension
        # forwards it as `-ccbin ${CXX}`).
        export NVCC_CCBIN="${CONDA_GXX}"
        echo "Host compiler for nvcc: ${CONDA_GXX}"
        "${CONDA_GXX}" --version | head -n1 || true
    else
        echo "WARNING: conda gcc not found; build will likely fail against GCC 13+."
    fi

    # Point the build at the env's CUDA, shadowing any system CUDA on PATH.
    export CUDA_HOME="${ENV_PREFIX}"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    # FORCE_CUDA=1 makes detectron2's setup.py always build CUDA ops even if
    # `torch.cuda.is_available()` returns False at build time (e.g. headless
    # login node without a visible GPU).
    export FORCE_CUDA=1
    # Build for a reasonable set of compute capabilities. Override via env if
    # you target something exotic. 8.0 covers A100, 8.6 covers A10/RTX 30xx,
    # 9.0 covers H100.
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;9.0}"

    echo "Using nvcc: $(command -v nvcc || echo '<missing>')"
    nvcc --version || true

    # --no-build-isolation so the build uses OUR torch instead of fetching a
    # fresh one in an isolated env (which would break the ABI link).
    python -m pip install --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git'
fi

# ---------------------------------------------------------------------------
# [5/5] Fetch upstream ViTDet project tree and final sanity check
# ---------------------------------------------------------------------------
echo "=== [5/5] Fetch upstream ViTDet tree ==="
VITDET_DIR="${HERE}/upstream_vitdet"
if [[ -d "${VITDET_DIR}/.git" ]]; then
    echo "Already cloned -> git pull"
    git -C "${VITDET_DIR}" pull --ff-only || true
elif [[ -d "${VITDET_DIR}" ]] && [[ -n "$(ls -A "${VITDET_DIR}" 2>/dev/null || true)" ]]; then
    echo "Directory already populated -> skipping clone: ${VITDET_DIR}"
else
    TMP_CLONE="${HERE}/_detectron2_sparse"
    rm -rf "${TMP_CLONE}"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/facebookresearch/detectron2.git "${TMP_CLONE}"
    git -C "${TMP_CLONE}" sparse-checkout set projects/ViTDet configs/common tools
    mkdir -p "${VITDET_DIR}"
    cp -r "${TMP_CLONE}/projects/ViTDet/"* "${VITDET_DIR}/"
    cp -r "${TMP_CLONE}/tools" "${VITDET_DIR}/tools"
    rm -rf "${TMP_CLONE}"
fi

echo "=== Sanity check ==="
python - <<'PY'
import detectron2, torch, torchvision
print("detectron2  :", detectron2.__version__)
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("cuda avail  :", torch.cuda.is_available())
# Force-load a CUDA op to verify the extension actually links.
try:
    from detectron2 import _C  # noqa: F401
    print("detectron2._C: loaded OK")
except Exception as e:
    print("detectron2._C: FAILED ->", e)
    raise SystemExit(1)
PY

echo ""
echo "Done."
echo "  Env prefix      : ${ENV_PREFIX}"
echo "  Activate with   : conda activate ${ENV_PREFIX}"
echo "  ViTDet code in  : ${VITDET_DIR}"
echo "  Train entry pt  : ${VITDET_DIR}/tools/lazyconfig_train_net.py"
echo ""
echo "Re-running this script on a fresh node will now be a no-op because"
echo "the env, the conda pkg cache and the pip cache all live under"
echo "  ${PERSIST_ROOT}"
echo "(set FORCE=1 to reinstall from scratch)."
