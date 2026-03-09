#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
CONDA_ROOT="$PROJECT_ROOT/miniconda3"
CONDA_BIN="$CONDA_ROOT/bin/conda"
PROXY_URL="http://star-proxy.oa.com:3128"

cd "$PROJECT_ROOT"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"

# Make sure conda is callable
if ! command -v conda >/dev/null 2>&1; then
    export PATH="$CONDA_ROOT/bin:$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not callable. Expected at $CONDA_BIN" >&2
    exit 1
fi

# Persist conda setup for future bash shells
conda init bash

# Persist proxy for future shells, without duplicating
if ! grep -q '# >>> labelmix proxy >>>' "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<EOF

# >>> labelmix proxy >>>
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"
# <<< labelmix proxy <<<
EOF
fi

# Optional: auto-activate labelmix in future interactive bash shells
if ! grep -q '^conda activate labelmix$' "$HOME/.bashrc"; then
    printf '\nconda activate labelmix\n' >> "$HOME/.bashrc"
fi

# Enable conda in this current shell
eval "$("$CONDA_BIN" shell.bash hook)"

# Activate env now
conda activate labelmix

echo "Done. Current env: ${CONDA_DEFAULT_ENV:-<none>}"