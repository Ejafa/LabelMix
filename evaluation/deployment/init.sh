#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
CONDA_ROOT="$PROJECT_ROOT/miniconda3"
CONDA_BIN="$CONDA_ROOT/bin/conda"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
BASHRC="$HOME/.bashrc"
PROXY_URL="http://star-proxy.oa.com:3128"

cd "$PROJECT_ROOT"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"

touch "$BASHRC"

if ! grep -q '# >>> labelmix proxy >>>' "$BASHRC"; then
  cat >> "$BASHRC" <<EOF

# >>> labelmix proxy >>>
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"
# <<< labelmix proxy <<<
EOF
fi

if ! grep -q '# >>> labelmix conda init >>>' "$BASHRC"; then
  cat >> "$BASHRC" <<EOF

# >>> labelmix conda init >>>
__conda_setup="\$("$CONDA_BIN" shell.bash hook 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__conda_setup"
else
    if [ -f "$CONDA_SH" ]; then
        . "$CONDA_SH"
    else
        export PATH="$CONDA_ROOT/bin:\$PATH"
    fi
fi
unset __conda_setup
# <<< labelmix conda init <<<
EOF
fi

if ! grep -q '^conda activate labelmix$' "$BASHRC"; then
  echo 'conda activate labelmix' >> "$BASHRC"
fi

# Activate labelmix in the current script too
__conda_setup="$("$CONDA_BIN" shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_SH" ]; then
        . "$CONDA_SH"
    else
        export PATH="$CONDA_ROOT/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate labelmix

echo "Done. Current env: ${CONDA_DEFAULT_ENV:-<none>}"