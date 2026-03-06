#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
cd "$PROJECT_ROOT"

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

# Load/create the conda env from the shared drive at PROJECT_ROOT/..
SHARED_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
ENV_PATH="${ENV_PATH:-$SHARED_ROOT/labelmix_env}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/env.yaml}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: conda env file not found at ${ENV_FILE}." >&2
  echo "Set ENV_FILE to your env yaml path (e.g. export ENV_FILE=/path/to/environment.yml)." >&2
  exit 1
fi

if [[ ! -d "$ENV_PATH" ]]; then
  conda env create -f "$ENV_FILE" -p "$ENV_PATH"
else
  conda env update -f "$ENV_FILE" -p "$ENV_PATH" --prune
fi

# Guarantee env-local libs are searched first whenever this env is activated.
mkdir -p "$ENV_PATH/etc/conda/activate.d" "$ENV_PATH/etc/conda/deactivate.d"
cat > "$ENV_PATH/etc/conda/activate.d/ld_library_path.sh" <<'EOF'
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
cat > "$ENV_PATH/etc/conda/deactivate.d/ld_library_path.sh" <<'EOF'
export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH:-}"
unset _OLD_LD_LIBRARY_PATH
EOF

# Clear Ray cache
# 2) Remove local Ray metadata/state (default temp dir)
rm -rf /tmp/ray/session_* /tmp/ray/session_latest /tmp/ray/ray_current_cluster
# optional: if you want a totally clean local cache too
rm -rf ~/.cache/ray

add_bashrc_line() {
  local line="$1"
  grep -Fqx "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}

echo ""
add_bashrc_line "conda activate $ENV_PATH"
add_bashrc_line "cd $PROJECT_ROOT"
add_bashrc_line "export http_proxy=http://star-proxy.oa.com:3128"
add_bashrc_line "export https_proxy=http://star-proxy.oa.com:3128"
add_bashrc_line "export ftp_proxy=http://star-proxy.oa.com:3128"
