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