#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
cd "$PROJECT_ROOT"

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

# Load/create the conda env from the shared drive at PROJECT_ROOT/..
SHARED_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
ENV_PATH="$SHARED_ROOT/labelmix_env"

if [[ ! -d "$ENV_PATH" ]]; then
  conda env create -f env.yaml -p "$ENV_PATH"
fi

echo ""
echo "conda activate $ENV_PATH" >> ~/.bashrc
echo "cd $PROJECT_ROOT" >> ~/.bashrc
echo "export http_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
echo "export https_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
echo "export ftp_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
