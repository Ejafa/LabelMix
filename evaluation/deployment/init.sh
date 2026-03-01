#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
cd "$PROJECT_ROOT"

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

conda env create -f env.yaml -n labelmix

echo ""
echo "conda activate labelmix" >> ~/.bashrc
echo "cd $PROJECT_ROOT" >> ~/.bashrc
echo "export http_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
echo "export https_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
echo "export ftp_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc