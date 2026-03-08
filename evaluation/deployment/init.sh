# #!/usr/bin/env bash
# set -euo pipefail

# PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix/"
# cd "$PROJECT_ROOT"

# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"
# export ftp_proxy="http://star-proxy.oa.com:3128"

# # Load/create the conda env from the shared drive at PROJECT_ROOT/..
# SHARED_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
# # ENV_PATH="$SHARED_ROOT/labelmix_env"

# # if [[ ! -d "$ENV_PATH" ]]; then
# #   conda env create -f env.yaml -p "$ENV_PATH"
# # fi

# echo "
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# " >> ~/.bashrc

# echo ""
# conda activate labelmix
# #echo "conda activate $ENV_PATH" >> ~/.bashrc
# # echo "cd $PROJECT_ROOT" >> ~/.bashrc
# echo "export http_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
# echo "export https_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc
# echo "export ftp_proxy=http://star-proxy.oa.com:3128" >> ~/.bashrc

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/labelmix"
CONDA_ROOT="$PROJECT_ROOT/miniconda3"
CONDA_BIN="$CONDA_ROOT/bin/conda"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
ENV_NAME="labelmix"
PROXY_URL="http://star-proxy.oa.com:3128"

cd "$PROJECT_ROOT"

# 1) Set proxy first
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export FTP_PROXY="$PROXY_URL"

# Optional for apt specifically
echo "Configuring apt proxy..."
sudo mkdir -p /etc/apt/apt.conf.d
cat <<EOF | sudo tee /etc/apt/apt.conf.d/95proxy >/dev/null
Acquire::http::Proxy "$PROXY_URL";
Acquire::https::Proxy "$PROXY_URL";
EOF

# 2) Update and upgrade
echo "Updating package lists..."
sudo apt-get update

echo "Upgrading installed packages..."
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# 3) Install bash
echo "Installing bash..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y bash

# 4) Add proxy exports to ~/.bashrc if not already present
if ! grep -q 'star-proxy.oa.com:3128' ~/.bashrc; then
  cat <<EOF >> ~/.bashrc

# Proxy settings
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export ftp_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export FTP_PROXY="$PROXY_URL"
EOF
fi

# 5) Add conda init block to ~/.bashrc if not already present
if ! grep -q '# >>> conda initialize >>>' ~/.bashrc; then
  cat <<EOF >> ~/.bashrc

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$("$CONDA_BIN" 'shell.bash' 'hook' 2> /dev/null)"
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
# <<< conda initialize <<<
EOF
fi

# Add auto-activation for bash shells if not already present
if ! grep -q "conda activate $ENV_NAME" ~/.bashrc; then
  cat <<EOF >> ~/.bashrc

# Auto-activate conda environment
conda activate $ENV_NAME
EOF
fi

# Activate conda for the current script session too
if [ -f "$CONDA_SH" ]; then
  . "$CONDA_SH"
else
  export PATH="$CONDA_ROOT/bin:$PATH"
fi

echo "Activating conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "Done."