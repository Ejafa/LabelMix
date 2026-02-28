#!/usr/bin/env bash
set -euo pipefail

dnf update -y
dnf upgrade -y

VENV_DIR=".venv"

# 1) Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# 2) Activate venv (must be sourced to affect current shell)
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3) Install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Load .env into current shell
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

echo "Venv active and .env loaded."
