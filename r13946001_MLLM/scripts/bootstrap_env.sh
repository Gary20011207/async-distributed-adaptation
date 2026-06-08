#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-fed-mllm-r139}"
BASE_ENV="${BASE_ENV:-fedpath-r139}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME already exists"
else
  echo "Cloning $BASE_ENV to $ENV_NAME"
  conda create --name "$ENV_NAME" --clone "$BASE_ENV" -y
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -e "$ROOT_DIR"
echo "Ready: conda activate $ENV_NAME"
