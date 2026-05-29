#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-fedpath-r139}"
BASE_ENV="${BASE_ENV:-a2a_local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$HOME/anaconda3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[bootstrap] cloning conda env $BASE_ENV -> $ENV_NAME"
  conda create --name "$ENV_NAME" --clone "$BASE_ENV" -y
else
  echo "[bootstrap] conda env $ENV_NAME already exists"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_DIR"
python - <<'PY'
import importlib.util as u
mods = ["torch", "torchvision", "flwr", "medmnist", "pandas", "matplotlib"]
missing = [m for m in mods if u.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing dependencies: {missing}")
print("dependencies ok")
PY
