#!/usr/bin/env bash
# Build the garment-45 conda env: Isaac Sim 4.5.0 + DexGarmentLab deps.
# Idempotent: re-running skips the conda create if the env already exists.
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
ENV_NAME="garment-45"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] env $ENV_NAME already exists — skipping create"
else
  echo "[setup] creating conda env $ENV_NAME"
  conda env create -f "$REPO_ROOT/envs/garment-45.yml"
fi

conda activate "$ENV_NAME"

echo "[setup] upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing Isaac Sim 4.5.0"
pip install "isaacsim[all,extscache]==4.5.0" \
  --extra-index-url https://pypi.nvidia.com

echo "[setup] installing DexGarmentLab requirements"
pip install -r "$REPO_ROOT/vendor/DexGarmentLab/requirements.txt"

echo "[setup] done. activate with:  conda activate $ENV_NAME"
