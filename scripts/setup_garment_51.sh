#!/usr/bin/env bash
# Build the garment-51 conda env: Isaac Sim 5.1.0 + Isaac Lab v2.3.x from source.
# Idempotent: re-running skips the conda create if the env already exists.
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
ENV_NAME="garment-51"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] env $ENV_NAME already exists — skipping create"
else
  echo "[setup] creating conda env $ENV_NAME"
  conda env create -f "$REPO_ROOT/envs/garment-51.yml"
fi

conda activate "$ENV_NAME"

echo "[setup] upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing torch 2.7.0 (cu128)"
pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[setup] installing Isaac Sim 5.1.0"
pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

echo "[setup] installing Isaac Lab from source (./isaaclab.sh -i)"
cd "$REPO_ROOT/vendor/IsaacLab"
./isaaclab.sh -i

echo "[setup] done. activate with:  conda activate $ENV_NAME"
