#!/usr/bin/env bash
# Launch a starVLA inference server (websocket on $PORT).
#
# Runs in the `starVLA` conda env, separate from `garment-45` — Isaac Sim and
# starVLA's torch stack are version-incompatible, so we keep them in different
# processes and bridge over a websocket.
#
# Usage:
#   bash scripts/run_starvla_server.sh <CKPT_PATH> [--port 5694]
#
# CKPT_PATH should point to a starVLA HuggingFace checkpoint trained on
# RoboTwin (bimanual). See docs/vla_integration.md for the search query.

set -euo pipefail

# Default to the public RoboTwin-2.0 checkpoint (OFT variant — Qwen-PI ckpt is
# starVLA team-internal). Override by passing a path as $1, e.g. a local
# fine-tune or a different HuggingFace repo.
DEFAULT_CKPT="StarVLA/Qwen3-VL-OFT-Robotwin2"

if [[ $# -ge 1 && "$1" != --* ]]; then
  CKPT_PATH="$1"
  shift
else
  CKPT_PATH="$DEFAULT_CKPT"
fi

STARVLA_ROOT="/home/simo/Documents/starVLA"
ENV_NAME="starVLA"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$STARVLA_ROOT"

# --use_bf16: matches starVLA's run_policy_server.sh; halves VRAM (24 GB RTX 3090 fits comfortably).
echo "[run_starvla_server] ckpt=$CKPT_PATH"
python -u deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT_PATH" \
  --use_bf16 \
  "$@"
