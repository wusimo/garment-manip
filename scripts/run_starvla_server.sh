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

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <CKPT_PATH> [--port PORT] [--use_bf16]"
  echo "  CKPT_PATH: local path or HuggingFace repo (e.g. StarVLA/Qwen-PI-RoboTwin)"
  exit 1
fi

CKPT_PATH="$1"
shift

STARVLA_ROOT="/home/simo/Documents/starVLA"
ENV_NAME="starVLA"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$STARVLA_ROOT"

python -u deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT_PATH" \
  "$@"
