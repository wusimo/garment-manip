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

# Default: the public RoboTwin-2.0 checkpoint (OFT variant). Override by
# passing a .pt path as $1 (e.g. a local fine-tune).
#
# baseframework.from_pretrained needs a Path to the .pt file, not the HF repo
# ID, so we resolve the HF snapshot dir and append checkpoints/*.pt.
DEFAULT_REPO="StarVLA/Qwen3-VL-OFT-Robotwin2"

if [[ $# -ge 1 && "$1" != --* ]]; then
  CKPT_PATH="$1"
  shift
else
  CKPT_PATH=$(conda run -n starVLA python -c "
import os, glob
from huggingface_hub import snapshot_download
root = snapshot_download(repo_id='$DEFAULT_REPO')
ckpts = sorted(glob.glob(os.path.join(root, 'checkpoints', '*.pt')))
assert ckpts, f'no .pt file under {root}/checkpoints'
print(ckpts[-1])
" 2>/dev/null | tail -1)
  if [[ -z "$CKPT_PATH" || ! -f "$CKPT_PATH" ]]; then
    echo "[run_starvla_server] failed to resolve default ckpt; pass a path as \$1." >&2
    exit 1
  fi
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
