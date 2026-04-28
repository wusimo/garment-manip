#!/usr/bin/env bash
# Download DexGarmentLab's core asset zips (Garment, Robots, Scene, LeapMotion, Human)
# from HuggingFace and unzip them into vendor/DexGarmentLab/Assets/.
# Idempotent: snapshot_download resumes; unzip is skipped if the target dir exists.
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
ASSETS_DIR="$REPO_ROOT/vendor/DexGarmentLab/Assets"
ENV_NAME="garment-45"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$ASSETS_DIR"
cd "$ASSETS_DIR"

echo "[assets] snapshot_download from HuggingFace (resumable) -> $ASSETS_DIR"
# HF connections occasionally drop mid-stream on multi-GB files. Loop until it
# completes — snapshot_download resumes from existing partial files.
attempt=0
until python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="wayrise/DexGarmentLab",
    allow_patterns=["Garment.zip", "Robots.zip", "Scene.zip", "LeapMotion.zip", "Human.zip"],
    local_dir=".",
    repo_type="dataset",
    max_workers=2,
)
PY
do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 8 ]; then
    echo "[assets] giving up after $attempt attempts" >&2
    exit 1
  fi
  echo "[assets] download interrupted, retrying ($attempt)..." >&2
  sleep 5
done

for z in Garment Robots Scene LeapMotion Human; do
  if [ -d "$ASSETS_DIR/$z" ]; then
    echo "[assets] $z/ already exists — skipping unzip"
  elif [ -f "$ASSETS_DIR/$z.zip" ]; then
    echo "[assets] unzipping $z.zip"
    unzip -q "$ASSETS_DIR/$z.zip" -d "$ASSETS_DIR"
  else
    echo "[assets] WARN: $z.zip not found"
  fi
done

echo "[assets] done"
ls -lh "$ASSETS_DIR"
