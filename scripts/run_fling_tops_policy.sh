#!/usr/bin/env bash
# Launch the Policy-driven Fling_Tops env. Picks the policy via --policy {halo,starvla}.
#
# Examples:
#   DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy halo
#   DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy starvla \
#     --instruction "fling the t-shirt to flatten it"
#
# For --policy starvla, the starVLA inference server must already be running
# (see scripts/run_starvla_server.sh) and reachable at $STARVLA_HOST:$STARVLA_PORT
# (defaults: 127.0.0.1:5694).
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DGL="$REPO_ROOT/vendor/DexGarmentLab"
ENV_NAME="garment-45"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Both sys.paths need to be present:
#   1. cwd = $DGL so DexGarmentLab's sys.path.append(os.getcwd()) finds Env_Config.* etc.
#   2. $REPO_ROOT/src on PYTHONPATH so `policies.*` and `envs.*` are importable.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$DGL"

yes Yes | python -u "$REPO_ROOT/src/envs/fling_tops_policy_env.py" "$@"
