#!/usr/bin/env bash
# Launch DexGarmentLab's Fling_Tops_Env.py in the garment-45 env with GUI.
# Must be run from a session that has $DISPLAY set (any GNOME terminal does).
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DGL="$REPO_ROOT/vendor/DexGarmentLab"
ENV_NAME="garment-45"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# DexGarmentLab uses sys.path.append(os.getcwd()) to resolve `Env_Config.*`
# and `Env_StandAlone.*`, so the working dir must be the repo root.
cd "$DGL"

# Pipe Yes for the Omniverse EULA in case the user hasn't accepted it yet.
yes Yes | python -u Env_StandAlone/Fling_Tops_Env.py "$@"
