# garment-manip

Research workspace for **bimanual garment manipulation** with Isaac Sim / Isaac Lab and DexGarmentLab.

## Layout

```
garment-manip/
├── src/                # your algorithms / policies / wrappers
├── configs/            # task / training configs
├── notebooks/          # exploration
├── scripts/            # one-off runners (training, eval, data collection)
├── envs/               # conda env definitions
├── data/               # (gitignored) garment assets, demos, replay buffers
└── vendor/
    ├── DexGarmentLab/  # submodule — pinned to Isaac Sim 4.5.0
    └── IsaacLab/       # submodule — pinned to v2.3.2 (Isaac Sim 5.1.0)
```

## Two conda envs

DexGarmentLab pins **Isaac Sim 4.5.0 / Python 3.10**, while the latest stable Isaac Lab (v2.3.2) is on **Isaac Sim 5.1.0 / Python 3.11**. We keep them isolated:

| Env           | Python | Isaac Sim | Isaac Lab | Purpose                                |
|---------------|--------|-----------|-----------|----------------------------------------|
| `garment-45`  | 3.10   | 4.5.0     | —         | Run / extend DexGarmentLab tasks       |
| `garment-51`  | 3.11   | 5.1.0     | v2.3.x    | New algorithms on latest Isaac Lab     |

> Isaac Lab 3.0 (built on Isaac Sim 6.0) is still beta — switch to it later by checking out the `feature/3.0` branch in `vendor/IsaacLab` and bumping `setup_garment_51.sh`.

Activate with `conda activate garment-45` or `garment-51`.

## First-time setup

```bash
# clone with submodules
git submodule update --init --recursive

# create both envs (slow — pulls ~10 GB each)
bash scripts/setup_garment_45.sh
bash scripts/setup_garment_51.sh
```

## Smoke tests

```bash
conda activate garment-45 && python scripts/smoke_test.py
conda activate garment-51 && python scripts/smoke_test.py
```

## Download DexGarmentLab assets

DexGarmentLab tasks load ~22 GB of garments / robots / scenes from HuggingFace. The `Garment.zip` (~5 GB) tends to drop mid-download, so the script wraps `snapshot_download` in a retry loop:

```bash
bash scripts/download_dexgarment_assets.sh
# downloads into vendor/DexGarmentLab/Assets/{Garment,Robots,Scene,LeapMotion,Human}
```

## Run a DexGarmentLab task (verified)

```bash
DISPLAY=:1 bash scripts/run_fling_tops.sh
```

Launches `Env_StandAlone/Fling_Tops_Env.py` in `garment-45`: bimanual UR10e + LeapHand flings a tops garment, the HALO/GAM affordance model picks two grasp points, the executed fling reaches **~0.94 flatten proportion** and the judge returns `True`. The viewer stays open in an idle loop until you close it.

To run a different task, follow the same pattern (substitute the task script):

```bash
DISPLAY=:1 \
  conda run -n garment-45 \
  bash -c "cd vendor/DexGarmentLab && yes Yes | python -u Env_StandAlone/<TaskName>_Env.py"
```

Available tasks live in `vendor/DexGarmentLab/Env_StandAlone/` (Fold_Tops, Hang_Coat, Wear_Baseballcap, …).

## Setup gotchas (already handled by the scripts)

1. **EULA prompt** — `isaaclab.sh -i` and the first `from isaacsim import SimulationApp` both prompt for the Omniverse EULA; the setup scripts pipe `yes Yes |` to keep them non-interactive.
2. **CMake 4.x + egl_probe** — `egl_probe` (a `robomimic` dep) needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` to build under CMake 4.x; exported in `setup_garment_51.sh`.
3. **HF connection drops** — `download_dexgarment_assets.sh` retries `snapshot_download` up to 8 times so the 5 GB Garment archive resumes after stalls.
4. **cwd requirement** — DexGarmentLab task scripts `sys.path.append(os.getcwd())`, so they must be launched with cwd = `vendor/DexGarmentLab` (the runner script handles this).

## Why two envs

DexGarmentLab pins Isaac Sim 4.5; the latest stable Isaac Lab is on Isaac Sim 5.1; Isaac Lab 3.0 / Sim 6.0 is still beta-only. The split avoids version conflicts — keep DexGarmentLab tasks on `garment-45` and new Isaac Lab algorithm work on `garment-51`.
