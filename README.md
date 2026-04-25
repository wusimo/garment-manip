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
    ├── DexGarmentLab/  # submodule — pinned to Isaac Sim 4.5
    └── IsaacLab/       # submodule — main branch (Isaac Sim 6.0)
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
