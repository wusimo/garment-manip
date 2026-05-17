"""Forked Fling_Tops driver that runs through the Policy interface.

Compared to vendor/DexGarmentLab/Env_StandAlone/Fling_Tops_Env.py, this:
  - keeps the FlingTops_Env scene construction (imported from upstream)
  - moves the episode logic (the body of FlingTops()) into a Policy
  - adds --policy {halo,starvla} so we can swap implementations from CLI

Must be launched with cwd = vendor/DexGarmentLab (DexGarmentLab modules
self-register via sys.path.append(os.getcwd())). The launcher script
scripts/run_fling_tops_policy.sh handles this plus PYTHONPATH for src/.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import os
import sys
import time

import numpy as np
from termcolor import cprint

sys.path.append(os.getcwd())

from Env_StandAlone.Fling_Tops_Env import FlingTops_Env

# garment-manip src/ is on PYTHONPATH via the launcher script.
from policies.halo_policy import HALOFlingTopsPolicy


def build_policy(name: str):
    if name == "halo":
        return HALOFlingTopsPolicy()
    if name == "starvla":
        # Imported lazily so the websocket client is only required when used.
        from policies.starvla_policy import StarVLAPolicy

        return StarVLAPolicy(
            host=os.environ.get("STARVLA_HOST", "127.0.0.1"),
            port=int(os.environ.get("STARVLA_PORT", "5694")),
        )
    raise ValueError(f"unknown policy: {name!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["halo", "starvla"], default="halo")
    p.add_argument("--instruction", type=str, default="fling and flatten the t-shirt")
    p.add_argument("--garment-random", action="store_true")
    p.add_argument("--keep-window-open", action="store_true",
                   help="leave the viewer running after the episode (for visual inspection)")
    return p.parse_args()


def sample_initial_state(garment_random: bool):
    pos = np.array([0.0, 0.5, 0.2])
    ori = np.array([65.0, 0.0, 0.0])
    usd_path = None
    if garment_random:
        np.random.seed(int(time.time()))
        x = np.random.uniform(-0.1, 0.1)
        y = np.random.uniform(0.5, 0.7)
        angle = np.random.uniform(65.0, 80.0)
        pos = np.array([x, y, 0.0])
        ori = np.array([angle, 0.0, 0.0])
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        listfile = os.path.join(
            base, "vendor/DexGarmentLab/Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_training_list.txt"
        )
        if os.path.exists(listfile):
            with open(listfile) as f:
                assets = [ln.rstrip("\n") for ln in f]
            usd_path = os.path.join(os.getcwd(), np.random.choice(assets))
    return pos, ori, usd_path


def main():
    args = parse_args()
    pos, ori, usd_path = sample_initial_state(args.garment_random)

    env = FlingTops_Env(pos, ori, usd_path, ground_material_usd=None, record_video_flag=False)
    policy = build_policy(args.policy)
    policy.reset(args.instruction)

    cprint(f"[run] policy={args.policy}  instruction={args.instruction!r}", color="yellow")
    metrics = policy.run_episode(env)
    cprint(f"[run] metrics={metrics}", color="yellow")

    if args.keep_window_open:
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
