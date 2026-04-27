"""Smoke test — runs in either env. Confirms Isaac Sim launches headless and a stage steps.

The Isaac Sim Python API was renamed in 5.0: pre-5.0 used `omni.isaac.core`,
5.0+ uses `isaacsim.core.api`. We probe both so this works in garment-45 and garment-51.
"""
import sys


def main() -> None:
    from isaacsim import SimulationApp  # type: ignore

    app = SimulationApp({"headless": True})

    import torch

    try:
        from isaacsim.core.api import World  # Isaac Sim >= 5.0
    except ImportError:
        from omni.isaac.core import World  # Isaac Sim 4.x

    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")

    world = World()
    world.scene.add_default_ground_plane()
    world.reset()
    for _ in range(60):
        world.step(render=False)
    print("stepped 60 frames OK")

    app.close()


if __name__ == "__main__":
    main()
