"""Smoke test — runs in either env. Confirms Isaac Sim launches headless and a stage steps."""
import sys


def main() -> None:
    from isaacsim import SimulationApp  # type: ignore

    app = SimulationApp({"headless": True})

    import torch
    from omni.isaac.core import World  # type: ignore

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
