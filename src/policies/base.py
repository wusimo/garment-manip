"""Policy interface for garment-manip experiments.

The abstraction is intentionally minimal: a Policy owns the full interaction
pattern with a DexGarmentLab `Env`. We do not impose a gym-style step()/obs()
boundary on top of DexGarmentLab — the existing tasks operate at the
"waypoint + open/close" abstraction, while VLA policies operate at the
"per-step joint command" abstraction. Forcing both through the same step()
would distort one or the other.

Each Policy implementation is responsible for:
  - perception (calling env.<camera>.get_*)
  - decision-making (model inference or hardcoded heuristic)
  - actuation (calling env.bimanual_dex.* or env.<arm>.set_joint_positions)
  - sim stepping (env.step() / env.world.step())
  - termination judgment

The Env only provides primitives; the Policy orchestrates.
"""

from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    @abstractmethod
    def reset(self, instruction: str) -> None:
        """Reset internal state before each episode. `instruction` is the
        natural-language task description (used by language-conditioned
        policies; ignored by HALO)."""

    @abstractmethod
    def run_episode(self, env: Any) -> dict:
        """Run a full episode against `env` (a DexGarmentLab task env, e.g.
        FlingTops_Env). Return a metrics dict; required keys:
            - "success": bool
            - "policy_name": str
        Implementations may add task-specific fields (flatten_proportion,
        step_count, etc.)."""
