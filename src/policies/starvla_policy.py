"""StarVLAPolicy — closed-loop control via a starVLA websocket server.

ACTION-SPACE CONTRACT (load-bearing — read before debugging):

  starVLA (RoboTwin-2.0 pretrained checkpoint):
    output: (chunk_size, 14)
    layout after model2robotwin re-ordering:
      [0:6]   left arm joint positions  (6-DoF)
      [6:12]  right arm joint positions (6-DoF)
      [12]    left gripper             (continuous, 0=closed → 1=open, parallel-jaw)
      [13]    right gripper            (continuous, 0=closed → 1=open, parallel-jaw)

  DexGarmentLab (UR10e + LeapHand):
    expected per arm: 6 UR10e arm joints + 24 LeapHand joints = 30-DoF
    bimanual total: 60-DoF
    The arm joint convention may differ from RoboTwin's ARX X5 — first-attempt
    assumption is that both follow standard URDF order, which will not hold for
    the actual joint values.

  Bridge (first-attempt, lossy by design — see docs/vla_integration.md):
    1. starVLA arm joints[0:6]  → dexleft.set_joint_positions(arm_only=True)
       starVLA arm joints[6:12] → dexright.set_joint_positions(arm_only=True)
       (No retargeting between ARX and UR10e kinematics — values will be off.)
    2. starVLA gripper[12,13] → discrete LeapHand state:
       gripper >= 0.5 → set_both_hand_state("open")
       gripper <  0.5 → set_both_hand_state("close")
       (Collapses 24-DoF dexterous hand to 1-bit gripper command.)

  Expected first-run outcome: the robot will move but actions will be
  nonsensical for the task. The goal of run #1 is to verify the loop
  end-to-end and surface concrete failure modes, not to succeed at fling.
"""

import os
from typing import Optional

import numpy as np

from .base import Policy


class StarVLAPolicy(Policy):
    """Talks to a starVLA Qwen-PI websocket server. The server is launched
    separately in the `starVLA` conda env via scripts/run_starvla_server.sh —
    keeping it out-of-process avoids the Isaac Sim ↔ PyTorch env conflict."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5694,
        unnorm_key: Optional[str] = "robotwin",
        action_mode: str = "abs",
        max_steps: int = 200,
        sim_steps_per_action: int = 5,
    ):
        # Import here to keep the import cost out of the env-init path; the
        # websocket client only loads when the policy is actually instantiated.
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.action_mode = action_mode
        self.max_steps = max_steps
        self.sim_steps_per_action = sim_steps_per_action
        self.instruction: Optional[str] = None
        self.step_idx = 0
        self.action_chunk: Optional[np.ndarray] = None
        self.action_chunk_size: Optional[int] = None

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.step_idx = 0
        self.action_chunk = None

    def run_episode(self, env) -> dict:
        from termcolor import cprint
        from Env_Config.Utils_Project.Flatten_Judge import judge_fling

        assert self.instruction is not None, "Call reset(instruction) before run_episode."

        image_judge = env.judge_camera.get_rgb_graph()

        for self.step_idx in range(self.max_steps):
            obs = self._collect_observation(env)
            action_14d = self._predict_action(obs, self.step_idx)
            self._apply_action(env, action_14d)

            for _ in range(self.sim_steps_per_action):
                env.step()

        image_end = env.garment_camera.get_rgb_graph()
        success = bool(judge_fling(image_judge, image_end, threshold=0.2))
        cprint(f"[StarVLAPolicy] steps={self.step_idx + 1}  success={success}", color="cyan")

        return {
            "success": success,
            "policy_name": "starvla_robotwin",
            "steps": self.step_idx + 1,
        }

    def _collect_observation(self, env) -> dict:
        """Build the obs dict expected by starVLA's RoboTwin interface:
            {"lang": str, "image": [head, left, right], "state": (14,) joint vec}
        We map DexGarmentLab cameras → RoboTwin's expected camera triplet:
            env_camera  → "head" (overhead third-person view)
            garment_camera → "left" (top-down placeholder)
            judge_camera → "right" (side placeholder)
        These placeholders are not training-distribution-correct; see doc.
        """
        head_rgb = env.env_camera.get_rgb_graph(save_or_not=False)
        left_rgb = env.garment_camera.get_rgb_graph(save_or_not=False)
        right_rgb = env.judge_camera.get_rgb_graph(save_or_not=False)

        # Drop alpha if present.
        head_rgb = head_rgb[..., :3] if head_rgb.shape[-1] == 4 else head_rgb
        left_rgb = left_rgb[..., :3] if left_rgb.shape[-1] == 4 else left_rgb
        right_rgb = right_rgb[..., :3] if right_rgb.shape[-1] == 4 else right_rgb

        # 14-D state vector: 6 left arm + 6 right arm + 2 gripper-state placeholders.
        # We read the full joint vector and extract the arm portion via
        # dexleft.arm_dof_indices (the UR10e arm has its own index list, kept
        # separate from hand_dof_indices). Gripper-state placeholders are 0.0
        # (closed) at episode start — refined once first action is applied.
        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()

        arm_idx_L = np.asarray(env.bimanual_dex.dexleft.arm_dof_indices)
        arm_idx_R = np.asarray(env.bimanual_dex.dexright.arm_dof_indices)

        state_14 = np.concatenate(
            [
                np.asarray(joint_pos_L)[arm_idx_L][:6],
                np.asarray(joint_pos_R)[arm_idx_R][:6],
                np.array([0.0, 0.0]),  # gripper open/closed placeholder
            ]
        ).astype(np.float32)

        return {
            "lang": self.instruction,
            "image": [head_rgb, left_rgb, right_rgb],
            "state": state_14,
        }

    def _predict_action(self, obs: dict, step: int) -> np.ndarray:
        """Query the websocket server. Caches the action chunk and indexes
        into it across calls — same pattern as model2robotwin_interface."""
        if self.action_chunk is None or step % (self.action_chunk_size or 1) == 0:
            vla_input = {
                "examples": [
                    {"lang": obs["lang"], "image": obs["image"]}
                ],
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": 10,
            }
            response = self.client.predict_action(vla_input)
            normalized = np.asarray(response["data"]["normalized_actions"])  # (B, chunk, D)
            self.action_chunk = normalized[0]  # (chunk, D)
            self.action_chunk_size = self.action_chunk.shape[0]
            # NOTE: we skip the unnormalize-via-norm-stats step here for the
            # first attempt — the action stats from the RoboTwin ckpt don't
            # match our setup anyway. Treat outputs as raw and clamp.
            self.action_chunk = np.clip(self.action_chunk, -1.0, 1.0)

        idx = step % (self.action_chunk_size or 1)
        return self.action_chunk[idx]

    def _apply_action(self, env, action_14d: np.ndarray) -> None:
        """Bridge: 14-D starVLA action → DexGarmentLab bimanual_dex calls.
        See module docstring for the contract and caveats."""
        arm_L = action_14d[0:6]
        arm_R = action_14d[6:12]
        grip_L = float(action_14d[12])
        grip_R = float(action_14d[13])

        # Direct joint-position command on each arm. The set_joint_positions
        # signature on Articulation supports indices to write only the arm DOFs.
        arm_idx_L = np.asarray(env.bimanual_dex.dexleft.arm_dof_indices)
        arm_idx_R = np.asarray(env.bimanual_dex.dexright.arm_dof_indices)
        env.bimanual_dex.dexleft.set_joint_positions(np.asarray(arm_L, dtype=np.float32), joint_indices=arm_idx_L)
        env.bimanual_dex.dexright.set_joint_positions(np.asarray(arm_R, dtype=np.float32), joint_indices=arm_idx_R)

        # Discrete hand-state command derived from continuous gripper signals.
        left_state = "open" if grip_L >= 0.5 else "close"
        right_state = "open" if grip_R >= 0.5 else "close"
        env.bimanual_dex.set_both_hand_state(left_hand_state=left_state, right_hand_state=right_state)
