"""HALOFlingTopsPolicy — wraps the existing DexGarmentLab Fling_Tops pipeline.

This is a refactor, not a rewrite: the body of `FlingTops()` from
`vendor/DexGarmentLab/Env_StandAlone/Fling_Tops_Env.py` is moved into
`run_episode()` here so the same end-to-end behavior is reachable through the
generic Policy interface.

Other tasks (Fold_Tops, Hang_Coat, ...) will need their own *Policy subclasses
when we extend beyond Fling_Tops; the GAM model itself is task-agnostic but
the post-affordance trajectory is hardcoded per task in the original repo.
"""

import os
import numpy as np
from termcolor import cprint

from isaacsim.core.utils.prims import set_prim_visibility, get_prim_at_path

from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Flatten_Judge import judge_fling
from Env_Config.Room.Object_Tools import set_prim_visible_group

from .base import Policy


class HALOFlingTopsPolicy(Policy):
    """Baseline: GAM affordance + hardcoded 2-stage fling trajectory."""

    def __init__(self):
        self.instruction = None

    def reset(self, instruction: str) -> None:
        self.instruction = instruction  # HALO ignores it

    def run_episode(self, env) -> dict:
        image_judge = env.judge_camera.get_rgb_graph()

        # Stage 1: grasp + lift + spread + release
        self._hide_robot(env)
        pcd = self._capture_garment_pcd(env)
        env.garment_pcd = pcd
        self._show_robot(env)

        manipulation_points, _, points_similarity = env.model.get_manipulation_points(
            input_pcd=pcd, index_list=[838, 179]
        )
        env.points_affordance_feature = normalize_columns(points_similarity.T)
        manipulation_points[:, 2] = 0.002

        env.bimanual_dex.dense_move_both_ik(
            left_pos=manipulation_points[0],
            left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
            right_pos=manipulation_points[1],
            right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        )
        env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

        distance = np.sqrt(
            (manipulation_points[0][0] - manipulation_points[1][0]) ** 2
            + (manipulation_points[0][1] - manipulation_points[1][1]) ** 2
        ) / 2

        left_lift, right_lift = np.array([-0.1, 0.5, 0.85]), np.array([0.1, 0.5, 0.85])
        env.bimanual_dex.dense_move_both_ik(
            left_pos=left_lift,
            left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
            right_pos=right_lift,
            right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        )

        left_lift, right_lift = np.array([-distance - 0.02, 1.4, 0.15]), np.array([distance + 0.02, 1.4, 0.15])
        env.bimanual_dex.dense_move_both_ik(
            left_pos=left_lift,
            left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
            right_pos=right_lift,
            right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        )

        env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
        env.garment.particle_material.set_gravity_scale(10.0)
        for _ in range(100):
            env.step()
        env.garment.particle_material.set_gravity_scale(1.0)

        left_lift, right_lift = np.array([-0.5, 1.3, 0.65]), np.array([0.5, 1.3, 0.65])
        env.bimanual_dex.dense_move_both_ik(
            left_pos=left_lift,
            left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
            right_pos=right_lift,
            right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        )

        # Stage 2: re-grasp at 4 points + sleeve-flatten
        self._hide_robot(env)
        pcd = self._capture_garment_pcd(env)
        env.garment_pcd = pcd
        self._show_robot(env)

        manipulation_points, _, points_similarity = env.model.get_manipulation_points(
            input_pcd=pcd, index_list=[1635, 954, 838, 179]
        )
        env.points_affordance_feature = normalize_columns(points_similarity[0:2].T)
        manipulation_points[:, 2] = 0.002

        env.bimanual_dex.dense_move_both_ik(
            left_pos=manipulation_points[0],
            left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
            right_pos=manipulation_points[1],
            right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        )
        env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

        left_len = np.sqrt(
            (manipulation_points[0][0] - manipulation_points[2][0]) ** 2
            + (manipulation_points[0][1] - manipulation_points[2][1]) ** 2
        ) / 2
        right_len = np.sqrt(
            (manipulation_points[2][0] - manipulation_points[3][0]) ** 2
            + (manipulation_points[1][1] - manipulation_points[3][1]) ** 2
        ) / 2
        sleeve_len = (left_len + right_len) / 2 + 0.06
        manipulation_points[0][0] -= sleeve_len
        manipulation_points[1][0] += sleeve_len
        manipulation_points[:, 1] = 1.28
        manipulation_points[:, 2] = 0.1

        env.bimanual_dex.dense_move_both_ik(
            left_pos=manipulation_points[0],
            left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
            right_pos=manipulation_points[1],
            right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        )
        env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
        env.garment.particle_material.set_gravity_scale(10.0)
        for _ in range(150):
            env.step()
        env.garment.particle_material.set_gravity_scale(1.0)

        set_prim_visibility(get_prim_at_path("/World/DexLeft"), False)
        set_prim_visibility(get_prim_at_path("/World/DexRight"), False)
        for _ in range(50):
            env.step()

        image_end = env.garment_camera.get_rgb_graph()
        success = judge_fling(image_judge, image_end, threshold=0.2)
        cprint(f"[HALOFlingTopsPolicy] final result: {success}", color="green", on_color="on_green")

        return {"success": bool(success), "policy_name": "halo_fling_tops"}

    @staticmethod
    def _hide_robot(env):
        set_prim_visible_group(prim_path_list=["/World/DexLeft", "/World/DexRight"], visible=False)
        for _ in range(50):
            env.step()

    @staticmethod
    def _show_robot(env):
        set_prim_visible_group(prim_path_list=["/World/DexLeft", "/World/DexRight"], visible=True)
        for _ in range(50):
            env.step()

    @staticmethod
    def _capture_garment_pcd(env):
        pcd, _ = env.garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path=get_unique_filename("data", extension=".ply"),
            real_time_watch=False,
        )
        return pcd
