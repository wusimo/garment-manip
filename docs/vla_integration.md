# VLA integration — starVLA + DexGarmentLab

Status as of 2026-05-17: infrastructure written, smoke-test and rollout blocked on local GPU/NVIDIA driver being offline.

## Architecture

```
┌─────────────────────────────────┐                  ┌───────────────────────────┐
│  garment-45 conda env           │                  │  starVLA conda env        │
│  (Isaac Sim 4.5, DexGarmentLab) │                  │  (torch 2.6+cu124, Qwen)  │
│                                 │                  │                           │
│  src/envs/fling_tops_policy_env │  websocket :5694 │  deployment/model_server/ │
│   └─ FlingTops_Env (scene)      │  ←─────────────→ │   └─ Qwen-PI inference    │
│   └─ StarVLAPolicy              │   JSON examples  │     (RoboTwin checkpoint) │
│       └─ closed-loop control    │   ↔ action chunk │                           │
└─────────────────────────────────┘                  └───────────────────────────┘
```

Two conda envs, two processes, connected by a websocket. This split is
non-negotiable — DexGarmentLab pins Isaac Sim 4.5 + Python 3.10 + torch 2.11
(cu130 wheel), while starVLA needs torch 2.6 + cu124 + Qwen2.5-VL. Trying to
merge them caused dependency conflicts in earlier attempts.

The websocket protocol is the same one used in starVLA's RoboTwin eval (see
`examples/Robotwin/eval_files/model2robotwin_interface.py`):
  - Client sends `{"examples": [{"lang": str, "image": [head, left, right]}], "use_ddim": true, ...}`.
  - Server returns `{"data": {"normalized_actions": (B, chunk, 14)}}`.
  - Client unnormalizes (skipped for first-attempt; see "Known gaps" below)
    and indexes into the action chunk across sim steps.

## Action-space contract

This is the load-bearing piece — the integration is correct iff the bridge
below is correct.

| | starVLA (RoboTwin) | DexGarmentLab (UR10e + LeapHand) |
|---|---|---|
| Arm DoF/side | 6 (ARX X5/X7) | 6 (UR10e) |
| Hand DoF/side | 1 (parallel-jaw gripper) | 24 (LeapHand) |
| Action total | 14 (6+6+1+1) | 60 (30+30) |
| Cameras | head + left + right (3× RGB) | env + garment + judge |
| Action mode | abs / delta / rel (configurable) | EE waypoints (HALO) **or** direct joint pos |

**Bridge (StarVLAPolicy._apply_action), first-attempt:**

  | starVLA index | Action | DexGarmentLab call |
  |---|---|---|
  | `[0:6]`  | left arm joint positions  | `dexleft.set_joint_positions(action[0:6], arm_dof_indices)` |
  | `[6:12]` | right arm joint positions | `dexright.set_joint_positions(action[6:12], arm_dof_indices)` |
  | `[12]`   | left gripper (continuous) | `set_both_hand_state(left="open" if ≥0.5 else "close", ...)` |
  | `[13]`   | right gripper (continuous) | `... right="open" if ≥0.5 else "close")` |

**Camera mapping** is a worse approximation — DexGarmentLab cameras are placed
for cloth perception (one overhead, one judge angle), not for the
head/left/right wrist-cam layout RoboTwin was trained on. We pass our cameras
through `head/left/right` slots for now to keep the API working.

## Known gaps (what will fail and why)

These are predicted before running. The first rollout will tell us which actually bite.

1. **Joint convention mismatch.** UR10e and ARX X5 are both 6-DoF arms, but
   their URDF joint orderings and zero/limit conventions differ. starVLA's
   actions are calibrated to ARX kinematics — applying them as UR10e joint
   targets will produce wrong end-effector poses.
2. **Action normalization stats.** We `np.clip(action, -1, 1)` and skip
   unnormalization, because the stored RoboTwin norm stats describe the ARX
   distribution and aren't meaningful for UR10e. The arm joints will likely
   sit at near-saturated values — robot may pose in extreme positions.
3. **LeapHand collapsed to 1 bit.** A 24-DoF dexterous hand reduced to a
   single open/close command can't actually grasp cloth at a specific
   affordance point. Even if the arms reach the right place, the hand can't
   pinch a sleeve corner.
4. **Camera viewpoint distribution shift.** starVLA was trained with cameras
   in a specific RoboTwin layout. Our `env_camera` is overhead, `judge_camera`
   is a side angle for flatness scoring. The VLA's vision encoder sees
   out-of-distribution images and the language grounding will be poor.
5. **No closed-loop reset semantics.** starVLA expects per-episode state reset
   between rollouts. Our `reset()` only zeroes the action-chunk cache; the
   action-chunk semantics (delta vs abs) interact with `initial_state` and
   `prev_action` tracking that we haven't implemented.

## What to investigate after first rollout

Order of importance (each addresses one gap above):
1. **Print the action distribution.** Are values near ±1 (clip-saturated) or in
   a sensible range? This tells us whether the bridge is even producing
   plausible joint targets.
2. **Visualize the chosen waypoint.** Compute forward kinematics of the
   commanded joint targets and overlay on the scene — confirms whether the
   end-effector goes anywhere near the cloth.
3. **Replace `set_joint_positions` with IK on inferred EE.** If actions are
   nonsense, an intermediate experiment is to interpret the 14-D action as
   left/right EE deltas (6+6 DoF for 6D pose), and call `dense_move_both_ik`.
   This isn't what starVLA was trained for, but it might land closer to the
   ballpark than direct joint replay.

## Choosing a checkpoint

**Practical choice: `StarVLA/Qwen3-VL-OFT-Robotwin2`** (HuggingFace, public).
This is the OFT variant — parallel-decoded continuous actions, **not**
flow-matching. starVLA does ship a Qwen-PI/RoboTwin checkpoint, but it lives
on the starVLA team's internal storage (see `deploy_policy.yml`'s
`/mnt/data/gaoning/...` path), not on HuggingFace.

The websocket protocol is identical between OFT and PI variants — same 14-D
action chunk output, same input dict — so `StarVLAPolicy` works against
either. We call the integration "starVLA Qwen-PI" loosely, but in practice
we're running OFT until a public PI ckpt appears or we train one.

The default in `scripts/run_starvla_server.sh` points at this checkpoint; pass
a positional argument to override (e.g. a local fine-tune).

Other ckpts (don't use yet):
  - LIBERO 4-in-1 (`Qwen3-VL-OFT-LIBERO-4in1`): single-arm 7-DoF; the bridge
    would need rewriting to apply only the left arm.

## Running (when GPU is back)

```bash
# Terminal 1 — start the inference server (default ckpt: StarVLA/Qwen3-VL-OFT-Robotwin2)
bash scripts/run_starvla_server.sh

# Terminal 2 — run the env. Wait for the server to print "server running ..."
DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy starvla \
  --instruction "fling and flatten the t-shirt"
```

Override the checkpoint by passing it as $1:
```bash
bash scripts/run_starvla_server.sh /path/to/fine-tuned-ckpt --port 5694
```

For the HALO baseline (sanity check that the refactor didn't break anything):
```bash
DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy halo
```

## Files added in this thread

| File | What |
|---|---|
| `src/policies/base.py` | Abstract `Policy` interface |
| `src/policies/halo_policy.py` | `HALOFlingTopsPolicy` — refactor of upstream `FlingTops()` |
| `src/policies/starvla_policy.py` | `StarVLAPolicy` — websocket client + action-space bridge |
| `src/envs/fling_tops_policy_env.py` | Forked Fling_Tops driver using `Policy` |
| `scripts/run_starvla_server.sh` | Launch starVLA inference server in starVLA conda env |
| `scripts/run_fling_tops_policy.sh` | Launch the policy-driven env in garment-45 conda env |
| `docs/vla_integration.md` | This file |
