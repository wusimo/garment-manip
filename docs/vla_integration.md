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

The first-attempt design assumes a RoboTwin-trained Qwen-PI checkpoint. To
find one, browse https://huggingface.co/StarVLA and filter for repos with
`robotwin` in the name. The README at `vendor/starVLA/examples/Robotwin/`
(or `/home/simo/Documents/starVLA/examples/Robotwin/README.md`) lists the
exact ckpt path used in their published RoboTwin 2.0 numbers. Set the path
via the first positional arg to `scripts/run_starvla_server.sh`.

If no suitable Qwen-PI/RoboTwin ckpt is available, fall back to:
  - **Qwen-OFT RoboTwin** — different action representation (parallel decoding
    rather than flow-matching), but same 14-D output, same websocket
    interface. The `StarVLAPolicy` code does not depend on which variant the
    server hosts.
  - **A LIBERO ckpt** — single-arm 7-DoF; would require rewriting the bridge
    to apply only the left arm and leave the right idle. Skip for now.

## Running (when GPU is back)

```bash
# Terminal 1 — start the inference server
bash scripts/run_starvla_server.sh <CKPT_PATH> --port 5694

# Terminal 2 — run the env. Wait for the server to print "server running ..."
DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy starvla \
  --instruction "fling and flatten the t-shirt"
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
