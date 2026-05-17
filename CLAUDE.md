# Claude Code context — garment-manip

Research workspace for bimanual garment manipulation. Public repo: https://github.com/wusimo/garment-manip

## Verified state (2026-04-29)

- Both conda envs green: `garment-45` (Isaac Sim 4.5, DexGarmentLab tasks) and `garment-51` (Isaac Sim 5.1 + Isaac Lab v2.3.2, new algorithm work).
- DexGarmentLab assets (~22 GB) downloaded at `vendor/DexGarmentLab/Assets/` — do not re-download.
- `bash scripts/run_fling_tops.sh` (with `DISPLAY=:1`) runs end-to-end: HALO/GAM picks 2 grasp points → fling executes → judge returns `True` at ~0.94 flatten proportion.

## VLA thread — starVLA integration (2026-05-17, in progress)

Goal: drop a real VLA in at the `Fling_Tops_Env.py:220` seam and observe what
happens. Chose **starVLA's Qwen-PI variant** (flow-matching, π0-style) on a
**RoboTwin-2.0 pretrained checkpoint**. starVLA source lives at
`/home/simo/Documents/starVLA`, env `starVLA` (separate from garment-45).

Architecture: out-of-process, websocket-bridged. Isaac Sim runs in garment-45,
inference server runs in starVLA env, they talk over `:5694`. Avoids the
unavoidable torch/Isaac-Sim version conflict.

**Status:**
- Infrastructure written and committed:
  - `src/policies/{base,halo_policy,starvla_policy}.py` — Policy interface + adapters
  - `src/envs/fling_tops_policy_env.py` — forked Fling_Tops driver using Policy
  - `scripts/run_starvla_server.sh`, `scripts/run_fling_tops_policy.sh`
  - `docs/vla_integration.md` — architecture + action-space contract + gap analysis
- Smoke test and first rollout BLOCKED on local NVIDIA driver (currently down).
- Action-space bridge is intentionally lossy (14-D starVLA action → 60-D
  bimanual UR10e+LeapHand). Expected to fail informatively on first run. See
  `docs/vla_integration.md` "Known gaps" for the predicted failure modes
  ordered by what to investigate first.

Run when GPU is back:
```bash
# T1
bash scripts/run_starvla_server.sh <CKPT_PATH> --port 5694
# T2 (wait for "server running ...")
DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy starvla
# Sanity baseline (no server needed)
DISPLAY=:1 bash scripts/run_fling_tops_policy.sh --policy halo
```

## Running a task

```bash
DISPLAY=:1 bash scripts/run_fling_tops.sh        # the verified one
# Any other of the 15 tasks under vendor/DexGarmentLab/Env_StandAlone/:
DISPLAY=:1 conda run -n garment-45 \
  bash -c "cd vendor/DexGarmentLab && yes Yes | python -u Env_StandAlone/<TaskName>_Env.py"
```

Categories: Fling / Fold / Hang (Tops, Dress, Trousers, Coat), Wear (Baseballcap, Bowlhat, Glove, Scarf), Store_Tops.

## HALO architecture — NOT a VLA

Two stages, both vision-only (point clouds). **No language conditioning anywhere.**

| Stage | Component | Architecture |
|---|---|---|
| Affordance | **GAM** (`Model_HALO/GAM/`) | PointNet++ → 512-d per-point features; cosine-similarity match to a demo PC picks grasp points |
| Trajectory | **SADP-G** (`Model_HALO/SADP_G/`) | ConditionalUnet1D + DDIM (100 steps) over `{garment_pc, env_pc, affordance_feats, joint_state}` |

SADP-G ships a full IL training loop (`Model_HALO/SADP_G/train.py`, Zarr buffer + EMA + WandB). **No RL loop included.**

## Key seams (in `Env_StandAlone/Fling_Tops_Env.py` — same shape across all tasks)

- `:220` — `env.model.get_manipulation_points(...)` — affordance picker; swap for a VLA here.
- `:244-253` — hardcoded lift/spread/release waypoints; swap for SADP-G or RL-finetuned policy.
- `:343` — `judge_fling()` returns sparse success; **flatten proportion is continuous and available as a dense reward**.

All tasks inherit `BaseEnv` (`vendor/DexGarmentLab/Env_StandAlone/BaseEnv.py:30-100`). A `src/policies/` wrapper can target this uniform interface.

## Setup gotchas (already handled by scripts)

- EULA: `yes Yes |` piped into `isaaclab.sh -i` and first `SimulationApp` import.
- CMake 4.x + egl_probe: `CMAKE_POLICY_VERSION_MINIMUM=3.5` set in setup script.
- HF asset stalls (5 GB Garment.zip): retry loop in `download_dexgarment_assets.sh`.
- Task scripts assume cwd = `vendor/DexGarmentLab` (they `sys.path.append(os.getcwd())`).

## Don't

- Don't merge the two conda envs (Sim 4.5 vs 5.1 pin conflict).
- Don't touch the legacy `isaaclab` conda env (Sim 4.1, unrelated).
- Don't re-download the 22 GB of assets.

---

# Research plan

Three threads, in suggested order. One variable at a time — don't stack VLA + RL + multi-garment changes.

## Thread 1 — VLA integration (replace the GAM picker)

Current "policy" is V-A, not VLA. Goal: drop a language-conditioned policy in at `Fling_Tops_Env.py:220`.

**Candidate ordering (by integration cost):**
- **GR00T N1** (NVIDIA) — same Isaac Sim stack, native PC support, bimanual training. Lowest pain.
- **RDT-1B** — diffusion, bimanual-native. Strong conceptual fit.
- **π0 / π0.5** — flow-matching, 50 Hz joint targets; action-space matches UR10e + LeapHand setup.
- **OpenVLA / Octo** — single-arm, EE-pose actions. Big mismatch — skip unless heavily adapting.

**Watchpoints:**
- HALO uses task-space waypoints + IK (`dense_move_both_ik`); most VLAs predict joint deltas at 30–50 Hz. Either retarget waypoints → joint streams, or replace IK altogether.
- LeapHand is 16-DoF; most public VLAs were trained on parallel-jaw or Allegro/Shadow. Need a hand-action adapter or new fine-tuning data.
- Prefer PC-capable VLAs (GR00T, π0) to keep the cloth-geometric prior.

## Thread 2 — RL post-training on SADP-G

Infra already there except the RL loop. Reward shaping: flatten proportion (dense) + judge success (sparse terminal).

**Method choices:**
- **DDPO** (Black et al.) — simplest, REINFORCE-style on the diffusion sampler. Sanity-check first.
- **DPPO** (Ren et al., 2024) — PPO over diffusion timesteps. Stronger but more complex.
- **IDQL / Diffusion-QL** — offline RL reusing the existing IL Zarr dataset, reweighted by reward.

Start with DDPO on Fling_Tops to validate the loop end-to-end before scaling.

## Thread 3 — Multi-garment disentangle → fold (the actual research arc)

No multi-object infra exists. Only a single `Particle_Garment(...)` is loaded at `Fling_Tops_Env.py:65`. Genuine research direction — published cloth-unfolding work (FlingBot, ClothFunnels) is single-garment.

**Build order:**
1. **Sim extension** — instantiate N garments with random drop poses, tune PhysX particle contact, curriculum from 2 lightly overlapping → pile of 5+. Watch VRAM on RTX 3090.
2. **Per-garment segmentation** — GAM has no instance labels. Cheapest experiment: cluster on GAM features. Alternatives: SAM-2 on RGB back-projected, Mask3D.
3. **Hierarchical policy** — high-level picks which garment + action class (disentangle / isolate / fold); low-level is HALO-style picker conditioned on instance mask. Disentangle is a new skill — its success criterion is separation distance, not flatten proportion.
4. **Benchmark** — held-out garment combinations. Metric: (# fully folded) / total + time-to-completion.

Estimated 6–12 month arc, not a weekend extension.

## Suggested ordering

| When | Task |
|---|---|
| Week 1 | Wrap HALO behind a `Policy` interface in `src/policies/`; smoke 2–3 more tasks (Fold_Tops, Hang_Coat) to confirm the abstraction holds. |
| Weeks 2–4 | Drop in **GR00T N1** as a second `Policy` implementation. Lowest integration cost, forces the language-conditioning question, gives a baseline VLA to beat. |
| Month 2 | DDPO loop on SADP-G with flatten-proportion as dense reward — single-task, single-garment. |
| Month 3+ | Multi-garment sim extension. The research contribution lives here. |
