# UR3e RLT Stage2

This folder contains the UR3e RLT Stage2 tools.

The current online trainer follows the HIL-SERL/RLPD idea:

```text
online_buffer:
  all confirmed transitions inside RL_gate=1

intervention_buffer:
  confirmed transitions where the human used VR override inside RL_gate=1

training batch:
  critic / Q actor loss: online samples
  BC regularizer: oversampled intervention samples
```

The frozen SmolVLA still proposes the reference action. Stage2 learns a small
residual actor:

```text
ref_action_chunk = frozen VLA(obs)[:H]
z_rl             = frozen Stage1 token encoder(obs, ref_action_chunk)
action_chunk_xyz = ref_action_chunk_xyz + RLT_actor(z_rl, state, ref_action_chunk_xyz)
gripper          = frozen VLA gripper, or human VR gripper during override
```

`H` is controlled by `--rlt-action-chunk-steps` and defaults to `10`, matching
the default `--execution-horizon`. The refined chunk is queued and executed one
action at a time at `--action-step-hz`.

The Stage2 update is TD3-style: twin critics, target actor/critic networks,
delayed actor updates, soft target updates, and target policy smoothing via
`--rlt-target-noise-xyz` / `--rlt-target-noise-clip-xyz`.

Stage2 currently trains only the TCP position part of each chunk action: `x, y, z`.
The gripper is recorded for traceability and executed from VLA/VR, but it is not
part of the actor/critic action. This keeps the RL problem focused on insertion
alignment instead of mixing insertion with release timing.

The current chunk-level Stage2 is not checkpoint-compatible with earlier
single-step Stage2 actor/critic checkpoints. Do not resume old Stage2
checkpoints after switching to this version. Replay buffers saved by older runs
can be loaded for inspection, but for real training it is cleaner to recollect
Stage2 buffers with the new sequence Stage1 token and chunk-level actor.

Return-home pulses open the gripper by default using `--home-gripper-value 0.0`.
Set this argument to another value if a trial needs to preserve or close the
gripper while returning home.

The critic is trained from confirmed gate-window transitions in `online_buffer`.
Human intervention data is not treated as magic success by itself; intervention
frames are additionally stored in `intervention_buffer` and are used as a BC
regularizer for the actor. This keeps reward learning grounded in full
episodes, while still emphasizing the human correction direction.

## Buttons

- `A`: start/stop inference for the current trial.
- `Y`: select terminal reward `1` for the current gate-window episode.
- `X`: select terminal reward `0`.
- `B`: first press freezes the episode and adds the terminal `done=1` transition; second press confirms and commits it to replay buffers.
- Left upper trigger: discard the current staged episode or cancel pending save.
- Left lower trigger: stop inference, clear queued actions, and send a return-home pulse.
  Internally this is the VR SDK field named `left_grip`.
- Right lower trigger: VR override, only active while `RL_gate=1`.
- Right upper trigger: gripper during VR override.

For safety, transitions are staged in memory during the episode. They are only
inserted into replay buffers and used for updates after `B` is pressed twice.
This differs slightly from the original HIL-SERL actor, which streams
transitions immediately, but it preserves the ability to discard bad real-robot
runs.

After a confirmed save, repeated B events are ignored briefly using
`--save-button-cooldown-s` so a physical double-trigger does not produce a false
"no transitions" warning.

## Runtime Flow

1. Start the robot impedance node only.
2. Start the raw VR node.
3. Start the RealSense camera nodes.
4. Start the Stage2 trainer.
5. Press `A` to begin VLA/RLT rollout.
6. When `RL_gate=1`, VR override can correct the insertion phase.
7. When the gate exits, inference pauses and a short return-home pulse is sent.
8. Select reward with `Y` or `X`.
9. Press `B` once to stage, press `B` again to commit/save/update.
10. Press `A` again for the next trial.

## Commands

Robot node:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
conda activate ur3e_rlt
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node robot \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop
```

Raw VR node:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
conda activate ur3e_rlt
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node vr \
  --vr-output-topic /ur3e_vr/vr_command_raw
```

Cameras:

```bash
conda activate ur3e_rlt
scripts/collect_data/run_realsense_cameras.sh
```

Stage2 HIL-SERL trainer:

```bash
conda activate ur3e_rlt
python scripts/rlt_train/train_hil_serl_stage2.py --execute
```

`--sync-reference both` means the trainer triggers sampling from the current
time and then requires both camera frames plus robot state to be close enough in
time. The observation always contains both cameras; `front`/`wrist`/`timer`/`both`
only choose how the synchronization timestamp is produced.

Useful conservative first test:

```bash
python scripts/rlt_train/train_hil_serl_stage2.py \
  --execute \
  --no-rlt-enable-actor \
  --rlt-updates-per-step 1
```

This runs VLA + VR intervention and fills the HIL-SERL buffers, but does not let
the learned actor affect commands yet.

If the robot is in an awkward pose and you want manual VR authority even outside
`RL_gate=1`, add:

```bash
--vr-override-anytime
```

This allows right-lower-trigger VR control at any time, but transitions are still
recorded only while `RL_gate=1`.

To resume a Stage2 checkpoint:

```bash
python scripts/rlt_train/train_hil_serl_stage2.py \
  --execute \
  --rlt-checkpoint outputs/rlt_stage2/<run>/checkpoints/last.pt
```

## Default Paths

- VLA: `outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model`
- Stage1 token: `outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260614_165914/best.pt`
- RL gate: `outputs/rlt_gate/rlt_gate_20260614_154758/best.pt`
- Stage2 output: `outputs/rlt_stage2/hil_serl_stage2_*`

## Outputs

Each confirmed episode saves:

- `episode_XXXXXX/episode_transitions.npz`
- `episode_XXXXXX/metadata.json`
- `episodes.jsonl`
- `checkpoints/last.pt`
- optional replay snapshots in `buffers/`

Transition fields:

```text
z_rl, state, ref_action, action,
reward, next_z_rl, next_state, next_ref_action,
done, is_intervention
```

`ref_action` is the frozen VLA action chunk. `action` is the executed/refined
action chunk after RLT actor or VR override. Both are stored as `[H, 4]`
`[x, y, z, gripper]`, but the Stage2 actor/critic update uses only the first
three dimensions of each chunk step.
