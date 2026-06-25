# UR3e RLT Pipeline

This is the main guide for the UR3e RLT workflow around the Ethernet insertion task.

The current system is intentionally split into small pieces:

```text
Frozen VLA              outputs no-rotvec action chunks
RL gate                 decides when the insertion-refinement phase is active
Stage1 RL token         compresses frozen VLA context/action information into z_rl
Stage2 RLT actor        adds a small xyz residual to the VLA action chunk while RL gate is active
Ready gate              decides when the scene is ready for the next rollout
Impedance robot node    receives end-effector targets and handles soft execution
```

The most important separation is:

```text
ready_gate: "Can the next trial start?"
RL gate:    "Should RLT refine the VLA right now?"
z_rl:       "What compact representation does the RL actor see?"
RLT actor:  "How should the VLA action chunk be corrected?"
```

## Current Defaults

Most scripts default to the current Ethernet task setup:

```text
VLA policy:
  outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model

Stage1 token:
  outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260616_163323/best.pt

RL gate:
  outputs/rlt_gate/rlt_gate_20260615_142442/best.pt

Ready gate:
  outputs/ready_gate/ready_gate_20260617_150953/best.pt

Good Stage2 checkpoint:
  outputs/rlt_stage2/hil_serl_stage2_20260616_204313/checkpoints/stage2_ep000050.pt
```

If you are comparing behavior, keep the robot node and camera setup fixed and only swap the rollout/training command.

## Start Hardware

Use this robot command for model rollout or RLT rollout without VR override:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
source /home/arts/anaconda3/etc/profile.d/conda.sh
conda activate ur3e_rlt

python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node robot \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop \
  --no-twin
```

For Stage2 HIL collection/training with possible VR override, also start the raw VR node:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node vr \
  --vr-output-topic /ur3e_vr/vr_command_raw
```

Start cameras:

```bash
scripts/collect_data/run_realsense_cameras.sh
```

Do not run two publishers that command `/ur3e_vr/ik_target` or `/ur3e_vr/vr_command` at the same time.

## Baseline: Plain VLA With Ready Loop

Before judging RLT, test the frozen VLA alone. This script has the same ready-gate and impedance reset behavior as the closed-loop RLT rollout, but it does not use the RLT actor.

Sync mode:

```bash
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_ready_loop.py \
  --mode sync \
  --wait-ready-on-start \
  --execute
```

RTC mode:

```bash
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_ready_loop.py \
  --mode rtc \
  --wait-ready-on-start \
  --execute
```

Useful options:

```bash
--trial-duration-s 18          # auto return home after this many seconds; 0 disables auto end
--wait-ready-on-start          # wait for ready_gate before first trial
--no-auto-start-on-ready       # show ready_gate but do not start automatically
--no-reset-impedance-on-trial-start
--no-reset-impedance-during-home
```

This baseline answers: "Does the frozen VLA already succeed under the current robot/camera/impedance setup?"

## Closed-Loop RLT Rollout

Use this when you want to evaluate a trained Stage2 RLT actor without VR and without online learning.

The rollout loads four frozen components:

```text
SmolVLA policy        produces the reference action chunk
Stage1 encoder       produces z_rl from VLA internal embeddings
RL gate              enables RLT only during the critical insertion phase
Stage2 actor          refines the VLA xyz action chunk

Ready gate            is optional control logic for starting the next trial
```

No actor/critic update is performed during rollout. The Stage2 critic is loaded
from the checkpoint for compatibility, but it does not choose commands. Only the
trained actor affects robot motion.

### Required Processes

Before starting either the sync or RTC RLT rollout, run these mandatory
processes in separate terminals and keep them running:

```bash
# Terminal 1: RealSense cameras
scripts/collect_data/run_realsense_cameras.sh

# Terminal 2: UR3e impedance robot controller
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node robot \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop \
  --no-twin
```

Only after both processes are ready should you start
`rollout_rlt_no_vr.py` or `rollout_rlt_no_vr_rtc.py` in a third terminal.

Do not start another rollout, collector, or VR control process that also
publishes `/ur3e_vr/ik_target` or `/ur3e_vr/vr_command`.

### Sync Rollout

The sync rollout obtains one aligned observation, runs VLA inference, builds
`z_rl`, applies the Stage2 actor when the RL gate is active, and then executes
the selected portion of the resulting chunk.

Sync version:

```bash
python scripts/rlt_train/rollout_rlt_no_vr.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model \
  --stage1-checkpoint outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260616_163323/best.pt \
  --gate-checkpoint outputs/rlt_gate/rlt_gate_20260615_142442/best.pt \
  --rlt-checkpoint outputs/rlt_stage2/hil_serl_stage2_20260616_204313/checkpoints/stage2_ep000050.pt \
  --ready-gate-checkpoint outputs/ready_gate/ready_gate_20260617_150953/best.pt \
  --wait-ready-on-start \
  --execute
```

Without `--execute`, the script is a dry run: models and topics are exercised,
but robot commands are not published.

### RTC Rollout

The RTC-style rollout keeps ROS observation/publish callbacks responsive while a
background worker performs VLA inference. It refreshes the action queue when the
queue becomes short. This usually gives smoother continuous motion, but actions
may be based on a slightly older observation than strict sync rollout.

RTC-style version:

```bash
python scripts/rlt_train/rollout_rlt_no_vr_rtc.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model \
  --stage1-checkpoint outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260616_163323/best.pt \
  --gate-checkpoint outputs/rlt_gate/rlt_gate_20260615_142442/best.pt \
  --rlt-checkpoint outputs/rlt_stage2/hil_serl_stage2_20260616_204313/checkpoints/stage2_ep000050.pt \
  --ready-gate-checkpoint outputs/ready_gate/ready_gate_20260617_150953/best.pt \
  --wait-ready-on-start \
  --rtc-infer-count 10 \
  --rtc-queue-refill-threshold 3 \
  --execute
```

### Per-Trial Execution Flow

```text
1. Before model loading, send return_home and open the gripper.
2. Wait for fresh front image, wrist image, and robot state.
3. If --wait-ready-on-start is enabled, wait for ready_gate=1.
4. Start a trial and reset impedance state.
5. SmolVLA predicts ref_action_chunk from the aligned observation.
6. Stage1 reads VLA hidden features and produces z_rl.
7. RL gate evaluates the camera image.
8. If RL_gate=0:
     executed_chunk = ref_action_chunk
9. If RL_gate=1:
     executed_xyz = Stage2Actor(z_rl, state, ref_action_chunk)
     executed_gripper = VLA gripper
10. Publish fixed-orientation xyz targets to the impedance robot node.
11. Once RL gate exits, lock re-entry for this trial, clear queued actions,
    invalidate any in-flight inference and cached RTC observation, return home,
    reset impedance, and wait for ready_gate before the next trial.
```

The Stage2 actor checkpoint stores its own architecture configuration, including
action chunk length, residual scale, and `direct` or `projected` fusion mode.
Rollout reconstructs the network from that checkpoint. Do not manually force a
different fusion mode during rollout.

### Sync Versus RTC

```text
sync:
  observation -> inference -> action execution -> next observation
  easier to reason about and best for model diagnosis

RTC:
  observation callbacks continue while a background worker performs inference
  action queue is refreshed before it becomes empty
  usually smoother, but introduces observation/action latency
```

Recommended order:

```text
1. Test plain VLA sync.
2. Test RLT sync and compare behavior inside RL_gate=1.
3. Use RLT RTC only after sync behavior is correct.
```

### Important Rollout Parameters

```text
--policy-path
  Frozen SmolVLA pretrained_model directory.

--stage1-checkpoint
  Frozen RL-token encoder checkpoint used to compute z_rl.

--rlt-checkpoint
  Trained Stage2 actor/critic checkpoint. The actor controls refinement.

--gate-checkpoint
  RL gate classifier. RLT can affect actions only while this gate is active.

--ready-gate-checkpoint
  Scene-ready classifier used to start the next trial.

--execution-horizon
  Number of actions taken from each VLA inference result in sync mode.

--action-step-hz
  Rate at which queued action steps are consumed.

--command-hz
  Rate at which the current target is published to the robot node.

--sync-reference front|wrist|timer|both
  Observation alignment trigger. Keep it consistent with collection unless
  intentionally testing another alignment policy.

--wait-ready-on-start
  Do not begin the first trial until ready_gate is positive.

--auto-start-on-ready / --no-auto-start-on-ready
  Automatically start a new trial after ready_gate becomes positive.

--min-action-z
  Final safety floor for published target z.

--action-pose-filter-alpha
  Output position smoothing. Larger values follow new targets more strongly.

--max-action-pos-step
  Maximum accepted position change between published targets.

--reset-impedance-on-trial-start
  Reset the impedance reference before a new trial.

--reset-impedance-during-home
  Reset impedance while returning home after the gate exits.

--preview / --no-preview
  Enable or disable the OpenCV camera/gate preview.
```

RTC-specific parameters:

```text
--rtc-infer-count
  Actions requested per background inference refresh. It is clamped to at least
  the Stage2 actor chunk length.

--rtc-queue-refill-threshold
  Start another inference when queued actions fall to this count.

--rtc-replace-queue-on-infer
  Replace old pending actions with the newest inference result.
```

### Reading Rollout Behavior

```text
RL_gate=0:
  Robot should match the plain frozen VLA baseline.

RL_gate=1:
  Stage2 actor may alter xyz, but gripper remains from the VLA.

After gate exit:
  RLT must not re-enter during the same trial. The robot returns home and waits.
```

If the robot behaves differently before `RL_gate=1`, first check that no second
command publisher is running and compare against the plain VLA ready-loop. The
no-VR RLT rollout is the cleanest test of the learned actor.

## Stage1: Train And Check RL Token

Stage1 freezes the trained SmolVLA and trains a small encoder/decoder that produces `z_rl`.

Train:

```bash
python scripts/rlt_token/train_rlt_stage1.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model \
  --batch-size 16 \
  --num-workers 8 \
  --steps 10000 \
  --device cuda
```

Evaluate token smoothness on demonstration episodes:

```bash
python scripts/rlt_token/eval_rlt_stage1.py \
  --checkpoint outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260616_163323/best.pt \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model \
  --episodes 0 1 2 3 4 5 \
  --device cuda
```

Interpretation:

```text
z_norm: should be finite and stable.
dz_mean / dz_p95: token motion should be smooth, with meaningful bumps near task changes.
loss: reconstruction sanity check, not robot success rate.
```

See `scripts/rlt_token/README.md` for the detailed Stage1 theory.

## RL Gate And Ready Gate

RL gate labels where RLT is allowed to refine the action. For the insertion task, this should cover the difficult final alignment/insertion window, not the whole pick-and-place episode.

Annotate/train/evaluate RL gate:

```bash
python scripts/rlt_gate/annotate_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --camera both

python scripts/rlt_gate/train_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --model resnet18 \
  --camera front \
  --image-size 160

python scripts/rlt_gate/eval_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --checkpoint outputs/rlt_gate/<run>/best.pt \
  --episode 12 \
  --show
```

Ready gate labels whether the plug is ready for the next trial.

Collect and train:

```bash
python scripts/ready_gate/collect_ready_gate.py

python scripts/ready_gate/train_ready_gate.py \
  --dataset datasets/ready_gate/<run> \
  --model resnet18 \
  --camera both \
  --device cuda
```

Live ready-gate check:

```bash
python scripts/ready_gate/live_ready_gate_eval.py \
  --checkpoint outputs/ready_gate/<run>/best.pt
```

## Stage2: HIL-SERL Style Online RLT

Stage2 trains a small TD3-style residual actor/critic. The frozen VLA still proposes the reference action chunk:

```text
ref_action_chunk = frozen VLA(obs)[:H]
z_rl             = frozen Stage1 encoder(obs, ref_action_chunk)
RLT residual     = actor(z_rl, state, ref_action_chunk)
executed xyz     = ref_action_chunk_xyz + residual_xyz
executed gripper = VLA gripper, or VR gripper during override
```

The actor currently trains only xyz residuals. It does not train gripper actions. This keeps the RL problem focused on insertion alignment.

Replay buffers:

```text
online_buffer:
  confirmed transitions inside RL_gate=1

intervention_buffer:
  confirmed transitions where human VR override was active inside RL_gate=1

training:
  TD3 critic/actor updates use replay samples
  intervention samples are oversampled for the BC regularizer
```

Confirmed means: the episode is staged and then saved by pressing `B` twice.

## Stage2 Buttons

```text
A                    start / stop inference
Y                    select terminal reward 1
X                    select terminal reward 0
B                    first press: stage pending save; second press: confirm and commit
Left upper trigger   discard current staged episode or cancel pending save
Left lower trigger   stop inference, clear queued actions, return home
Right lower trigger  VR override
Right upper trigger  gripper during VR override
```

The current design stages transitions in memory. Nothing is inserted into replay buffers until `B` is confirmed twice. This is slower than fully streaming HIL-SERL, but safer on the real robot because a bad collision run can be discarded.

## Stage2: Collect Warm Buffers Only

Use this when you want intervention data without letting the actor control the robot:

```bash
python scripts/rlt_train/train_hil_serl_stage2.py \
  --execute \
  --stage1-checkpoint outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260616_163323/best.pt \
  --no-rlt-enable-actor \
  --rlt-updates-per-step 0 \
  --rlt-action-chunk-steps 10 \
  --vr-override-anytime
```

Recommended usage:

```text
1. Press A to start model rollout.
2. When trend is wrong, use right lower trigger to correct with VR.
3. If final result is acceptable, press Y, then B, then B.
4. If bad or unsafe, use left upper trigger to discard.
5. Use left lower trigger to return home if the pose is awkward.
```

The output directory is printed at startup, for example:

```text
outputs/rlt_stage2/hil_serl_stage2_YYYYMMDD_HHMMSS
```

Warm buffer files live under:

```text
outputs/rlt_stage2/<run>/buffers/
```

## Stage2: Train From Warm Buffers

Start from a known buffer directory:

```bash
python scripts/rlt_train/train_hil_serl_stage2.py \
  --execute \
  --rlt-buffer-dir outputs/rlt_stage2/<warm_run>/buffers \
  --rlt-updates-per-step 1 \
  --rlt-startup-updates 300 \
  --rlt-bc-weight 0.1 \
  --rlt-replay-demo-ratio 0.25 \
  --rlt-action-delta-scale-xyz 0.002 \
  --vr-override-anytime
```

Conservative first online run:

```bash
python scripts/rlt_train/train_hil_serl_stage2.py \
  --execute \
  --rlt-buffer-dir outputs/rlt_stage2/<warm_run>/buffers \
  --rlt-updates-per-step 1 \
  --rlt-startup-updates 300 \
  --rlt-action-delta-scale-xyz 0.002 \
  --vr-override-anytime
```

More aggressive online update:

```bash
--rlt-updates-per-step 2
```

Use `2` only after checking that control latency and GPU load are stable.

## Resume Or Roll Back Stage2

Resume a checkpoint:

```bash
python scripts/rlt_train/train_hil_serl_stage2.py \
  --execute \
  --rlt-checkpoint outputs/rlt_stage2/<run>/checkpoints/stage2_ep000050.pt \
  --rlt-buffer-dir outputs/rlt_stage2/<run>/buffers \
  --rlt-updates-per-step 1 \
  --vr-override-anytime
```

Roll back to an earlier good checkpoint by using it as `--rlt-checkpoint`. If a later run degraded after episode 60, start from the earlier `stage2_ep000050.pt` and either load the older buffers or collect a cleaner warm buffer.

Start with empty buffers:

```bash
--no-rlt-buffer-dir
```

This is useful for debugging, but not usually useful for learning.

## Important Parameters

```text
--rlt-action-chunk-steps
  Number of VLA/RLT chunk steps used by Stage2. Default 10.

--rlt-action-delta-scale-xyz
  Maximum residual scale in meters. Smaller is safer. 0.002 means about 2 mm residual scale.

--rlt-updates-per-step
  Online gradient updates after each confirmed episode or update trigger. 1 is conservative.

--rlt-startup-updates
  Offline updates before actor control starts when replay buffers are loaded.

--rlt-replay-demo-ratio
  Fraction of each training batch drawn from intervention buffer.

--rlt-bc-weight
  Strength of behavior-cloning regularization on intervention samples.

--rlt-fusion-mode direct|projected
  Actor/critic input fusion mode. `direct` keeps the original path:
  concat([z_rl, state, ref_action]) followed by one LayerNorm+MLP. This is the
  default and remains compatible with existing Stage2 checkpoints.
  `projected` first maps each stream through its own LayerNorm+Linear before
  concatenation. This gives z_rl, robot state, and action chunks comparable
  hidden feature sizes, but requires training a fresh Stage2 actor/critic or a
  checkpoint trained with the same fusion mode.

--rlt-fusion-dim
  Per-stream projection width used only by `--rlt-fusion-mode projected`.
  Default 128.

--gate-hold-frames
  Hysteresis frame count for RL gate enter/exit.

--home-pulse-s
  How long return_home command is held after gate exit or manual home.

--reset-impedance-on-trial-start
  Sends reset_impedance before a new trial. Keep enabled unless debugging.

--reset-impedance-during-home
  Sends reset_impedance during return_home pulses. Keep enabled to reduce drift after contacts.
```

## What To Watch In Logs

Stage2 training logs often include:

```text
src=rlt_actor       RLT actor is affecting commands
src=vla_gate0      frozen VLA command, outside gate
rlt=1              RLT residual is active in published action
res=6.9mm          size of residual correction
buf=online/intvn   replay buffer sizes
upd=1116           trainer update count
c                  critic loss
act                actor loss
bc                 BC regularizer loss
q                  actor Q objective
```

If `upd` is increasing, the network is being updated.

If `src=rlt_actor` appears while `RL_gate=0`, that is a bug. RLT should only modify commands during the gated refinement window.

## Common Failure Modes

### VLA succeeds sometimes, then drifts lower over repeated trials

Likely impedance drift after contact. Keep both reset options enabled:

```bash
--reset-impedance-on-trial-start
--reset-impedance-during-home
```

Use the plain VLA ready-loop to verify whether the issue exists without RLT:

```bash
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_ready_loop.py \
  --mode sync \
  --execute
```

### Actor gets worse after many online episodes

Roll back to a good checkpoint and reduce update aggressiveness:

```text
--rlt-updates-per-step 1
--rlt-action-delta-scale-xyz 0.002
--rlt-bc-weight 0.1
```

Also check whether the warm buffer distribution is too narrow. If the cable orientation changes, collect warm interventions covering those poses.

### Robot does not move in rollout

Check:

```text
1. Is the robot impedance node running?
2. Is any other node publishing /ur3e_vr/ik_target or /ur3e_vr/vr_command?
3. Are camera topics publishing?
4. Is the script stuck waiting for ready_gate?
5. Is min_action_z too high?
```

Use no-VR rollout first to avoid VR blocking:

```bash
python scripts/rlt_train/rollout_rlt_no_vr.py --execute
```

### Computer stalls during startup updates

Lower startup/update load:

```bash
--rlt-startup-updates 100
--rlt-updates-per-step 1
--rlt-batch-size 64
```

Large buffers, OpenCV preview, and VLA inference can all compete for CPU/GPU time.

## Output Files

Each confirmed Stage2 episode saves:

```text
episode_XXXXXX/episode_transitions.npz
episode_XXXXXX/metadata.json
episodes.jsonl
checkpoints/last.pt
checkpoints/stage2_epXXXXXX.pt
buffers/online_buffer_latest.npz
buffers/intervention_buffer_latest.npz
```

Transition fields:

```text
z_rl
state
ref_action
action
reward
next_z_rl
next_state
next_ref_action
done
is_intervention
```

`ref_action` is the frozen VLA chunk. `action` is the executed/refined chunk. Both are stored as `[H, 4]` with `[x, y, z, gripper]`, but Stage2 actor/critic currently uses only xyz.

## Related Docs

- Stage1 token details: `scripts/rlt_token/README.md`
- RL gate annotation/training: `scripts/rlt_gate/README.md`
- Ready gate data/training: `scripts/ready_gate/README.md`
- No-rotvec VLA training: `scripts/train_smolvla_no_rotvec/README.md`
- Plain no-rotvec rollout: `scripts/rollout_smolvla_no_rotvec/README.md`
