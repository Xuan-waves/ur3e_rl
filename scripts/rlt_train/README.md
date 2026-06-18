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

Sync version:

```bash
python scripts/rlt_train/rollout_rlt_no_vr.py \
  --wait-ready-on-start \
  --execute
```

RTC-style version:

```bash
python scripts/rlt_train/rollout_rlt_no_vr_rtc.py \
  --execute \
  --wait-ready-on-start \
  --rtc-infer-count 10 \
  --rtc-queue-refill-threshold 3
```

What happens:

```text
1. Program starts with return_home and open gripper.
2. If --wait-ready-on-start is set, it waits for ready_gate=1.
3. Frozen VLA produces action chunks.
4. While RL_gate=0, VLA actions are executed unchanged.
5. While RL_gate=1, Stage2 actor adds an xyz residual to the VLA chunk.
6. When RL_gate exits, the robot returns home, resets impedance, then waits for ready_gate.
```

The no-VR RLT rollout is the cleanest test of the learned actor. If this behaves worse than plain VLA, inspect the Stage2 checkpoint or gate timing first.

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
