# VR ServoJ SmolVLA Rollout

This rollout is for policies trained from the pure VR servoJ collector.

The policy observation/action layout is jointspace:

- `observation.images.cam_front`: D455 RGB
- `observation.images.cam_wrist`: D405 RGB
- `observation.state`: `[q1, q2, q3, q4, q5, q6, gripper, RL_mark]`
- `action`: `[commanded_q1, commanded_q2, commanded_q3, commanded_q4, commanded_q5, commanded_q6, target_gripper, RL_mark]`

The rollout publishes model actions to `/ur3e_vr/joint_target`. The existing
`ur3e_robot` node still owns servoJ execution, robot-side filtering, speed
limits, and RTDE communication.

## Start Dependencies

Start the camera topics and the servoJ robot node first. Do not start the VR
or IK nodes for rollout.

Example terminals:

```bash
conda activate ur3e_rlt
scripts/collect_data/run_realsense_cameras.sh
```

```bash
conda activate ur3e_rlt
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node robot \
  --robot-ip 192.168.5.1 \
  --control-mode servoj
```

## Dry Run

Dry run loads the model, synchronizes observations, runs policy inference, and
prints target joints without publishing to the robot.

```bash
conda activate ur3e_rlt
python vr_servoj_test/rollout/rollout_ur3e_servoj_smolvla.py \
  --policy-path outputs/train/ur3e_smolvla_0604/checkpoints/020000
```

## Execute

```bash
python vr_servoj_test/rollout/rollout_ur3e_servoj_smolvla.py \
  --policy-path outputs/train/ur3e_smolvla_0604/checkpoints/020000 \
  --execute
```

With `--execute`, rollout sends a one-shot return-to-home command before loading
the policy, opens the gripper, then starts synchronous inference.

By default rollout uses the same project-local Hugging Face cache as training:
`.cache/huggingface`, and runs in offline mode. If you intentionally want to
download a missing backbone, pass `--no-offline`.

## RTC Execute

RTC uses LeRobot's real-time chunking path: the model still predicts an action
chunk, but the unexecuted tail of the previous chunk is fed back into the next
chunk as a prefix constraint. This is meant to reduce chunk-boundary reversal
and late-inference discontinuity.

```bash
python vr_servoj_test/rollout/rollout_ur3e_servoj_smolvla_rtc.py \
  --policy-path outputs/train/ur3e_smolvla_0604/checkpoints/020000 \
  --execute
```

Default RTC settings follow the reference project style:

- `--rtc-execution-horizon 10`: overlap horizon used by RTC guidance.
- `--rtc-max-guidance-weight 10.0`: maximum prefix guidance strength.
- `--rtc-prefix-attention-schedule linear`: prefix weighting schedule.
- `--action-step-hz 15`: rate used to consume queued policy actions. Raise this only if inference is consistently faster than the chunk execution window.
- `--command-hz 100`: keep-alive publish rate for the current joint target.

## RTC EEpose Execute

Use this for policies trained with:

```bash
--state-mode eepose --action-mode eepose --ee-action-position-mode absolute
# or
--state-mode eepose --action-mode eepose --ee-action-position-mode relative
```

For absolute eepose policies, the model action is interpreted as:

```text
[target_tcp_x, target_tcp_y, target_tcp_z, target_tcp_rx, target_tcp_ry, target_tcp_rz, gripper, RL_mark]
```

For relative eepose policies, the model action is interpreted as:

```text
[delta_tcp_x, delta_tcp_y, delta_tcp_z, target_tcp_rx, target_tcp_ry, target_tcp_rz, gripper, RL_mark]
```

The rollout converts the target TCP pose to a joint target with the local
Mink IK solver, then publishes `/ur3e_vr/joint_target`. It also publishes the
model TCP target to `/ur3e_vr/ik_target` for debugging/visualization. Start the
camera topics and the robot node only; do not start the VR or IK nodes for this
rollout.

```bash
python vr_servoj_test/rollout/rollout_ur3e_servoj_eepose_smolvla_rtc.py \
  --policy-path outputs/train/ur3e_smolvla_0605/checkpoints/020000 \
  --execute
```

EEpose rotvec observations use the same fixed branch reference as collection:
`TeleopConfig.hardware_home_q` FK. This avoids episode-dependent `+pi/-pi`
branch choices.

`--ee-action-position-mode auto` is the default. It reads the checkpoint's
training dataset sync report and chooses `relative` or `absolute` automatically.
Override it manually only when testing a mismatched or edited dataset.

## Useful Parameters

- `--execution-horizon`: number of queued actions to request from each SmolVLA chunk. Default `50`.
- `--action-step-hz`: rate used to advance through queued policy actions. Default `15`; `--command-hz` still keeps publishing the current target.
- `--prefetch-actions`: start the next inference once the queue has this many or fewer actions left. Default `40`.
- `--replace-queue-on-infer`: replace the old queued tail when a fresh chunk arrives. Enabled by default.
- `--replan-every-step`: ignore the chunk queue and run a fresh inference for every published action.
- `--fps`: observation/inference tick rate. Default `30`.
- `--command-hz`: joint target keep-alive publish rate. Default `100`; queued actions still advance at `--fps`.
- `--sync-reference`: `front`, `wrist`, or `timer`. Default `front`, matching collection.
- `--max-dt-front-image`, `--max-dt-wrist-image`, `--max-dt-state`: strict sync windows.
- `--rl-mark`: fixed RL mark appended to the rollout state. Default `0`.
- `--no-preview`: disable the OpenCV two-camera preview.
- `--hf-home`: Hugging Face cache directory. Default `.cache/huggingface`.
- `--no-offline`: allow Hugging Face/Transformers to download missing files.
- `--action-q-filter-alpha`: low-pass coefficient for model joint targets. Lower is smoother. Default `0.35`.
- `--max-action-joint-step`: max per-action jump before publishing to the robot node. Default `0.06` rad.
- `--ee-action-position-mode`: eepose action position decoding, `auto`, `relative`, or `absolute`.
