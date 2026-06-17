# UR3e No-Rotvec SmolVLA RTC Rollout

This rollout is for checkpoints trained on datasets where the fixed end-effector
rotvec was removed.

Model interface:

- Current RLT VLA no-rotvec checkpoints: `observation.state=[tcp_x, tcp_y, tcp_z, gripper]`,
  `action=[target_tcp_x, target_tcp_y, target_tcp_z, target_gripper]`.
- Older no-rotvec checkpoints with `RL_mark` are still supported. The rollout reads
  `config.json` from the checkpoint and automatically builds 4D or 5D state.

The rollout restores a fixed end-effector orientation from `TeleopConfig.hardware_home_q`
and publishes impedance targets on `/ur3e_vr/ik_target`.

## Full Command Flow

Start these in separate terminals. Keep the robot stack running in impedance
mode; this rollout publishes fixed-orientation TCP targets to that stack.

1. Start the RealSense ROS camera topics:

```bash
conda activate ur3e_rlt
scripts/collect_data/run_realsense_cameras.sh
```

2. Start only the UR3e robot node in impedance mode:

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

3. Optional for RLT/human intervention: start the VR node on a raw topic.

Do not start the IK node. The rollout is the only node that should publish
`/ur3e_vr/ik_target`; VR is only an input for intervention.

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
conda activate ur3e_rlt
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node vr \
  --vr-output-topic /ur3e_vr/vr_command_raw
```

4. Start the no-rotvec RTC rollout:

```bash
conda activate ur3e_rlt
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_rtc.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0610/checkpoints/050000/pretrained_model \
  --execute
```

With `--vr-override` enabled, holding the right-hand lower trigger switches the
published target from model output to anchored VR position control. Releasing it
drains the pending RTC action queue briefly and resumes model control from the
current TCP. Gripper takeover is relative to the trigger value at the moment
intervention starts, so a closed gripper will not suddenly open.

Do not use `scripts/hardware/run_ur3e_vr_tabs.sh` for rollout tests. That
launcher starts the VR and IK nodes too; the IK node publishes hold/VR pose
targets to `/ur3e_vr/ik_target` and will compete with the rollout target.
`--no-twin` only disables the MuJoCo viewer, not the IK target publisher.

Dry-run rollout without publishing robot commands:

```bash
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_rtc.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0610/checkpoints/050000/pretrained_model
```

Useful overrides:

```bash
--action-position-mode relative   # or absolute; auto is default
--rtc-execution-horizon 10
--action-step-hz 15
--max-action-pos-step 0.035
--min-action-z 0.042
--vr-override / --no-vr-override
--vr-raw-topic /ur3e_vr/vr_command_raw
--no-preview
```

For future RLT experiments, this RTC rollout is a good plain-VLA baseline. When
the RL residual is added, start with a short horizon such as
`--rtc-execution-horizon 3` to reduce stale actions near contact.

## Synchronous Diagnostic Rollout

Use this version to check whether RTC queue latency is making grasp timing worse.
It runs ordinary synchronous SmolVLA inference and replaces old queued actions
whenever a new inference finishes. It also supports the same raw-VR intervention
path as the RTC rollout: hold the right-hand lower trigger to take over the TCP
target, release it to drain stale model actions and resume synchronous inference.

```bash
conda activate ur3e_rlt
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_sync.py \
  --policy-path outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/050000/pretrained_model \
  --n-obs-steps 2 \
  --execution-horizon 3 \
  --action-step-hz 30 \
  --action-pose-filter-alpha 0.75 \
  --max-action-pos-step 0.06 \
  --execute
```

For the VR intervention test, keep the robot node and raw VR node from the full
command flow running. Do not start the IK node. The sync rollout subscribes to
`/ur3e_vr/vr_command_raw` and publishes the final fixed-orientation target on
`/ur3e_vr/ik_target`.

Useful sync-only overrides:

```bash
--n-obs-steps 1|2             # omit to follow the checkpoint config
--vr-override / --no-vr-override
--vr-raw-topic /ur3e_vr/vr_command_raw
--vr-override-resume-delay-s 0.30
--execution-horizon 1|3|10
```
