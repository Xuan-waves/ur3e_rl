# UR3e No-Rotvec SmolVLA RTC Rollout

This rollout is for checkpoints trained on datasets where the fixed end-effector
rotvec was removed.

Model interface:

- `observation.state`: `[tcp_x, tcp_y, tcp_z, gripper, RL_mark]`
- `action`: `[target_tcp_x, target_tcp_y, target_tcp_z, target_gripper, RL_mark]`

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

3. Start the no-rotvec RTC rollout:

```bash
conda activate ur3e_rlt
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_rtc.py \
  --policy-path outputs/train/ur3e_smolvla_060520/checkpoints/020000 \
  --execute
```

Do not use `scripts/hardware/run_ur3e_vr_tabs.sh` for rollout tests. That
launcher starts the VR and IK nodes too; the IK node publishes hold/VR pose
targets to `/ur3e_vr/ik_target` and will compete with the rollout target.
`--no-twin` only disables the MuJoCo viewer, not the IK target publisher.

Dry-run rollout without publishing robot commands:

```bash
python scripts/rollout_smolvla_no_rotvec/rollout_ur3e_smolvla_no_rotvec_rtc.py \
  --policy-path outputs/train/ur3e_smolvla_060520/checkpoints/020000
```

Useful overrides:

```bash
--action-position-mode relative   # or absolute; auto is default
--rtc-execution-horizon 10
--action-step-hz 15
--max-action-pos-step 0.035
--min-action-z 0.08
--no-preview
```
