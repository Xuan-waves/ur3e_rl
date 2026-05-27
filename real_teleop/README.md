# UR3e VR servoJ ROS2 teleop

This is an independent ROS2 teleop stack. The old `/home/xuan/ur3e_vr` project is only used as design reference; this code owns its VR input, IK, safety, robot, and MuJoCo twin layers.

## Controls

- Right lower trigger / grip: hold to enable arm tracking.
- Right upper trigger / index trigger: gripper command, `0.0` open to `1.0` closed.
- A button: `moveJ` return to the hardware home pose.

## ROS2 topics

High-rate control topics use `std_msgs/msg/Float64MultiArray`, best-effort `keep_last(1)` QoS, and "latest sample wins" semantics so stale frames are dropped instead of queued. Low-rate debug remains JSON in `std_msgs/msg/String`.

- `/ur3e_vr/vr_command`: 100 Hz VR pose, enable, gripper, home button.
- `/ur3e_vr/robot_state`: 200 Hz current joint state plus FK TCP pose.
- `/ur3e_vr/joint_target`: 200 Hz IK output joint target for servoJ.
- `/ur3e_vr/ik_target`: target TCP pose for MuJoCo visualization.

## Run

Dry run with ROS2 communication and MuJoCo twin:

```bash
conda activate ur3e_rlt
scripts/hardware/run_ur3e_vr_tabs.sh --dry-run
```

Real robot:

```bash
scripts/hardware/run_ur3e_vr_tabs.sh --robot-ip 192.168.5.1
```

The bash launcher opens three GNOME Terminal tabs for `vr`, `ik`, and `robot`, with separate Python processes and shared Ctrl+C cleanup.

Run nodes separately:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node vr
python scripts/hardware/ur3e_vr_servoj_ros2.py --node ik
python scripts/hardware/ur3e_vr_servoj_ros2.py --node robot --robot-ip 192.168.5.1
```

To verify only the MuJoCo viewer path, run:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node twin
```

Core dependencies are ROS2 `rclpy`, `std_msgs`, `numpy`, `scipy`, `mujoco`, `mink`, `daqp`, `ur_rtde`, and the local `Xrobot_tool` Python package.

## Offline checks

These checks do not connect to the real robot:

```bash
conda run -n Xrobot python -m compileall real_teleop scripts/hardware/ur3e_vr_servoj_ros2.py
conda run -n Xrobot python -c "from real_teleop.config import TeleopConfig; from real_teleop.kinematics import RobotKinematics; cfg=TeleopConfig(); kin=RobotKinematics(cfg); q=kin.model.key('home').qpos[:6].copy(); print(kin.forward(q)[0])"
conda run -n Xrobot env ROS_LOG_DIR=/tmp/ros_logs python scripts/hardware/ur3e_vr_servoj_ros2.py --help
```

Use `--dry-run` for the robot node until the UR controller is connected. In sandboxed environments, ROS2 may need `ROS_LOG_DIR=/tmp/ros_logs` because `$HOME/.ros/log` can be read-only.

If `/ur3e_vr/joint_target` repeatedly alternates between the tracking flag and hold reason codes, the VR command stream is dropping or arriving late. Start by checking:

```bash
ros2 topic hz --qos-reliability best_effort /ur3e_vr/vr_command
ros2 topic hz --qos-reliability best_effort /ur3e_vr/joint_target
ros2 topic echo --qos-reliability best_effort /ur3e_vr/robot_state
```

The MuJoCo twin now lives inside the IK node, so the ROS graph has three independent nodes: `vr`, `ik`, and `robot`. It follows `/ur3e_vr/robot_state`, while `/ur3e_vr/ik_target` drives the visual target marker.

For latency testing, prefer split-node launch:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node vr
python scripts/hardware/ur3e_vr_servoj_ros2.py --node ik
python scripts/hardware/ur3e_vr_servoj_ros2.py --node robot --dry-run
```

The Python entry still supports split-node launch for quick debugging:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node all-tabs --dry-run
```

Equivalent form:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node all --split-tabs --dry-run
```

The default launcher uses `tmux` when available and attaches in the current terminal. This avoids GNOME Terminal tab creation errors. To force a specific launcher:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node all-tabs --dry-run --split-launcher tmux
python scripts/hardware/ur3e_vr_servoj_ros2.py --node all-tabs --dry-run --split-launcher gnome-tabs
python scripts/hardware/ur3e_vr_servoj_ros2.py --node all-tabs --dry-run --split-launcher print
```

Smoothing parameters live in `real_teleop/config.py`:

- `ctrl_filter_alpha`: VR controller pose EMA.
- `target_filter_alpha`: TCP target pose EMA before IK.
- `joint_target_alpha`: robot-side joint target EMA before servoJ.
- `fixed_ee_orientation`: if `True`, VR controls TCP position only and keeps TCP orientation fixed at the hardware-home end-effector orientation.
- `target_pos_hold_epsilon`, `target_rot_hold_epsilon`, `ik_joint_deadband`: hold tiny target/IK changes to suppress micro-jitter.
- `max_joint_speed` / `max_joint_step`: robot-side joint speed limiting.
- `robot_state_hz` / `actual_read_hz` / `gripper_hz`: robot node timers split away from the 200 Hz servoJ loop.

Lower alpha is smoother with more lag; higher alpha is more responsive with more jitter.

## Position impedance test

Use this standalone script to tune end-effector xyz position impedance before
integrating it into the VR teleop loop:

```bash
python scripts/hardware/ur3e_position_impedance_test.py --print-config
```

The default parameters live in `real_teleop/impedance/config.py`.
`real_teleop/impedance_config.py` is kept as a compatibility import for older
scripts. Tune the `passive`, `spring`, and `zero-force` profiles in the new
config file first; command-line flags are only temporary overrides.

Read the real robot TCP state and print the computed force without commanding
forceMode:

```bash
python scripts/hardware/ur3e_position_impedance_test.py --read-robot --robot-ip 192.168.5.1 --duration 5 --offset 0 0 0
```

Send real UR forceMode commands only after confirming the robot workspace is
clear and the VR/servoJ robot node is stopped:

```bash
python scripts/hardware/ur3e_position_impedance_test.py --execute --robot-ip 192.168.5.1 --duration 5 --offset 0 0 0 --kp 160 160 160 --kd 18 18 18
```

Move to the API home pose first, then hold the home TCP position with impedance:

```bash
python scripts/hardware/ur3e_position_impedance_test.py --execute --mode spring --robot-ip 192.168.5.1 --duration 5
```

For hand-guided testing at the API home pose, use passive mode. This keeps a
soft virtual spring around the home TCP position while xyz axes are compliant:

```bash
python scripts/hardware/ur3e_position_impedance_test.py --execute --mode passive --robot-ip 192.168.5.1
```

The current hardware path uses UR's RTDE TCP pose and TCP speed by default:
`--state-source rtde`. The Jacobian source is still available for later work
with `--state-source jacobian`, but do not use it on the real robot until the
MuJoCo base/TCP frame is aligned with the UR controller TCP frame.

If the TCP still slowly sinks, set the tool payload correctly with
`--payload-mass` / `--payload-cog` or add a small upward `--force-bias 0 0 FZ`.
`--mode zero-force` is available for damping-only tests, but it can drift/drop
because it intentionally applies no position hold.

The implementation is xyz Cartesian impedance:
`F_xyz = Kp * (target_xyz - tcp_xyz) - Kd * tcp_linear_velocity - M * tcp_acceleration + force_bias`,
with per-axis force clipping and a startup force ramp. `M` is optional and can
be left at zero while tuning `Kp`, `Kd`, `max_force`, payload, and UR
`force_mode_limits`. For UR forceMode, the first three `force_mode_limits`
values are TCP speed limits for compliant xyz axes, so keep them in m/s-scale
values such as `0.2` to `0.8`, not tens.

The reusable motion API now lives in `real_teleop/impedance/`.  Future VR
control should update `RtdeImpedanceMotion.set_target_pose(...)` at the VR rate,
while the robot loop keeps calling `RtdeImpedanceMotion.step()` at 200 Hz.

## VR impedance mode

The tab launcher now defaults to impedance execution:

```bash
scripts/hardware/run_ur3e_vr_tabs.sh --robot-ip 192.168.5.1 --control-mode impedance
```

In this mode the IK tab is used as a VR-to-TCP-target node: it keeps the
existing VR anchoring, dead zones, pose filters, jump limits, fixed-orientation
option, and MuJoCo twin target visualization, then publishes the desired TCP
pose. The robot tab subscribes to that pose and drives the real UR through
`RtdeImpedanceMotion.step()` at 200 Hz. The direct robot state and workspace
limits are RTDE-frame values; tune `impedance_workspace_min/max` in
`real_teleop/config.py` separately from the MuJoCo/IK workspace.

The old servoJ path is still available with:

```bash
scripts/hardware/run_ur3e_vr_tabs.sh --control-mode servoj
```
