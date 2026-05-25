# UR3e VR servoJ ROS2 teleop

This is an independent ROS2 teleop stack. The old `/home/xuan/ur3e_vr` project is only used as design reference; this code owns its VR input, IK, safety, robot, and MuJoCo twin layers.

## Controls

- Right lower trigger / grip: hold to enable arm tracking.
- Right upper trigger / index trigger: gripper command, `0.0` open to `1.0` closed.
- A button: `moveJ` return to the hardware home pose.

## ROS2 topics

All topics use `std_msgs/msg/String` with compact JSON payloads to avoid custom message generation while iterating.
High-rate control topics use best-effort `keep_last(1)` QoS so stale frames are dropped instead of queued.

- `/ur3e_vr/vr_command`: VR pose, enable, gripper, home button.
- `/ur3e_vr/robot_state`: current joint state plus FK TCP pose.
- `/ur3e_vr/joint_target`: IK output joint target for servoJ.
- `/ur3e_vr/ik_target`: target TCP pose for MuJoCo visualization.

## Run

Dry run with ROS2 communication and MuJoCo twin:

```bash
conda activate Xrobot
python scripts/hardware/ur3e_vr_servoj_ros2.py --dry-run
```

Real robot:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --robot-ip 192.168.5.1
```

Run nodes separately:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node vr
python scripts/hardware/ur3e_vr_servoj_ros2.py --node ik
python scripts/hardware/ur3e_vr_servoj_ros2.py --node robot --robot-ip 192.168.5.1
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

If `/ur3e_vr/joint_target` repeatedly alternates between `tracking:true`, `stale_vr`, and `anchored`, the VR command stream is dropping or arriving late. Start by checking:

```bash
ros2 topic hz --qos-reliability best_effort /ur3e_vr/vr_command
ros2 topic hz --qos-reliability best_effort /ur3e_vr/joint_target
ros2 topic echo --qos-reliability best_effort /ur3e_vr/robot_state
```

The MuJoCo twin follows `/ur3e_vr/robot_state`, not `/ur3e_vr/joint_target` directly.

For latency testing, prefer split-node launch:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py --node vr
python scripts/hardware/ur3e_vr_servoj_ros2.py --node ik
python scripts/hardware/ur3e_vr_servoj_ros2.py --node robot --dry-run
python scripts/hardware/ur3e_vr_servoj_ros2.py --node twin
```

Launch those nodes in a `tmux` session automatically:

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

Lower alpha is smoother with more lag; higher alpha is more responsive with more jitter.
