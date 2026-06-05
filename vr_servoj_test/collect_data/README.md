# UR3e Pure VR ServoJ LeRobot Collection

这个目录用于测试“不上阻抗控制”的 VR servoJ 数据采集。它属于仓库根目录下的
`vr_servoj_test` 测试工程，后续 train 和 rollout 也放在这个目录内。

控制链路仍然是：

```text
VR node -> IK node -> joint_target -> robot commanded_joint_target -> robot servoJ
```

collector 只订阅 ROS topic，不直接控制机械臂，也不直接打开 RealSense。

## 数据表示

`observation.images.cam_front` 和 `observation.images.cam_wrist` 与当前 impedance collector 一致。

`observation.state` 可选：

```bash
--state-mode eepose      # [tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper, RL_mark]
--state-mode jointspace  # [q1, q2, q3, q4, q5, q6, gripper, RL_mark]
```

`action` 可选：

```bash
--action-mode jointspace # [commanded_q1..commanded_q6, target_gripper, RL_mark]
--action-mode eepose     # [tcp position, target_tcp_rx, target_tcp_ry, target_tcp_rz, target_gripper, RL_mark]
```

当 `--action-mode eepose` 时，位置还有两种表示：

```bash
--ee-action-position-mode relative # target_pos - current_tcp_pos
--ee-action-position-mode absolute # target_pos
```

eepose 姿态使用 rotvec。由于 UR3e home 附近的末端姿态接近 pi 分支，collector 会用
`TeleopConfig.hardware_home_q` 的 FK 姿态作为固定 rotvec 分支参考；每个 episode 内再
保持时间连续。这样 collect 和 rollout 都可以独立复现同一个分支规则，不依赖某一轮数据
的第一帧。

## 启动流程

1. 启动纯 VR servoJ 控制：

```bash
scripts/hardware/run_ur3e_vr_tabs.sh \
  --robot-ip 192.168.5.1 \
  --control-mode servoj \
  --no-twin \
  --conda-env ur3e_rlt
```

2. 启动 RealSense ROS：

```bash
scripts/collect_data/run_realsense_cameras.sh
```

3. 启动采集：

```bash
vr_servoj_test/collect_data/run_collect_servoj_tabs.sh \
  --state-mode eepose \
  --action-mode jointspace
```

如果想采末端动作：

```bash
vr_servoj_test/collect_data/run_collect_servoj_tabs.sh \
  --state-mode eepose \
  --action-mode eepose \
  --ee-action-position-mode relative
```

## 同步方式

collector 默认以 front/D455 相机帧作为统一采样时间，在每个采样点上从各 topic 的
ring buffer 里取离当前采样时间最近的一帧：

- front image: `/camera/d455/color/image_raw`
- wrist image: `/camera/d405/color/image_raw`
- robot state: `/ur3e_vr/robot_state`
- VR command: `/ur3e_vr/vr_command`
- commanded joint action: `/ur3e_vr/commanded_joint_target`
- EE action: `/ur3e_vr/ik_target`

当 `--action-mode jointspace` 时，`commanded_joint_target` 是必需的，`ik_target` 只作为
诊断 topic；当 `--action-mode eepose` 时，`ik_target` 是必需的，`commanded_joint_target`
只作为夹爪/诊断备用来源。

默认时间窗口：

```bash
front/wrist image <= 0.04s
state/action/vr   <= 0.02s
```

超过窗口会 drop frame，不会复用 stale 数据。图像优先使用 ROS image
header stamp；`robot_state`、action 和 VR topic 按参考工程做法使用 collector
回调接收时间。默认 `--sync-reference front`，也就是由 front/D455 图像帧触发
采样；如果要回到固定 timer 触发，可传 `--sync-reference timer`。

## 手柄按键

- 右手下扳机: 控制机械臂运动
- 右手上扳机: 控制夹爪
- A: return_to_home
- X: 开始记录
- Y: 切换 `RL_mark`
- B: 保存当前 episode
- 左手上扳机完全按下: 丢弃当前 episode 并停止记录
- 左手下扳机完全按下: 退出采集；如果正在记录，则先丢弃当前 episode
