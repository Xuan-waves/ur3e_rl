# UR3e VR Impedance LeRobot Data Collection

本目录用于采集 UR3e VR 阻抗遥操作数据，并保存为 LeRobot v3 格式数据集。
collector 只订阅 ROS topic，不直接打开 RealSense 设备；相机节点和预览窗口都
应作为独立进程运行。

默认任务描述、数据集路径、同步窗口等参数在
`scripts/collect_data/config.py` 的 `CollectConfig` 中定义，命令行参数会覆盖
这些默认值。

## 当前默认

- 采集频率: `30 Hz`
- 同步基准: `front`，即 D455 `/camera/d455/color/image_raw`
- 图像同步窗口: `0.04 s`
- state/action/VR 同步窗口: `0.02 s`
- action 位置: `relative`，即 `target_tcp_pos - state_tcp_pos`
- action 姿态: `state`，即 action 姿态与当前 state 姿态一致
- rotvec 分支参考: `TeleopConfig.hardware_home_q` 的 FK 末端姿态

## 数据格式

- `observation.images.cam_front`: D455 RGB, `/camera/d455/color/image_raw`
- `observation.images.cam_wrist`: D405 RGB, `/camera/d405/color/image_raw`
- `observation.state`: `[tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper, RL_mark]`
- `action`: `[delta_tcp_x, delta_tcp_y, delta_tcp_z, tcp_rx, tcp_ry, tcp_rz, target_gripper, RL_mark]`

默认情况下：

- `state[0:3]` 来自 `/ur3e_vr/robot_state` 的实际 TCP 位置
- `state[3:6]` 是实际 TCP 四元数转换得到的连续 rotvec
- `action[0:3]` 是 `/ur3e_vr/ik_target` 目标位置减去当前 `state[0:3]`
- `action[3:6]` 直接复制 `state[3:6]`
- `action[6]` 优先来自 `/ur3e_vr/joint_target` 的夹爪目标，其次使用 VR command 或 robot state
- `state[7]` 和 `action[7]` 是 `RL_mark`

可以切换 action 表达：

```bash
--action-position-mode relative|absolute
--action-orientation-source state|ik_target
```

如果使用 `--action-position-mode absolute`，`action[0:3]` 会变为绝对目标 TCP
位置，feature names 会写成 `target_tcp_x/y/z`。

## 同步策略

collector 默认以 front/D455 图像为同步基准：

- 相机使用 ROS `Image.header.stamp`
- robot state、IK target、joint target、VR command 使用 collector 收到消息时的 ROS 时刻
- 每次基准相机有新帧时，collector 在 `30 Hz` 上限内采样一次
- 每次采样从 ring buffer 中选取离基准时间最近的数据
- 超出同步窗口的帧会被丢弃并打印延迟信息

同步基准可切换：

```bash
--reference-camera front|wrist|timer
```

一般建议保持默认 `front`，与 `vr_servoj_test` 的 eepose collect/rollout 对齐。
`timer` 仅用于调试，会退回固定 30 Hz 定时采样。

每个 episode 保存后会写入：

```text
meta/sync_report_episode_XXXXXX.json
```

里面包含同步窗口、基准相机、action 表达、rotvec 分支参考和同步统计。训练和
rollout 应以这个 report 为准，而不是只看当前 `config.py`。

## 启动流程

### 1. 启动 VR 阻抗遥操作

```bash
conda activate ur3e_rlt
scripts/hardware/run_ur3e_vr_tabs.sh \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop \
  --no-twin
```

`--control-mode` 只有 `impedance` 和 `servoj` 两个选项。`teleop` 是阻抗控制的
参数 profile，需要通过 `--impedance-profile teleop` 选择；`passive` 更适合
单独测试手推阻抗手感，不建议作为 VR 数据采集默认参数。

确认 VR 可以控制机械臂，A 键可以 return to home，右手 trigger 可以控制夹爪。

### 2. 启动 RealSense ROS 相机

```bash
conda activate ur3e_rlt
scripts/collect_data/run_realsense_cameras.sh
```

等待 D455 和 D405 都打印 `RealSense Node Is Up!`。collector 不负责启动相机，
也不要同时用其他程序打开相机设备。

### 3. 检查相机 topic

```bash
conda activate ur3e_rlt
source /opt/ros/humble/setup.bash
python scripts/collect_data/test_realsense_ros.py \
  --mode ros \
  --camera both \
  --no-launch \
  --duration 10
```

期望两个 topic 都接近 `30 Hz`：

```bash
ros2 topic hz /camera/d455/color/image_raw
ros2 topic hz /camera/d405/color/image_raw
```

### 4. 检查夹爪数据

按右手夹爪 trigger，同时运行：

```bash
conda activate ur3e_rlt
source /opt/ros/humble/setup.bash
python scripts/collect_data/test_vr_gripper.py --duration 10
```

重点看：

```text
vr_gripper=... joint_gripper=... state_gripper=...
```

如果 `joint_gripper` 会跟随 trigger 变化，采集出的 `action[6]` 应该有夹爪数据。

### 5. 启动采集窗口

```bash
conda activate ur3e_rlt
scripts/collect_data/run_collect_data_tabs.sh
```

默认打开两个窗口：

- `UR3e LeRobot Collector`: 保存数据
- `Collection Preview`: 显示 D455/D405 图像

collector tab 会显示固定刷新的状态面板，包括按键提示、episode 帧数、drop 数、
同步延迟、topic 状态和夹爪值。开始、保存、丢弃和 drop 警告仍会正常打印。

## VR 按键

- Right lower trigger / grip: 控制机械臂运动
- Right upper trigger / index trigger: 控制夹爪，保存到 `[0, 0.93]`
- A: 通过 robot node 返回 API home
- X: 开始记录当前 episode
- Y: 切换 `RL_mark`
- B: 结束并保存当前 episode
- Left upper trigger 完全按下: 停止记录并丢弃当前 episode，回到等待 X 的状态
- Left lower trigger / grip 完全按下: 结束整个采集程序；如果正在记录，先丢弃当前 episode 再退出

## 常用命令

设置数据集名字和任务描述：

```bash
scripts/collect_data/run_collect_data_tabs.sh \
  --dataset-name ur3e_yellow_duck_bowl \
  --task "Pick up the yellow toy duck and place it into the grey bowl."
```

限制最大 episode 数：

```bash
scripts/collect_data/run_collect_data_tabs.sh --max-episodes 20
```

明确使用默认相对位置 action：

```bash
scripts/collect_data/run_collect_data_tabs.sh \
  --action-position-mode relative \
  --action-orientation-source state
```

切换到绝对位置 action：

```bash
scripts/collect_data/run_collect_data_tabs.sh \
  --action-position-mode absolute \
  --action-orientation-source state
```

关闭预览窗口：

```bash
scripts/collect_data/run_collect_data_tabs.sh --no-preview
```

关闭固定状态面板：

```bash
scripts/collect_data/run_collect_data_tabs.sh --no-status-panel
```

只运行 collector，不打开 tab：

```bash
conda activate ur3e_rlt
source /opt/ros/humble/setup.bash
python scripts/collect_data/collect_ur3e_vr_impedance.py \
  --camera-source ros \
  --dataset-root datasets \
  --dataset-name ur3e_yellow_duck_bowl \
  --task "Pick up the yellow toy duck and place it into the grey bowl." \
  --reference-camera front \
  --action-position-mode relative \
  --action-orientation-source state \
  --max-episodes 20 \
  --no-launch-realsense-ros
```

手动打开相机预览：

```bash
conda activate ur3e_rlt
source /opt/ros/humble/setup.bash
python scripts/collect_data/preview_collection_topic.py \
  --front-topic /camera/d455/color/image_raw \
  --wrist-topic /camera/d405/color/image_raw
```

## 相机诊断

只枚举设备，不开流：

```bash
conda activate ur3e_rlt
python scripts/collect_data/test_realsense_ros.py --mode detect
```

直接用 RealSense SDK 测单相机：

```bash
python scripts/collect_data/test_realsense_ros.py --mode sdk --camera d455 --cleanup
python scripts/collect_data/test_realsense_ros.py --mode sdk --camera d405 --cleanup
```

直接用 RealSense SDK 同时测双相机：

```bash
python scripts/collect_data/test_realsense_ros.py \
  --mode sdk \
  --camera both \
  --parallel-sdk \
  --cleanup
```

启动 ROS 相机并测试 ROS 图像：

```bash
python scripts/collect_data/test_realsense_ros.py --mode ros --camera both --cleanup
```

判断方式：

- SDK 单相机失败: 设备、USB、权限或驱动问题
- SDK 双相机失败: USB 带宽、供电或设备并发问题
- SDK 成功但 ROS 失败: `realsense2_camera` 或 ROS launch 参数问题
- ROS 成功但采集失败: collector 同步、控制 topic 或写入问题

## 注意事项

- 不要让 collector、RealSense SDK 测试程序、`realsense-viewer` 同时打开同一个相机。
- 新旧数据集的 action 表达可能不同，训练和 rollout 必须按对应数据集的 sync report 解码。
- 如果 drop 很多，不要先放宽同步窗口；先检查相机频率、robot state 频率和 IK target 是否稳定。
- 重新采集 impedance 数据后，建议重新训练对应模型，旧模型不会自动适配新 action 表达。
