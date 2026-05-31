# UR3e VR Impedance LeRobot Data Collection

本目录用于把当前 UR3e VR 阻抗遥操作数据保存成 LeRobot 数据集。采集频率为
30 Hz，相机通过 ROS topic 读取，collector 不直接打开 RealSense 设备。

## 数据格式

- `observation.images.cam_front`: D455 RGB, `/camera/d455/color/image_raw`
- `observation.images.cam_wrist`: D405 RGB, `/camera/d405/color/image_raw`
- `observation.state`: `[tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper, RL_mark]`
- `action`: `[target_tcp_x, target_tcp_y, target_tcp_z, target_tcp_rx, target_tcp_ry, target_tcp_rz, target_gripper, RL_mark]`

`state` 的 TCP 来自 `/ur3e_vr/robot_state`。`action` 的 TCP 目标来自
`/ur3e_vr/ik_target`。夹爪目标优先来自 `/ur3e_vr/joint_target`，并保留
`/ur3e_vr/vr_command` 与 `/ur3e_vr/robot_state` 作为诊断/备用来源。

默认数据路径、dataset 名字、task 描述、最大 episode 数等配置在
`scripts/collect_data/config.py` 的 `CollectConfig` 中定义。命令行参数会覆盖
这些默认值。

## 采集流程

### 1. 启动 VR 阻抗遥操作

在第一个终端启动 VR、IK、Robot 三个 tab：

```bash
conda activate ur3e_rlt
scripts/hardware/run_ur3e_vr_tabs.sh \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile passive
```

确认 VR 可以控制机械臂，A 键可以回 home，右手 trigger 可以控制夹爪。

### 2. 启动 RealSense ROS 相机

在第二个终端单独启动相机 ROS 节点：

```bash
conda activate ur3e_rlt
scripts/collect_data/run_realsense_cameras.sh
```

等待两个节点都打印 `RealSense Node Is Up!`。

### 3. 检查相机 topic

在第三个终端检查已经启动的相机 topic，不要重复启动相机：

```bash
conda activate ur3e_rlt
source /opt/ros/humble/setup.bash
python scripts/collect_data/test_realsense_ros.py \
  --mode ros \
  --camera both \
  --no-launch \
  --duration 10
```

期望 D455/D405 都接近 `30 Hz`。

也可以用 ROS 自带命令看频率：

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /camera/d455/color/image_raw
ros2 topic hz /camera/d405/color/image_raw
```

### 4. 检查夹爪数据

按右手食指 trigger，同时运行：

```bash
conda activate ur3e_rlt
source /opt/ros/humble/setup.bash
python scripts/collect_data/test_vr_gripper.py --duration 10
```

重点看：

```text
vr_gripper=... joint_gripper=... state_gripper=...
```

如果 `joint_gripper` 会跟随 trigger 变化，采集出来的 `action[6]` 应该有夹爪数据。

### 5. 启动采集窗口

相机和 VR 遥操作都正常后，启动 collector 与 OpenCV 预览：

```bash
conda activate ur3e_rlt
scripts/collect_data/run_collect_data_tabs.sh
```

默认会打开：

- `UR3e LeRobot Collector`: 负责保存数据
- `Collection Preview`: 显示 D455/D405 图像

collector tab 会显示一个固定刷新的状态面板，包含按键提示、当前 episode
帧数、drop 数、同步延迟、topic 状态和夹爪值，不再持续滚动打印周期状态。
事件日志，例如开始、保存、丢弃或 drop 警告，仍会正常打印。

不要默认使用 `--launch-realsense-ros`。推荐相机始终由
`run_realsense_cameras.sh` 单独启动，collector 只订阅 topic。

## VR 按键

- Right lower trigger / grip: 控制机械臂运动
- Right upper trigger / index trigger: 控制夹爪，保存到 `[0, 0.93]`
- A: 通过 robot node 返回 API home
- X: 开始记录当前 episode
- Y: 切换 `RL_mark`
- B: 结束并保存当前 episode
- Left upper trigger 完全按下: 停止记录并丢弃当前 episode，回到等待 X 开始的状态
- Left lower trigger / grip 完全按下: 结束整个采集程序；如果正在记录，先丢弃当前 episode 再退出

## 常用参数

指定数据集名字和任务描述：

```bash
scripts/collect_data/run_collect_data_tabs.sh \
  --dataset-name ur3e_ethernet_insert \
  --task "Insert the Ethernet connector into the matching slot."
```

设置最大采集轮次，保存到指定 episode 数后自动退出：

```bash
scripts/collect_data/run_collect_data_tabs.sh --max-episodes 20
```

如果 `CollectConfig.max_episodes = 0` 或命令行为 `--max-episodes 0`，表示不限制轮次。

关闭预览窗口：

```bash
scripts/collect_data/run_collect_data_tabs.sh --no-preview
```

关闭 collector 的固定状态面板，恢复普通日志风格：

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
  --dataset-name ur3e_ethernet_insert \
  --task "Insert the Ethernet connector into the matching slot." \
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

如果相机异常，按层级排查。

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
- SDK 双相机失败: USB 带宽/供电/设备并发问题
- SDK 成功但 ROS 失败: `realsense2_camera` 或 ROS launch 参数问题
- ROS 成功但采集失败: collector 同步、控制 topic 或写入问题

## 同步策略

collector 以固定 30 Hz tick 采样，在每个 tick 上从 ring buffer 中选择最近的：

- front image
- wrist image
- robot state
- IK target
- joint target
- VR command

默认最大时间差为：

```bash
--max-dt-front 0.08
--max-dt-wrist 0.08
--max-dt-state 0.08
--max-dt-action 0.08
```

超过时间窗口的帧会被丢弃并打印延迟信息。这样可以暴露相机或控制 topic
卡顿，而不是把 stale 数据悄悄写进数据集。
