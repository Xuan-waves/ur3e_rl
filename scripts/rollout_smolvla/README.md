# UR3e SmolVLA Rollout

本目录用于把训练好的 SmolVLA policy 接到当前 UR3e 阻抗控制链路上做闭环验证。

rollout 只负责：

- 订阅两个相机图像和 `/ur3e_vr/robot_state`
- 按 collector 相同方式做时间窗口对齐
- 异步或同步运行 SmolVLA 推理
- 发布模型输出的末端目标到 `/ur3e_vr/ik_target`
- 发布夹爪值到 `/ur3e_vr/vr_command`
- 在 `--execute` 模式启动时请求一次 return_to_home
- 用 OpenCV 显示 rollout 实际使用的两个相机视角

Robot node 仍然负责真实机械臂阻抗控制、workspace clamp、target filter、gripper 执行。

## 前置进程

启动 RealSense ROS 相机：

```bash
scripts/collect_data/run_realsense_cameras.sh
```

启动 robot node，注意这里只需要 robot：

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node robot \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop
```

## Dry Run

默认不会发布控制命令，只打印模型输出：

```bash
python scripts/rollout_smolvla/rollout_ur3e_smolvla.py \
  --policy-path outputs/train/YOUR_RUN/checkpoints/last/pretrained_model
```

Dry-run 不会触发启动 return_to_home。

如果不传 `--policy-path`，脚本会尝试从 `outputs/train/*/checkpoints/*/pretrained_model`
里找最新 checkpoint。

## Execute

确认 dry-run 输出合理后，再加 `--execute`：

```bash
python scripts/rollout_smolvla/rollout_ur3e_smolvla.py \
  --policy-path outputs/train/YOUR_RUN/checkpoints/last/pretrained_model \
  --execute
```

默认会在程序启动约 1s 后只请求一次 return_to_home，然后释放 home 信号。
`--execute` 模式下，这个启动序列会发生在 policy/model 加载之前：先请求
return_to_home，等待 home 动作沉降，再打开夹爪一小段时间，然后才加载模型、
开始推理和发布模型 action。
如需关闭：

```bash
python scripts/rollout_smolvla/rollout_ur3e_smolvla.py \
  --policy-path outputs/train/YOUR_RUN/checkpoints/last/pretrained_model \
  --execute \
  --no-return-home-on-start
```

## 关键参数

- `--fps`: 观测采样和推理请求频率，默认 30Hz。
- `--command-hz`: 发布最新模型 action 的频率，默认 30Hz。
- `--inference-mode`: `async` 或 `sync`，默认 `async`。
- `--execution-horizon`: 每个 SmolVLA chunk 实际执行的动作数，默认 10；checkpoint 的 chunk 仍是 50。
- `--replan-every-step`: 每帧重置 SmolVLA action queue，强制用当前观测重新规划。调试用，通常先不要开。
- `--sync-inference-hz`: 仅在 `--replan-every-step` 时限制完整模型推理频率，默认 30Hz。
- `--return-home-on-start` / `--no-return-home-on-start`: 是否在 `--execute` 启动时请求一次 home。
- `--start-home-delay-s`: 程序启动后延迟多久发送 home 请求，默认 1.0s。
- `--start-home-pulse-s`: home 请求保持时间，默认 0.6s。
- `--start-home-settle-s`: home 请求释放后等待机械臂回到 home 的时间，默认 1.5s。
- `--start-open-gripper-s`: home 完成后持续发送夹爪打开命令的时间，默认 0.8s。
- `--start-open-gripper-value`: 启动阶段夹爪打开值，默认 0.0。
- `--preview` / `--no-preview`: 是否显示 OpenCV 双相机窗口，默认开启。
- `--preview-hz`: OpenCV 窗口刷新频率，默认 10Hz。
- `--max-dt-front-image` / `--max-dt-wrist-image` / `--max-dt-state`: 对齐时间窗，默认 80ms。
- `--max-action-age-s`: 模型 action 超过这个年龄后停止发布，默认 0.5s。
- `--max-position-step-m`: 发布端的单步末端位置限制，默认 4cm，用于抑制模型尖峰。
- `--action-position-mode`: action 前三维解释方式，默认 `relative`，即
  `target_tcp_pos = current_tcp_pos + action[0:3]`。旧的绝对位置模型用 `absolute`。
- `--action-orientation-source`: 姿态目标来源，默认 `state`，即保持当前实际 TCP
  绝对姿态；如果训练数据里的 action[3:6] 是目标姿态，用 `ik_target`。
- `--rl-mark`: rollout 时喂给 policy 的 state 最后一维，默认 0。

当前新采集数据默认 action 表示为：

```text
[delta_tcp_x, delta_tcp_y, delta_tcp_z, tcp_rx, tcp_ry, tcp_rz, target_gripper, RL_mark]
```

因此 rollout 默认也按这个语义执行。使用旧数据训练的模型需要显式加：

```bash
--action-position-mode absolute --action-orientation-source ik_target
```

## 对齐方式

与 collector 一致：

- 图像使用 ROS 接收时刻 `time.monotonic()`
- robot state 使用消息内的 monotonic stamp
- 每个 30Hz tick 以当前 monotonic time 为基准，分别在三个 buffer 中找最近样本
- 任一源超过对应 `max_dt_*`，该 tick 不送入模型

这会暴露相机或 robot state 的延迟问题，而不是用旧帧掩盖。
