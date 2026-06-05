# UR3e SmolVLA Training

本目录用于在当前 UR3e VR 阻抗采集数据上启动 LeRobot SmolVLA 训练。

默认训练参数集中放在：

```bash
scripts/train_smolvla/config.py
```

命令行参数仍可临时覆盖 config 里的默认值。

默认数据集：

```bash
datasets/ur3e_lerobot_vr_impedance_20260531_172425
```

数据集内容：

- `observation.images.cam_front`: D455 RGB
- `observation.images.cam_wrist`: D405 RGB
- `observation.state`: `[tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper, RL_mark]`
- `action`: `[target_tcp_x, target_tcp_y, target_tcp_z, target_tcp_rx, target_tcp_ry, target_tcp_rz, target_gripper, RL_mark]`

## 环境检查

先在 `ur3e_rlt` 环境运行 dry-run：

```bash
conda activate ur3e_rlt
python scripts/train_smolvla/train_ur3e_smolvla.py --dry-run
```

当前环境中需要注意两个点：

- SmolVLA 需要 `transformers`，如果缺失，训练前安装 LeRobot 的 SmolVLA extras。
- 本机 `torchcodec` 与 FFmpeg 共享库可能不匹配，训练脚本固定使用 `--dataset.video_backend=pyav`。

建议安装：

```bash
pip install "lerobot[smolvla]"
```

如果网络或版本锁定不方便，也可以先只补：

```bash
pip install transformers
```

## Smoke Train

第一次建议先跑短训练，确认视频解码、tokenizer、显存和 checkpoint 都正常：

```bash
python scripts/train_smolvla/train_ur3e_smolvla.py \
  --steps 200 \
  --batch-size 1 \
  --save-freq 100 \
  --log-freq 10
```

输出默认保存到：

```bash
outputs/train/ur3e_smolvla_YYYYmmdd_HHMMSS
```

## 常用训练命令

小数据集起步训练：

```bash
python scripts/train_smolvla/train_ur3e_smolvla.py
```

当前默认按 24GB 显存设置：

- `batch_size=4`
- `steps=5000`
- `chunk_size=50`
- `n_action_steps=50`
- `image_size=256`
- `num_workers=4`

如果 CUDA OOM，优先把 `batch_size` 从 4 降到 2；如果显存仍有余量，可尝试把
`image_size` 从 256 提到 384 或 512。

使用预训练 SmolVLM 权重初始化 VLM：

```bash
python scripts/train_smolvla/train_ur3e_smolvla.py \
  --load-vlm-weights \
  --steps 3000 \
  --batch-size 2
```

## 关键参数

- `--dataset`: 本地 LeRobot 数据集路径。
- `--steps`: policy update 步数。
- `--batch-size`: DataLoader batch size。默认 4，面向 24GB 显存。
- `--chunk-size` / `--n-action-steps`: 一次预测的动作窗口，默认 50 帧，即 30Hz 下约 1.67 秒。
- `--image-size`: SmolVLA 输入图像 padding 后尺寸，默认 256。
- `--num-vlm-layers`: 使用的 VLM 层数，默认 8，用于降低初期训练成本。
- `--load-vlm-weights`: 从 `--vlm-model-name` 加载 VLM 预训练权重。默认关闭，避免第一次训练必须下载大权重。
- `--output-dir`: 指定训练输出目录。

额外 LeRobot 参数可以放在 `--` 后面透传，例如：

```bash
python scripts/train_smolvla/train_ur3e_smolvla.py \
  --steps 1000 \
  -- \
  --policy.optimizer_lr=5e-5
```

## 当前数据提醒

脚本会读取 `meta/stats.json` 并打印 `RL_mark` 的范围。如果 `RL_mark min=max=0`，
说明这份数据里没有 RL 阶段标记变化，模型暂时学不到“何时切换到 RL”的判断。
