# VR ServoJ SmolVLA Training

This directory trains SmolVLA on datasets collected with the pure VR servoJ collector.

Default config:

```bash
vr_servoj_test/train/config.py
```

Default dataset:

```bash
datasets/ur3e_lerobot_vr_servoj_20260604_113221
```

The default dataset is jointspace:

- `observation.images.cam_front`: D455 RGB
- `observation.images.cam_wrist`: D405 RGB
- `observation.state`: `[q1, q2, q3, q4, q5, q6, gripper, RL_mark]`
- `action`: `[commanded_q1, commanded_q2, commanded_q3, commanded_q4, commanded_q5, commanded_q6, target_gripper, RL_mark]`

## Dry Run

```bash
conda activate ur3e_rlt
python vr_servoj_test/train/train_servoj_smolvla.py --dry-run
```

The dry run prints dataset metadata and the exact `lerobot-train` command.

## Smoke Train

Run a short training job before launching a long run:

```bash
python vr_servoj_test/train/train_servoj_smolvla.py \
  --steps 200 \
  --batch-size 2 \
  --num-workers 4 \
  --save-freq 100 \
  --log-freq 10
```

## Full Train

The default config targets a 24 GB GPU:

```bash
python vr_servoj_test/train/train_servoj_smolvla.py
```

Defaults:

- `batch_size=32`
- `steps=20000`
- `chunk_size=50`
- `n_action_steps=50`
- `n_obs_steps=1`
- `image_size=256`
- `num_vlm_layers=16`

If CUDA OOM appears, reduce in this order:

```bash
--batch-size 16
--num-workers 16
--image-size 224
```

Extra LeRobot args can be forwarded after `--`, for example:

```bash
python vr_servoj_test/train/train_servoj_smolvla.py \
  --steps 1000 \
  -- \
  --policy.optimizer_lr=5e-5
```
