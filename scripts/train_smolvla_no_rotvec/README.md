# UR3e No-Rotvec SmolVLA Training

This directory trains SmolVLA on the impedance dataset with fixed orientation
removed from `observation.state` and `action`.

Default config:

```bash
scripts/train_smolvla_no_rotvec/config.py
```

Default dataset:

```bash
datasets/ur3e_lerobot_vr_impedance_20260605_170753_no_rotvec
```

Dataset features:

- `observation.images.cam_front`: D455 RGB
- `observation.images.cam_wrist`: D405 RGB
- `observation.state`: `[tcp_x, tcp_y, tcp_z, gripper, RL_mark]`
- `action`: `[target_tcp_x, target_tcp_y, target_tcp_z, target_gripper, RL_mark]`

The end-effector orientation is assumed fixed by the controller/rollout side.
Train and rollout must therefore use the matching no-rotvec interface.

## Dry Run

```bash
conda activate ur3e_rlt
python scripts/train_smolvla_no_rotvec/train_ur3e_smolvla_no_rotvec.py --dry-run
```

The dry run prints dataset metadata and the exact `lerobot-train` command.

## Smoke Train

```bash
python scripts/train_smolvla_no_rotvec/train_ur3e_smolvla_no_rotvec.py \
  --steps 200 \
  --batch-size 2 \
  --num-workers 4 \
  --save-freq 100 \
  --log-freq 10
```

## Full Train

```bash
python scripts/train_smolvla_no_rotvec/train_ur3e_smolvla_no_rotvec.py
```

Defaults target the 24 GB training GPU:

- `batch_size=16`
- `steps=20000`
- `chunk_size=50`
- `n_action_steps=50`
- `n_obs_steps=1`
- `image_size=256`
- `num_vlm_layers=16`

If CUDA OOM appears, reduce in this order:

```bash
--batch-size 8
--num-workers 4
--image-size 224
```

Extra LeRobot args can be forwarded after `--`, for example:

```bash
python scripts/train_smolvla_no_rotvec/train_ur3e_smolvla_no_rotvec.py \
  --steps 1000 \
  -- \
  --policy.optimizer_lr=5e-5
```
