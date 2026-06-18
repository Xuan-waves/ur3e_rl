# RLT Gate Manual Annotation

This tool creates frame-level `rlt_phase` labels for training an RL/RLT gate model.
It does not modify the original LeRobot `data/*.parquet` files.

For the full UR3e RLT workflow and rollout/training commands, start from
`scripts/rlt_train/README.md`.

## Input

Default input is the per-episode video export:

```bash
datasets/lerobot-export(2)
```

Expected layout:

```text
data/chunk-000/file-000.parquet
meta/episodes/chunk-000/file-000.parquet
videos/observation.images.cam_front/chunk-000/file-000.mp4
videos/observation.images.cam_wrist/chunk-000/file-000.mp4
...
```

## Check Videos

Before annotation, verify that every per-episode mp4 has the same frame count as
the LeRobot episode length:

```bash
python scripts/rlt_gate/annotate_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --check-videos
```

## Annotate

```bash
python scripts/rlt_gate/annotate_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --camera both
```

Controls:

```text
R       toggle RLT phase marking from current frame to the next R
Z       undo active start, or remove the last completed interval
Space   play / pause
A / D   step one frame backward / forward
J / L   jump one second backward / forward
B       replay the latest marked interval in this episode
E       replay current episode from the beginning
N / P   next / previous episode
S       save annotations
Q       save and quit
```

## Output

The tool writes:

```text
meta/rlt_gate_annotations.json
meta/rlt_gate_labels.parquet
```

`rlt_gate_annotations.json` stores human-editable intervals:

```json
{
  "episodes": {
    "12": {
      "length": 300,
      "intervals": [{"start": 180, "end": 260}]
    }
  }
}
```

`rlt_gate_labels.parquet` expands intervals to one row per LeRobot frame:

```text
episode_index
frame_index
index
timestamp
rlt_phase
```

`rlt_phase=1` means this frame should be considered part of the future
RL/refinement phase.

## Train Gate

The training script reads `meta/rlt_gate_labels.parquet` and the exported
episode videos. The split is episode-level, so frames from the same episode do
not appear in both train and validation sets.

Fast deployment baseline:

```bash
python scripts/rlt_gate/train_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --model tiny \
  --camera front
```

More robust baseline:

```bash
python scripts/rlt_gate/train_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --model resnet18 \
  --camera front \
  --image-size 160
```

Training shows a per-epoch progress bar by default. Use `--no-progress` if you
want plain logs:

```bash
python scripts/rlt_gate/train_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --model tiny \
  --camera front \
  --no-progress
```

Training reads labels from parquet but images from mp4 files. Random seeking
inside H264 mp4 is slow, so the default is:

```text
--cache-images ram
```

This decodes the needed frames once, episode by episode, resizes them, and keeps
them as uint8 images in RAM. The first startup is slower, but later epochs avoid
random mp4 seeking. The script prints an estimated RAM cost before caching. To
use the old direct-video path:

```bash
python scripts/rlt_gate/train_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --model resnet18 \
  --camera front \
  --cache-images none
```

For a faster first pass on RTX 3070:

```bash
python scripts/rlt_gate/train_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --model resnet18 \
  --camera front \
  --image-size 128 \
  --batch-size 96 \
  --num-workers 4 \
  --device cuda \
  --sample-stride 2
```

`--camera front` is the recommended first pass because the task phase is usually
visible from the global view. Use `--camera wrist` if the close-up view contains
the decisive insertion/contact cue. Use `--camera both` only if a single camera
is clearly ambiguous; it doubles the input channels and can be less convenient
for deployment.

For runtime use, avoid switching the RL/RLT phase on one noisy frame. A simple
hysteresis works well: enter when probability is above `positive_threshold`
for several consecutive frames, and exit when it is below `negative_threshold`
for several consecutive frames.

Model choice:

```text
tiny      fastest, good default for online gating
resnet18  stronger visual baseline, useful to check whether tiny is underfit
```

If ResNet18 only improves validation F1 slightly, deploy `tiny`. If ResNet18 is
much better, deploy ResNet18 first or use it as a teacher for a smaller model.

## Evaluate And Preview

Compare checkpoints on the labeled videos:

```bash
python scripts/rlt_gate/eval_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --checkpoint outputs/rlt_gate/rlt_gate_20260610_162104/best.pt \
  --checkpoint outputs/rlt_gate/rlt_gate_20260610_164453/best.pt \
  --device cuda \
  --batch-size 256 \
  --hold-frames 3 \
  --save-csv outputs/rlt_gate/compare_resnet_tiny.csv
```

Preview one episode with GT, raw probability, and hysteresis phase overlaid:

```bash
python scripts/rlt_gate/eval_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --checkpoint outputs/rlt_gate/rlt_gate_20260610_164453/best.pt \
  --episode 12 \
  --show
```

For checkpoints trained with `--camera both`, the preview displays `front` and
`wrist` side by side while using both views for prediction.

When `--episode` is set, the script evaluates only that episode by default so
the preview starts quickly. Add `--full-eval` if you also want full-dataset
metrics before previewing.

Save an overlay mp4 instead of opening a window:

```bash
python scripts/rlt_gate/eval_rlt_gate.py \
  --dataset "datasets/lerobot-export(2)" \
  --checkpoint outputs/rlt_gate/rlt_gate_20260610_164453/best.pt \
  --episode 12 \
  --output-video outputs/rlt_gate/episode_012_gate_preview.mp4
```

Saved preview videos use H264/yuv420p when PyAV is available, which is more
compatible with VSCode and common video players than OpenCV's default `mp4v`.

## Live VR Impedance Test

Use the live monitor while controlling the robot with the VR impedance stack:

```bash
scripts/rlt_gate/run_vr_impedance_gate_test.sh \
  --robot-ip 192.168.5.1 \
  --checkpoint outputs/rlt_gate/rlt_gate_20260610_172234/best.pt \
  --max-infer-hz 15
```

This opens:

```text
RealSense ROS Cameras
RLT Gate Live Monitor
UR3e VR Input / IK / Robot tabs in impedance mode
```

The monitor subscribes to:

```text
/camera/d455/color/image_raw
/camera/d405/color/image_raw
```

It shows front and wrist views side by side, overlays raw `prob`, hysteresis
phase, inference Hz, draw Hz, inference latency, and camera frame ages. Press
`q` in the monitor window to close it.

Manual equivalent:

```bash
scripts/collect_data/run_realsense_cameras.sh

scripts/hardware/run_ur3e_vr_tabs.sh \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop \
  --no-twin \
  --conda-env ur3e_rlt

python scripts/rlt_gate/live_rlt_gate_monitor.py \
  --checkpoint outputs/rlt_gate/rlt_gate_20260610_172234/best.pt \
  --max-infer-hz 15
```
