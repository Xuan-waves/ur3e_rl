# UR3e Ready-Gate Data Collection

This collector records camera frames for a small binary classifier:

- `ready_gate=1`: the Ethernet plug is in the start holder/fixture and the next rollout can begin.
- `ready_gate=0`: the scene is not ready, for example the plug is missing, already inserted, held by the gripper, dropped, or badly posed.

The script starts with a return-home pulse, relays raw VR commands to the robot, and records synchronized front/wrist camera images at 30 Hz.

## Run

Start cameras as usual, then start the UR3e stack with the VR node publishing to the raw topic:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node all \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop \
  --no-twin \
  --vr-output-topic /ur3e_vr/vr_command_raw
```

In another terminal:

```bash
python scripts/ready_gate/collect_ready_gate.py
```

Do not run another node that publishes directly to `/ur3e_vr/vr_command` at the same time.

## Controls

- `X`: start recording one episode using the current next-label.
- `B`: stop and save the current episode.
- `Y`: toggle the label for the next episode.
- `A`: return the robot to home.

The current recording keeps the label it had when `X` was pressed. Pressing `Y` while recording only changes the next episode's label.

## Output

Data is saved under:

```text
datasets/ready_gate/ready_gate_YYYYMMDD_HHMMSS/
```

Each episode contains:

```text
episode_000000/
  front/frame_000000.jpg
  wrist/frame_000000.jpg
  labels.jsonl
  metadata.json
metadata.json
```

Each `labels.jsonl` row stores `ready_gate`, frame paths, camera timestamps, and camera ages.
