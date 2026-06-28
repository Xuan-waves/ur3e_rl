# UR3e RL / VR Teleoperation / RLT

This repository contains the UR3e real-robot teleoperation, data collection, SmolVLA training/rollout, and RLT-style refinement tools used for Ethernet-plug insertion experiments.

You can view the demo through 
## Demo
[![Bilibili Demo](https://img.shields.io/badge/Bilibili-Watch%20Demo-00A1D6?logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1uVLX6uE3E/)


or 小红书：
https://www.xiaohongshu.com/explore/6a3264fb0000000016026af9?xsec_token=ABlF0bPI8QfGO9RDdHlkzPIfpA1JW6R0chscDVKbozabk=&xsec_source=pc_user

The project is built around:

- UR3e end-effector teleoperation with XRoboToolkit VR input.
- Impedance-mode robot execution for safer contact-rich manipulation.
- LeRobot-format data collection from two RealSense RGB cameras.
- SmolVLA behavior cloning training and rollout.
- RLT / HIL-SERL-style online refinement for the difficult insertion phase.

## External XRoboToolkit Files

First clone the XRoboToolkit Python teleoperation sample:

```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
```

Then copy these three folders from the cloned repository into this repository's `Xrobot_tool/` directory:

```text
xrobotoolkit_teleop
XRoboToolkit-PC-Service
XRoboToolkit-PC-Service-Pybind
```

Expected local layout:

```text
Xrobot_tool/
  xrobotoolkit_teleop/
  XRoboToolkit-PC-Service/
  XRoboToolkit-PC-Service-Pybind/
```

These files are external XR-Robotics components and are not developed in this repository.

## Environment Setup

Follow the environment guidance from `XRoboToolkit-Teleop-Sample-Python`, but this project is tested around Python 3.10. Conda is recommended because ROS2 Humble, RealSense, PyTorch, and LeRobot dependencies are easier to keep isolated.

Example:

```bash
conda create -n ur3e_rlt python=3.10
conda activate ur3e_rlt
```

Install PyTorch for your CUDA version. For example, choose the official PyTorch command that matches your GPU driver/CUDA runtime.

Then install the project-level robotics and learning dependencies:

```bash
pip install mink
pip install "lerobot==0.4.4"
```

Depending on your local machine, you may also need ROS2 Humble, RealSense ROS, `pyrealsense2`, OpenCV with GUI support, and the UR RTDE dependencies used by the hardware scripts.

Typical shell setup before running robot/camera tools:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
source /home/arts/anaconda3/etc/profile.d/conda.sh
conda activate ur3e_rlt
```

## Main Runtime Commands

Start RealSense cameras:

```bash
scripts/collect_data/run_realsense_cameras.sh
```

Start the UR3e impedance robot node without VR twin:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node robot \
  --robot-ip 192.168.5.1 \
  --control-mode impedance \
  --impedance-profile teleop \
  --no-twin
```

For VR teleoperation or HIL/RLT intervention, start the VR raw-command node:

```bash
python scripts/hardware/ur3e_vr_servoj_ros2.py \
  --node vr \
  --vr-output-topic /ur3e_vr/vr_command_raw
```

Do not run multiple nodes that publish robot targets to `/ur3e_vr/ik_target` or `/ur3e_vr/vr_command` at the same time.

## Code Structure

```text
real_teleop/
  Shared teleoperation, safety, kinematics, messages, and impedance-control utilities.

real_teleop/impedance/
  Basic impedance-controller API and config used by hardware scripts.

scripts/hardware/
  UR3e hardware entry points, VR input node, IK/robot node, and impedance tests.

scripts/collect_data/
  Main impedance-mode LeRobot data collection tools.

scripts/train_smolvla/
  SmolVLA training utilities for full pose/state variants.

scripts/train_smolvla_no_rotvec/
  SmolVLA training utilities for the no-rotvec representation:
  state/action use xyz + gripper, with fixed end-effector orientation.

scripts/rollout_smolvla/
  Original SmolVLA rollout tools.

scripts/rollout_smolvla_no_rotvec/
  No-rotvec SmolVLA rollout tools, including sync, RTC, and ready-gate closed-loop rollout.

scripts/rlt_gate/
  Manual RL-gate annotation, gate classifier training, evaluation, and live monitoring.

scripts/rlt_token/
  RLT Stage1 token extraction/training/evaluation.

scripts/rlt_train/
  RLT Stage2 / HIL-SERL-style online refinement, replay buffers, rollout, and README workflow.

scripts/ready_gate/
  Ready-gate data collection/training/live evaluation. The ready gate decides whether the scene is ready for the next trial.

vr_servoj_test/
  Isolated ServoJ teleoperation/data-collection/train/rollout testbed.

mujoco_env/
  MuJoCo assets and hardware helper code.

datasets/
  Local collected datasets. These are usually large and machine-specific.

outputs/
  Local model checkpoints, training runs, RLT buffers, and evaluation outputs.

Xrobot_tool/
  Local copy of XRoboToolkit runtime folders.
```

## Data And Learning Flow

The current Ethernet insertion workflow is roughly:

1. Use VR + impedance control to collect LeRobot-format demonstrations from two RealSense cameras.
2. Train a no-rotvec SmolVLA policy on images, xyz TCP position, and gripper state/action.
3. Validate plain VLA rollout with sync/RTC execution.
4. Train an RL gate that marks the difficult insertion-refinement phase.
5. Train Stage1 RLT token encoder on the frozen VLA.
6. Collect HIL intervention buffers in the RL-gated phase.
7. Train Stage2 RLT actor/critic to add small xyz residuals to the frozen VLA action chunk.
8. Use ready-gate closed-loop rollout to repeat trials only when the scene is ready.

The most detailed guide for the RLT portion is:

```text
scripts/rlt_train/README.md
```

## Important Notes

- The robot can move unexpectedly if multiple controllers publish targets at once. Keep one active controller.
- Use impedance mode for contact-rich insertion experiments.
- Keep `reset_impedance_on_trial_start` and `reset_impedance_during_home` enabled unless debugging drift.
- `ready_gate` is not the same as `RL_gate`.
  - `ready_gate`: whether the next trial can begin.
  - `RL_gate`: whether RLT should refine the VLA action.
- RLT Stage2 currently refines only xyz TCP action chunks; gripper remains from VLA or VR override.

## References

- XRoboToolkit / XR-Robotics: https://xr-robotics.github.io/
- XRoboToolkit Python teleoperation sample: https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python
- SmolVLA / LeRobot project: https://github.com/huggingface/lerobot
- RLT research page: https://www.pi.website/research/rlt

## License

This repository is intended to be released under the MIT License.

External components copied into `Xrobot_tool/` keep their original upstream licenses. Check the upstream XR-Robotics repository and bundled files before redistributing those components.
