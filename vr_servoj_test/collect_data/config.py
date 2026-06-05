from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VrServoJCollectConfig:
    fps: float = 30.0
    max_dt_image: float = 0.04
    max_dt_front_image: float = 0.04
    max_dt_wrist_image: float = 0.04
    max_dt_state: float = 0.02
    max_dt_action: float = 0.02
    buffer_maxlen: int = 600
    sync_reference: str = "front"  # "front", "wrist", or "timer".

    front_image_topic: str = "/camera/d455/color/image_raw"
    wrist_image_topic: str = "/camera/d405/color/image_raw"
    robot_state_topic: str = "/ur3e_vr/robot_state"
    vr_command_topic: str = "/ur3e_vr/vr_command"
    ik_target_topic: str = "/ur3e_vr/ik_target"
    joint_target_topic: str = "/ur3e_vr/joint_target"
    commanded_joint_target_topic: str = "/ur3e_vr/commanded_joint_target"

    dataset_root: Path = REPO_ROOT / "datasets"
    dataset_name: str = "ur3e_lerobot_vr_servoj"
    repo_id: str = "local/ur3e_vr_servoj"
    robot_type: str = "ur3e_robotiq"
    task: str = "Pick up the yellow toy duck and place it into the grey bowl."
    max_episodes: int = 0

    state_mode: str = "eepose"  # "eepose" or "jointspace".
    action_mode: str = "jointspace"  # "eepose" or "jointspace".
    ee_action_position_mode: str = "relative"  # "relative" stores target_pos - state_pos; "absolute" stores target_pos.

    use_videos: bool = True
    image_writer_threads: int = 4
    video_codec: str = "h264"
    resize: tuple[int, int] | None = None
    gripper_max: float = 0.93
    status_panel: bool = True
    status_hz: float = 4.0
    stop_left_grip_threshold: float = 0.95
