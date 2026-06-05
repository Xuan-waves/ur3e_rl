from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CollectConfig:
    fps: float = 30.0
    max_dt_image: float = 0.04
    max_dt_front_image: float = 0.04
    max_dt_wrist_image: float = 0.04
    max_dt_state: float = 0.02
    max_dt_action: float = 0.02
    allow_stale_front: bool = False
    front_stale_warn_after: float = 2.0
    buffer_maxlen: int = 2000

    front_image_topic: str = "/camera/d455/color/image_raw"
    wrist_image_topic: str = "/camera/d405/color/image_raw"
    robot_state_topic: str = "/ur3e_vr/robot_state"
    vr_command_topic: str = "/ur3e_vr/vr_command"
    ik_target_topic: str = "/ur3e_vr/ik_target"
    joint_target_topic: str = "/ur3e_vr/joint_target"
    camera_source: str = "ros"  # "ros" or "realsense"
    front_camera_serial: str = "151422253456"
    wrist_camera_serial: str = "218722270648"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    reference_camera: str = "front"  # "front", "wrist", or "timer"
    launch_realsense_ros: bool = False
    cleanup_stale_realsense_ros: bool = False
    camera_startup_wait: float = 3.0

    dataset_root: Path = REPO_ROOT / "datasets"
    dataset_name: str = "ur3e_lerobot_vr_impedance"
    repo_id: str = "local/ur3e_vr_impedance"
    robot_type: str = "ur3e_robotiq"
    # task: str = "Insert the Ethernet connector into the matching slot."
    task: str = "Pick up the yellow toy duck and place it into the grey bowl."

    max_episodes: int = 0  # 0 means unlimited.
    action_position_mode: str = "relative"  # "relative" stores target_pos - state_pos; "absolute" stores target_pos.
    action_orientation_source: str = "state"  # "state" aligns action[3:6] with state[3:6]; "ik_target" stores target orientation.

    use_videos: bool = True
    image_writer_threads: int = 4
    video_codec: str = "h264"
    resize: tuple[int, int] | None = None
    preview: bool = True
    preview_window: bool = False
    preview_hz: float = 30.0
    preview_width: int = 960
    preview_topic: str = "/ur3e_vr/collection_preview"
    gripper_max: float = 0.93
    status_panel: bool = True
    status_hz: float = 4.0

    x_key_name: str = "X"
    y_key_name: str = "Y"
    b_key_name: str = "B"
    a_key_name: str = "A"
    cancel_left_trigger_threshold: float = 0.95
    stop_left_grip_threshold: float = 0.95
