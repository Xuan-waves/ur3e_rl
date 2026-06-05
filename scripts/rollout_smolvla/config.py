from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.collect_data.config import CollectConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SmolVLARolloutConfig:
    policy_path: Path | None = None
    output_root: Path = REPO_ROOT / "outputs" / "train"
    task: str = CollectConfig.task

    front_image_topic: str = CollectConfig.front_image_topic
    wrist_image_topic: str = CollectConfig.wrist_image_topic
    robot_state_topic: str = CollectConfig.robot_state_topic
    ik_target_topic: str = CollectConfig.ik_target_topic
    vr_command_topic: str = CollectConfig.vr_command_topic

    fps: float = CollectConfig.fps
    command_hz: float = CollectConfig.fps
    # command_hz: float = 100.0

    inference_mode: str = "async"
    replan_every_step: bool = False
    execution_horizon: int = 10
    sync_inference_hz: float = 30.0
    return_home_on_start: bool = True
    start_home_delay_s: float = 1.0
    start_home_pulse_s: float = 0.6
    start_home_settle_s: float = 1.5
    start_open_gripper_s: float = 0.8
    start_open_gripper_value: float = 0.0
    preview: bool = True
    preview_hz: float = 10.0
    max_dt_front_image: float = CollectConfig.max_dt_front_image
    max_dt_wrist_image: float = CollectConfig.max_dt_wrist_image
    max_dt_state: float = CollectConfig.max_dt_state
    buffer_maxlen: int = CollectConfig.buffer_maxlen

    device: str = "cuda"
    gripper_max: float = CollectConfig.gripper_max
    max_action_age_s: float = 1.00
    max_position_step_m: float = 0.040
    # None means infer from the checkpoint's training dataset metadata/stats.
    action_position_mode: str | None = None
    action_orientation_source: str | None = None
    dry_run_log_hz: float = 2.0

    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"


CFG = SmolVLARolloutConfig()
