from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.collect_data.config import CollectConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NoRotvecSmolVLARTCRolloutConfig:
    policy_path: Path = REPO_ROOT / "outputs" / "train" / "ur3e_smolvla_060520" / "checkpoints" / "020000"
    task: str = CollectConfig.task

    front_image_topic: str = CollectConfig.front_image_topic
    wrist_image_topic: str = CollectConfig.wrist_image_topic
    robot_state_topic: str = CollectConfig.robot_state_topic
    ik_target_topic: str = CollectConfig.ik_target_topic
    vr_command_topic: str = CollectConfig.vr_command_topic

    fps: float = CollectConfig.fps
    command_hz: float = CollectConfig.fps
    action_step_hz: float = 15.0

    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "linear"
    rtc_latency_window: int = 50
    rtc_idle_sleep_s: float = 0.002
    rtc_queue_refill_threshold: int = -1
    rtc_debug: bool = False

    sync_reference: str = CollectConfig.reference_camera
    max_dt_front_image: float = CollectConfig.max_dt_front_image
    max_dt_wrist_image: float = CollectConfig.max_dt_wrist_image
    max_dt_state: float = CollectConfig.max_dt_state
    buffer_maxlen: int = CollectConfig.buffer_maxlen

    device: str = "cuda"
    gripper_max: float = CollectConfig.gripper_max
    rl_mark: float = 0.0

    # None/auto means infer from the checkpoint training dataset metadata.
    action_position_mode: str = "auto"
    action_pose_filter_alpha: float = 0.35
    action_gripper_filter_alpha: float = 0.50
    max_action_pos_step: float = 0.035
    min_action_z: float = 0.08

    return_home_on_start: bool = True
    start_home_delay_s: float = 1.0
    start_home_pulse_s: float = 0.6
    start_home_settle_s: float = 1.5
    start_open_gripper_s: float = 0.8
    start_open_gripper_value: float = 0.0

    preview: bool = True
    preview_hz: float = 10.0
    log_hz: float = 2.0
    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"
    offline: bool = True


CFG = NoRotvecSmolVLARTCRolloutConfig()
