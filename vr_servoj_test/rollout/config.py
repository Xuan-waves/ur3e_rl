from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServoJSmolVLARolloutConfig:
    policy_path: Path = REPO_ROOT / "outputs/train/ur3e_smolvla_0604/checkpoints/020000"
    task: str = "Pick up the yellow toy duck and place it into the grey bowl."

    front_image_topic: str = "/camera/d455/color/image_raw"
    wrist_image_topic: str = "/camera/d405/color/image_raw"
    robot_state_topic: str = "/ur3e_vr/robot_state"
    ik_target_topic: str = "/ur3e_vr/ik_target"
    joint_target_topic: str = "/ur3e_vr/joint_target"
    vr_command_topic: str = "/ur3e_vr/vr_command"

    fps: float = 30.0
    command_hz: float = 100.0
    action_step_hz: float = 15.0
    execution_horizon: int = 50
    prefetch_actions: int = 40
    replace_queue_on_infer: bool = True
    replan_every_step: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "linear"
    rtc_latency_window: int = 50
    rtc_idle_sleep_s: float = 0.002
    rtc_queue_refill_threshold: int = -1
    rtc_debug: bool = False
    sync_reference: str = "front"  # "front", "wrist", or "timer".
    max_dt_front_image: float = 0.04
    max_dt_wrist_image: float = 0.04
    max_dt_state: float = 0.02
    buffer_maxlen: int = 600

    device: str = "cuda"
    amp: bool = True
    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"
    offline: bool = True
    rl_mark: float = 0.0
    gripper_max: float = 0.93
    max_action_age_s: float = 10.0
    action_q_filter_alpha: float = 0.35
    action_gripper_filter_alpha: float = 0.50
    max_action_joint_step: float = 0.06
    log_hz: float = 2.0

    preview: bool = True
    preview_hz: float = 15.0

    return_home_on_start: bool = True
    start_home_delay_s: float = 0.5
    start_home_pulse_s: float = 0.25
    start_home_settle_s: float = 2.0
    start_open_gripper_s: float = 0.8
    start_open_gripper_value: float = 0.0


CFG = ServoJSmolVLARolloutConfig()
