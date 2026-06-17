from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RLTInterventionCollectConfig:
    policy_path: Path = REPO_ROOT / "outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model"
    stage1_checkpoint: Path = (
        REPO_ROOT / "outputs/rlt_stage1/rlt_stage1_ur3e_smolvla_0614_030000_20260616_163323/best.pt"
    )
    gate_checkpoint: Path = REPO_ROOT / "outputs/rlt_gate/rlt_gate_20260615_142442/best.pt"
    output_dir: Path = REPO_ROOT / "outputs/rlt_interventions"
    stage2_output_dir: Path = REPO_ROOT / "outputs/rlt_stage2"

    task: str = "Pick up the Ethernet plug and insert it into the port on the black module."
    device: str = "cuda"
    amp: bool = True
    offline: bool = True
    hf_home: Path = REPO_ROOT / ".cache/huggingface"

    front_image_topic: str = "/camera/d455/color/image_raw"
    wrist_image_topic: str = "/camera/d405/color/image_raw"
    robot_state_topic: str = "/ur3e_vr/robot_state"
    ik_target_topic: str = "/ur3e_vr/ik_target"
    vr_command_topic: str = "/ur3e_vr/vr_command"
    vr_raw_topic: str = "/ur3e_vr/vr_command_raw"

    fps: float = 30.0
    command_hz: float = 30.0
    action_step_hz: float = 30.0
    n_obs_steps: int = 1
    execution_horizon: int = 10
    sync_reference: str = "front"
    buffer_maxlen: int = 240
    max_dt_front_image: float = 0.08
    max_dt_wrist_image: float = 0.08
    max_dt_state: float = 0.08

    action_position_mode: str = "absolute"
    gripper_max: float = 0.93
    min_action_z: float = 0.01
    action_pose_filter_alpha: float = 0.75
    action_gripper_filter_alpha: float = 0.55
    max_action_pos_step: float = 0.06
    max_action_age_s: float = 1.0
    prefetch_actions: int = 0
    replace_queue_on_infer: bool = True

    gate_positive_threshold: float | None = None
    gate_negative_threshold: float | None = None
    gate_hold_frames: int = 3
    gate_infer_hz: float = 15.0

    vr_override_stale_s: float = 0.25
    vr_override_resume_delay_s: float = 0.25
    vr_override_gripper_gain: float = 1.0
    vr_override_anytime: bool = False
    manual_home_pulse_s: float = 1.5
    home_gripper_value: float = 0.0

    home_pulse_s: float = 1.2
    home_after_gate_exit: bool = True
    block_model_during_home: bool = True
    reset_impedance_on_trial_start: bool = True
    reset_impedance_during_home: bool = True
    return_home_on_start: bool = True
    start_home_delay_s: float = 0.5
    start_home_pulse_s: float = 2.0
    start_home_settle_s: float = 0.7
    start_open_gripper_s: float = 1.0
    start_open_gripper_value: float = 0.0

    preview: bool = True
    preview_hz: float = 20.0
    log_hz: float = 2.0
    save_button_cooldown_s: float = 0.8

    # HIL-SERL style online Stage2. The actor learns a small residual on top of
    # the frozen VLA action. Intervention samples are also stored in a separate
    # buffer and oversampled during updates, mirroring HIL-SERL's RLPD setup.
    rlt_enable_actor: bool = True
    rlt_warmup_steps: int = 64
    rlt_min_actor_updates: int = 1
    rlt_startup_updates: int = 500
    rlt_startup_log_interval: int = 50
    rlt_startup_empty_cache_interval: int = 0
    rlt_replay_capacity: int = 200000
    rlt_batch_size: int = 128
    rlt_replay_demo_ratio: float = 0.5
    rlt_updates_per_step: int = 1
    rlt_policy_delay: int = 2
    rlt_actor_lr: float = 3e-4
    rlt_critic_lr: float = 3e-4
    rlt_gamma: float = 0.98
    rlt_tau: float = 0.005
    rlt_bc_weight: float = 0.2
    rlt_target_noise_xyz: float = 0.002
    rlt_target_noise_clip_xyz: float = 0.006
    rlt_actor_hidden_dim: int = 256
    rlt_critic_hidden_dim: int = 256
    rlt_train_action_dim: int = 3
    rlt_action_chunk_steps: int = 10
    rlt_action_delta_scale_xyz: float = 0.004
    rlt_checkpoint: Path | None = None
    rlt_buffer_dir: Path | None = REPO_ROOT / "outputs/rlt_stage2/hil_serl_stage2_20260616_184857/buffers"
    rlt_save_every_episodes: int = 1
    rlt_snapshot_buffers: bool = True


CFG = RLTInterventionCollectConfig()
