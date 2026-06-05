from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TeleopConfig:
    robot_ip: str = "192.168.5.1"
    xml_path: str = str(REPO_ROOT / "mujoco_env/assets/scenes/scene.xml")
    ee_frame: str = "gripper_tcp_site"
    control_hz: float = 200.0
    servoj_control_hz: float = 400.0
    vr_hz: float = 100.0
    collection_input_hz: float = 30.0
    robot_state_hz: float = 100.0
    actual_read_hz: float = 50.0
    gripper_hz: float = 25.0
    twin_hz: float = 60.0
    robot_control_mode: str = "impedance"

    enable_threshold: float = 0.85
    enable_release_threshold: float = 0.65
    gripper_command_max: float = 0.93
    scale: float = 1.2
    vr_control_position_sign: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float)
    )
    servoj_control_position_sign: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, -1.0, 1.0], dtype=float)
    )
    servoj_control_rotation_sign: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, -1.0, 1.0], dtype=float)
    )
    servoj_ctrl_filter_alpha: float = 0.30
    servoj_target_filter_alpha: float = 0.60
    servoj_joint_target_alpha: float = 0.25
    servoj_dead_zone_pos: float = 0.003
    servoj_dead_zone_rot: float = 0.008
    servoj_target_pos_hold_epsilon: float = 0.0005
    servoj_target_rot_hold_epsilon: float = 0.002
    servoj_ik_joint_deadband: float = 0.0003
    servoj_max_joint_speed: float = 5.0
    servoj_max_joint_step: float = 0.008
    servoj_max_target_jump: float = 0.12
    servoj_stale_target_s: float = 0.50
    servoj_lookahead_time: float = 0.10
    servoj_gain: float = 100.0
    dead_zone_pos: float = 0.0035
    dead_zone_rot: float = 0.010
    ctrl_filter_alpha: float = 0.62
    target_filter_alpha: float = 0.58
    joint_target_alpha: float = 0.30
    impedance_target_alpha: float = 0.68
    impedance_profile: str = "teleop"
    impedance_state_source: str = "rtde"
    impedance_max_fk_rtde_delta_m: float = 0.05
    impedance_ramp_duration_s: float = 0.08
    impedance_target_pos_error_limit: float = 0.5
    impedance_target_rot_error_limit: float = 0.0
    vr_track_orientation: bool = False
    fixed_ee_orientation: bool = False
    stale_command_s: float = 0.50
    stale_target_s: float = 0.30
    impedance_soft_hold_on_lost: bool = True
    target_pos_hold_epsilon: float = 0.0012
    target_rot_hold_epsilon: float = 0.004
    ik_joint_deadband: float = 0.00015
    vr_controller_pivot_offset_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )

    gripper_close_mj: float = 0.6
    home_move_duration_s: float = 1.5
    max_joint_speed: float = 2.0
    max_joint_step: float = 0.008
    max_target_jump: float = 0.03
    workspace_min: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.05, 0.76], dtype=float)
    )
    workspace_max: np.ndarray = field(
        default_factory=lambda: np.array([1.35, 1.15, 1.35], dtype=float)
    )
    impedance_workspace_min: np.ndarray = field(
        default_factory=lambda: np.array([-0.15, -0.65, 0.01], dtype=float)
    )
    impedance_workspace_max: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.10, 0.45], dtype=float)
    )
    joint_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-2.0 * np.pi, 2.0 * np.pi],
                [-2.0 * np.pi, 2.0 * np.pi],
                [-np.pi, np.pi],
                [-2.0 * np.pi, 2.0 * np.pi],
                [-2.0 * np.pi, 2.0 * np.pi],
                [-2.0 * np.pi, 2.0 * np.pi],
            ],
            dtype=float,
        )
    )
    hardware_home_q: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.pi / 2.0,
                -np.pi / 2.0,
                np.pi / 2.0,
                -np.pi / 2.0,
                -np.pi / 2.0,
                np.pi,
            ],
            dtype=float,
        )
    )

    ik_position_cost: float = 0.8
    ik_orientation_cost: float = 0.5
    
    ik_posture_cost: float = 0.02
    ik_damping_cost: float = 0.01
    ik_lm_damping: float = 1.0
    ik_solve_damping: float = 0.001
    ik_iters: int = 1

    headset_to_world: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )
    headset_orientation_to_world: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )


TOPIC_VR_COMMAND = "/ur3e_vr/vr_command"
TOPIC_ROBOT_STATE = "/ur3e_vr/robot_state"
TOPIC_JOINT_TARGET = "/ur3e_vr/joint_target"
TOPIC_COMMANDED_JOINT_TARGET = "/ur3e_vr/commanded_joint_target"
TOPIC_IK_TARGET = "/ur3e_vr/ik_target"
TOPIC_ROBOT_DEBUG = "/ur3e_vr/robot_debug"
