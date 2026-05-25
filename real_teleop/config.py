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
    vr_hz: float = 100.0
    robot_state_hz: float = 100.0
    actual_read_hz: float = 50.0
    gripper_hz: float = 25.0
    twin_hz: float = 60.0

    enable_threshold: float = 0.85
    scale: float = 1.2
    dead_zone_pos: float = 0.0005
    dead_zone_rot: float = 0.002
    ctrl_filter_alpha: float = 0.65
    target_filter_alpha: float = 0.55
    joint_target_alpha: float = 0.35
    fixed_ee_orientation: bool = True
    stale_command_s: float = 0.50
    stale_target_s: float = 0.30

    gripper_close_mj: float = 0.6
    home_move_duration_s: float = 1.5
    max_joint_speed: float = 2.0
    max_joint_step: float = 0.010
    max_target_jump: float = 0.05
    workspace_min: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.05, 0.76], dtype=float)
    )
    workspace_max: np.ndarray = field(
        default_factory=lambda: np.array([1.35, 1.15, 1.35], dtype=float)
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

    ik_position_cost: float = 1.0
    ik_orientation_cost: float = 0.5
    ik_posture_cost: float = 0.02
    ik_damping_cost: float = 0.01
    ik_lm_damping: float = 1.0
    ik_solve_damping: float = 0.001
    ik_iters: int = 4

    headset_to_world: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )
    headset_orientation_to_world: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )


TOPIC_VR_COMMAND = "/ur3e_vr/vr_command"
TOPIC_ROBOT_STATE = "/ur3e_vr/robot_state"
TOPIC_JOINT_TARGET = "/ur3e_vr/joint_target"
TOPIC_IK_TARGET = "/ur3e_vr/ik_target"
TOPIC_ROBOT_DEBUG = "/ur3e_vr/robot_debug"
