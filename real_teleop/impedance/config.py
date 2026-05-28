from __future__ import annotations

from dataclasses import dataclass, field


Vector3 = tuple[float, float, float]
Vector6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ImpedanceProfile:
    """Cartesian impedance gains and safety limits for one tuning profile."""

    kp: Vector3
    kd: Vector3
    max_force: Vector3
    max_position_error: float
    virtual_mass: Vector3 = (0.0, 0.0, 0.0)
    accel_filter_alpha: float = 0.20
    max_accel: Vector3 = (4.0, 4.0, 4.0)
    force_bias: Vector3 = (0.0, 0.0, 0.0)
    force_deadband: Vector3 = (0.0, 0.0, 0.0)
    force_mode_limits: Vector6 = (0.05, 0.05, 0.05, 0.15, 0.15, 0.15)
    force_mode_damping: float = 0.12
    force_mode_gain_scaling: float = 0.65

    enable_orientation: bool = False
    rot_kp: Vector3 = (0.0, 0.0, 0.0)
    rot_kd: Vector3 = (0.0, 0.0, 0.0)
    rot_virtual_mass: Vector3 = (0.0, 0.0, 0.0)
    torque_bias: Vector3 = (0.0, 0.0, 0.0)
    torque_deadband: Vector3 = (0.0, 0.0, 0.0)
    max_torque: Vector3 = (3.0, 3.0, 3.0)
    max_rot_error: float = 0.8


@dataclass(frozen=True)
class ImpedanceRuntimeConfig:
    """Robot-side runtime settings shared by tests and future teleop nodes."""

    robot_ip: str = "192.168.5.1"
    control_hz: float = 200.0
    ramp_duration_s: float = 0.3
    state_source: str = "rtde"
    max_fk_rtde_delta_m: float = 0.05

    move_home_first: bool = True
    zero_ft_sensor: bool = True
    payload_mass_kg: float = 0.0
    payload_cog_m: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ImpedanceTestConfig:
    """Standalone impedance test defaults."""

    robot_ip: str = "192.168.5.1"
    default_mode: str = "passive"
    duration_s: float = 20.0
    control_hz: float = 200.0
    ramp_duration_s: float = 0.3
    log_interval_s: float = 0.25

    move_home_first: bool = True
    zero_ft_sensor: bool = True
    payload_mass_kg: float = 0.0
    payload_cog_m: Vector3 = (0.0, 0.0, 0.0)

    simulated_mass_kg: float = 3.0
    start_offset_m: Vector3 = (0.0, 0.0, 0.0)

    profiles: dict[str, ImpedanceProfile] = field(
        default_factory=lambda: {
            "teleop": ImpedanceProfile(
                kp=(1000.0, 1000.0, 1100.0),
                virtual_mass=(0.0, 0.0, 0.0),
                accel_filter_alpha=0.25,
                max_accel=(8.0, 8.0, 8.0),
                kd=(18.0, 18.0, 20.0),
                force_deadband=(0.5, 0.5, 0.6),
                max_force=(200.0, 200.0, 200.0),
                max_position_error=0.60,
                force_mode_limits=(1.20, 1.20, 1.20, 1.20, 1.20, 1.20),
                force_mode_damping=0.06,
                force_mode_gain_scaling=0.85,
                enable_orientation=False,
                rot_kp=(220.0, 220.0, 220.0),
                rot_kd=(8.0, 8.0, 8.0),
                rot_virtual_mass=(0.0, 0.0, 0.0),
                torque_deadband=(0.08, 0.08, 0.08),
                max_torque=(28.0, 28.0, 28.0),
                max_rot_error=3.14,
            ),
            "passive": ImpedanceProfile(
                kp=(500.0, 500.0, 500.0),
                virtual_mass=(0.0, 0.0, 0.0),
                accel_filter_alpha=0.2,
                max_accel=(4.0, 4.0, 4.0),
                kd=(1.0, 1.0, 1.0),
                max_force=(28.0, 28.0, 32.0),
                max_position_error=0.60,
                force_mode_limits=(1.20, 1.20, 1.20, 0.15, 0.15, 0.15),
                force_mode_damping=0.01,
                force_mode_gain_scaling=0.98,
                enable_orientation=True,
                rot_kp=(30.0, 30.0, 30.0),
                rot_kd=(0.2, 0.2, 0.2),
                rot_virtual_mass=(0.0, 0.0, 0.0),
                max_torque=(10.0, 10.0, 10.0),
                max_rot_error=0.6,
            ),
            "spring": ImpedanceProfile(
                kp=(350.0, 350.0, 450.0),
                kd=(28.0, 28.0, 34.0),
                max_force=(20.0, 20.0, 20.0),
                max_position_error=0.08,
                force_mode_damping=0.12,
            ),
            "zero-force": ImpedanceProfile(
                kp=(0.0, 0.0, 0.0),
                kd=(3.0, 3.0, 3.0),
                max_force=(8.0, 8.0, 8.0),
                max_position_error=0.20,
                force_mode_damping=0.20,
            ),
        }
    )


DEFAULT_IMPEDANCE_TEST_CONFIG = ImpedanceTestConfig()
