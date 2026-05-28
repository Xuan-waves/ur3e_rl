from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import ImpedanceProfile, ImpedanceRuntimeConfig


def _vec(values: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf: {arr}.")
    return arr


@dataclass(slots=True)
class CartesianState:
    position: np.ndarray
    rotation_vector: np.ndarray
    speed: np.ndarray


@dataclass(slots=True)
class ImpedanceCommand:
    wrench: np.ndarray
    position_error: np.ndarray
    rotation_error: np.ndarray
    acceleration: np.ndarray


class CartesianImpedanceCore:
    """Pure Cartesian impedance law with no robot dependency."""

    def __init__(
        self,
        profile: ImpedanceProfile,
        *,
        mode: str = "passive",
        ramp_duration_s: float = 0.3,
    ) -> None:
        self.profile = profile
        self.mode = mode
        self.ramp_duration_s = max(float(ramp_duration_s), 0.0)
        self.target_position = np.zeros(3, dtype=float)
        self.target_rotation = R.identity()
        self._started_at = time.monotonic()
        self._last_time: float | None = None
        self._last_speed: np.ndarray | None = None
        self._acceleration = np.zeros(6, dtype=float)
        self.last_command = ImpedanceCommand(
            wrench=np.zeros(6, dtype=float),
            position_error=np.zeros(3, dtype=float),
            rotation_error=np.zeros(3, dtype=float),
            acceleration=np.zeros(6, dtype=float),
        )

    @property
    def selection_vector(self) -> tuple[int, int, int, int, int, int]:
        return (1, 1, 1, 1, 1, 1) if self.profile.enable_orientation else (1, 1, 1, 0, 0, 0)

    def set_target_pose(self, position: Any, rotation_vector: Any | None = None) -> None:
        self.target_position = _vec(position, 3, "target_position")
        if rotation_vector is not None:
            self.target_rotation = R.from_rotvec(_vec(rotation_vector, 3, "target_rotation_vector"))

    def reset_ramp(self) -> None:
        self._started_at = time.monotonic()

    def compute(self, state: CartesianState) -> ImpedanceCommand:
        pos = _vec(state.position, 3, "state.position")
        rotvec = _vec(state.rotation_vector, 3, "state.rotation_vector")
        speed = _vec(state.speed, 6, "state.speed")
        accel = self._filtered_acceleration(speed)

        position_error = self.target_position - pos
        err_norm = float(np.linalg.norm(position_error))
        if err_norm > self.profile.max_position_error:
            raise RuntimeError(
                f"Position error {err_norm:.4f} m exceeds safety limit "
                f"{self.profile.max_position_error:.4f} m."
            )

        kp = _vec(self.profile.kp, 3, "profile.kp")
        kd = _vec(self.profile.kd, 3, "profile.kd")
        virtual_mass = _vec(self.profile.virtual_mass, 3, "profile.virtual_mass")
        if self.mode == "zero-force":
            force = -kd * speed[:3] - virtual_mass * accel[:3]
        else:
            force = kp * position_error - kd * speed[:3] - virtual_mass * accel[:3]
        force += _vec(self.profile.force_bias, 3, "profile.force_bias")
        force = _deadband(force, _vec(self.profile.force_deadband, 3, "profile.force_deadband"))
        force = np.clip(force, -_vec(self.profile.max_force, 3, "profile.max_force"), _vec(self.profile.max_force, 3, "profile.max_force"))

        rotation_error = np.zeros(3, dtype=float)
        torque = np.zeros(3, dtype=float)
        if self.profile.enable_orientation:
            rotation_error = (self.target_rotation * R.from_rotvec(rotvec).inv()).as_rotvec()
            rot_err_norm = float(np.linalg.norm(rotation_error))
            if rot_err_norm > self.profile.max_rot_error:
                raise RuntimeError(
                    f"Rotation error {rot_err_norm:.4f} rad exceeds safety limit "
                    f"{self.profile.max_rot_error:.4f} rad."
                )
            rot_kp = _vec(self.profile.rot_kp, 3, "profile.rot_kp")
            rot_kd = _vec(self.profile.rot_kd, 3, "profile.rot_kd")
            rot_mass = _vec(self.profile.rot_virtual_mass, 3, "profile.rot_virtual_mass")
            if self.mode == "zero-force":
                torque = -rot_kd * speed[3:] - rot_mass * accel[3:]
            else:
                torque = rot_kp * rotation_error - rot_kd * speed[3:] - rot_mass * accel[3:]
            torque += _vec(self.profile.torque_bias, 3, "profile.torque_bias")
            torque = _deadband(torque, _vec(self.profile.torque_deadband, 3, "profile.torque_deadband"))
            max_torque = _vec(self.profile.max_torque, 3, "profile.max_torque")
            torque = np.clip(torque, -max_torque, max_torque)

        wrench = np.r_[force, torque]
        if self.ramp_duration_s > 0.0:
            elapsed = time.monotonic() - self._started_at
            wrench *= min(1.0, max(0.0, elapsed / self.ramp_duration_s))

        self.last_command = ImpedanceCommand(
            wrench=wrench,
            position_error=position_error,
            rotation_error=rotation_error,
            acceleration=accel.copy(),
        )
        return self.last_command

    def _filtered_acceleration(self, speed: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        if self._last_time is None or self._last_speed is None:
            self._last_time = now
            self._last_speed = speed.copy()
            return self._acceleration.copy()

        dt = max(now - self._last_time, 1e-4)
        raw_accel = (speed - self._last_speed) / dt
        accel_limit = np.r_[_vec(self.profile.max_accel, 3, "profile.max_accel"), _vec(self.profile.max_accel, 3, "profile.max_accel")]
        raw_accel = np.clip(raw_accel, -accel_limit, accel_limit)
        alpha = float(np.clip(self.profile.accel_filter_alpha, 0.0, 1.0))
        self._acceleration = (1.0 - alpha) * self._acceleration + alpha * raw_accel
        self._last_time = now
        self._last_speed = speed.copy()
        return self._acceleration.copy()


def _deadband(values: np.ndarray, band: np.ndarray) -> np.ndarray:
    band = np.maximum(np.asarray(band, dtype=float), 0.0)
    result = np.asarray(values, dtype=float).copy()
    result[np.abs(result) < band] = 0.0
    return result


class RtdeImpedanceMotion:
    """UR RTDE motion API that drives the TCP through forceMode impedance."""

    TASK_FRAME = np.zeros(6, dtype=float)
    FORCE_TYPE = 2

    def __init__(
        self,
        robot: Any,
        profile: ImpedanceProfile,
        *,
        runtime: ImpedanceRuntimeConfig | None = None,
        mode: str = "passive",
        kinematics: Any | None = None,
    ) -> None:
        self.robot = robot
        self.runtime = runtime or ImpedanceRuntimeConfig()
        self.profile = profile
        self.kinematics = kinematics if self.runtime.state_source == "jacobian" else None
        self.core = CartesianImpedanceCore(
            profile,
            mode=mode,
            ramp_duration_s=self.runtime.ramp_duration_s,
        )

    @classmethod
    def connect(
        cls,
        robot_ip: str,
        profile: ImpedanceProfile,
        *,
        runtime: ImpedanceRuntimeConfig | None = None,
        mode: str = "passive",
        kinematics: Any | None = None,
    ) -> "RtdeImpedanceMotion":
        from mujoco_env.hardware.ur3e_api import UR3eController

        robot = UR3eController(robot_ip, auto_connect=True, connect_gripper=False)
        return cls(robot, profile, runtime=runtime, mode=mode, kinematics=kinematics)

    @property
    def selection_vector(self) -> tuple[int, int, int, int, int, int]:
        return self.core.selection_vector

    def close(self, *, stop_force_mode: bool = True) -> None:
        if stop_force_mode:
            self.stop()
        self.robot.close()

    def move_to_home(self) -> bool:
        return bool(self.robot.move_to_home())

    def configure_force_mode(self) -> None:
        if self.runtime.payload_mass_kg > 0.0:
            self.robot.set_payload(self.runtime.payload_mass_kg, self.runtime.payload_cog_m)
        if self.runtime.zero_ft_sensor:
            self.robot.zero_ft_sensor()
            time.sleep(0.1)
        self.robot.set_force_mode_damping(self.profile.force_mode_damping)
        self.robot.set_force_mode_gain_scaling(self.profile.force_mode_gain_scaling)

    def read_state(self) -> CartesianState:
        if self.kinematics is None:
            pose = self.robot.get_tcp_pose()
            return CartesianState(position=pose[:3], rotation_vector=pose[3:], speed=self.robot.get_tcp_speed())

        q = self.robot.get_joint_positions()
        qd = self.robot.get_joint_speeds()
        pos, quat, linear_speed, angular_speed = self.kinematics.forward_with_velocity(q, qd)
        return CartesianState(
            position=pos,
            rotation_vector=R.from_quat(quat).as_rotvec(),
            speed=np.r_[linear_speed, angular_speed],
        )

    def assert_state_source_aligned(self) -> None:
        if self.kinematics is None:
            return
        fk_pos = self.read_state().position
        rtde_pos = self.robot.get_tcp_pose()[:3]
        delta = float(np.linalg.norm(fk_pos - rtde_pos))
        if delta > self.runtime.max_fk_rtde_delta_m:
            raise RuntimeError(
                f"Jacobian FK and RTDE TCP differ by {delta:.3f} m, "
                f"above {self.runtime.max_fk_rtde_delta_m:.3f} m. "
                "Use state_source='rtde' until the MuJoCo base/TCP frame is aligned."
            )

    def set_target_pose(self, position: Any, rotation_vector: Any | None = None, *, reset_ramp: bool = False) -> None:
        self.core.set_target_pose(position, rotation_vector)
        if reset_ramp:
            self.core.reset_ramp()

    def set_target_from_current(self, offset: Any | None = None, *, keep_orientation: bool = True) -> CartesianState:
        state = self.read_state()
        delta = np.zeros(3, dtype=float) if offset is None else _vec(offset, 3, "offset")
        rot = state.rotation_vector if keep_orientation else None
        self.set_target_pose(state.position + delta, rot, reset_ramp=True)
        return state

    def compute(self, state: CartesianState | None = None) -> ImpedanceCommand:
        return self.core.compute(state or self.read_state())

    def step(self, *, execute: bool = True, state: CartesianState | None = None) -> ImpedanceCommand:
        command = self.compute(state)
        if execute:
            self.robot.force_mode(
                self.TASK_FRAME,
                self.selection_vector,
                command.wrench,
                self.FORCE_TYPE,
                self.profile.force_mode_limits,
            )
        return command

    def stop(self) -> None:
        try:
            self.robot.force_mode_stop()
        except Exception:
            pass
