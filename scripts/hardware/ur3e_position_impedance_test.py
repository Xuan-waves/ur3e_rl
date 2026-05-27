#!/usr/bin/env python3
"""Standalone UR3e Cartesian impedance tuning entry point."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_teleop.impedance import (  # noqa: E402
    DEFAULT_IMPEDANCE_TEST_CONFIG,
    CartesianImpedanceCore,
    CartesianState,
    ImpedanceProfile,
    ImpedanceRuntimeConfig,
    RtdeImpedanceMotion,
)


TEST_CONFIG = DEFAULT_IMPEDANCE_TEST_CONFIG


def _vec(values: Sequence[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf: {arr}.")
    return arr


def _fmt(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v): .5f}" for v in values) + "]"


class SimulatedTcp:
    def __init__(self, start_pos: np.ndarray, mass: float) -> None:
        self.pos = _vec(start_pos, 3, "start_pos").copy()
        self.vel = np.zeros(3, dtype=float)
        self.mass = max(float(mass), 1e-6)

    def step(self, force: np.ndarray, dt: float) -> None:
        self.vel += np.asarray(force, dtype=float) / self.mass * dt
        self.pos += self.vel * dt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test UR3e end-effector impedance. Default mode is offline simulation; "
            "use --read-robot or --execute explicitly for hardware."
        )
    )
    parser.add_argument("--robot-ip", default=TEST_CONFIG.robot_ip)
    parser.add_argument("--mode", choices=tuple(TEST_CONFIG.profiles.keys()), default=TEST_CONFIG.default_mode)
    parser.add_argument("--execute", action="store_true", help="Send forceMode commands to the real robot.")
    parser.add_argument("--read-robot", action="store_true", help="Read robot state and print wrench without sending.")
    parser.add_argument(
        "--move-home-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="MoveJ to this project's API home pose before the impedance test.",
    )
    parser.add_argument(
        "--state-source",
        choices=("rtde", "jacobian"),
        default="rtde",
        help="Use UR RTDE TCP state or MuJoCo Jacobian state. Keep rtde for the current robot setup.",
    )
    parser.add_argument("--max-fk-rtde-delta", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--offset", type=float, nargs=3, default=TEST_CONFIG.start_offset_m)
    parser.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--kp", type=float, nargs=3, default=None)
    parser.add_argument("--kd", type=float, nargs=3, default=None)
    parser.add_argument("--virtual-mass", type=float, nargs=3, default=None)
    parser.add_argument("--accel-filter-alpha", type=float, default=None)
    parser.add_argument("--max-accel", type=float, nargs=3, default=None)
    parser.add_argument("--force-bias", type=float, nargs=3, default=None)
    parser.add_argument("--max-force", type=float, nargs=3, default=None)
    parser.add_argument("--force-mode-limits", type=float, nargs=6, default=None)
    parser.add_argument("--max-position-error", type=float, default=None)
    parser.add_argument("--ramp-duration", type=float, default=None)
    parser.add_argument("--log-interval", type=float, default=None)
    parser.add_argument("--payload-mass", type=float, default=None)
    parser.add_argument("--payload-cog", type=float, nargs=3, default=None)
    parser.add_argument("--zero-ft-sensor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-mode-damping", type=float, default=None)
    parser.add_argument("--force-mode-gain-scaling", type=float, default=None)
    parser.add_argument("--enable-orientation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rot-kp", type=float, nargs=3, default=None)
    parser.add_argument("--rot-kd", type=float, nargs=3, default=None)
    parser.add_argument("--rot-virtual-mass", type=float, nargs=3, default=None)
    parser.add_argument("--torque-bias", type=float, nargs=3, default=None)
    parser.add_argument("--max-torque", type=float, nargs=3, default=None)
    parser.add_argument("--max-rot-error", type=float, default=None)
    parser.add_argument("--simulated-mass", type=float, default=None)
    parser.add_argument("--print-config", action="store_true")
    return parser


def _resolved_profile(args: argparse.Namespace) -> ImpedanceProfile:
    profile = TEST_CONFIG.profiles[str(args.mode)]
    updates: dict[str, Any] = {}
    vector_fields = {
        "kp": (args.kp, 3),
        "kd": (args.kd, 3),
        "virtual_mass": (args.virtual_mass, 3),
        "max_accel": (args.max_accel, 3),
        "force_bias": (args.force_bias, 3),
        "max_force": (args.max_force, 3),
        "force_mode_limits": (args.force_mode_limits, 6),
        "rot_kp": (args.rot_kp, 3),
        "rot_kd": (args.rot_kd, 3),
        "rot_virtual_mass": (args.rot_virtual_mass, 3),
        "torque_bias": (args.torque_bias, 3),
        "max_torque": (args.max_torque, 3),
    }
    for name, (value, size) in vector_fields.items():
        if value is not None:
            updates[name] = tuple(float(v) for v in _vec(value, size, name))

    scalar_fields = {
        "accel_filter_alpha": args.accel_filter_alpha,
        "max_position_error": args.max_position_error,
        "force_mode_damping": args.force_mode_damping,
        "force_mode_gain_scaling": args.force_mode_gain_scaling,
        "max_rot_error": args.max_rot_error,
    }
    for name, value in scalar_fields.items():
        if value is not None:
            updates[name] = float(value)

    if args.enable_orientation is not None:
        updates["enable_orientation"] = bool(args.enable_orientation)

    resolved = replace(profile, **updates)
    _validate_profile(resolved)
    return resolved


def _runtime_from_args(args: argparse.Namespace) -> ImpedanceRuntimeConfig:
    control_hz = TEST_CONFIG.control_hz if args.control_hz is None else float(args.control_hz)
    ramp_duration = TEST_CONFIG.ramp_duration_s if args.ramp_duration is None else float(args.ramp_duration)
    payload_mass = TEST_CONFIG.payload_mass_kg if args.payload_mass is None else float(args.payload_mass)
    payload_cog = TEST_CONFIG.payload_cog_m if args.payload_cog is None else tuple(float(v) for v in args.payload_cog)
    zero_ft = TEST_CONFIG.zero_ft_sensor if args.zero_ft_sensor is None else bool(args.zero_ft_sensor)
    if control_hz <= 0.0:
        raise ValueError("--control-hz must be positive.")
    if ramp_duration < 0.0:
        raise ValueError("--ramp-duration must be non-negative.")
    if payload_mass < 0.0:
        raise ValueError("--payload-mass must be non-negative.")
    return ImpedanceRuntimeConfig(
        robot_ip=str(args.robot_ip),
        control_hz=control_hz,
        ramp_duration_s=ramp_duration,
        state_source=str(args.state_source),
        max_fk_rtde_delta_m=float(args.max_fk_rtde_delta),
        move_home_first=_should_move_home_first(args),
        zero_ft_sensor=zero_ft,
        payload_mass_kg=payload_mass,
        payload_cog_m=payload_cog,
    )


def _validate_profile(profile: ImpedanceProfile) -> None:
    if profile.max_position_error <= 0.0:
        raise ValueError("--max-position-error must be positive.")
    if profile.max_rot_error <= 0.0:
        raise ValueError("--max-rot-error must be positive.")
    if not 0.0 <= profile.accel_filter_alpha <= 1.0:
        raise ValueError("--accel-filter-alpha must be in [0, 1].")


def _target_from_start(args: argparse.Namespace, start_pos: np.ndarray) -> np.ndarray:
    if args.target is not None:
        return _vec(args.target, 3, "target")
    return start_pos + _vec(args.offset, 3, "offset")


def _should_move_home_first(args: argparse.Namespace) -> bool:
    if args.move_home_first is None:
        return bool(args.execute and TEST_CONFIG.move_home_first)
    return bool(args.move_home_first)


def _sleep_until(next_time: float) -> float:
    now = time.monotonic()
    if next_time > now:
        time.sleep(next_time - now)
        return next_time
    return now


def _print_resolved_config(profile: ImpedanceProfile, runtime: ImpedanceRuntimeConfig, args: argparse.Namespace) -> None:
    duration = TEST_CONFIG.duration_s if args.duration is None else float(args.duration)
    log_interval = TEST_CONFIG.log_interval_s if args.log_interval is None else float(args.log_interval)
    print("Resolved impedance config:")
    print(f"  mode                  : {args.mode}")
    print(f"  state_source          : {runtime.state_source}")
    print(f"  robot_ip              : {runtime.robot_ip}")
    print(f"  move_home_first        : {runtime.move_home_first}")
    print(f"  duration_s            : {duration}")
    print(f"  control_hz            : {runtime.control_hz}")
    print(f"  kp_n_per_m            : {_fmt(profile.kp)}")
    print(f"  virtual_mass_kg       : {_fmt(profile.virtual_mass)}")
    print(f"  accel_filter_alpha    : {profile.accel_filter_alpha}")
    print(f"  max_accel_m_s2        : {_fmt(profile.max_accel)}")
    print(f"  kd_n_s_per_m          : {_fmt(profile.kd)}")
    print(f"  force_bias_n          : {_fmt(profile.force_bias)}")
    print(f"  max_force_n           : {_fmt(profile.max_force)}")
    print(f"  max_position_error_m  : {profile.max_position_error}")
    print(f"  force_mode_limits     : {_fmt(profile.force_mode_limits)}")
    print(f"  force_mode_damping    : {profile.force_mode_damping}")
    print(f"  force_mode_gain       : {profile.force_mode_gain_scaling}")
    print(f"  enable_orientation    : {profile.enable_orientation}")
    print(f"  rot_kp_n_m_per_rad    : {_fmt(profile.rot_kp)}")
    print(f"  rot_kd_n_m_s_per_rad  : {_fmt(profile.rot_kd)}")
    print(f"  max_torque_n_m        : {_fmt(profile.max_torque)}")
    print(f"  max_rot_error_rad     : {profile.max_rot_error}")
    print(f"  payload_mass_kg       : {runtime.payload_mass_kg}")
    print(f"  payload_cog_m         : {_fmt(runtime.payload_cog_m)}")
    print(f"  zero_ft_sensor         : {runtime.zero_ft_sensor}")
    print(f"  log_interval_s        : {log_interval}")


def _run_simulation(profile: ImpedanceProfile, runtime: ImpedanceRuntimeConfig, args: argparse.Namespace) -> int:
    duration = TEST_CONFIG.duration_s if args.duration is None else float(args.duration)
    simulated_mass = TEST_CONFIG.simulated_mass_kg if args.simulated_mass is None else float(args.simulated_mass)
    log_interval = TEST_CONFIG.log_interval_s if args.log_interval is None else float(args.log_interval)
    if duration <= 0.0:
        raise ValueError("--duration must be positive.")

    start_pos = np.zeros(3, dtype=float)
    target_pos = _target_from_start(args, start_pos)
    core = CartesianImpedanceCore(profile, mode=str(args.mode), ramp_duration_s=runtime.ramp_duration_s)
    core.set_target_pose(target_pos, np.zeros(3, dtype=float))
    sim = SimulatedTcp(start_pos, simulated_mass)

    print("Mode: offline simulation")
    print(f"control_mode={args.mode}, target_pos={_fmt(target_pos)} m")
    print(f"kp={_fmt(profile.kp)} N/m, kd={_fmt(profile.kd)} N*s/m, max_force={_fmt(profile.max_force)} N")
    _loop(
        read_state=lambda: CartesianState(sim.pos, np.zeros(3, dtype=float), np.r_[sim.vel, np.zeros(3, dtype=float)]),
        compute_command=core.compute,
        command_callback=lambda command: sim.step(command.wrench[:3], 1.0 / runtime.control_hz),
        duration=duration,
        control_hz=runtime.control_hz,
        log_interval=log_interval,
    )
    return 0


def _run_robot(profile: ImpedanceProfile, runtime: ImpedanceRuntimeConfig, args: argparse.Namespace) -> int:
    from real_teleop.config import TeleopConfig
    from real_teleop.kinematics import RobotKinematics

    duration = TEST_CONFIG.duration_s if args.duration is None else float(args.duration)
    log_interval = TEST_CONFIG.log_interval_s if args.log_interval is None else float(args.log_interval)
    if duration <= 0.0:
        raise ValueError("--duration must be positive.")

    kinematics = RobotKinematics(TeleopConfig(robot_ip=runtime.robot_ip)) if runtime.state_source == "jacobian" else None
    motion = RtdeImpedanceMotion.connect(
        runtime.robot_ip,
        profile,
        runtime=runtime,
        mode=str(args.mode),
        kinematics=kinematics,
    )
    try:
        if runtime.move_home_first:
            print("Moving to API home pose with moveJ before impedance test...")
            motion.move_to_home()
            time.sleep(0.2)

        start_state = motion.set_target_from_current(keep_orientation=True)
        target_pos = _target_from_start(args, start_state.position)
        motion.set_target_pose(target_pos, start_state.rotation_vector, reset_ramp=True)
        motion.assert_state_source_aligned()

        mode = "real robot forceMode" if args.execute else "real robot read-only"
        print(f"Mode: {mode}")
        print(f"control_mode={args.mode}")
        print(f"state_source={runtime.state_source}")
        print("Do not run the VR servoJ robot node at the same time.")
        print(f"start_pos={_fmt(start_state.position)} m, target_pos={_fmt(target_pos)} m")
        print(f"target_rotvec={_fmt(start_state.rotation_vector)} rad, orientation_enabled={profile.enable_orientation}")
        print(
            f"kp={_fmt(profile.kp)} N/m, virtual_mass={_fmt(profile.virtual_mass)} kg, "
            f"kd={_fmt(profile.kd)} N*s/m, force_bias={_fmt(profile.force_bias)} N, "
            f"max_force={_fmt(profile.max_force)} N"
        )
        if args.mode == "passive":
            print("Passive mode: a virtual spring holds the start TCP; push by hand to feel compliance.")
        elif args.mode == "zero-force":
            print("Zero-force mode: no position hold is applied, so the TCP may drift or drop.")

        if args.execute:
            motion.configure_force_mode()

        _loop(
            read_state=motion.read_state,
            compute_command=motion.compute,
            command_callback=None,
            duration=duration,
            control_hz=runtime.control_hz,
            log_interval=log_interval,
            execute_step=(motion.step if args.execute else None),
        )
    finally:
        motion.close(stop_force_mode=bool(args.execute))
    return 0


def _loop(
    *,
    read_state: Any,
    compute_command: Any,
    command_callback: Any | None = None,
    duration: float,
    control_hz: float,
    log_interval: float,
    execute_step: Any | None = None,
) -> None:
    period = 1.0 / control_hz
    start = time.monotonic()
    next_tick = start
    next_log = start
    samples = 0
    max_late = 0.0

    while True:
        now = time.monotonic()
        if now - start >= duration:
            break

        if execute_step is not None:
            command = execute_step(execute=True)
            state = read_state() if now >= next_log else None
        else:
            state = read_state()
            command = compute_command(state)
            if command_callback is not None:
                command_callback(command)
        samples += 1

        if now >= next_log:
            pos = state.position if state is not None else read_state().position
            print(
                f"t={now - start:6.3f}s pos={_fmt(pos)} "
                f"err={_fmt(command.position_error)} rot_err={_fmt(command.rotation_error)} "
                f"acc={_fmt(command.acceleration)} wrench={_fmt(command.wrench)}"
            )
            next_log += log_interval

        next_tick += period
        before_sleep = time.monotonic()
        max_late = max(max_late, max(0.0, before_sleep - next_tick))
        _sleep_until(next_tick)

    actual = max(time.monotonic() - start, 1e-9)
    print(f"Finished: {samples} samples, avg_rate={samples / actual:.1f} Hz, max_late={max_late * 1000.0:.2f} ms")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.execute and args.read_robot:
            raise ValueError("--execute already reads the robot; use only one mode flag.")
        if args.move_home_first is True and not args.execute:
            raise ValueError("--move-home-first commands moveJ, so it must be used together with --execute.")
        profile = _resolved_profile(args)
        runtime = _runtime_from_args(args)
        if args.print_config:
            _print_resolved_config(profile, runtime, args)
        if args.execute or args.read_robot:
            return _run_robot(profile, runtime, args)
        return _run_simulation(profile, runtime, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        _print_error(exc)
        return 1


def _print_error(exc: Exception) -> None:
    message = str(exc)
    print(f"ERROR: {message}", file=sys.stderr)
    if "RTDE input registers are already in use" in message:
        print(
            "Hint: stop every other process connected through rtde_control first, "
            "especially the VR robot/servoJ tab. If no local process is running, "
            "disable EtherNet/IP, PROFINET, and configured MODBUS units on the UR controller.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
