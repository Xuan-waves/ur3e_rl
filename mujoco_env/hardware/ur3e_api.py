"""
UR3e real-machine low-level control API.

Self-contained module — only dependencies are numpy and ur_rtde.
Provides direct RTDE control of the UR3e arm and socket-based control
of the Robotiq 2F-85 gripper.

Usage::

    from mujoco_env.hardware.ur3e_api import UR3eController

    robot = UR3eController("192.168.5.1")
    robot.go_home()
    robot.move_joints([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    robot.servo_joints([1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    robot.move_gripper_by_trigger(0.5)   # half-close
    robot.close()
"""

from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import rtde_control
import rtde_receive

# ---------------------------------------------------------------------------
#  Default parameters (matching the production hardware YAML config)
# ---------------------------------------------------------------------------

# fmt: off
_HOME_Q = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi])
_JOINT_LIMITS = np.array([
    [-2.0 * np.pi, 2.0 * np.pi],
    [-2.0 * np.pi, 2.0 * np.pi],
    [-np.pi,         np.pi       ],
    [-2.0 * np.pi, 2.0 * np.pi],
    [-2.0 * np.pi, 2.0 * np.pi],
    [-2.0 * np.pi, 2.0 * np.pi],
])
# fmt: on


# ===================================================================
#  RobotiqGripper  —  raw socket driver for 2F-85 / HAND-E
# ===================================================================

class RobotiqGripper:
    """Communicates with a Robotiq gripper directly via TCP socket."""

    # register names
    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    ADR = "ADR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"
    STA = "STA"
    PRE = "PRE"
    OBJ = "OBJ"
    FLT = "FLT"

    ENCODING = "UTF-8"

    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self) -> None:
        self.socket: Optional[socket.socket] = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    # -- connection --------------------------------------------------

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    # -- register I/O ------------------------------------------------

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]) -> bool:
        cmd = "SET"
        for var, val in var_dict.items():
            cmd += f" {var} {val}"
        cmd += "\n"
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return data == b"ack"

    def _set_var(self, variable: str, value: Union[int, float]) -> bool:
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str) -> int:
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response: {data!r}")
        return int(value_str)

    # -- lifecycle ---------------------------------------------------

    def _reset(self) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)

    def activate(self, auto_calibrate: bool = True) -> None:
        if self.is_active():
            return
        self._reset()
        while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
            time.sleep(0.01)
        self._set_var(self.ACT, 1)
        time.sleep(1.0)
        while self._get_var(self.ACT) != 1 or self._get_var(self.STA) != 3:
            time.sleep(0.01)
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self) -> bool:
        return RobotiqGripper.GripperStatus(self._get_var(self.STA)) == RobotiqGripper.GripperStatus.ACTIVE

    # -- calibration -------------------------------------------------

    def auto_calibrate(self, log: bool = True) -> None:
        self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        pos, _ = self.move_and_wait_for_pos(self.get_closed_position(), 64, 1)
        self._max_position = pos
        pos, _ = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        self._min_position = pos
        if log:
            print(f"Gripper auto-calibrated to [{self._min_position}, {self._max_position}]")

    # -- motion ------------------------------------------------------

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        def _clip(lo, v, hi):
            return max(lo, min(v, hi))

        clip_pos = _clip(self._min_position, position, self._max_position)
        clip_spe = _clip(self._min_speed, speed, self._max_speed)
        clip_for = _clip(self._min_force, force, self._max_force)
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
        return self._set_vars(var_dict), clip_pos

    def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> Tuple[int, ObjectStatus]:
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set gripper move variables.")
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)
        cur_obj = self._get_var(self.OBJ)
        while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
            cur_obj = self._get_var(self.OBJ)
        return self._get_var(self.POS), RobotiqGripper.ObjectStatus(cur_obj)

    # -- position helpers --------------------------------------------

    def get_min_position(self) -> int:
        return self._min_position

    def get_max_position(self) -> int:
        return self._max_position

    def get_open_position(self) -> int:
        return self._min_position

    def get_closed_position(self) -> int:
        return self._max_position

    def get_current_position(self) -> int:
        return self._get_var(self.POS)

    def is_open(self) -> bool:
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self) -> bool:
        return self.get_current_position() >= self.get_closed_position()


# ===================================================================
#  RobotiqGripperController  —  trigger-based wrapper (0=open, 1=closed)
# ===================================================================

class RobotiqGripperController:
    """Convenience wrapper that accepts a normalised *trigger* value in [0, 1].

    ``0.0`` → fully open,  ``1.0`` → fully closed.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        auto_connect: bool = False,
        gripper_port: int = 63352,
        gripper_speed: int = 255,
        gripper_force: int = 128,
        open_position: int = 0,
        closed_position: int = 255,
        open_angle_rad: float = 0.0,
        closed_angle_rad: float = 0.6,
    ) -> None:
        self.robot_ip = robot_ip
        self.port = gripper_port
        self.speed = gripper_speed
        self.force = gripper_force
        self.open_pos = open_position
        self.closed_pos = closed_position
        self.open_angle = open_angle_rad
        self.closed_angle = closed_angle_rad

        self.gripper: Optional[RobotiqGripper] = None
        self.last_trigger = 0.0
        self.last_angle = open_angle_rad

        if auto_connect:
            self.connect()

    @property
    def is_connected(self) -> bool:
        return self.gripper is not None

    # -- connection --------------------------------------------------

    def connect(self) -> None:
        if self.is_connected:
            return
        g = RobotiqGripper()
        g.connect(self.robot_ip, self.port)
        g.activate()
        self.gripper = g

    def disconnect(self) -> None:
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception:
                pass
        self.gripper = None

    # -- conversion helpers ------------------------------------------

    def trigger_to_position(self, trigger: float) -> int:
        t = float(np.clip(trigger, 0.0, 1.0))
        return int(round(self.open_pos + t * (self.closed_pos - self.open_pos)))

    def trigger_to_angle(self, trigger: float) -> float:
        t = float(np.clip(trigger, 0.0, 1.0))
        return self.open_angle + t * (self.closed_angle - self.open_angle)

    def position_to_trigger(self, position: float) -> float:
        r = (float(position) - self.open_pos) / max(self.closed_pos - self.open_pos, 1e-9)
        return float(np.clip(r, 0.0, 1.0))

    def position_to_angle(self, position: float) -> float:
        return self.trigger_to_angle(self.position_to_trigger(position))

    def angle_to_trigger(self, angle: float) -> float:
        r = (float(angle) - self.open_angle) / max(self.closed_angle - self.open_angle, 1e-9)
        return float(np.clip(r, 0.0, 1.0))

    # -- motion ------------------------------------------------------

    def move_by_trigger(self, trigger: float) -> None:
        t = float(np.clip(trigger, 0.0, 1.0))
        self.last_trigger = t
        self.last_angle = self.trigger_to_angle(t)
        if self.gripper is None:
            return
        target = self.trigger_to_position(t)
        if hasattr(self.gripper, "move"):
            self.gripper.move(target, self.speed, self.force)
        elif hasattr(self.gripper, "move_and_wait_for_pos"):
            self.gripper.move_and_wait_for_pos(target, self.speed, self.force)
        else:
            raise RuntimeError("Attached Robotiq gripper does not provide a move method.")

    def open(self) -> None:
        self.move_by_trigger(0.0)

    def close(self) -> None:
        self.move_by_trigger(1.0)

    # -- read-back ---------------------------------------------------

    def get_position(self) -> Optional[int]:
        if self.gripper is None:
            return None
        for m in ("get_current_position", "get_current_pos", "get_position"):
            if hasattr(self.gripper, m):
                try:
                    return int(getattr(self.gripper, m)())
                except Exception:
                    break
        return None

    def get_angle(self) -> float:
        pos = self.get_position()
        if pos is not None:
            self.last_angle = self.position_to_angle(float(pos))
        return self.last_angle

    def get_trigger(self) -> float:
        pos = self.get_position()
        if pos is not None:
            self.last_trigger = self.position_to_trigger(float(pos))
        return self.angle_to_trigger(self.last_angle)


# ===================================================================
#  UR3eController  —  main entry point for UR3e + gripper control
# ===================================================================

class UR3eController:
    """Low-level RTDE controller for a UR3e arm with optional Robotiq 2F-85 gripper.

    Parameters
    ----------
    robot_ip:
        IP address of the UR3e controller (default ``"192.168.5.1"``).
    auto_connect:
        Call ``connect()`` immediately on construction.
    home_q:
        Home joint configuration (6-element array-like, rad).
    joint_limits:
        (6, 2) array of lower/upper joint limits (rad).
    movej_speed:
        Default moveJ speed (rad/s of the tool-space equivalent).
    movej_acceleration:
        Default moveJ acceleration.
    servo_time:
        servoJ timestep (s).  Default ``0.002`` → 500 Hz.
    lookahead_time:
        servoJ lookahead (s).
    servo_gain:
        servoJ proportional gain.
    connect_gripper:
        Whether to attempt gripper attachment on ``attach_gripper()``.
    gripper_port / gripper_speed / gripper_force:
        Passed through to ``RobotiqGripperController``.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.5.1",
        *,
        auto_connect: bool = True,
        # arm
        home_q: Sequence[float] | np.ndarray = _HOME_Q,
        joint_limits: np.ndarray = _JOINT_LIMITS,
        movej_speed: float = 0.25,
        movej_acceleration: float = 0.5,
        servo_time: float = 0.002,
        lookahead_time: float = 0.10,
        servo_gain: float = 100.0,
        # gripper
        connect_gripper: bool = True,
        gripper_port: int = 63352,
        gripper_speed: int = 255,
        gripper_force: int = 128,
    ) -> None:
        self.robot_ip = robot_ip
        self.home_q = np.asarray(home_q, dtype=float)
        self.joint_limits = np.asarray(joint_limits, dtype=float)
        self.movej_speed = movej_speed
        self.movej_acceleration = movej_acceleration
        self.servo_time = servo_time
        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain

        self._connect_gripper = connect_gripper

        self.rtde_c: Optional[rtde_control.RTDEControlInterface] = None
        self.rtde_r: Optional[rtde_receive.RTDEReceiveInterface] = None

        self.gripper = RobotiqGripperController(
            robot_ip,
            auto_connect=False,
            gripper_port=gripper_port,
            gripper_speed=gripper_speed,
            gripper_force=gripper_force,
        )

        if auto_connect:
            self.connect()

    # -- connection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.rtde_c is not None and self.rtde_r is not None

    def connect(self) -> None:
        """Establish RTDE connection to the arm controller."""
        if self.is_connected:
            return
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

    def close(self) -> None:
        """Stop servo / moveJ and disconnect from the arm and gripper."""
        if self.rtde_c is not None:
            self._safe(self.rtde_c.servoStop)
            self._safe(self.rtde_c.speedStop)
            self._safe(self.rtde_c.stopScript)
        self._safe(self.gripper.disconnect)
        self.rtde_c = None
        self.rtde_r = None

    # -- validation --------------------------------------------------

    def validate_joints(self, q: Sequence[float]) -> np.ndarray:
        """Validate and return a 6-element joint array; raises on failure."""
        qa = np.asarray(list(q), dtype=float)
        if qa.shape != (6,):
            raise ValueError(f"Expected 6 joint values, got shape {qa.shape}.")
        if not np.all(np.isfinite(qa)):
            raise ValueError(f"Joint target contains NaN or inf: {qa}.")
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        if np.any(qa < lo) or np.any(qa > hi):
            raise ValueError(f"Joint target outside software limits: {qa}.")
        return qa

    # -- state read-back ---------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Return current joint angles as a (6,) array (rad)."""
        self._require()
        return np.asarray(self.rtde_r.getActualQ(), dtype=float)

    # -- moveJ (point-to-point trajectory) ---------------------------

    def move_joints(self, q: Sequence[float], asynchronous: bool = False) -> bool:
        """Move to *q* via the UR controller's built-in moveJ trajectory.

        When *asynchronous* is False the call blocks until the move finishes.
        """
        self._require()
        qa = self.validate_joints(q)
        return bool(
            self.rtde_c.moveJ(
                qa.tolist(),
                self.movej_speed,
                self.movej_acceleration,
                bool(asynchronous),
            )
        )

    def go_home(self) -> bool:
        """moveJ back to the configured home pose."""
        return self.move_joints(self.home_q, asynchronous=False)

    def move_to_home(self) -> bool:
        """moveJ back to the configured home pose."""
        return self.go_home()

    def wait_until_joints_reached(
        self,
        target_q: Sequence[float],
        tolerance: float = 0.02,
        timeout: float = 10.0,
    ) -> bool:
        """Poll ``get_joint_positions()`` until *target_q* is reached.

        Returns True if the target was reached within *timeout* seconds.
        """
        target = self.validate_joints(target_q)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if np.max(np.abs(self.get_joint_positions() - target)) < tolerance:
                return True
            time.sleep(0.02)
        return False

    # -- servoJ (streaming high-frequency control) -------------------

    def servo_joints(self, q: Sequence[float]) -> None:
        """Send a single servoJ command.  Call in a loop at the desired control rate."""
        self._require()
        qa = self.validate_joints(q)
        self.rtde_c.servoJ(
            qa.tolist(),
            0.0,
            0.0,
            self.servo_time,
            self.lookahead_time,
            self.servo_gain,
        )

    def servo_stop(self) -> None:
        """Stop the servoJ stream."""
        self._require()
        self.rtde_c.servoStop()

    # -- gripper -----------------------------------------------------

    def attach_gripper(self) -> None:
        """Connect and activate the Robotiq gripper (no-op if disabled)."""
        if not self._connect_gripper:
            return
        self.gripper.connect()

    def move_gripper_by_trigger(self, trigger: float) -> None:
        """Move gripper: 0.0 = open, 1.0 = closed."""
        self.gripper.move_by_trigger(trigger)

    def get_gripper_angle(self) -> float:
        return self.gripper.get_angle()

    def get_gripper_trigger(self) -> float:
        return self.gripper.get_trigger()

    # -- internal ----------------------------------------------------

    def _require(self) -> None:
        if not self.is_connected:
            raise RuntimeError("UR3eController is not connected. Call connect() first.")

    @staticmethod
    def _safe(fn, *a, **kw) -> None:
        try:
            fn(*a, **kw)
        except Exception:
            pass
