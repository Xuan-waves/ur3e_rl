from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import (
    TOPIC_IK_TARGET,
    TOPIC_JOINT_TARGET,
    TOPIC_ROBOT_DEBUG,
    TOPIC_ROBOT_STATE,
    TOPIC_VR_COMMAND,
    TeleopConfig,
)
from .filters import PoseFilter
from .messages import as_vec, dumps, loads, now
from .ros_qos import latest_qos
from .safety import SafetyLimiter
from .vr import XrobotVrReader


class VrNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig):
        from std_msgs.msg import String

        self.node = rclpy_node
        self.cfg = cfg
        self.reader = XrobotVrReader(cfg)
        self.pub = self.node.create_publisher(String, TOPIC_VR_COMMAND, latest_qos())
        self.timer = self.node.create_timer(1.0 / cfg.vr_hz, self._tick)
        self.node.get_logger().info(f"VR node publishing at {cfg.vr_hz:.0f} Hz")

    def _tick(self) -> None:
        from std_msgs.msg import String

        payload = self.reader.read()
        payload["stamp"] = now()
        msg = String()
        msg.data = dumps(payload)
        self.pub.publish(msg)


class IkNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig):
        from std_msgs.msg import String

        self.node = rclpy_node
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.safety = SafetyLimiter(cfg)
        from .kinematics import MinkIkSolver

        self.ik = MinkIkSolver(cfg)
        self.ctrl_filter = PoseFilter(cfg.ctrl_filter_alpha, cfg.ctrl_filter_alpha)
        self.target_filter = PoseFilter(cfg.target_filter_alpha, cfg.target_filter_alpha)

        self.state: Optional[dict] = None
        self.command: Optional[dict] = None
        self.anchor_ctrl_pos: Optional[np.ndarray] = None
        self.anchor_ctrl_quat: Optional[np.ndarray] = None
        self.anchor_tcp_pos: Optional[np.ndarray] = None
        self.anchor_tcp_quat: Optional[np.ndarray] = None
        self.last_target_pos: Optional[np.ndarray] = None
        self.last_home = False

        qos = latest_qos()
        self.target_pub = self.node.create_publisher(String, TOPIC_JOINT_TARGET, qos)
        self.ik_target_pub = self.node.create_publisher(String, TOPIC_IK_TARGET, qos)
        self.state_sub = self.node.create_subscription(String, TOPIC_ROBOT_STATE, self._on_state, qos)
        self.command_sub = self.node.create_subscription(String, TOPIC_VR_COMMAND, self._on_command, qos)
        self.timer = self.node.create_timer(self.dt, self._tick)
        self.node.get_logger().info(f"IK node running mink at {cfg.control_hz:.0f} Hz")

    def _on_state(self, msg) -> None:
        try:
            self.state = loads(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad robot state: {exc}")

    def _on_command(self, msg) -> None:
        try:
            self.command = loads(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad VR command: {exc}")

    @property
    def tracking(self) -> bool:
        return self.anchor_tcp_pos is not None

    def _release(self) -> None:
        self.anchor_ctrl_pos = None
        self.anchor_ctrl_quat = None
        self.anchor_tcp_pos = None
        self.anchor_tcp_quat = None
        self.last_target_pos = None
        self.ctrl_filter.reset()
        self.target_filter.reset()

    def _tick(self) -> None:
        if self.state is None or self.command is None:
            return
        if now() - float(self.command.get("stamp", 0.0)) > self.cfg.stale_command_s:
            self._release()
            self._publish_hold("stale_vr")
            return

        home = bool(self.command.get("home", False))
        if home and not self.last_home:
            self._release()
            self._publish_hold("home")
        self.last_home = home

        pose = self.command.get("pose")
        if not bool(self.command.get("enable", False)) or pose is None:
            self._release()
            self._publish_hold("disabled")
            return

        try:
            ctrl_pose = as_vec(pose, 7)
            q = as_vec(self.state.get("q"), 6)
            tcp_pos = as_vec(self.state.get("tcp_pos"), 3)
            tcp_quat = as_vec(self.state.get("tcp_quat"), 4)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad IK input: {exc}")
            return

        ctrl_pos, ctrl_quat = ctrl_pose[:3], ctrl_pose[3:]
        if not self.tracking:
            self._anchor(ctrl_pos, ctrl_quat, tcp_pos, tcp_quat)
            self._publish_hold("anchored")
            return

        target_pos, target_quat = self._target_from_controller(ctrl_pos, ctrl_quat)
        target_pos = self.safety.clamp_workspace(target_pos)
        if self.last_target_pos is not None:
            jump = target_pos - self.last_target_pos
            norm = float(np.linalg.norm(jump))
            if norm > self.cfg.max_target_jump:
                target_pos = self.last_target_pos + jump / norm * self.cfg.max_target_jump
        self.last_target_pos = target_pos.copy()

        q_raw, ok = self.ik.solve(target_pos, target_quat, q, self.dt)
        q_delta = float(np.max(np.abs(q_raw[:6] - q)))
        q_safe = self.safety.clamp_joints(q_raw)
        self._publish_target(q_safe, target_pos, target_quat, ok, q_delta)

    def _anchor(
        self,
        ctrl_pos: np.ndarray,
        ctrl_quat: np.ndarray,
        tcp_pos: np.ndarray,
        tcp_quat: np.ndarray,
    ) -> None:
        self.anchor_ctrl_pos = ctrl_pos.copy()
        self.anchor_ctrl_quat = _norm_quat(ctrl_quat)
        self.anchor_tcp_pos = tcp_pos.copy()
        self.anchor_tcp_quat = _norm_quat(tcp_quat)
        self.ctrl_filter.reset()
        self.target_filter.reset()
        self.last_target_pos = None

    def _target_from_controller(self, ctrl_pos: np.ndarray, ctrl_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctrl_pos_f, ctrl_quat_f = self.ctrl_filter(ctrl_pos, ctrl_quat)
        dpos = (ctrl_pos_f - self.anchor_ctrl_pos) * self.cfg.scale
        drot = (R.from_quat(ctrl_quat_f) * R.from_quat(self.anchor_ctrl_quat).inv()).as_rotvec()
        if np.linalg.norm(dpos) < self.cfg.dead_zone_pos:
            dpos[:] = 0.0
        if np.linalg.norm(drot) < self.cfg.dead_zone_rot:
            drot[:] = 0.0
        pos = self.anchor_tcp_pos + dpos
        quat = (R.from_rotvec(drot) * R.from_quat(self.anchor_tcp_quat)).as_quat()
        return self.target_filter(pos, quat)

    def _publish_hold(self, reason: str) -> None:
        from std_msgs.msg import String

        msg = String()
        msg.data = dumps(
            {
                "stamp": now(),
                "tracking": False,
                "reason": reason,
                "gripper": float(self.command.get("gripper", 0.0)) if self.command else 0.0,
            }
        )
        self.target_pub.publish(msg)

    def _publish_target(
        self,
        q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        ok: bool,
        q_delta: float,
    ) -> None:
        from std_msgs.msg import String

        msg = String()
        msg.data = dumps(
            {
                "stamp": now(),
                "tracking": True,
                "ok": bool(ok),
                "q": q.tolist(),
                "gripper": float(self.command.get("gripper", 0.0)),
                "q_delta": float(q_delta),
            }
        )
        self.target_pub.publish(msg)

        target_msg = String()
        target_msg.data = dumps(
            {
                "stamp": now(),
                "pos": target_pos.tolist(),
                "quat": target_quat.tolist(),
            }
        )
        self.ik_target_pub.publish(target_msg)


class RobotNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig, *, dry_run: bool = False):
        from std_msgs.msg import String

        self.node = rclpy_node
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.dry_run = dry_run
        self.safety = SafetyLimiter(cfg)
        from .kinematics import RobotKinematics

        self.kin = RobotKinematics(cfg)
        qos = latest_qos()
        self.pub = self.node.create_publisher(String, TOPIC_ROBOT_STATE, qos)
        self.debug_pub = self.node.create_publisher(String, TOPIC_ROBOT_DEBUG, 10)
        self.target_sub = self.node.create_subscription(String, TOPIC_JOINT_TARGET, self._on_target, qos)
        self.command_sub = self.node.create_subscription(String, TOPIC_VR_COMMAND, self._on_command, qos)

        self.robot = None
        self.current_q = self._model_home()
        self.target_q: Optional[np.ndarray] = None
        self.filtered_target_q: Optional[np.ndarray] = None
        self.target_stamp = 0.0
        self.target_tracking = False
        self.target_reason = "no_target"
        self.target_count = 0
        self.last_target_delta = 0.0
        self.last_q_step = 0.0
        self.last_debug_log = 0.0
        self.last_debug_pub = 0.0
        self.servo_active = False
        self.desired_gripper = 0.0
        self.last_gripper = -1.0
        self.last_gripper_send = 0.0
        self.last_home = False
        self.homing = False
        self.home_thread: Optional[threading.Thread] = None

        if not dry_run:
            from mujoco_env.hardware.ur3e_api import UR3eController

            self.robot = UR3eController(
                cfg.robot_ip,
                auto_connect=True,
                servo_time=self.dt,
                lookahead_time=0.06,
                servo_gain=120.0,
            )
            try:
                self.robot.attach_gripper()
            except Exception as exc:
                self.node.get_logger().warn(f"Gripper not attached: {exc}")
            self.current_q = self.robot.get_joint_positions()

        self.timer = self.node.create_timer(self.dt, self._tick)
        mode = "dry-run" if dry_run else f"real robot {cfg.robot_ip}"
        self.node.get_logger().info(f"Robot node controlling {mode} at {cfg.control_hz:.0f} Hz")

    def _model_home(self) -> np.ndarray:
        try:
            return self.kin.model.key("home").qpos[:6].copy()
        except Exception:
            return np.zeros(6, dtype=float)

    def _on_target(self, msg) -> None:
        try:
            payload = loads(msg.data)
            self.target_tracking = bool(payload.get("tracking", False))
            self.target_stamp = float(payload.get("stamp", 0.0))
            self.target_reason = str(payload.get("reason", "tracking" if self.target_tracking else "hold"))
            self.target_count += 1
            if "q" in payload:
                q = as_vec(payload["q"], 6)
                if self.safety.check_joints(q):
                    self.target_q = q
                    self.last_target_delta = float(np.max(np.abs(q - self.current_q)))
                    if self.filtered_target_q is None:
                        self.filtered_target_q = q.copy()
                    else:
                        alpha = float(np.clip(self.cfg.joint_target_alpha, 0.0, 1.0))
                        self.filtered_target_q = alpha * q + (1.0 - alpha) * self.filtered_target_q
            if "gripper" in payload:
                self.desired_gripper = float(np.clip(payload["gripper"], 0.0, 1.0))
        except Exception as exc:
            self.node.get_logger().warn(f"Bad joint target: {exc}")

    def _on_command(self, msg) -> None:
        try:
            payload = loads(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad robot command: {exc}")
            return
        self.desired_gripper = float(np.clip(payload.get("gripper", self.desired_gripper), 0.0, 1.0))
        home = bool(payload.get("home", False))
        if home and not self.last_home:
            self._start_home()
        self.last_home = home

    def _start_home(self) -> None:
        if self.homing:
            return
        self.homing = True
        self.target_tracking = False
        self.target_q = None
        self.filtered_target_q = None
        self._servo_stop()
        self.home_thread = threading.Thread(target=self._home_worker, daemon=True)
        self.home_thread.start()

    def _home_worker(self) -> None:
        try:
            if self.dry_run:
                self.current_q = self._model_home()
                return
            self.node.get_logger().info("A pressed: moveJ return to home")
            self.robot.go_home()
        except Exception as exc:
            self.node.get_logger().error(f"Home move failed: {exc}")
        finally:
            self.homing = False

    def _tick(self) -> None:
        if not self.dry_run:
            try:
                self.current_q = self.robot.get_joint_positions()
            except Exception as exc:
                self.node.get_logger().error(f"Failed to read robot joints: {exc}")
                self._servo_stop()
                return

        self._apply_servo()
        self._apply_gripper()
        self._publish_state()

    def _apply_servo(self) -> None:
        fresh = now() - self.target_stamp <= self.cfg.stale_target_s
        if self.homing or not self.target_tracking or self.target_q is None or not fresh:
            self._servo_stop()
            return
        target_q = self.filtered_target_q if self.filtered_target_q is not None else self.target_q
        q_cmd = self.safety.limit_step(self.current_q, target_q, self.dt)
        self.last_target_delta = float(np.max(np.abs(target_q - self.current_q)))
        self.last_q_step = float(np.max(np.abs(q_cmd - self.current_q)))
        if self.dry_run:
            self.current_q = q_cmd
            self.servo_active = True
            return
        try:
            self.robot.servo_joints(q_cmd)
            self.servo_active = True
        except Exception as exc:
            self.node.get_logger().error(f"servoJ failed: {exc}")
            self._servo_stop()

    def _servo_stop(self) -> None:
        if not self.servo_active:
            return
        if not self.dry_run:
            try:
                self.robot.servo_stop()
            except Exception as exc:
                self.node.get_logger().warn(f"servoStop failed: {exc}")
        self.servo_active = False

    def _apply_gripper(self) -> None:
        if now() - self.last_gripper_send < 0.04:
            return
        if abs(self.desired_gripper - self.last_gripper) < 0.02:
            return
        self.last_gripper = self.desired_gripper
        self.last_gripper_send = now()
        if self.dry_run:
            return
        try:
            self.robot.move_gripper_by_trigger(self.desired_gripper)
        except Exception as exc:
            self.node.get_logger().warn(f"Gripper command failed: {exc}")

    def _publish_state(self) -> None:
        from std_msgs.msg import String

        try:
            tcp_pos, tcp_quat = self.kin.forward(self.current_q)
        except Exception as exc:
            self.node.get_logger().warn(f"FK failed: {exc}")
            return
        msg = String()
        msg.data = dumps(
            {
                "stamp": now(),
                "q": self.current_q.tolist(),
                "tcp_pos": tcp_pos.tolist(),
                "tcp_quat": tcp_quat.tolist(),
                "gripper": float(self.desired_gripper),
                "servo_active": bool(self.servo_active),
                "homing": bool(self.homing),
                "mode": "dry" if self.dry_run else "real",
                "target_tracking": bool(self.target_tracking),
                "target_age": float(now() - self.target_stamp) if self.target_stamp > 0.0 else None,
                "target_reason": self.target_reason,
                "target_count": int(self.target_count),
                "target_delta": float(self.last_target_delta),
            }
        )
        self.pub.publish(msg)
        if now() - self.last_debug_pub >= 0.1:
            self.last_debug_pub = now()
            self._publish_debug()

    def _publish_debug(self) -> None:
        from std_msgs.msg import String

        age = float(now() - self.target_stamp) if self.target_stamp > 0.0 else None
        payload = {
            "count": int(self.target_count),
            "tracking": bool(self.target_tracking),
            "age": age,
            "reason": self.target_reason,
            "delta": float(self.last_target_delta),
            "step": float(self.last_q_step),
            "servo": bool(self.servo_active),
            "q0": float(self.current_q[0]),
            "q1": float(self.current_q[1]),
        }
        msg = String()
        msg.data = dumps(payload)
        self.debug_pub.publish(msg)

        t = now()
        if t - self.last_debug_log > 1.0:
            self.last_debug_log = t
            self.node.get_logger().info(
                "robot_debug "
                f"count={payload['count']} tracking={payload['tracking']} "
                f"age={payload['age']} reason={payload['reason']} "
                f"delta={payload['delta']:.6f} step={payload['step']:.6f} "
                f"servo={payload['servo']}"
            )

    def close(self) -> None:
        self._servo_stop()
        if self.robot is not None:
            self.robot.close()


class TwinNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig):
        from std_msgs.msg import String
        global mujoco
        import mujoco
        import mujoco.viewer

        self.node = rclpy_node
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.state: Optional[dict] = None
        self.target: Optional[dict] = None
        self.target_mocap_id = self._find_target_mocap()
        self.gripper_substeps = max(1, round((1.0 / cfg.twin_hz) / self.model.opt.timestep))

        qos = latest_qos()
        self.state_sub = self.node.create_subscription(String, TOPIC_ROBOT_STATE, self._on_state, qos)
        self.target_sub = self.node.create_subscription(String, TOPIC_IK_TARGET, self._on_target, qos)
        self.timer = self.node.create_timer(1.0 / cfg.twin_hz, self._tick)
        self.node.get_logger().info(f"MuJoCo digital twin rendering at {cfg.twin_hz:.0f} Hz")

    def _find_target_mocap(self) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_target")
        if body_id < 0:
            return -1
        return int(self.model.body_mocapid[body_id])

    def _on_state(self, msg) -> None:
        try:
            self.state = loads(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad twin state: {exc}")

    def _on_target(self, msg) -> None:
        try:
            self.target = loads(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad twin target: {exc}")

    def _tick(self) -> None:
        if self.state is None:
            return
        try:
            q = as_vec(self.state.get("q"), 6)
            gripper_ctrl = float(self.state.get("gripper", 0.0)) * self.cfg.gripper_close_mj
            self.data.qpos[:6] = q
            self.data.qvel[:6] = 0.0
            if self.model.nu > 0:
                self.data.ctrl[: min(6, self.model.nu)] = q[: min(6, self.model.nu)]
            if self.model.nu > 6:
                self.data.ctrl[6] = gripper_ctrl
            self._update_target()
            for _ in range(self.gripper_substeps):
                self.data.qpos[:6] = q
                self.data.qvel[:6] = 0.0
                mujoco.mj_step(self.model, self.data)
            self.data.qpos[:6] = q
            self.data.qvel[:6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
        except Exception as exc:
            self.node.get_logger().warn(f"Twin update failed: {exc}")

    def _update_target(self) -> None:
        if self.target is None or self.target_mocap_id < 0:
            return
        pos = as_vec(self.target.get("pos"), 3)
        quat = as_vec(self.target.get("quat"), 4)
        self.data.mocap_pos[self.target_mocap_id] = pos
        self.data.mocap_quat[self.target_mocap_id] = np.roll(_norm_quat(quat), 1)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n
