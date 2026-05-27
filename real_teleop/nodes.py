from __future__ import annotations

import os
import threading
import traceback
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
from .messages import (
    as_vec,
    dumps,
    make_ik_target,
    make_joint_target,
    make_robot_state,
    make_vr_command,
    now,
    parse_joint_target,
    parse_ik_target,
    parse_robot_state,
    parse_vr_command,
)
from .ros_qos import latest_qos
from .safety import SafetyLimiter
from .vr import XrobotVrReader


class VrNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig):
        from std_msgs.msg import Float64MultiArray

        self.node = rclpy_node
        self.cfg = cfg
        self.reader = XrobotVrReader(cfg)
        self.pub = self.node.create_publisher(Float64MultiArray, TOPIC_VR_COMMAND, latest_qos())
        self.timer = self.node.create_timer(1.0 / cfg.vr_hz, self._tick)
        self.node.get_logger().info(f"VR node publishing typed commands at {cfg.vr_hz:.0f} Hz")

    def _tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        payload = self.reader.read()
        payload["stamp"] = now()
        msg = Float64MultiArray()
        msg.data = make_vr_command(payload)
        self.pub.publish(msg)


class IkNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig, *, enable_twin: bool = True):
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from std_msgs.msg import Float64MultiArray

        self.node = rclpy_node
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.safety = SafetyLimiter(cfg)
        from .kinematics import MinkIkSolver, RobotKinematics

        self.use_impedance_target = cfg.robot_control_mode == "impedance"
        self.pose_kin = RobotKinematics(cfg)
        self.ik = None if self.use_impedance_target else MinkIkSolver(cfg)
        self.home_tcp_pos, self.home_tcp_quat = self.pose_kin.forward(cfg.hardware_home_q)
        self.ctrl_filter = PoseFilter(cfg.ctrl_filter_alpha, cfg.ctrl_filter_alpha)
        self.target_filter = PoseFilter(cfg.target_filter_alpha, cfg.target_filter_alpha)

        self.state: Optional[dict] = None
        self.command: Optional[dict] = None
        self.anchor_ctrl_pos: Optional[np.ndarray] = None
        self.anchor_ctrl_quat: Optional[np.ndarray] = None
        self.anchor_tcp_pos: Optional[np.ndarray] = None
        self.anchor_tcp_quat: Optional[np.ndarray] = None
        self.last_target_pos: Optional[np.ndarray] = None
        self.last_target_quat: Optional[np.ndarray] = None
        self.last_q_target: Optional[np.ndarray] = None
        self.last_home = False
        self.twin_target: Optional[dict] = None

        self.receive_group = MutuallyExclusiveCallbackGroup()
        self.solve_group = MutuallyExclusiveCallbackGroup()
        self.twin_group = MutuallyExclusiveCallbackGroup()

        qos = latest_qos()
        self.target_pub = self.node.create_publisher(Float64MultiArray, TOPIC_JOINT_TARGET, qos)
        self.ik_target_pub = self.node.create_publisher(Float64MultiArray, TOPIC_IK_TARGET, qos)
        self.state_sub = self.node.create_subscription(
            Float64MultiArray,
            TOPIC_ROBOT_STATE,
            self._on_state,
            qos,
            callback_group=self.receive_group,
        )
        self.command_sub = self.node.create_subscription(
            Float64MultiArray,
            TOPIC_VR_COMMAND,
            self._on_command,
            qos,
            callback_group=self.receive_group,
        )
        self.timer = self.node.create_timer(self.dt, self._tick, callback_group=self.solve_group)

        self.twin_enabled = False
        self.mujoco = None
        self.twin_model = None
        self.twin_data = None
        self.twin_viewer = None
        self.target_mocap_id = -1
        self.gripper_substeps = 1
        if enable_twin:
            self._init_twin()
        self.node.get_logger().info(
            f"IK node running typed ROS2 {'pose targets' if self.use_impedance_target else 'mink'} at {cfg.control_hz:.0f} Hz"
            + (f", MuJoCo twin at {cfg.twin_hz:.0f} Hz" if self.twin_enabled else "")
            + (", fixed EE orientation from hardware home" if cfg.fixed_ee_orientation else "")
        )

    def _init_twin(self) -> None:
        try:
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise RuntimeError("No DISPLAY or WAYLAND_DISPLAY is set; MuJoCo viewer needs a graphical session.")
            import mujoco
            import mujoco.viewer

            self.mujoco = mujoco
            self.twin_model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
            self.twin_data = mujoco.MjData(self.twin_model)
            self.twin_viewer = mujoco.viewer.launch_passive(self.twin_model, self.twin_data)
            self.target_mocap_id = self._find_target_mocap()
            self.gripper_substeps = max(
                1,
                round((1.0 / self.cfg.twin_hz) / self.twin_model.opt.timestep),
            )
            self.twin_timer = self.node.create_timer(
                1.0 / self.cfg.twin_hz,
                self._tick_twin,
                callback_group=self.twin_group,
            )
            self.twin_enabled = True
        except Exception as exc:
            self.node.get_logger().error(f"MuJoCo twin disabled: {exc}")
            self.node.get_logger().debug(traceback.format_exc())

    def _find_target_mocap(self) -> int:
        if self.mujoco is None or self.twin_model is None:
            return -1
        body_id = self.mujoco.mj_name2id(self.twin_model, self.mujoco.mjtObj.mjOBJ_BODY, "right_target")
        if body_id < 0:
            return -1
        return int(self.twin_model.body_mocapid[body_id])

    def _on_state(self, msg) -> None:
        try:
            self.state = parse_robot_state(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad robot state: {exc}")

    def _on_command(self, msg) -> None:
        try:
            self.command = parse_vr_command(msg.data)
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
        self.last_target_quat = None
        self.last_q_target = None
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
        target_pos = (
            self.safety.clamp_impedance_workspace(target_pos)
            if self.use_impedance_target
            else self.safety.clamp_workspace(target_pos)
        )
        target_pos, target_quat = self._stabilize_target_pose(target_pos, target_quat)

        if self.use_impedance_target:
            self._publish_pose_target(target_pos, target_quat)
            return

        q_raw, ok = self.ik.solve(target_pos, target_quat, q, self.dt)
        q_safe = self.safety.clamp_joints(q_raw)
        q_safe = self._stabilize_joint_target(q_safe)
        q_delta = float(np.max(np.abs(q_safe[:6] - q)))
        self._publish_target(q_safe, target_pos, target_quat, ok, q_delta)

    def _publish_pose_target(self, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        from std_msgs.msg import Float64MultiArray

        self.twin_target = {"pos": target_pos.copy(), "quat": _norm_quat(target_quat)}
        target_msg = Float64MultiArray()
        target_msg.data = make_ik_target(target_pos, target_quat)
        self.ik_target_pub.publish(target_msg)

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
        self.last_target_quat = None
        self.last_q_target = None

    def _target_from_controller(self, ctrl_pos: np.ndarray, ctrl_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctrl_pos_f, ctrl_quat_f = self.ctrl_filter(ctrl_pos, ctrl_quat)
        dpos = (ctrl_pos_f - self.anchor_ctrl_pos) * self.cfg.scale
        drot = (R.from_quat(ctrl_quat_f) * R.from_quat(self.anchor_ctrl_quat).inv()).as_rotvec()
        if np.linalg.norm(dpos) < self.cfg.dead_zone_pos:
            dpos[:] = 0.0
        if np.linalg.norm(drot) < self.cfg.dead_zone_rot:
            drot[:] = 0.0
        pos = self.anchor_tcp_pos + dpos
        if self.cfg.fixed_ee_orientation:
            quat = self.anchor_tcp_quat.copy() if self.use_impedance_target else self.home_tcp_quat.copy()
        else:
            quat = (R.from_rotvec(drot) * R.from_quat(self.anchor_tcp_quat)).as_quat()
        return self.target_filter(pos, quat)

    def _stabilize_target_pose(self, target_pos: np.ndarray, target_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target_quat = _norm_quat(target_quat)
        if self.last_target_pos is None or self.last_target_quat is None:
            self.last_target_pos = target_pos.copy()
            self.last_target_quat = target_quat.copy()
            return target_pos, target_quat

        jump = target_pos - self.last_target_pos
        norm = float(np.linalg.norm(jump))
        if norm < self.cfg.target_pos_hold_epsilon:
            target_pos = self.last_target_pos.copy()
        elif norm > self.cfg.max_target_jump:
            target_pos = self.last_target_pos + jump / norm * self.cfg.max_target_jump

        rot_delta = (
            R.from_quat(target_quat) * R.from_quat(self.last_target_quat).inv()
        ).as_rotvec()
        if float(np.linalg.norm(rot_delta)) < self.cfg.target_rot_hold_epsilon:
            target_quat = self.last_target_quat.copy()

        self.last_target_pos = target_pos.copy()
        self.last_target_quat = target_quat.copy()
        return target_pos, target_quat

    def _stabilize_joint_target(self, q: np.ndarray) -> np.ndarray:
        if self.last_q_target is not None:
            if float(np.max(np.abs(q - self.last_q_target))) < self.cfg.ik_joint_deadband:
                return self.last_q_target.copy()
        self.last_q_target = q.copy()
        return q

    def _publish_hold(self, reason: str) -> None:
        from std_msgs.msg import Float64MultiArray

        gripper = float(self.command.get("gripper", 0.0)) if self.command else 0.0
        msg = Float64MultiArray()
        msg.data = make_joint_target(tracking=False, gripper=gripper, reason=reason)
        self.target_pub.publish(msg)

    def _publish_target(
        self,
        q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        ok: bool,
        q_delta: float,
    ) -> None:
        from std_msgs.msg import Float64MultiArray

        gripper = float(self.command.get("gripper", 0.0))
        msg = Float64MultiArray()
        msg.data = make_joint_target(
            tracking=True,
            q=q,
            gripper=gripper,
            reason="tracking",
            ok=ok,
            q_delta=q_delta,
        )
        self.target_pub.publish(msg)

        self.twin_target = {"pos": target_pos.copy(), "quat": _norm_quat(target_quat)}
        target_msg = Float64MultiArray()
        target_msg.data = make_ik_target(target_pos, target_quat)
        self.ik_target_pub.publish(target_msg)

    def _tick_twin(self) -> None:
        if not self.twin_enabled or self.state is None:
            return
        try:
            q = as_vec(self.state.get("q"), 6)
            gripper_ctrl = float(self.state.get("gripper", 0.0)) * self.cfg.gripper_close_mj
            self.twin_data.qpos[:6] = q
            self.twin_data.qvel[:6] = 0.0
            if self.twin_model.nu > 0:
                n = min(6, self.twin_model.nu)
                self.twin_data.ctrl[:n] = q[:n]
            if self.twin_model.nu > 6:
                self.twin_data.ctrl[6] = gripper_ctrl
            self._update_twin_target()
            for _ in range(self.gripper_substeps):
                self.twin_data.qpos[:6] = q
                self.twin_data.qvel[:6] = 0.0
                self.mujoco.mj_step(self.twin_model, self.twin_data)
            self.twin_data.qpos[:6] = q
            self.twin_data.qvel[:6] = 0.0
            self.mujoco.mj_forward(self.twin_model, self.twin_data)
            self.twin_viewer.sync()
        except Exception as exc:
            self.node.get_logger().warn(f"Twin update failed: {exc}")

    def _update_twin_target(self) -> None:
        if self.twin_target is None or self.target_mocap_id < 0:
            return
        pos = as_vec(self.twin_target.get("pos"), 3)
        quat = as_vec(self.twin_target.get("quat"), 4)
        self.twin_data.mocap_pos[self.target_mocap_id] = pos
        self.twin_data.mocap_quat[self.target_mocap_id] = np.roll(_norm_quat(quat), 1)

    def close(self) -> None:
        if self.twin_viewer is not None:
            self.twin_viewer.close()


class RobotNode:
    def __init__(self, rclpy_node, cfg: TeleopConfig, *, dry_run: bool = False):
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from std_msgs.msg import Float64MultiArray, String

        self.node = rclpy_node
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.dry_run = dry_run
        self.control_mode = cfg.robot_control_mode
        self.safety = SafetyLimiter(cfg)
        from .kinematics import RobotKinematics

        self.kin = RobotKinematics(cfg)
        self.impedance_target_filter = PoseFilter(cfg.impedance_target_alpha, cfg.impedance_target_alpha)
        qos = latest_qos()
        self.data_lock = threading.Lock()
        self.receive_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.state_group = MutuallyExclusiveCallbackGroup()
        self.actual_group = MutuallyExclusiveCallbackGroup()
        self.gripper_group = MutuallyExclusiveCallbackGroup()
        self.pub = self.node.create_publisher(Float64MultiArray, TOPIC_ROBOT_STATE, qos)
        self.debug_pub = self.node.create_publisher(String, TOPIC_ROBOT_DEBUG, 10)
        self.target_sub = self.node.create_subscription(
            Float64MultiArray,
            TOPIC_JOINT_TARGET,
            self._on_target,
            qos,
            callback_group=self.receive_group,
        )
        self.pose_target_sub = self.node.create_subscription(
            Float64MultiArray,
            TOPIC_IK_TARGET,
            self._on_pose_target,
            qos,
            callback_group=self.receive_group,
        )
        self.command_sub = self.node.create_subscription(
            Float64MultiArray,
            TOPIC_VR_COMMAND,
            self._on_command,
            qos,
            callback_group=self.receive_group,
        )

        self.robot = None
        self.impedance_motion = None
        self.impedance_configured = False
        self.current_q = self._home_q()
        self.target_q: Optional[np.ndarray] = None
        self.filtered_target_q: Optional[np.ndarray] = None
        self.target_tcp_pos: Optional[np.ndarray] = None
        self.target_tcp_quat: Optional[np.ndarray] = None
        self.target_pose_stamp = 0.0
        self.soft_hold_active = False
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
            self._set_current_q(self.robot.get_joint_positions())
            if self.control_mode == "impedance":
                self._init_impedance_motion()

        self.control_timer = self.node.create_timer(self.dt, self._control_tick, callback_group=self.control_group)
        self.state_timer = self.node.create_timer(
            1.0 / cfg.robot_state_hz,
            self._publish_state,
            callback_group=self.state_group,
        )
        self.gripper_timer = self.node.create_timer(
            1.0 / cfg.gripper_hz,
            self._apply_gripper,
            callback_group=self.gripper_group,
        )
        self.actual_timer = None
        if not dry_run:
            self.actual_timer = self.node.create_timer(
                1.0 / cfg.actual_read_hz,
                self._read_actual_state,
                callback_group=self.actual_group,
            )
        mode = "dry-run" if dry_run else f"real robot {cfg.robot_ip}"
        self.node.get_logger().info(
            f"Robot node {self.control_mode} controlling {mode} at {cfg.control_hz:.0f} Hz, "
            f"state publishing at {cfg.robot_state_hz:.0f} Hz"
        )

    def _init_impedance_motion(self) -> None:
        from real_teleop.impedance import DEFAULT_IMPEDANCE_TEST_CONFIG, ImpedanceRuntimeConfig, RtdeImpedanceMotion

        profile = DEFAULT_IMPEDANCE_TEST_CONFIG.profiles.get(self.cfg.impedance_profile)
        if profile is None:
            raise ValueError(f"Unknown impedance profile: {self.cfg.impedance_profile}")
        runtime = ImpedanceRuntimeConfig(
            robot_ip=self.cfg.robot_ip,
            control_hz=self.cfg.control_hz,
            ramp_duration_s=DEFAULT_IMPEDANCE_TEST_CONFIG.ramp_duration_s,
            state_source=self.cfg.impedance_state_source,
            max_fk_rtde_delta_m=self.cfg.impedance_max_fk_rtde_delta_m,
            move_home_first=False,
            zero_ft_sensor=DEFAULT_IMPEDANCE_TEST_CONFIG.zero_ft_sensor,
            payload_mass_kg=DEFAULT_IMPEDANCE_TEST_CONFIG.payload_mass_kg,
            payload_cog_m=DEFAULT_IMPEDANCE_TEST_CONFIG.payload_cog_m,
        )
        self.impedance_motion = RtdeImpedanceMotion(
            self.robot,
            profile,
            runtime=runtime,
            mode=self.cfg.impedance_profile,
            kinematics=self.kin if runtime.state_source == "jacobian" else None,
        )
        self.impedance_motion.assert_state_source_aligned()

    def _home_q(self) -> np.ndarray:
        return np.asarray(self.cfg.hardware_home_q, dtype=float).copy()

    def _get_current_q(self) -> np.ndarray:
        with self.data_lock:
            return self.current_q.copy()

    def _set_current_q(self, q: np.ndarray) -> None:
        with self.data_lock:
            self.current_q = np.asarray(q, dtype=float).copy()

    def _on_target(self, msg) -> None:
        try:
            payload = parse_joint_target(msg.data)
            if self.control_mode == "impedance":
                self.desired_gripper = float(np.clip(payload.get("gripper", self.desired_gripper), 0.0, 1.0))
                if not bool(payload.get("tracking", False)):
                    self.target_tracking = False
                    self.target_reason = str(payload.get("reason", "hold"))
                    self.impedance_target_filter.reset()
                    if not (self.servo_active or self.soft_hold_active or self.target_pose_stamp > 0.0):
                        self.target_tcp_pos = None
                        self.target_tcp_quat = None
                return
            self.target_tracking = bool(payload.get("tracking", False))
            self.target_stamp = float(payload.get("stamp", 0.0))
            self.target_reason = str(payload.get("reason", "tracking" if self.target_tracking else "hold"))
            self.target_count += 1
            q = payload.get("q")
            if q is not None:
                q = as_vec(q, 6)
                if self.safety.check_joints(q):
                    self.target_q = q
                    self.last_target_delta = float(np.max(np.abs(q - self._get_current_q())))
                    if self.filtered_target_q is None:
                        self.filtered_target_q = q.copy()
                    else:
                        alpha = float(np.clip(self.cfg.joint_target_alpha, 0.0, 1.0))
                        self.filtered_target_q = alpha * q + (1.0 - alpha) * self.filtered_target_q
            else:
                self.target_q = None
                self.filtered_target_q = None
            self.desired_gripper = float(np.clip(payload.get("gripper", self.desired_gripper), 0.0, 1.0))
        except Exception as exc:
            self.node.get_logger().warn(f"Bad joint target: {exc}")

    def _on_pose_target(self, msg) -> None:
        if self.control_mode != "impedance":
            return
        try:
            payload = parse_ik_target(msg.data)
            pos = self.safety.clamp_impedance_workspace(as_vec(payload.get("pos"), 3))
            quat = _norm_quat(as_vec(payload.get("quat"), 4))
            pos, quat = self.impedance_target_filter(pos, quat)
            self.target_tcp_pos = pos
            self.target_tcp_quat = quat
            self.target_pose_stamp = float(payload.get("stamp", now()))
            self.soft_hold_active = False
            self.target_stamp = self.target_pose_stamp
            self.target_tracking = True
            self.target_reason = "tracking"
            self.target_count += 1
        except Exception as exc:
            self.node.get_logger().warn(f"Bad impedance pose target: {exc}")

    def _on_command(self, msg) -> None:
        try:
            payload = parse_vr_command(msg.data)
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
        self.target_tcp_pos = None
        self.target_tcp_quat = None
        self.target_pose_stamp = 0.0
        self.target_stamp = 0.0
        self.soft_hold_active = False
        self.impedance_target_filter.reset()
        self._servo_stop()
        self._force_stop()
        self.home_thread = threading.Thread(target=self._home_worker, daemon=True)
        self.home_thread.start()

    def _home_worker(self) -> None:
        try:
            if self.dry_run:
                self.node.get_logger().info("A pressed: dry-run moveJ-style return to hardware home")
                self._dry_run_move_to_home()
                return
            self.node.get_logger().info("A pressed: moveJ return to home")
            move_to_home = getattr(self.robot, "move_to_home", self.robot.go_home)
            move_to_home()
            self._set_current_q(self.robot.get_joint_positions())
        except Exception as exc:
            self.node.get_logger().error(f"Home move failed: {exc}")
        finally:
            self.homing = False

    def _dry_run_move_to_home(self) -> None:
        start_q = self._get_current_q()
        target_q = self._home_q()
        duration = max(self.cfg.home_move_duration_s, self.dt)
        steps = max(1, int(round(duration / self.dt)))
        import time

        for i in range(steps):
            if not self.homing:
                return
            s = float(i + 1) / float(steps)
            s = s * s * (3.0 - 2.0 * s)
            self._set_current_q((1.0 - s) * start_q + s * target_q)
            time.sleep(self.dt)

    def _control_tick(self) -> None:
        if self.control_mode == "impedance":
            self._apply_impedance()
        else:
            self._apply_servo()

    def _read_actual_state(self) -> None:
        if self.dry_run or self.robot is None:
            return
        try:
            actual_q = self.robot.get_joint_positions()
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to read robot joints: {exc}")
            return
        if self.control_mode == "impedance" or self.homing or not self.servo_active:
            self._set_current_q(actual_q)

    def _apply_servo(self) -> None:
        fresh = now() - self.target_stamp <= self.cfg.stale_target_s
        if self.homing or not self.target_tracking or self.target_q is None or not fresh:
            self._servo_stop()
            return
        target_q = self.filtered_target_q if self.filtered_target_q is not None else self.target_q
        current_q = self._get_current_q()
        q_cmd = self.safety.limit_step(current_q, target_q, self.dt)
        self.last_target_delta = float(np.max(np.abs(target_q - current_q)))
        self.last_q_step = float(np.max(np.abs(q_cmd - current_q)))
        if self.dry_run:
            self._set_current_q(q_cmd)
            self.servo_active = True
            return
        try:
            self.robot.servo_joints(q_cmd)
            self._set_current_q(q_cmd)
            self.servo_active = True
        except Exception as exc:
            self.node.get_logger().error(f"servoJ failed: {exc}")
            self._servo_stop()

    def _servo_stop(self) -> None:
        if self.control_mode == "impedance":
            return
        if not self.servo_active:
            return
        if not self.dry_run:
            try:
                self.robot.servo_stop()
            except Exception as exc:
                self.node.get_logger().warn(f"servoStop failed: {exc}")
        self.servo_active = False

    def _apply_impedance(self) -> None:
        if self.homing:
            self._force_stop()
            return

        fresh = now() - self.target_pose_stamp <= self.cfg.stale_target_s
        if not self.target_tracking or self.target_tcp_pos is None or self.target_tcp_quat is None or not fresh:
            reason = "stale_vr" if self.target_tracking and not fresh else self.target_reason
            had_control = self.target_pose_stamp > 0.0 or self.servo_active or self.soft_hold_active
            if self.cfg.impedance_soft_hold_on_lost and had_control:
                self._apply_soft_hold(reason)
            else:
                if not fresh and self.target_tracking:
                    self.target_tracking = False
                    self.target_reason = "stale_vr"
                self._force_stop()
            return

        target_rotvec = R.from_quat(_norm_quat(self.target_tcp_quat)).as_rotvec()
        self.soft_hold_active = False
        if self.dry_run:
            self.servo_active = True
            self.last_target_delta = 0.0
            self.last_q_step = 0.0
            return
        if self.impedance_motion is None:
            self.node.get_logger().error("Impedance motion is not initialized.")
            return

        try:
            if not self.impedance_configured:
                self.impedance_motion.configure_force_mode()
                self.impedance_configured = True
            self.impedance_motion.set_target_pose(
                self.target_tcp_pos,
                target_rotvec,
                reset_ramp=not self.servo_active,
            )
            command = self.impedance_motion.step(execute=True)
            self.last_target_delta = float(np.linalg.norm(command.position_error))
            self.last_q_step = float(np.linalg.norm(command.wrench[:3]))
            self.servo_active = True
        except Exception as exc:
            self.node.get_logger().error(f"impedance forceMode failed: {exc}")
            self._force_stop()

    def _apply_soft_hold(self, reason: str) -> None:
        self.target_tracking = False
        self.target_reason = reason
        self.impedance_target_filter.reset()

        if self.dry_run:
            self.servo_active = False
            return
        if self.impedance_motion is None:
            self._force_stop()
            return

        try:
            if not self.impedance_configured:
                self.impedance_motion.configure_force_mode()
                self.impedance_configured = True
            reset_ramp = not self.servo_active or not self.soft_hold_active
            if self.target_tcp_pos is None or self.target_tcp_quat is None:
                state = self.impedance_motion.read_state()
                self.target_tcp_pos = state.position.copy()
                self.target_tcp_quat = R.from_rotvec(state.rotation_vector).as_quat()
            self.soft_hold_active = True
            target_rotvec = R.from_quat(_norm_quat(self.target_tcp_quat)).as_rotvec()
            self.impedance_motion.set_target_pose(
                self.target_tcp_pos,
                target_rotvec,
                reset_ramp=reset_ramp,
            )
            command = self.impedance_motion.step(execute=True)
            self.last_target_delta = float(np.linalg.norm(command.position_error))
            self.last_q_step = float(np.linalg.norm(command.wrench[:3]))
            self.servo_active = True
        except Exception as exc:
            self.node.get_logger().warn(f"impedance soft-hold failed: {exc}")
            self._force_stop()

    def _force_stop(self) -> None:
        if self.control_mode != "impedance" or not self.servo_active:
            return
        if not self.dry_run and self.impedance_motion is not None:
            self.impedance_motion.stop()
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
        from std_msgs.msg import Float64MultiArray

        q = self._get_current_q()
        try:
            if self.control_mode == "impedance" and not self.dry_run and self.robot is not None:
                pose = self.robot.get_tcp_pose()
                tcp_pos = pose[:3]
                tcp_quat = R.from_rotvec(pose[3:]).as_quat()
            else:
                tcp_pos, tcp_quat = self.kin.forward(q)
        except Exception as exc:
            self.node.get_logger().warn(f"TCP state failed: {exc}")
            return
        age = float(now() - self.target_stamp) if self.target_stamp > 0.0 else -1.0
        msg = Float64MultiArray()
        msg.data = make_robot_state(
            q=q,
            tcp_pos=tcp_pos,
            tcp_quat=tcp_quat,
            gripper=self.desired_gripper,
            servo_active=self.servo_active,
            homing=self.homing,
            target_tracking=self.target_tracking,
            target_age=age,
        )
        self.pub.publish(msg)
        if now() - self.last_debug_pub >= 0.1:
            self.last_debug_pub = now()
            self._publish_debug()

    def _publish_debug(self) -> None:
        from std_msgs.msg import String

        age = float(now() - self.target_stamp) if self.target_stamp > 0.0 else None
        q = self._get_current_q()
        payload = {
            "count": int(self.target_count),
            "tracking": bool(self.target_tracking),
            "age": age,
            "reason": self.target_reason,
            "delta": float(self.last_target_delta),
            "step": float(self.last_q_step),
            "servo": bool(self.servo_active),
            "soft_hold": bool(self.soft_hold_active),
            "mode": self.control_mode,
            "q0": float(q[0]),
            "q1": float(q[1]),
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
                f"servo={payload['servo']} soft_hold={payload['soft_hold']} mode={payload['mode']}"
            )

    def close(self) -> None:
        self._servo_stop()
        self._force_stop()
        if self.robot is not None:
            self.robot.close()


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n
