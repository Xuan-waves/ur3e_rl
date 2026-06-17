#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.messages import make_ik_target, make_vr_command, parse_robot_state, parse_vr_command
from real_teleop.safety import SafetyLimiter
from scripts.rollout_smolvla_no_rotvec.config import CFG
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_rtc import (
    home_orientation,
    infer_no_rotvec_position_mode,
    radial_deadzone,
    read_policy_feature_dims,
    robot_state_to_no_rotvec_tensor,
)
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import (
    ActionPacket,
    ObservationPacket,
    SmolVLASyncRunner,
    SyncBuffer,
    fmt_vec,
    image_msg_to_tensor,
    make_preview_frame,
    resolve_policy_path,
    run_pre_model_startup_sequence,
    stamp_to_sec,
)


DEFAULT_POLICY_PATH = (
    REPO_ROOT
    / "outputs"
    / "rlt_vla"
    / "ur3e_smolvla_0612"
    / "checkpoints"
    / "030000"
    / "pretrained_model"
)


def read_policy_n_obs_steps(policy_path: Path) -> int:
    config_path = policy_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing policy config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return max(1, int(config.get("n_obs_steps", 1)))


class NoRotvecSmolVLASyncRollout:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.policy_path = resolve_policy_path(args.policy_path)
        self.policy_state_dim, self.policy_action_dim = read_policy_feature_dims(self.policy_path)
        model_n_obs_steps = read_policy_n_obs_steps(self.policy_path)
        if args.n_obs_steps is None:
            self.n_obs_steps = model_n_obs_steps
            self.n_obs_steps_source = "model"
        else:
            self.n_obs_steps = max(1, int(args.n_obs_steps))
            self.n_obs_steps_source = f"manual(model={model_n_obs_steps})"
        if args.action_position_mode == "auto":
            inferred_mode, inferred_source = infer_no_rotvec_position_mode(self.policy_path)
            args.action_position_mode = inferred_mode or "absolute"
            args.action_position_mode_source = inferred_source or "fallback:absolute"
        else:
            args.action_position_mode_source = "manual"
        self.sync_reference = str(args.sync_reference).lower()
        if self.sync_reference not in {"front", "wrist", "timer"}:
            raise ValueError("--sync-reference must be front, wrist, or timer")

        self.teleop_cfg = TeleopConfig()
        self.safety = SafetyLimiter(self.teleop_cfg)
        self.fixed_quat, self.fixed_rotvec = home_orientation()
        self.pending_reference_stamps: queue.SimpleQueue[float] = queue.SimpleQueue()
        self.action_queue: deque[ActionPacket] = deque()
        self.action_lock = threading.Lock()
        self.current_action: ActionPacket | None = None
        self.infer_lock = threading.Lock()
        self.infer_busy = False
        self.last_action_step_time = 0.0
        self.last_published_pos: np.ndarray | None = None
        self.last_published_gripper: float | None = None
        self.last_log_time = 0.0
        self.last_dt_map: dict[str, float | None] = {}
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=self.n_obs_steps)
        self.vr_command: dict[str, Any] | None = None
        self.vr_recv_mono = 0.0
        self.vr_override_active = False
        self.vr_anchor_ctrl_pos: np.ndarray | None = None
        self.vr_anchor_tcp_pos: np.ndarray | None = None
        self.vr_anchor_gripper = 0.0
        self.vr_anchor_trigger = 0.0
        self.model_resume_block_until = 0.0
        self.topic_counts = {"front": 0, "wrist": 0, "state": 0}
        self.topic_last = {"front": 0.0, "wrist": 0.0, "state": 0.0}
        self.preview_stop = threading.Event()
        self.preview_failed = False
        self.preview_thread: threading.Thread | None = None

        self.buffers = {
            "front": SyncBuffer(args.buffer_maxlen),
            "wrist": SyncBuffer(args.buffer_maxlen),
            "state": SyncBuffer(args.buffer_maxlen),
        }
        self.runner = SmolVLASyncRunner(
            self.policy_path,
            args.device,
            execution_horizon=args.execution_horizon,
            replan_every_step=args.replan_every_step,
            amp=args.amp,
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        data_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.callback_group = ReentrantCallbackGroup()
        self.ik_target_pub = node.create_publisher(Float64MultiArray, args.ik_target_topic, data_qos)
        self.command_pub = node.create_publisher(Float64MultiArray, args.vr_command_topic, data_qos)
        self.subs = [
            node.create_subscription(Image, args.front_image_topic, self._on_front, image_qos, callback_group=self.callback_group),
            node.create_subscription(Image, args.wrist_image_topic, self._on_wrist, image_qos, callback_group=self.callback_group),
            node.create_subscription(
                Float64MultiArray,
                args.robot_state_topic,
                self._on_state,
                data_qos,
                callback_group=self.callback_group,
            ),
        ]
        if args.vr_override:
            self.subs.append(
                node.create_subscription(
                    Float64MultiArray,
                    args.vr_raw_topic,
                    self._on_raw_vr,
                    data_qos,
                    callback_group=self.callback_group,
                )
            )
        self.infer_timer = node.create_timer(1.0 / max(args.fps, 1.0), self._infer_tick, callback_group=self.callback_group)
        self.publish_timer = node.create_timer(
            1.0 / max(args.command_hz, 1.0),
            self._publish_tick,
            callback_group=self.callback_group,
        )
        if args.preview:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()

        mode = "EXECUTE" if self.execute else "DRY-RUN"
        node.get_logger().info(
            f"UR3e no-rotvec SmolVLA SYNC rollout ready ({mode}). policy={self.policy_path}, "
            f"task={args.task!r}, state_dim={self.policy_state_dim}, action_dim={self.policy_action_dim}, "
            f"action_position_mode={args.action_position_mode}, "
            f"action_mode_source={args.action_position_mode_source}, fps={args.fps:.1f}, "
            f"command_hz={args.command_hz:.1f}, action_step_hz={args.action_step_hz:.1f}, "
            f"n_obs_steps={self.n_obs_steps}({self.n_obs_steps_source}), "
            f"sync_reference={self.sync_reference}, horizon={self.runner.execution_horizon}/"
            f"{self.runner.ckpt_action_horizon}, replan_every_step={args.replan_every_step}, "
            f"fixed_home_rotvec={fmt_vec(self.fixed_rotvec)}, "
            f"vr_override={args.vr_override} raw_topic={args.vr_raw_topic}"
        )

    def close(self) -> None:
        self.preview_stop.set()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)
        if not self.preview_failed:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    def now_sec(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _msg_time_sec(self, msg: Any) -> float:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        return self.now_sec() if stamp is None else stamp_to_sec(stamp)

    def _push(self, key: str, stamp: float, value: Any) -> None:
        self.buffers[key].push(stamp, value)
        self.topic_counts[key] += 1
        self.topic_last[key] = self.now_sec()

    def _on_front(self, msg) -> None:
        stamp = self._msg_time_sec(msg)
        self._push("front", stamp, msg)
        if self.sync_reference == "front":
            self.pending_reference_stamps.put(stamp)

    def _on_wrist(self, msg) -> None:
        stamp = self._msg_time_sec(msg)
        self._push("wrist", stamp, msg)
        if self.sync_reference == "wrist":
            self.pending_reference_stamps.put(stamp)

    def _on_state(self, msg) -> None:
        try:
            self._push("state", self.now_sec(), parse_robot_state(msg.data))
        except Exception as exc:
            self.node.get_logger().warn(f"Bad robot_state: {exc}")

    def _on_raw_vr(self, msg) -> None:
        try:
            self.vr_command = parse_vr_command(msg.data)
            self.vr_recv_mono = time.monotonic()
        except Exception as exc:
            self.node.get_logger().warn(f"Bad raw VR command: {exc}")

    def _take_reference_stamp(self) -> float | None:
        if self.sync_reference == "timer":
            return self.now_sec()
        stamp = None
        while True:
            try:
                stamp = self.pending_reference_stamps.get_nowait()
            except queue.Empty:
                break
        return stamp

    def _infer_tick(self) -> None:
        with self.action_lock:
            if len(self.action_queue) > int(self.args.prefetch_actions):
                return
        if self.now_sec() < self.model_resume_block_until:
            return
        if self.infer_busy:
            return
        stamp = self._take_reference_stamp()
        if stamp is None:
            self._log_wait("waiting for reference image")
            return
        packet = self._build_observation(stamp)
        if packet is None:
            return
        count = 1 if self.args.replan_every_step else self.runner.execution_horizon
        self.infer_busy = True
        threading.Thread(
            target=self._run_inference,
            args=(packet, count),
            name="no_rotvec_smolvla_sync_inference",
            daemon=True,
        ).start()

    def _run_inference(self, packet: ObservationPacket, count: int) -> None:
        try:
            actions = self.runner.infer_sequence(packet, count)
        except Exception as exc:
            self.node.get_logger().error(f"SmolVLA sync inference failed: {exc}")
            self.infer_busy = False
            return
        if self.vr_override_active or self.now_sec() < self.model_resume_block_until:
            self.infer_busy = False
            return
        with self.action_lock:
            if self.args.replace_queue_on_infer:
                self.action_queue.clear()
            self.action_queue.extend(actions)
            ready_stamp = self.now_sec()
            for action in self.action_queue:
                action.ready_stamp = ready_stamp
        self.infer_busy = False
        self._log_info(
            f"sync infer actions={len(actions)} infer={actions[0].inference_s * 1000.0:.0f}ms "
            f"{self._format_status()}"
        )

    def _build_observation(self, stamp: float) -> ObservationPacket | None:
        front, dt_front = self.buffers["front"].nearest(stamp, self.args.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.args.max_dt_wrist_image)
        state, dt_state = self.buffers["state"].nearest(stamp, self.args.max_dt_state)
        self.last_dt_map = {"front": dt_front, "wrist": dt_wrist, "state": dt_state}
        if front is None or wrist is None or state is None:
            self._log_wait(f"sync miss {self._format_status()}")
            return None
        try:
            frame_batch = {
                "observation.images.cam_front": image_msg_to_tensor(front),
                "observation.images.cam_wrist": image_msg_to_tensor(wrist),
                "observation.state": robot_state_to_no_rotvec_tensor(
                    state,
                    rl_mark=self.args.rl_mark,
                    gripper_max=self.args.gripper_max,
                    state_dim=self.policy_state_dim,
                ),
            }
            self.observation_history.append(frame_batch)
            history = list(self.observation_history)
            if len(history) < self.n_obs_steps:
                history = [history[0]] * (self.n_obs_steps - len(history)) + history
            batch = self._stack_observation_history(history)
            batch["task"] = self.args.task
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build no-rotvec observation: {exc}")
            return None
        return ObservationPacket(stamp=stamp, batch=batch, dt_map=self.last_dt_map.copy())

    def _stack_observation_history(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            "observation.images.cam_front": torch.stack(
                [item["observation.images.cam_front"] for item in history],
                dim=0,
            ).unsqueeze(0),
            "observation.images.cam_wrist": torch.stack(
                [item["observation.images.cam_wrist"] for item in history],
                dim=0,
            ).unsqueeze(0),
            "observation.state": torch.stack(
                [item["observation.state"] for item in history],
                dim=0,
            ).unsqueeze(0),
        }

    def _publish_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        stamp = self.now_sec()
        state, state_age = self.buffers["state"].latest(now=stamp)
        if state is None or state_age is None or state_age > self.args.max_dt_state:
            self._log_wait(f"holding: no fresh robot state age={state_age}")
            return
        current_pos = np.asarray(state["tcp_pos"], dtype=float).reshape(3)
        current_gripper = float(np.clip(state.get("gripper", 0.0), 0.0, self.args.gripper_max))

        override = self._vr_override_target(current_pos=current_pos, current_gripper=current_gripper)
        source = "model"
        action_age = 0.0
        if override is not None:
            target_pos, gripper = override
            source = "vr_override"
        else:
            with self.action_lock:
                if self.action_queue and stamp - self.last_action_step_time >= 1.0 / max(self.args.action_step_hz, 1e-6):
                    self.current_action = self.action_queue.popleft()
                    self.last_action_step_time = stamp
                packet = self.current_action
            if packet is None:
                return
            action_age = stamp - packet.ready_stamp
            if action_age > self.args.max_action_age_s:
                self._log_wait(f"drop stale action age={action_age:.3f}s")
                with self.action_lock:
                    self.current_action = None
                return
            action = np.asarray(packet.action, dtype=np.float32).reshape(-1)
            if action.size < 4 or action.size < self.policy_action_dim:
                self.node.get_logger().warn(
                    f"Bad no-rotvec action shape: {action.shape}, expected at least {self.policy_action_dim}"
                )
                return
            target_pos, gripper = self._decode_and_smooth_action(action, current_pos=current_pos)
            action_age = stamp - packet.ready_stamp
        if self.execute:
            pose_msg = Float64MultiArray()
            pose_msg.data = make_ik_target(target_pos, self.fixed_quat)
            self.ik_target_pub.publish(pose_msg)

            command_msg = Float64MultiArray()
            command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
            self.command_pub.publish(command_msg)

        prefix = "publish" if self.execute else "dry-run"
        self._log_info(
            f"{prefix} source={source} pos={fmt_vec(target_pos)} fixed_rot={fmt_vec(self.fixed_rotvec)} "
            f"gripper={gripper:.3f} queue={len(self.action_queue)} action_age={action_age * 1000.0:.0f}ms"
        )

    def _vr_override_target(
        self,
        *,
        current_pos: np.ndarray,
        current_gripper: float,
    ) -> tuple[np.ndarray, float] | None:
        if not self.args.vr_override or self.vr_command is None:
            self._release_vr_override(current_pos)
            return None
        if time.monotonic() - self.vr_recv_mono > self.args.vr_override_stale_s:
            self._release_vr_override(current_pos)
            return None

        pose = self.vr_command.get("pose")
        enabled = bool(self.vr_command.get("enable", False))
        if not enabled or pose is None:
            self._release_vr_override(current_pos)
            return None

        ctrl_pose = np.asarray(pose, dtype=float).reshape(7)
        ctrl_pos = self._vr_control_pos(ctrl_pose[:3], ctrl_pose[3:])
        raw_gripper = float(np.clip(self.vr_command.get("gripper", 0.0), 0.0, self.args.gripper_max))

        if not self.vr_override_active:
            self.vr_override_active = True
            self.vr_anchor_ctrl_pos = ctrl_pos.copy()
            self.vr_anchor_tcp_pos = np.asarray(current_pos, dtype=float).reshape(3).copy()
            self.vr_anchor_gripper = float(np.clip(current_gripper, 0.0, self.args.gripper_max))
            self.vr_anchor_trigger = raw_gripper
            self.last_published_pos = self.vr_anchor_tcp_pos.copy()
            self.last_published_gripper = self.vr_anchor_gripper
            self._drain_action_queue()
            self.node.get_logger().info("VR override engaged; anchored controller to current TCP.")

        sign = np.asarray(self.teleop_cfg.vr_control_position_sign, dtype=float)
        if sign.shape != (3,):
            sign = np.ones(3, dtype=float)
        dpos = (ctrl_pos - self.vr_anchor_ctrl_pos) * sign * float(self.teleop_cfg.scale)
        dpos = radial_deadzone(dpos, float(self.teleop_cfg.dead_zone_pos))
        pos = self.vr_anchor_tcp_pos + dpos
        pos = self.safety.clamp_impedance_workspace(pos)
        min_action_z = float(self.args.min_action_z)
        if np.isfinite(min_action_z):
            pos[2] = max(float(pos[2]), min_action_z)
        pos = self._smooth_position(pos)

        gripper = self.vr_anchor_gripper + (raw_gripper - self.vr_anchor_trigger) * float(self.args.vr_override_gripper_gain)
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * gripper + (1.0 - alpha_g) * self.last_published_gripper
        self.last_published_gripper = gripper
        return pos.astype(np.float32), gripper

    def _release_vr_override(self, current_pos: np.ndarray) -> None:
        if not self.vr_override_active:
            return
        self.vr_override_active = False
        self.vr_anchor_ctrl_pos = None
        self.vr_anchor_tcp_pos = None
        self.last_published_pos = np.asarray(current_pos, dtype=float).reshape(3).copy()
        self._drain_action_queue()
        self.model_resume_block_until = self.now_sec() + float(self.args.vr_override_resume_delay_s)
        self.node.get_logger().info("VR override released; model queue drained and rollout re-anchored to current TCP.")

    def _drain_action_queue(self) -> None:
        with self.action_lock:
            self.action_queue.clear()
            self.current_action = None

    def _vr_control_pos(self, ctrl_pos: np.ndarray, ctrl_quat: np.ndarray) -> np.ndarray:
        pos = np.asarray(ctrl_pos, dtype=float).reshape(3)
        offset = np.asarray(self.teleop_cfg.vr_controller_pivot_offset_m, dtype=float)
        if offset.shape != (3,) or float(np.linalg.norm(offset)) < 1e-9:
            return pos.copy()
        quat = np.asarray(ctrl_quat, dtype=float).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return pos.copy()
        return pos + R.from_quat(quat / norm).apply(offset)

    def _smooth_position(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float).reshape(3)
        if self.last_published_pos is not None:
            alpha = float(np.clip(self.args.action_pose_filter_alpha, 0.0, 1.0))
            pos = alpha * pos + (1.0 - alpha) * self.last_published_pos
            delta = pos - self.last_published_pos
            norm = float(np.linalg.norm(delta))
            max_step = float(max(self.args.max_action_pos_step, 1e-6))
            if norm > max_step:
                pos = self.last_published_pos + delta / norm * max_step
        self.last_published_pos = pos.copy()
        return pos

    def _decode_and_smooth_action(self, action: np.ndarray, *, current_pos: np.ndarray) -> tuple[np.ndarray, float]:
        raw_pos = np.asarray(action[:3], dtype=float).reshape(3)
        if self.args.action_position_mode == "relative":
            pos = np.asarray(current_pos, dtype=float).reshape(3) + raw_pos
        elif self.args.action_position_mode == "absolute":
            pos = raw_pos
        else:
            raise ValueError(f"Unsupported action_position_mode: {self.args.action_position_mode!r}")
        pos = self.safety.clamp_impedance_workspace(pos)
        min_action_z = float(self.args.min_action_z)
        if np.isfinite(min_action_z):
            pos[2] = max(float(pos[2]), min_action_z)
        gripper = float(np.clip(action[3], 0.0, self.args.gripper_max))

        pos = self._smooth_position(pos)

        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * gripper + (1.0 - alpha_g) * self.last_published_gripper
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        self.last_published_gripper = gripper
        return pos.astype(np.float32), gripper

    def _preview_loop(self) -> None:
        try:
            import cv2

            cv2.namedWindow("UR3e No-Rotvec SmolVLA SYNC Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e No-Rotvec SmolVLA SYNC Rollout", 1280, 480)
            self.node.get_logger().info("OpenCV no-rotvec sync rollout preview window started.")
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
            return

        period = 1.0 / max(self.args.preview_hz, 1.0)
        while not self.preview_stop.is_set() and not self.preview_failed:
            try:
                import cv2

                now = self.now_sec()
                front, front_age = self.buffers["front"].latest(now=now)
                wrist, wrist_age = self.buffers["wrist"].latest(now=now)
                cv2.imshow(
                    "UR3e No-Rotvec SmolVLA SYNC Rollout",
                    make_preview_frame(front, wrist, front_age, wrist_age),
                )
                cv2.waitKey(1)
            except Exception as exc:
                self.preview_failed = True
                self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
                return
            time.sleep(period)

    def _format_status(self) -> str:
        dt = " ".join(
            f"{key}={'none' if value is None else f'{value * 1000.0:.0f}ms'}"
            for key, value in self.last_dt_map.items()
        )
        now = self.now_sec()
        topics = " ".join(
            f"{key}#{self.topic_counts[key]}@"
            f"{'none' if self.topic_counts[key] <= 0 else f'{(now - self.topic_last[key]) * 1000.0:.0f}ms'}"
            for key in ("front", "wrist", "state")
        )
        return f"sync[{dt}] topics[{topics}]"

    def _log_wait(self, text: str) -> None:
        if self.now_sec() - self.last_log_time > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_log_time = self.now_sec()
            self.node.get_logger().warn(text)

    def _log_info(self, text: str) -> None:
        if self.now_sec() - self.last_log_time > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_log_time = self.now_sec()
            self.node.get_logger().info(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous SmolVLA rollout for UR3e impedance no-rotvec control.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CFG.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=CFG.ik_target_topic)
    parser.add_argument("--vr-command-topic", default=CFG.vr_command_topic)
    parser.add_argument("--vr-raw-topic", default=CFG.vr_raw_topic)
    parser.add_argument("--fps", type=float, default=CFG.fps)
    parser.add_argument("--command-hz", type=float, default=CFG.command_hz)
    parser.add_argument("--action-step-hz", type=float, default=CFG.action_step_hz)
    parser.add_argument(
        "--n-obs-steps",
        type=int,
        default=None,
        help="Override observation history length. Default: read n_obs_steps from policy config.",
    )
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument("--prefetch-actions", type=int, default=0)
    parser.add_argument("--replace-queue-on-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replan-every-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sync-reference", choices=("front", "wrist", "timer"), default=CFG.sync_reference)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=CFG.max_dt_state)
    parser.add_argument("--buffer-maxlen", type=int, default=CFG.buffer_maxlen)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-home", type=Path, default=CFG.hf_home)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=CFG.offline)
    parser.add_argument("--rl-mark", type=float, default=CFG.rl_mark)
    parser.add_argument("--gripper-max", type=float, default=CFG.gripper_max)
    parser.add_argument("--max-action-age-s", type=float, default=1.0)
    parser.add_argument("--action-position-mode", choices=("auto", "relative", "absolute"), default=CFG.action_position_mode)
    parser.add_argument("--action-pose-filter-alpha", type=float, default=0.75)
    parser.add_argument("--max-action-pos-step", type=float, default=0.06)
    parser.add_argument("--min-action-z", type=float, default=CFG.min_action_z)
    parser.add_argument("--action-gripper-filter-alpha", type=float, default=CFG.action_gripper_filter_alpha)
    parser.add_argument("--vr-override", action=argparse.BooleanOptionalAction, default=CFG.vr_override)
    parser.add_argument("--vr-override-stale-s", type=float, default=CFG.vr_override_stale_s)
    parser.add_argument("--vr-override-resume-delay-s", type=float, default=CFG.vr_override_resume_delay_s)
    parser.add_argument("--vr-override-gripper-gain", type=float, default=CFG.vr_override_gripper_gain)
    parser.add_argument("--log-hz", type=float, default=CFG.log_hz)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=CFG.preview)
    parser.add_argument("--preview-hz", type=float, default=CFG.preview_hz)
    parser.add_argument("--return-home-on-start", action=argparse.BooleanOptionalAction, default=CFG.return_home_on_start)
    parser.add_argument("--start-home-delay-s", type=float, default=CFG.start_home_delay_s)
    parser.add_argument("--start-home-pulse-s", type=float, default=CFG.start_home_pulse_s)
    parser.add_argument("--start-home-settle-s", type=float, default=CFG.start_home_settle_s)
    parser.add_argument("--start-open-gripper-s", type=float, default=CFG.start_open_gripper_s)
    parser.add_argument("--start-open-gripper-value", type=float, default=CFG.start_open_gripper_value)
    parser.add_argument("--execute", action="store_true", help="Publish fixed-orientation model targets to robot topics.")
    return parser.parse_args()


def main() -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    args = parse_args()
    args.policy_path = resolve_policy_path(args.policy_path)
    hf_home = args.hf_home.expanduser()
    if not hf_home.is_absolute():
        hf_home = (REPO_ROOT / hf_home).resolve()
    args.hf_home = hf_home
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    rclpy.init()
    node = rclpy.create_node("ur3e_no_rotvec_smolvla_sync_rollout")
    rollout = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = NoRotvecSmolVLASyncRollout(node, args)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rollout is not None:
            rollout.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
