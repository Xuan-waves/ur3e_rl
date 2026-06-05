#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import parse_ik_target, parse_joint_target, parse_robot_state, parse_vr_command
from real_teleop.rotation_repr import continuous_rotvec_from_quat
from real_teleop.config import TeleopConfig
from real_teleop.kinematics import RobotKinematics
from scripts.collect_data.collect_ur3e_vr_impedance import ConsoleStatusPanel, SyncBuffer, SyncStats
from vr_servoj_test.collect_data.config import VrServoJCollectConfig
from vr_servoj_test.collect_data.lerobot_writer import CollectedFrame, LeRobotVrServoJWriter


class Ur3eVrServoJCollector:
    """LeRobot collector for the non-impedance VR servoJ teleop path."""

    def __init__(self, node, cfg: VrServoJCollectConfig) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.cfg = cfg
        self.sync_reference = str(cfg.sync_reference).strip().lower()
        if self.sync_reference not in {"front", "wrist", "timer"}:
            raise ValueError("sync_reference must be 'front', 'wrist', or 'timer'")
        self.writer = LeRobotVrServoJWriter(cfg)
        self.recording = False
        self.rl_mark = 0.0
        self.total_frames = 0
        self.last_buttons = {
            "record_start": False,
            "record_stop": False,
            "rl_toggle": False,
            "cancel_record": False,
            "stop_collection": False,
        }
        self.last_wait_log_time = 0.0
        self.last_drop_log_time = 0.0
        self.last_sample_time = 0.0
        self.last_gripper_status = {"state": 0.0, "action": 0.0, "vr": 0.0, "joint": 0.0}
        self.last_dt_map: dict[str, float | None] = {}
        self.eepose_rotvec_reference = self._make_eepose_rotvec_reference()
        self.last_state_rotvec: np.ndarray | None = None
        self.last_action_rotvec: np.ndarray | None = None
        self.pending_reference_stamps: deque[float] = deque(maxlen=cfg.buffer_maxlen)
        self.stats = SyncStats()
        self.status = ConsoleStatusPanel(enabled=cfg.status_panel, min_period=1.0 / max(cfg.status_hz, 1e-6))
        self.shutdown_requested = False
        self.shutdown_reason = ""

        self.buffers = {
            "front": SyncBuffer(cfg.buffer_maxlen),
            "wrist": SyncBuffer(cfg.buffer_maxlen),
            "state": SyncBuffer(cfg.buffer_maxlen),
            "ee_action": SyncBuffer(cfg.buffer_maxlen),
            "joint_action": SyncBuffer(cfg.buffer_maxlen),
            "vr": SyncBuffer(cfg.buffer_maxlen),
        }
        self.topic_counts = {key: 0 for key in self.buffers}
        self.topic_last = {key: 0.0 for key in self.buffers}

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

        self.subs = [
            node.create_subscription(Image, cfg.front_image_topic, self._on_front_image, image_qos, callback_group=self.callback_group),
            node.create_subscription(Image, cfg.wrist_image_topic, self._on_wrist_image, image_qos, callback_group=self.callback_group),
            node.create_subscription(Float64MultiArray, cfg.robot_state_topic, self._on_robot_state, data_qos, callback_group=self.callback_group),
            node.create_subscription(Float64MultiArray, cfg.ik_target_topic, self._on_ik_target, data_qos, callback_group=self.callback_group),
            node.create_subscription(
                Float64MultiArray,
                cfg.commanded_joint_target_topic,
                self._on_commanded_joint_target,
                data_qos,
                callback_group=self.callback_group,
            ),
            node.create_subscription(Float64MultiArray, cfg.vr_command_topic, self._on_vr_command, data_qos, callback_group=self.callback_group),
        ]
        timer_period = 1.0 / max(float(cfg.fps), 1.0)
        if self.sync_reference == "timer":
            self.collect_timer = node.create_timer(timer_period, self._collect_tick, callback_group=self.callback_group)
        else:
            self.collect_timer = node.create_timer(timer_period, self._reference_timer_tick, callback_group=self.callback_group)
        self._log_info(
            "UR3e VR servoJ LeRobot collector ready. "
            f"fps={cfg.fps:.1f}, state_mode={cfg.state_mode}, action_mode={cfg.action_mode}, "
            f"ee_action_position_mode={cfg.ee_action_position_mode}, sync_reference={self.sync_reference}, "
            f"output_parent={cfg.dataset_root}"
        )
        self._render_status(force=True)

    def _make_eepose_rotvec_reference(self) -> np.ndarray:
        teleop_cfg = TeleopConfig()
        _, home_quat = RobotKinematics(teleop_cfg).forward(teleop_cfg.hardware_home_q)
        return continuous_rotvec_from_quat(home_quat).astype(float)

    def now_sec(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _msg_time_sec(self, msg: Any) -> float:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return self.now_sec()
        sec = float(getattr(stamp, "sec", 0.0))
        nanosec = float(getattr(stamp, "nanosec", 0.0))
        return sec + nanosec * 1e-9

    def _push(self, key: str, stamp: float, value: Any) -> None:
        self.buffers[key].push(stamp, value)
        self.topic_counts[key] += 1
        self.topic_last[key] = self.now_sec()

    def _on_front_image(self, msg) -> None:
        stamp = self._msg_time_sec(msg)
        self._push("front", stamp, msg)
        if self.sync_reference == "front":
            self.pending_reference_stamps.append(stamp)

    def _on_wrist_image(self, msg) -> None:
        stamp = self._msg_time_sec(msg)
        self._push("wrist", stamp, msg)
        if self.sync_reference == "wrist":
            self.pending_reference_stamps.append(stamp)

    def _on_robot_state(self, msg) -> None:
        try:
            payload = parse_robot_state(msg.data)
            self._push("state", self.now_sec(), payload)
        except Exception as exc:
            self._log_warn(f"Bad robot_state frame: {exc}")

    def _on_ik_target(self, msg) -> None:
        try:
            payload = parse_ik_target(msg.data)
            self._push("ee_action", self.now_sec(), payload)
        except Exception as exc:
            self._log_warn(f"Bad ik_target frame: {exc}")

    def _on_commanded_joint_target(self, msg) -> None:
        try:
            payload = parse_joint_target(msg.data)
            self._push("joint_action", self.now_sec(), payload)
        except Exception as exc:
            self._log_warn(f"Bad commanded_joint_target frame: {exc}")

    def _on_vr_command(self, msg) -> None:
        try:
            payload = parse_vr_command(msg.data)
        except Exception as exc:
            self._log_warn(f"Bad vr_command frame: {exc}")
            return
        self._push("vr", self.now_sec(), payload)
        self._handle_vr_edges(payload)

    def _handle_vr_edges(self, payload: dict[str, Any]) -> None:
        current = {
            "record_start": bool(payload.get("record_start", False)),
            "record_stop": bool(payload.get("record_stop", False)),
            "rl_toggle": bool(payload.get("rl_toggle", False)),
            "cancel_record": bool(payload.get("cancel_record", False)),
            "stop_collection": bool(payload.get("stop_collection", False))
            or float(payload.get("left_grip", 0.0)) >= self.cfg.stop_left_grip_threshold,
        }
        if current["record_start"] and not self.last_buttons["record_start"]:
            self._start_recording()
        if current["rl_toggle"] and not self.last_buttons["rl_toggle"]:
            self.rl_mark = 0.0 if self.rl_mark > 0.5 else 1.0
            self.status.set_event(f"RL_mark toggled to {int(self.rl_mark)}")
            self._log_info(f"RL_mark toggled to {int(self.rl_mark)}")
        if current["record_stop"] and not self.last_buttons["record_stop"]:
            self._save_episode()
        if current["cancel_record"] and not self.last_buttons["cancel_record"]:
            self._discard_episode()
        if current["stop_collection"] and not self.last_buttons["stop_collection"]:
            self._stop_collection_from_vr()
        self.last_buttons = current

    def _start_recording(self) -> None:
        if self.recording:
            return
        self.recording = True
        self._clear_buffers()
        self._reset_rotation_continuity()
        self.stats.reset(self.now_sec())
        self.last_sample_time = 0.0
        self.status.set_event(f"recording started, RL_mark={int(self.rl_mark)}")
        self._log_info(f"Recording started. RL_mark={int(self.rl_mark)}")
        self._render_status(force=True)

    def _save_episode(self) -> None:
        self.recording = False
        self._reset_rotation_continuity()
        try:
            saved = self.writer.save_episode()
        except Exception as exc:
            self.status.set_event("save failed")
            self._log_error(f"Failed to save episode: {exc}")
            return
        if not saved:
            self.status.set_event("B pressed, no frames to save")
            self._log_warn("B pressed, but current episode has no frames.")
            return
        report_path = self._write_sync_report()
        self.status.set_event(f"episode saved, total={self.writer.total_saved_episodes}")
        self._log_info(f"Episode saved: root={self.writer.root}, report={report_path}")
        if self.cfg.max_episodes > 0 and self.writer.total_saved_episodes >= self.cfg.max_episodes:
            self._request_shutdown(f"max episodes reached ({self.cfg.max_episodes})")
        self._render_status(force=True)

    def _discard_episode(self) -> None:
        was_recording = self.recording
        discarded_frames = self.writer.episode_frame_count
        self.recording = False
        self.last_sample_time = 0.0
        self._clear_buffers()
        self._reset_rotation_continuity()
        self.stats.reset(self.now_sec())
        if discarded_frames <= 0:
            if was_recording:
                self.status.set_event("recording stopped, no frames discarded")
                self._log_warn("Recording stopped by left trigger. No frames were recorded.")
                self._render_status(force=True)
            return
        try:
            discarded = self.writer.discard_episode()
        except Exception as exc:
            self.status.set_event("discard failed")
            self._log_error(f"Failed to discard episode: {exc}")
            return
        if discarded:
            self.total_frames = max(0, self.total_frames - discarded_frames)
        self.status.set_event(f"recording stopped, discarded_frames={discarded_frames}")
        self._log_warn(f"Recording stopped and current episode discarded. removed_frames={discarded_frames}")
        self._render_status(force=True)

    def _stop_collection_from_vr(self) -> None:
        if self.recording or self.writer.episode_frame_count > 0:
            self._discard_episode()
            reason = "left lower trigger pressed; current episode discarded"
        else:
            reason = "left lower trigger pressed"
        self._request_shutdown(reason)

    def _request_shutdown(self, reason: str) -> None:
        self.shutdown_requested = True
        self.shutdown_reason = str(reason)
        self.status.set_event(f"shutdown requested: {self.shutdown_reason}")
        self._log_info(f"Collection shutdown requested: {self.shutdown_reason}")

    def _clear_buffers(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

    def _status_tick(self) -> None:
        if not self.recording:
            self._log_waiting_status()
            self._render_status()

    def _reference_timer_tick(self) -> None:
        if not self.recording:
            self._log_waiting_status()
            self._render_status()
            return
        if not self.pending_reference_stamps:
            self._render_status()
            return
        stamp = float(self.pending_reference_stamps[-1])
        self.pending_reference_stamps.clear()
        self._collect_tick(stamp)

    def _collect_tick(self, stamp: float | None = None) -> None:
        if not self.recording:
            self._log_waiting_status()
            self._render_status()
            return

        stamp = self.now_sec() if stamp is None else float(stamp)
        period_slack = 0.8 if self.sync_reference != "timer" else 1.0
        min_period = period_slack / max(float(self.cfg.fps), 1e-6)
        if stamp - self.last_sample_time < min_period:
            return
        self.last_sample_time = stamp

        sample, dt_map = self._nearest_sample(stamp)
        self.last_dt_map = dt_map
        if sample is None:
            self.stats.drop(stamp)
            if stamp - self.last_drop_log_time > 2.0:
                self.last_drop_log_time = stamp
                self.status.set_event("drop frame: stale/missing source")
                self._log_warn(f"Drop frame: {self._format_dt(dt_map)} {self._topic_status()}")
            self._render_status()
            return

        front_msg, wrist_msg, robot_state, ee_action, joint_action, vr_command = sample
        try:
            action_gripper = self._select_gripper(joint_action, vr_command, robot_state)
            state_vec = self._make_state(robot_state, fallback_gripper=action_gripper)
            action_vec = self._make_action(ee_action, joint_action, action_gripper, robot_state)
            frame = CollectedFrame(
                front_rgb=self._image_to_rgb(front_msg),
                wrist_rgb=self._image_to_rgb(wrist_msg),
                state=state_vec,
                action=action_vec,
                timestamp=stamp,
            )
            self.writer.add_frame(frame)
        except Exception as exc:
            self.stats.drop(stamp)
            self.status.set_event("failed to add frame")
            self._log_warn(f"Failed to add frame: {exc}")
            self._render_status(force=True)
            return

        self.total_frames += 1
        self.stats.success(stamp, dt_map)
        self.last_gripper_status = {
            "state": float(state_vec[6]),
            "action": float(action_vec[6]),
            "vr": float(vr_command.get("gripper", 0.0)),
            "joint": float(joint_action.get("gripper", 0.0)),
        }
        self._render_status()

    def _nearest_sample(self, stamp: float) -> tuple[tuple[Any, Any, dict, dict, dict, dict] | None, dict[str, float | None]]:
        front, dt_front = self.buffers["front"].nearest(stamp, self.cfg.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.cfg.max_dt_wrist_image)
        state, dt_state = self.buffers["state"].nearest(stamp, self.cfg.max_dt_state)
        ee_action, dt_ee_action = self.buffers["ee_action"].nearest(stamp, self.cfg.max_dt_action)
        joint_action, dt_joint_action = self.buffers["joint_action"].nearest(stamp, self.cfg.max_dt_action)
        vr, dt_vr = self.buffers["vr"].nearest(stamp, self.cfg.max_dt_action)
        dt_map = {
            "front": dt_front,
            "wrist": dt_wrist,
            "state": dt_state,
            "ee_action": dt_ee_action,
            "joint_action": dt_joint_action,
            "vr": dt_vr,
        }
        required = [front, wrist, state, vr]
        if self.cfg.action_mode == "jointspace":
            required.append(joint_action)
        else:
            required.append(ee_action)
        sample = (front, wrist, state, ee_action or {}, joint_action or {}, vr)
        return (None, dt_map) if any(value is None for value in required) else (sample, dt_map)

    def _log_waiting_status(self) -> None:
        now = self.now_sec()
        if now - self.last_wait_log_time < 5.0:
            return
        self.last_wait_log_time = now
        required = ["front", "wrist", "state", "vr", "joint_action" if self.cfg.action_mode == "jointspace" else "ee_action"]
        missing = [key for key in required if self.topic_counts.get(key, 0) <= 0]
        if missing:
            self.status.set_event(f"waiting for topics: {','.join(missing)}")

    def _render_status(self, *, force: bool = False) -> None:
        mode = "RECORDING" if self.recording else "WAITING"
        duration = max(self.stats.ended_at - self.stats.started_at, 0.0)
        fps = self.stats.frames_saved / max(duration, 1e-9) if self.recording else 0.0
        required = ["front", "wrist", "state", "vr", "joint_action" if self.cfg.action_mode == "jointspace" else "ee_action"]
        missing = [key for key in required if self.topic_counts.get(key, 0) <= 0]
        missing_text = "ok" if not missing else ",".join(missing)
        max_eps = "inf" if self.cfg.max_episodes <= 0 else str(self.cfg.max_episodes)
        grip = self.last_gripper_status
        lines = [
            "UR3e VR ServoJ Collector | X=start  Y=RL_mark  B=save  left_trigger=discard  left_grip=quit",
            (
                f"mode={mode:<9} episode_frames={self.writer.episode_frame_count:<5d} "
                f"total={self.total_frames:<5d} saved_eps={self.writer.total_saved_episodes}/{max_eps:<3} "
                f"fps={fps:>5.1f} drops={self.stats.frames_dropped:<4d} RL_mark={int(self.rl_mark)}"
            ),
            f"repr state={self.cfg.state_mode} action={self.cfg.action_mode} ee_pos={self.cfg.ee_action_position_mode}",
            (
                f"gripper state={grip['state']:.3f} action={grip['action']:.3f} "
                f"joint={grip['joint']:.3f} vr={grip['vr']:.3f}"
            ),
            f"sync {self._compact_dt(self.last_dt_map)}",
            f"topics missing={missing_text} | {self._compact_topic_status()}",
            f"dataset={self.writer.root}",
            f"event={self.status.last_event}",
        ]
        self.status.render(lines, force=force)

    def _select_gripper(self, joint_target: dict[str, Any], vr_command: dict[str, Any], robot_state: dict[str, Any]) -> np.float32:
        if "gripper" in joint_target:
            value = joint_target["gripper"]
        elif "gripper" in vr_command:
            value = vr_command["gripper"]
        else:
            value = robot_state.get("gripper", 0.0)
        return np.float32(np.clip(float(value), 0.0, self.cfg.gripper_max))

    def _make_state(self, robot_state: dict[str, Any], *, fallback_gripper: np.float32) -> np.ndarray:
        gripper = np.float32(np.clip(robot_state.get("gripper", 0.0), 0.0, self.cfg.gripper_max))
        if gripper <= 1e-4 and fallback_gripper > 1e-4:
            gripper = fallback_gripper
        if self.cfg.state_mode == "jointspace":
            q = np.asarray(robot_state["q"], dtype=np.float32)
            return np.concatenate([q, [gripper, np.float32(self.rl_mark)]]).astype(np.float32)
        if self.cfg.state_mode == "eepose":
            pos = np.asarray(robot_state["tcp_pos"], dtype=np.float32)
            quat = np.asarray(robot_state["tcp_quat"], dtype=float)
            previous_rotvec = self.last_state_rotvec
            if previous_rotvec is None:
                previous_rotvec = self.eepose_rotvec_reference
            rotvec = continuous_rotvec_from_quat(quat, previous_rotvec)
            self.last_state_rotvec = rotvec.astype(float)
            return np.concatenate([pos, rotvec, [gripper, np.float32(self.rl_mark)]]).astype(np.float32)
        raise ValueError(f"Unsupported state_mode: {self.cfg.state_mode!r}")

    def _make_action(
        self,
        ee_action: dict[str, Any],
        joint_target: dict[str, Any],
        gripper: np.float32,
        robot_state: dict[str, Any],
    ) -> np.ndarray:
        if self.cfg.action_mode == "jointspace":
            q = joint_target.get("q")
            if q is None:
                q = robot_state.get("q")
            if q is None:
                raise ValueError("jointspace action requested, but neither joint target nor robot state has q")
            return np.concatenate([np.asarray(q, dtype=np.float32), [gripper, np.float32(self.rl_mark)]]).astype(np.float32)
        if self.cfg.action_mode != "eepose":
            raise ValueError(f"Unsupported action_mode: {self.cfg.action_mode!r}")

        target_pos = np.asarray(ee_action["pos"], dtype=np.float32)
        if self.cfg.ee_action_position_mode == "relative":
            state_pos = np.asarray(robot_state["tcp_pos"], dtype=np.float32)
            pos = target_pos - state_pos
        elif self.cfg.ee_action_position_mode == "absolute":
            pos = target_pos
        else:
            raise ValueError(f"Unsupported ee_action_position_mode: {self.cfg.ee_action_position_mode!r}")

        quat = np.asarray(ee_action["quat"], dtype=float)
        previous_rotvec = self.last_action_rotvec
        if previous_rotvec is None and self.last_state_rotvec is not None:
            previous_rotvec = self.last_state_rotvec
        if previous_rotvec is None:
            previous_rotvec = self.eepose_rotvec_reference
        rotvec = continuous_rotvec_from_quat(quat, previous_rotvec)
        self.last_action_rotvec = rotvec.astype(float)
        return np.concatenate([pos, rotvec, [gripper, np.float32(self.rl_mark)]]).astype(np.float32)

    def _reset_rotation_continuity(self) -> None:
        self.last_state_rotvec = self.eepose_rotvec_reference.copy()
        self.last_action_rotvec = self.eepose_rotvec_reference.copy()

    def _image_to_rgb(self, msg) -> np.ndarray:
        image = decode_image_msg(msg)
        if self.cfg.resize is not None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required when --resize is used.") from exc
            width, height = self.cfg.resize
            image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        return image

    def _write_sync_report(self) -> Path:
        meta_dir = self.writer.root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        report_path = meta_dir / f"sync_report_episode_{self.writer.total_saved_episodes - 1:06d}.json"
        report = {
            "task": self.cfg.task,
            "fps": self.cfg.fps,
            "rl_mark_final": int(self.rl_mark),
            "stats": self.stats.as_dict(),
            "topics": {
                "front": self.cfg.front_image_topic,
                "wrist": self.cfg.wrist_image_topic,
                "robot_state": self.cfg.robot_state_topic,
                "ik_target": self.cfg.ik_target_topic,
                "joint_target": self.cfg.joint_target_topic,
                "commanded_joint_target": self.cfg.commanded_joint_target_topic,
                "vr_command": self.cfg.vr_command_topic,
            },
            "sync_policy": {
                "clock": "image header.stamp when available; receive-time for state/action/VR topics",
                "sync_reference": self.sync_reference,
                "max_dt_front_image": self.cfg.max_dt_front_image,
                "max_dt_wrist_image": self.cfg.max_dt_wrist_image,
                "max_dt_state": self.cfg.max_dt_state,
                "max_dt_action": self.cfg.max_dt_action,
                "strict": True,
            },
            "representation": {
                "control_mode": "servoj",
                "state_mode": self.cfg.state_mode,
                "action_mode": self.cfg.action_mode,
                "ee_action_position_mode": self.cfg.ee_action_position_mode if self.cfg.action_mode == "eepose" else None,
                "rotvec_reference_policy": "hardware_home_fk",
                "eepose_rotvec_reference": self.eepose_rotvec_reference.tolist(),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report_path

    def _compact_dt(self, dt_map: dict[str, float | None]) -> str:
        if not dt_map:
            return "front=-- wrist=-- state=-- ee=-- joint=-- vr=--"
        labels = {"front": "front", "wrist": "wrist", "state": "state", "ee_action": "ee", "joint_action": "joint", "vr": "vr"}
        return " ".join(
            f"{labels[key]}={'--' if dt_map.get(key) is None else f'{dt_map[key] * 1000.0:.0f}ms'}"
            for key in ("front", "wrist", "state", "ee_action", "joint_action", "vr")
        )

    def _compact_topic_status(self) -> str:
        now = self.now_sec()
        labels = {"front": "front", "wrist": "wrist", "state": "state", "ee_action": "ee", "joint_action": "joint", "vr": "vr"}
        parts = []
        for key in ("front", "wrist", "state", "ee_action", "joint_action", "vr"):
            count = self.topic_counts.get(key, 0)
            if count <= 0:
                parts.append(f"{labels[key]}#0")
            else:
                age_ms = (now - self.topic_last.get(key, 0.0)) * 1000.0
                parts.append(f"{labels[key]}#{count}@{age_ms:.0f}ms")
        return " ".join(parts)

    def _topic_status(self) -> str:
        now = self.now_sec()
        parts = []
        for key in ("front", "wrist", "state", "ee_action", "joint_action", "vr"):
            count = self.topic_counts.get(key, 0)
            age = None if count <= 0 else now - self.topic_last.get(key, 0.0)
            parts.append(f"{key}#={count} age={'none' if age is None else f'{age * 1000.0:.0f}ms'}")
        return " ".join(parts)

    @staticmethod
    def _format_dt(dt_map: dict[str, float | None]) -> str:
        return " ".join(f"{key}={'none' if value is None else f'{value * 1000.0:.1f}ms'}" for key, value in dt_map.items())

    def _log_info(self, text: str) -> None:
        self.status.clear()
        self.node.get_logger().info(text)

    def _log_warn(self, text: str) -> None:
        self.status.clear()
        self.node.get_logger().warn(text)

    def _log_error(self, text: str) -> None:
        self.status.clear()
        self.node.get_logger().error(text)

    def close(self) -> None:
        if self.recording:
            self.node.get_logger().warn("Collector closing while recording; unsaved current episode is discarded.")
            self._discard_episode()
        self.writer.finalize()


def decode_image_msg(msg) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    if encoding == "mono8":
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        gray = flat[: int(msg.step) * int(msg.height)].reshape((int(msg.height), int(msg.step)))
        gray = gray[:, : int(msg.width)]
        return np.repeat(gray[:, :, None], 3, axis=2).copy()
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")
    flat = np.frombuffer(msg.data, dtype=np.uint8)
    image = flat[: int(msg.step) * int(msg.height)].reshape((int(msg.height), int(msg.step)))
    image = image[:, : int(msg.width) * channels].reshape((int(msg.height), int(msg.width), channels))
    if encoding == "rgb8":
        return image[:, :, :3].copy()
    if encoding == "bgr8":
        return image[:, :, [2, 1, 0]].copy()
    if encoding == "rgba8":
        return image[:, :, :3].copy()
    return image[:, :, [2, 1, 0]].copy()


def parse_args() -> argparse.Namespace:
    cfg = VrServoJCollectConfig
    parser = argparse.ArgumentParser(description="LeRobot collector for pure VR servoJ teleoperation.")
    parser.add_argument("--dataset-root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--dataset-name", default=cfg.dataset_name)
    parser.add_argument("--repo-id", default=cfg.repo_id)
    parser.add_argument("--task", default=cfg.task)
    parser.add_argument("--max-episodes", type=int, default=cfg.max_episodes)
    parser.add_argument("--state-mode", choices=("jointspace", "eepose"), default=cfg.state_mode)
    parser.add_argument("--action-mode", choices=("jointspace", "eepose"), default=cfg.action_mode)
    parser.add_argument(
        "--ee-action-position-mode",
        choices=("relative", "absolute"),
        default=cfg.ee_action_position_mode,
        help="Only used when --action-mode eepose.",
    )
    parser.add_argument("--fps", type=float, default=cfg.fps)
    parser.add_argument("--front-image-topic", default=cfg.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=cfg.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=cfg.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=cfg.ik_target_topic)
    parser.add_argument("--joint-target-topic", default=cfg.joint_target_topic)
    parser.add_argument("--commanded-joint-target-topic", default=cfg.commanded_joint_target_topic)
    parser.add_argument("--vr-command-topic", default=cfg.vr_command_topic)
    parser.add_argument("--max-dt-image", type=float, default=cfg.max_dt_image)
    parser.add_argument("--max-dt-front-image", type=float, default=cfg.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=cfg.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=cfg.max_dt_state)
    parser.add_argument("--max-dt-action", type=float, default=cfg.max_dt_action)
    parser.add_argument("--sync-reference", choices=("front", "wrist", "timer"), default=cfg.sync_reference)
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--status-panel", dest="status_panel", action="store_true", default=cfg.status_panel)
    parser.add_argument("--no-status-panel", dest="status_panel", action="store_false")
    parser.add_argument("--status-hz", type=float, default=cfg.status_hz)
    parser.add_argument("--stop-left-grip-threshold", type=float, default=cfg.stop_left_grip_threshold)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> VrServoJCollectConfig:
    max_dt_image = float(args.max_dt_image)
    return VrServoJCollectConfig(
        fps=float(args.fps),
        max_dt_image=max_dt_image,
        max_dt_front_image=float(args.max_dt_front_image if args.max_dt_front_image is not None else max_dt_image),
        max_dt_wrist_image=float(args.max_dt_wrist_image if args.max_dt_wrist_image is not None else max_dt_image),
        max_dt_state=float(args.max_dt_state),
        max_dt_action=float(args.max_dt_action),
        sync_reference=str(args.sync_reference),
        front_image_topic=str(args.front_image_topic),
        wrist_image_topic=str(args.wrist_image_topic),
        robot_state_topic=str(args.robot_state_topic),
        ik_target_topic=str(args.ik_target_topic),
        joint_target_topic=str(args.joint_target_topic),
        commanded_joint_target_topic=str(args.commanded_joint_target_topic),
        vr_command_topic=str(args.vr_command_topic),
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        dataset_name=str(args.dataset_name),
        repo_id=str(args.repo_id),
        task=str(args.task),
        max_episodes=max(0, int(args.max_episodes)),
        state_mode=str(args.state_mode),
        action_mode=str(args.action_mode),
        ee_action_position_mode=str(args.ee_action_position_mode),
        use_videos=not bool(args.no_videos),
        resize=tuple(args.resize) if args.resize else None,
        status_panel=bool(args.status_panel),
        status_hz=float(args.status_hz),
        stop_left_grip_threshold=float(args.stop_left_grip_threshold),
    )


def main() -> int:
    args = parse_args()

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    cfg = make_config(args)
    rclpy.init()
    node = rclpy.create_node("ur3e_vr_servoj_lerobot_collector")
    collector = Ur3eVrServoJCollector(node, cfg)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while rclpy.ok() and not collector.shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if collector.shutdown_requested:
            collector.status.set_event(f"shutdown: {collector.shutdown_reason}")
            collector._render_status(force=True)
        collector.close()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
