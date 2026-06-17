#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import parse_ik_target, parse_joint_target, parse_robot_state, parse_vr_command
from scripts.collect_data.config import CollectConfig
from scripts.collect_data.lerobot_writer import CollectedFrame, LeRobotVrEpisodeWriter


class SyncBuffer:
    def __init__(self, maxlen: int) -> None:
        self._data = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, stamp: float, value: Any) -> None:
        with self._lock:
            self._data.append((float(stamp), value))

    def nearest(self, stamp: float, max_dt: float) -> tuple[Any | None, float | None]:
        with self._lock:
            if not self._data:
                return None, None
            sample_stamp, value = min(self._data, key=lambda item: abs(item[0] - stamp))
        dt = abs(float(sample_stamp) - float(stamp))
        return (value, dt) if dt <= max_dt else (None, dt)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class SyncStats:
    def __init__(self) -> None:
        self.reset(0.0)

    def reset(self, stamp: float) -> None:
        self.started_at = float(stamp)
        self.ended_at = float(stamp)
        self.frames_saved = 0
        self.frames_dropped = 0
        self.dt = {"front": [], "wrist": [], "state": [], "action": [], "joint_action": [], "vr": []}

    def success(self, stamp: float, dt_map: dict[str, float | None]) -> None:
        self.ended_at = float(stamp)
        self.frames_saved += 1
        for key, value in dt_map.items():
            if value is not None:
                self.dt.setdefault(key, []).append(float(value))

    def drop(self, stamp: float) -> None:
        self.ended_at = float(stamp)
        self.frames_dropped += 1

    def as_dict(self) -> dict[str, Any]:
        duration = max(self.ended_at - self.started_at, 0.0)
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": duration,
            "frames_saved": self.frames_saved,
            "frames_dropped": self.frames_dropped,
            "effective_fps": self.frames_saved / max(duration, 1e-9),
            "dt": {key: self._describe(values) for key, values in self.dt.items()},
        }

    @staticmethod
    def _describe(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "max": None, "p95": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "max": float(arr.max()), "p95": float(np.percentile(arr, 95))}


class ConsoleStatusPanel:
    def __init__(self, *, enabled: bool = True, min_period: float = 0.25) -> None:
        self.enabled = bool(enabled and sys.stdout.isatty())
        self.min_period = float(min_period)
        self.rendered_lines = 0
        self.last_render = 0.0
        self.last_event = "ready"
        self._lock = threading.Lock()

    def set_event(self, text: str) -> None:
        self.last_event = str(text)

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.rendered_lines <= 0:
                return
            sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            for _ in range(self.rendered_lines):
                sys.stdout.write("\x1b[2K\n")
            sys.stdout.flush()
            self.rendered_lines = 0

    def render(self, lines: list[str], *, force: bool = False) -> None:
        if not self.enabled:
            return
        now_stamp = time.monotonic()
        if not force and now_stamp - self.last_render < self.min_period:
            return
        with self._lock:
            if self.rendered_lines > 0:
                sys.stdout.write(f"\x1b[{self.rendered_lines}F")
            line_count = max(self.rendered_lines, len(lines))
            for index in range(line_count):
                sys.stdout.write("\x1b[2K")
                if index < len(lines):
                    sys.stdout.write(lines[index])
                sys.stdout.write("\n")
            sys.stdout.flush()
            self.rendered_lines = len(lines)
            self.last_render = now_stamp


class Ur3eVrImpedanceCollector:
    """Strict ROS-topic collector.

    This node intentionally does not open RealSense devices, launch camera nodes,
    publish preview images, or reuse stale images.  Camera startup and preview are
    separate processes; this node only samples already-published ROS topics.
    """

    def __init__(self, node, cfg: CollectConfig) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.cfg = cfg
        self.sync_reference = str(cfg.reference_camera).strip().lower()
        if self.sync_reference not in {"front", "wrist", "timer"}:
            raise ValueError("reference_camera must be 'front', 'wrist', or 'timer'")
        self.writer = LeRobotVrEpisodeWriter(cfg)
        self.recording = False
        self.pending_save = False
        self.total_frames = 0
        self.last_buttons = {
            "record_start": False,
            "record_stop": False,
            "cancel_record": False,
            "stop_collection": False,
        }
        self.last_log_time = self.now_sec()
        self.last_drop_log_time = 0.0
        self.last_wait_log_time = 0.0
        self.last_sample_time = 0.0
        self.b_released_since = self.now_sec()
        self.stats = SyncStats()
        self.last_gripper_status = {"state": 0.0, "action": 0.0, "vr": 0.0, "joint": 0.0}
        self.last_dt_map: dict[str, float | None] = {}
        self.pending_reference_stamps: deque[float] = deque(maxlen=cfg.buffer_maxlen)
        self.status = ConsoleStatusPanel(enabled=cfg.status_panel, min_period=1.0 / max(cfg.status_hz, 1e-6))
        self.shutdown_requested = False
        self.shutdown_reason = ""

        self.buffers = {
            "front": SyncBuffer(cfg.buffer_maxlen),
            "wrist": SyncBuffer(cfg.buffer_maxlen),
            "state": SyncBuffer(cfg.buffer_maxlen),
            "action": SyncBuffer(cfg.buffer_maxlen),
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
            node.create_subscription(Float64MultiArray, cfg.joint_target_topic, self._on_joint_target, data_qos, callback_group=self.callback_group),
            node.create_subscription(Float64MultiArray, cfg.vr_command_topic, self._on_vr_command, data_qos, callback_group=self.callback_group),
        ]
        timer_period = 1.0 / max(float(cfg.fps), 1.0)
        timer_callback = self._collect_tick if self.sync_reference == "timer" else self._reference_timer_tick
        self.collect_timer = node.create_timer(timer_period, timer_callback, callback_group=self.callback_group)

        self._log_info(
            "UR3e strict LeRobot collector ready. "
            f"fps={cfg.fps:.1f}, sync_reference={self.sync_reference}, "
            f"front={cfg.front_image_topic}, wrist={cfg.wrist_image_topic}, "
            f"output_parent={cfg.dataset_root}"
        )
        self._render_status(force=True)

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
            self._push("action", self.now_sec(), payload)
        except Exception as exc:
            self._log_warn(f"Bad ik_target frame: {exc}")

    def _on_joint_target(self, msg) -> None:
        try:
            payload = parse_joint_target(msg.data)
            self._push("joint_action", self.now_sec(), payload)
        except Exception as exc:
            self._log_warn(f"Bad joint_target frame: {exc}")

    def _on_vr_command(self, msg) -> None:
        try:
            payload = parse_vr_command(msg.data)
        except Exception as exc:
            self._log_warn(f"Bad vr_command frame: {exc}")
            return
        self._push("vr", self.now_sec(), payload)
        self._handle_vr_edges(payload)

    def _push(self, key: str, stamp: float, value: Any) -> None:
        self.buffers[key].push(stamp, value)
        self.topic_counts[key] += 1
        self.topic_last[key] = self.now_sec()

    def _handle_vr_edges(self, payload: dict[str, Any]) -> None:
        stamp = self.now_sec()
        current = {
            "record_start": bool(payload.get("record_start", False)),
            "record_stop": bool(payload.get("record_stop", False)),
            "cancel_record": bool(payload.get("cancel_record", False)),
            # In the shared VR message schema, stop_collection is still produced
            # by left_grip for older collectors. This impedance collector remaps
            # left_grip away from shutdown, so shutdown is intentionally Y-only here.
            "stop_collection": bool(payload.get("rl_toggle", False)),
        }
        if not current["record_stop"] and self.b_released_since <= 0.0:
            self.b_released_since = stamp
        if current["record_start"] and not self.last_buttons["record_start"]:
            self._start_recording()
        if current["record_stop"] and not self.last_buttons["record_stop"]:
            self._handle_save_button(stamp)
        if current["cancel_record"] and not self.last_buttons["cancel_record"]:
            self._discard_episode()
        if current["stop_collection"] and not self.last_buttons["stop_collection"]:
            self._stop_collection_from_vr()
        if current["record_stop"]:
            self.b_released_since = 0.0
        self.last_buttons = current

    def _start_recording(self) -> None:
        if self.recording:
            return
        if self.pending_save and self.writer.episode_frame_count > 0:
            frames = self.writer.episode_frame_count
            self.status.set_event(f"pending episode frames={frames}; B=save, left trigger=discard")
            self._log_warn(f"Pending episode has {frames} frames. Press B again to save or left trigger to discard.")
            self._render_status(force=True)
            return
        self.recording = True
        self.pending_save = False
        self._clear_buffers()
        self.stats.reset(self.now_sec())
        self.last_sample_time = 0.0
        self.last_log_time = self.now_sec()
        self.status.set_event("recording started")
        self._log_info("Recording started.")
        self._render_status(force=True)

    def _handle_save_button(self, stamp: float) -> None:
        if self.pending_save:
            released_for = stamp - self.b_released_since if self.b_released_since > 0.0 else 0.0
            if released_for < self.cfg.save_confirm_release_sec:
                frames = self.writer.episode_frame_count
                self.status.set_event(f"release B, then press again to save frames={frames}")
                self._log_warn(
                    f"Ignoring B confirm until B is released for {self.cfg.save_confirm_release_sec:.2f}s. "
                    f"pending_frames={frames}"
                )
                self._render_status(force=True)
                return
            self._save_episode()
            return
        if self.recording or self.writer.episode_frame_count > 0:
            self._stage_episode_for_save()
            return
        self.status.set_event("B pressed, no frames to stage")
        self._log_warn("B pressed, but current episode has no frames.")
        self._render_status(force=True)

    def _stage_episode_for_save(self) -> None:
        frames = self.writer.episode_frame_count
        self.recording = False
        self.last_sample_time = 0.0
        self._clear_buffers()
        if frames <= 0:
            self.pending_save = False
            self.status.set_event("B pressed, no frames to stage")
            self._log_warn("B pressed, but current episode has no frames.")
            self._render_status(force=True)
            return
        self.pending_save = True
        self.b_released_since = 0.0
        self.status.set_event(f"episode staged: frames={frames}; press B again to save")
        self._log_info(
            f"Episode staged in memory: frames={frames}. Release B, then press B again to save; "
            "or press left trigger to discard."
        )
        self._render_status(force=True)

    def _save_episode(self) -> None:
        self.recording = False
        try:
            saved = self.writer.save_episode()
        except Exception as exc:
            self.status.set_event("save failed")
            self._log_error(f"Failed to save episode: {exc}")
            return
        if not saved:
            self.pending_save = False
            self.status.set_event("B pressed, no frames to save")
            self._log_warn("B pressed, but current episode has no frames.")
            return
        self.pending_save = False
        report_path = self._write_sync_report()
        self.status.set_event(f"episode saved, total={self.writer.total_saved_episodes}")
        self._log_info(
            f"Episode saved: root={self.writer.root}, report={report_path}, "
            f"total_episodes={self.writer.total_saved_episodes}"
        )
        if self.cfg.max_episodes > 0 and self.writer.total_saved_episodes >= self.cfg.max_episodes:
            self._request_shutdown(f"max episodes reached ({self.cfg.max_episodes})")
        self._render_status(force=True)

    def _discard_episode(self) -> None:
        was_recording = self.recording
        was_pending = self.pending_save
        discarded_frames = self.writer.episode_frame_count
        self.recording = False
        self.pending_save = False
        self.last_sample_time = 0.0
        self._clear_buffers()
        self.stats.reset(self.now_sec())

        if discarded_frames <= 0:
            if not was_recording and not was_pending:
                return
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
        if self.recording or self.pending_save or self.writer.episode_frame_count > 0:
            self._discard_episode()
            reason = "Y pressed; current episode discarded"
        else:
            reason = "Y pressed"
        self._request_shutdown(reason)

    def _request_shutdown(self, reason: str) -> None:
        self.shutdown_requested = True
        self.shutdown_reason = str(reason)
        self.status.set_event(f"shutdown requested: {self.shutdown_reason}")
        self._log_info(f"Collection shutdown requested: {self.shutdown_reason}")

    def _clear_buffers(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

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

        front_msg, wrist_msg, robot_state, ik_target, joint_target, vr_command = sample
        try:
            action_gripper = self._select_gripper(joint_target, vr_command, robot_state)
            state_vec = self._make_state(robot_state, fallback_gripper=action_gripper)
            action_vec = self._make_action(ik_target, action_gripper, state_vec)
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
            "state": float(state_vec[3]),
            "action": float(action_vec[3]),
            "vr": float(vr_command.get("gripper", 0.0)),
            "joint": float(joint_target.get("gripper", 0.0)),
        }
        self._render_status()

    def _nearest_sample(self, stamp: float) -> tuple[tuple[Any, Any, dict, dict, dict, dict] | None, dict[str, float | None]]:
        front, dt_front = self.buffers["front"].nearest(stamp, self.cfg.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.cfg.max_dt_wrist_image)
        state, dt_state = self.buffers["state"].nearest(stamp, self.cfg.max_dt_state)
        action, dt_action = self.buffers["action"].nearest(stamp, self.cfg.max_dt_action)
        joint_action, dt_joint_action = self.buffers["joint_action"].nearest(stamp, self.cfg.max_dt_action)
        vr, dt_vr = self.buffers["vr"].nearest(stamp, self.cfg.max_dt_action)
        dt_map = {
            "front": dt_front,
            "wrist": dt_wrist,
            "state": dt_state,
            "action": dt_action,
            "joint_action": dt_joint_action,
            "vr": dt_vr,
        }
        if joint_action is None:
            joint_action = {}
        required_values = (front, wrist, state, action, vr)
        sample = (front, wrist, state, action, joint_action, vr)
        return (None, dt_map) if any(value is None for value in required_values) else (sample, dt_map)

    def _log_waiting_status(self) -> None:
        now = self.now_sec()
        if now - self.last_wait_log_time < 5.0:
            return
        self.last_wait_log_time = now
        required_keys = ("front", "wrist", "state", "action", "vr")
        missing = [key for key in required_keys if self.topic_counts.get(key, 0) <= 0]
        if missing:
            self.status.set_event(f"waiting for topics: {','.join(missing)}")

    def _topic_status(self) -> str:
        now = self.now_sec()
        parts = []
        for key in ("front", "wrist", "state", "action", "joint_action", "vr"):
            count = self.topic_counts.get(key, 0)
            age = None if count <= 0 else now - self.topic_last.get(key, 0.0)
            parts.append(f"{key}#={count} age={'none' if age is None else f'{age * 1000.0:.0f}ms'}")
        return " ".join(parts)

    def _render_status(self, *, force: bool = False) -> None:
        if self.recording:
            mode = "RECORDING"
        elif self.pending_save:
            mode = "STAGED"
        else:
            mode = "WAITING"
        duration = max(self.stats.ended_at - self.stats.started_at, 0.0)
        fps = self.stats.frames_saved / max(duration, 1e-9) if self.recording or self.pending_save else 0.0
        required_keys = ("front", "wrist", "state", "action", "vr")
        missing = [key for key in required_keys if self.topic_counts.get(key, 0) <= 0]
        missing_text = "ok" if not missing else ",".join(missing)
        max_eps = "inf" if self.cfg.max_episodes <= 0 else str(self.cfg.max_episodes)
        dt_text = self._compact_dt(self.last_dt_map)
        topic_text = self._compact_topic_status()
        grip = self.last_gripper_status
        lines = [
            "UR3e LeRobot Collector | X=start  B=stage/save  Y=discard+quit  left_trigger=discard",
            (
                f"mode={mode:<9} episode_frames={self.writer.episode_frame_count:<5d} "
                f"total={self.total_frames:<5d} saved_eps={self.writer.total_saved_episodes}/{max_eps:<3} "
                f"fps={fps:>5.1f} drops={self.stats.frames_dropped:<4d}"
            ),
            (
                f"gripper state={grip['state']:.3f} action={grip['action']:.3f} "
                f"joint={grip['joint']:.3f} vr={grip['vr']:.3f}"
            ),
            f"sync {dt_text}",
            f"topics missing={missing_text} | {topic_text}",
            f"dataset={self.writer.root}",
            f"event={self.status.last_event}",
        ]
        self.status.render(lines, force=force)

    def _compact_dt(self, dt_map: dict[str, float | None]) -> str:
        if not dt_map:
            return "front=-- wrist=-- state=-- action=-- joint=-- vr=--"
        labels = {
            "front": "front",
            "wrist": "wrist",
            "state": "state",
            "action": "pose",
            "joint_action": "joint",
            "vr": "vr",
        }
        parts = []
        for key in ("front", "wrist", "state", "action", "joint_action", "vr"):
            value = dt_map.get(key)
            text = "--" if value is None else f"{value * 1000.0:.0f}ms"
            parts.append(f"{labels[key]}={text}")
        return " ".join(parts)

    def _compact_topic_status(self) -> str:
        now = self.now_sec()
        labels = {
            "front": "front",
            "wrist": "wrist",
            "state": "state",
            "action": "pose",
            "joint_action": "joint",
            "vr": "vr",
        }
        parts = []
        for key in ("front", "wrist", "state", "action", "joint_action", "vr"):
            count = self.topic_counts.get(key, 0)
            if count <= 0:
                parts.append(f"{labels[key]}#0")
                continue
            age_ms = (now - self.topic_last.get(key, 0.0)) * 1000.0
            parts.append(f"{labels[key]}#{count}@{age_ms:.0f}ms")
        return " ".join(parts)

    def _log_info(self, text: str) -> None:
        self.status.clear()
        self.node.get_logger().info(text)

    def _log_warn(self, text: str) -> None:
        self.status.clear()
        self.node.get_logger().warn(text)

    def _log_error(self, text: str) -> None:
        self.status.clear()
        self.node.get_logger().error(text)

    def _select_gripper(
        self,
        joint_target: dict[str, Any],
        vr_command: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> np.float32:
        if "gripper" in joint_target:
            value = joint_target["gripper"]
        elif "gripper" in vr_command:
            value = vr_command["gripper"]
        else:
            value = robot_state.get("gripper", 0.0)
        return np.float32(np.clip(float(value), 0.0, self.cfg.gripper_max))

    def _make_state(self, robot_state: dict[str, Any], *, fallback_gripper: np.float32) -> np.ndarray:
        pos = np.asarray(robot_state["tcp_pos"], dtype=np.float32)
        gripper = np.float32(np.clip(robot_state.get("gripper", 0.0), 0.0, self.cfg.gripper_max))
        if gripper <= 1e-4 and fallback_gripper > 1e-4:
            gripper = fallback_gripper
        return np.concatenate([pos, [gripper]]).astype(np.float32)

    def _make_action(self, ik_target: dict[str, Any], gripper: np.float32, state_vec: np.ndarray) -> np.ndarray:
        target_pos = np.asarray(ik_target["pos"], dtype=np.float32)
        state_pos = np.asarray(state_vec[:3], dtype=np.float32)
        if self.cfg.action_position_mode == "relative":
            pos = target_pos - state_pos
        elif self.cfg.action_position_mode == "absolute":
            pos = target_pos
        else:
            raise ValueError(f"Unsupported action_position_mode: {self.cfg.action_position_mode!r}")

        return np.concatenate([pos, [gripper]]).astype(np.float32)

    def _image_to_rgb(self, msg) -> np.ndarray:
        image = self._decode_image(msg)
        if self.cfg.resize is not None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required when --resize is used.") from exc
            width, height = self.cfg.resize
            image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {image.shape}")
        return image

    @staticmethod
    def _decode_image(msg) -> np.ndarray:
        encoding = str(msg.encoding).lower()
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        if encoding == "mono8":
            flat = np.frombuffer(msg.data, dtype=np.uint8)
            expected = int(msg.step) * int(msg.height)
            gray = flat[:expected].reshape((int(msg.height), int(msg.step)))[:, : int(msg.width)]
            return np.repeat(gray[:, :, None], 3, axis=2)
        if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
            raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(msg.step) * int(msg.height)
        image = flat[:expected].reshape((int(msg.height), int(msg.step)))[:, : int(msg.width) * channels]
        image = image.reshape((int(msg.height), int(msg.width), channels))
        if encoding == "rgb8":
            return image.copy()
        if encoding == "bgr8":
            return image[:, :, ::-1].copy()
        if encoding == "rgba8":
            return image[:, :, :3].copy()
        return image[:, :, [2, 1, 0]].copy()

    def _write_sync_report(self) -> Path:
        meta_dir = self.writer.root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        report_path = meta_dir / f"sync_report_episode_{self.writer.total_saved_episodes - 1:06d}.json"
        report = {
            "task": self.cfg.task,
            "fps": self.cfg.fps,
            "stats": self.stats.as_dict(),
            "topics": {
                "front": self.cfg.front_image_topic,
                "wrist": self.cfg.wrist_image_topic,
                "robot_state": self.cfg.robot_state_topic,
                "ik_target": self.cfg.ik_target_topic,
                "joint_target": self.cfg.joint_target_topic,
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
            "action_representation": {
                "position": self.cfg.action_position_mode,
                "position_relative_to": "observation.state[0:3]"
                if self.cfg.action_position_mode == "relative"
                else None,
                "orientation": "removed_fixed_orientation",
            },
            "representation": {
                "control_mode": "impedance",
                "state_mode": "tcp_position_gripper",
                "action_mode": "tcp_position_gripper",
                "ee_action_position_mode": self.cfg.action_position_mode,
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report_path

    @staticmethod
    def _format_dt(dt_map: dict[str, float | None]) -> str:
        return " ".join(f"{key}={'none' if value is None else f'{value * 1000.0:.1f}ms'}" for key, value in dt_map.items())

    def close(self) -> None:
        if self.recording or self.pending_save:
            self.node.get_logger().warn("Collector closing while recording; unsaved current episode is discarded.")
            self._discard_episode()
        self.writer.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict LeRobot collector for UR3e VR impedance teleoperation.")
    parser.add_argument("--dataset-root", type=Path, default=CollectConfig.dataset_root)
    parser.add_argument("--dataset-name", default=CollectConfig.dataset_name)
    parser.add_argument("--repo-id", default=CollectConfig.repo_id)
    parser.add_argument("--task", default=CollectConfig.task)
    parser.add_argument("--max-episodes", type=int, default=CollectConfig.max_episodes)
    parser.add_argument(
        "--action-position-mode",
        choices=("relative", "absolute"),
        default=CollectConfig.action_position_mode,
        help="Store action position as target-state delta or absolute target TCP position.",
    )
    parser.add_argument(
        "--action-orientation-source",
        choices=("state", "ik_target"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fps", type=float, default=CollectConfig.fps)
    parser.add_argument("--front-image-topic", default=CollectConfig.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CollectConfig.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CollectConfig.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=CollectConfig.ik_target_topic)
    parser.add_argument("--joint-target-topic", default=CollectConfig.joint_target_topic)
    parser.add_argument("--vr-command-topic", default=CollectConfig.vr_command_topic)
    parser.add_argument("--max-dt-image", type=float, default=CollectConfig.max_dt_image)
    parser.add_argument("--max-dt-front-image", type=float, default=CollectConfig.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CollectConfig.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=CollectConfig.max_dt_state)
    parser.add_argument("--max-dt-action", type=float, default=CollectConfig.max_dt_action)
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--no-videos", action="store_true")

    # Deprecated compatibility options accepted by run_collect_data_tabs.sh.
    parser.add_argument("--camera-source", choices=("ros", "realsense"), default="ros")
    parser.add_argument("--reference-camera", choices=("front", "wrist", "timer"), default=CollectConfig.reference_camera)
    parser.add_argument("--camera-width", type=int, default=CollectConfig.camera_width)
    parser.add_argument("--camera-height", type=int, default=CollectConfig.camera_height)
    parser.add_argument("--camera-fps", type=int, default=CollectConfig.camera_fps)
    parser.add_argument("--front-camera-serial", default=CollectConfig.front_camera_serial)
    parser.add_argument("--wrist-camera-serial", default=CollectConfig.wrist_camera_serial)
    parser.add_argument("--preview-topic", default=CollectConfig.preview_topic)
    parser.add_argument("--preview", dest="preview", action="store_true", default=False)
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    parser.add_argument("--preview-window", dest="preview_window", action="store_true", default=False)
    parser.add_argument("--no-preview-window", dest="preview_window", action="store_false")
    parser.add_argument("--status-panel", dest="status_panel", action="store_true", default=CollectConfig.status_panel)
    parser.add_argument("--no-status-panel", dest="status_panel", action="store_false")
    parser.add_argument("--status-hz", type=float, default=CollectConfig.status_hz)
    parser.add_argument("--stop-left-grip-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--launch-realsense-ros", dest="launch_realsense_ros", action="store_true", default=False)
    parser.add_argument("--no-launch-realsense-ros", dest="launch_realsense_ros", action="store_false")
    parser.add_argument("--allow-stale-front", action="store_true", default=False)
    parser.add_argument("--strict-front", action="store_true", default=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> CollectConfig:
    if args.camera_source != "ros":
        raise ValueError("The rewritten collector only supports --camera-source ros.")
    max_dt_image = float(args.max_dt_image)
    return CollectConfig(
        fps=float(args.fps),
        max_dt_image=max_dt_image,
        max_dt_front_image=float(args.max_dt_front_image if args.max_dt_front_image is not None else max_dt_image),
        max_dt_wrist_image=float(args.max_dt_wrist_image if args.max_dt_wrist_image is not None else max_dt_image),
        max_dt_state=float(args.max_dt_state),
        max_dt_action=float(args.max_dt_action),
        allow_stale_front=False,
        front_image_topic=str(args.front_image_topic),
        wrist_image_topic=str(args.wrist_image_topic),
        robot_state_topic=str(args.robot_state_topic),
        ik_target_topic=str(args.ik_target_topic),
        joint_target_topic=str(args.joint_target_topic),
        vr_command_topic=str(args.vr_command_topic),
        reference_camera=str(args.reference_camera),
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        dataset_name=str(args.dataset_name),
        repo_id=str(args.repo_id),
        task=str(args.task),
        max_episodes=max(0, int(args.max_episodes)),
        action_position_mode=str(args.action_position_mode),
        use_videos=not bool(args.no_videos),
        resize=tuple(args.resize) if args.resize else None,
        status_panel=bool(args.status_panel),
        status_hz=float(args.status_hz),
    )


def main() -> int:
    args = parse_args()

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    cfg = make_config(args)
    rclpy.init()
    node = rclpy.create_node("ur3e_vr_impedance_lerobot_collector")
    collector = Ur3eVrImpedanceCollector(node, cfg)
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
