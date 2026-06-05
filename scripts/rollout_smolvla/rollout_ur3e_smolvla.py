#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.messages import make_ik_target, make_vr_command, parse_robot_state
from real_teleop.rotation_repr import continuous_rotvec_from_quat
from real_teleop.safety import SafetyLimiter
from scripts.rollout_smolvla.config import CFG


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

    def latest(self) -> tuple[Any | None, float | None]:
        with self._lock:
            if not self._data:
                return None, None
            sample_stamp, value = self._data[-1]
        return value, now_sec() - float(sample_stamp)


@dataclass(slots=True)
class ObservationPacket:
    stamp: float
    batch: dict[str, Any]
    dt_map: dict[str, float | None]


@dataclass(slots=True)
class ActionPacket:
    stamp: float
    action: np.ndarray
    inference_s: float


class SmolVLAInferencer:
    def __init__(
        self,
        policy_path: Path,
        device: str,
        *,
        replan_every_step: bool = False,
        execution_horizon: int = 10,
    ) -> None:
        # Importing this module registers SmolVLA processor steps used in saved processor configs.
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        self.replan_every_step = bool(replan_every_step)
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(
            policy_path,
            cli_overrides=[f"--device={device}"],
            local_files_only=True,
        )
        self.policy.eval()
        self.ckpt_action_horizon = int(getattr(self.policy.config, "n_action_steps", 1))
        chunk_size = int(getattr(self.policy.config, "chunk_size", self.ckpt_action_horizon))
        self.execution_horizon = max(1, min(int(execution_horizon), chunk_size))
        self.policy.config.n_action_steps = self.execution_horizon
        self.policy.reset()
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": device}},
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": "cpu"}},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.input_keys = {
            *self.policy.config.input_features.keys(),
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        }

    def infer(self, packet: ObservationPacket) -> ActionPacket:
        start = now_sec()
        with torch.inference_mode():
            batch = self.preprocessor(packet.batch)
            batch = {key: value for key, value in batch.items() if key in self.input_keys}
            if self.replan_every_step:
                self.policy.reset()
            action = self.policy.select_action(batch)
            action = self.postprocessor(action.detach().cpu())
        arr = np.asarray(action.detach().cpu(), dtype=np.float32).reshape(-1)
        return ActionPacket(stamp=packet.stamp, action=arr, inference_s=now_sec() - start)

    def infer_sequence(self, packet: ObservationPacket, count: int) -> list[ActionPacket]:
        start = now_sec()
        actions: list[np.ndarray] = []
        with torch.inference_mode():
            batch = self.preprocessor(packet.batch)
            batch = {key: value for key, value in batch.items() if key in self.input_keys}
            if self.replan_every_step:
                self.policy.reset()
            for _ in range(max(1, int(count))):
                action = self.policy.select_action(batch)
                action = self.postprocessor(action.detach().cpu())
                actions.append(np.asarray(action.detach().cpu(), dtype=np.float32).reshape(-1))
        inference_s = now_sec() - start
        return [ActionPacket(stamp=packet.stamp, action=action, inference_s=inference_s) for action in actions]


def now_sec() -> float:
    return time.monotonic()


def resolve_policy_path(path: Path | None, output_root: Path) -> Path:
    if path is not None:
        return normalize_policy_path(path.expanduser())

    candidates = []
    for pretrained in output_root.expanduser().glob("*/checkpoints/*/pretrained_model"):
        if (pretrained / "model.safetensors").exists():
            candidates.append(pretrained)
    if not candidates:
        raise FileNotFoundError(
            "No policy checkpoint found. Pass --policy-path pointing to a pretrained_model directory "
            "or a run directory containing checkpoints/last/pretrained_model."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def normalize_policy_path(path: Path) -> Path:
    path = path.resolve()
    if (path / "model.safetensors").exists():
        return path
    if (path / "pretrained_model" / "model.safetensors").exists():
        return path / "pretrained_model"
    last = path / "checkpoints" / "last" / "pretrained_model"
    if (last / "model.safetensors").exists():
        return last.resolve()
    checkpoints = list(path.glob("checkpoints/*/pretrained_model"))
    checkpoints = [p for p in checkpoints if (p / "model.safetensors").exists()]
    if checkpoints:
        return max(checkpoints, key=lambda p: p.stat().st_mtime).resolve()
    raise FileNotFoundError(f"Could not resolve a LeRobot pretrained_model directory from: {path}")


class AsyncSmolVLAWorker:
    def __init__(
        self,
        policy_path: Path,
        device: str,
        *,
        replan_every_step: bool = False,
        execution_horizon: int = 10,
    ) -> None:
        self.policy_path = policy_path
        self.device = device
        self.in_queue: queue.Queue[ObservationPacket] = queue.Queue(maxsize=1)
        self.latest: ActionPacket | None = None
        self.latest_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.inferencer = SmolVLAInferencer(
            policy_path,
            device,
            replan_every_step=replan_every_step,
            execution_horizon=execution_horizon,
        )
        self.thread = threading.Thread(target=self._run, name="smolvla_inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, packet: ObservationPacket) -> None:
        while True:
            try:
                self.in_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.in_queue.put_nowait(packet)
        except queue.Full:
            pass

    def get_latest(self) -> ActionPacket | None:
        with self.latest_lock:
            return self.latest

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    packet = self.in_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                action_packet = self.inferencer.infer(packet)
                with self.latest_lock:
                    self.latest = action_packet
        except BaseException as exc:
            self.error = exc


class SmolVLARolloutNode:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.safety = SafetyLimiter(TeleopConfig())
        self.last_sample_time = 0.0
        self.last_publish_time = 0.0
        self.last_log_time = 0.0
        self.last_sync_infer_start = 0.0
        self.sync_infer_busy = False
        self.sync_infer_lock = threading.Lock()
        self.last_target_pos: np.ndarray | None = None
        self.last_state_rotvec: np.ndarray | None = None
        self.last_publish_state_rotvec: np.ndarray | None = None
        self.latest_action: ActionPacket | None = None
        self.latest_action_lock = threading.Lock()
        self.sync_action_queue: deque[ActionPacket] = deque()
        self.start_time = now_sec()
        self.startup_sequence_enabled = bool(
            self.execute and args.return_home_on_start and not bool(getattr(args, "pre_model_startup_done", False))
        )
        self.policy_ready = not self.startup_sequence_enabled
        self.home_request_done = not self.startup_sequence_enabled
        self.home_release_done = self.home_request_done
        self.home_settle_done = self.home_request_done
        self.open_gripper_done = self.home_request_done
        self.preview_enabled = bool(args.preview)
        self.preview_failed = False
        self.preview_stop = threading.Event()
        self.preview_thread: threading.Thread | None = None
        self.topic_counts = {"front": 0, "wrist": 0, "state": 0}
        self.topic_last = {"front": 0.0, "wrist": 0.0, "state": 0.0}
        self.last_dt_map: dict[str, float | None] = {}

        self.buffers = {
            "front": SyncBuffer(args.buffer_maxlen),
            "wrist": SyncBuffer(args.buffer_maxlen),
            "state": SyncBuffer(args.buffer_maxlen),
        }

        self.inference_mode = str(args.inference_mode)
        self.worker: AsyncSmolVLAWorker | None = None
        self.inferencer: SmolVLAInferencer | None = None
        if self.inference_mode == "async":
            self.worker = AsyncSmolVLAWorker(
                args.policy_path,
                args.device,
                replan_every_step=args.replan_every_step,
                execution_horizon=args.execution_horizon,
            )
            self.worker.start()
            policy_config = self.worker.inferencer.policy.config
            ckpt_action_horizon = self.worker.inferencer.ckpt_action_horizon
            execution_horizon = self.worker.inferencer.execution_horizon
        else:
            self.inferencer = SmolVLAInferencer(
                args.policy_path,
                args.device,
                replan_every_step=args.replan_every_step,
                execution_horizon=args.execution_horizon,
            )
            policy_config = self.inferencer.policy.config
            ckpt_action_horizon = self.inferencer.ckpt_action_horizon
            execution_horizon = self.inferencer.execution_horizon
        self.execution_horizon = int(execution_horizon)

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
        self.sub_group = ReentrantCallbackGroup()
        self.sample_group = MutuallyExclusiveCallbackGroup()
        self.publish_group = MutuallyExclusiveCallbackGroup()
        self.home_group = MutuallyExclusiveCallbackGroup()

        self.pose_pub = node.create_publisher(Float64MultiArray, args.ik_target_topic, data_qos)
        self.command_pub = node.create_publisher(Float64MultiArray, args.vr_command_topic, data_qos)
        self.subs = [
            node.create_subscription(
                Image,
                args.front_image_topic,
                self._on_front,
                image_qos,
                callback_group=self.sub_group,
            ),
            node.create_subscription(
                Image,
                args.wrist_image_topic,
                self._on_wrist,
                image_qos,
                callback_group=self.sub_group,
            ),
            node.create_subscription(
                Float64MultiArray,
                args.robot_state_topic,
                self._on_state,
                data_qos,
                callback_group=self.sub_group,
            ),
        ]
        self.sample_timer = node.create_timer(
            1.0 / max(args.fps, 1.0),
            self._sample_tick,
            callback_group=self.sample_group,
        )
        self.publish_timer = node.create_timer(
            1.0 / max(args.command_hz, 1.0),
            self._publish_tick,
            callback_group=self.publish_group,
        )
        self.start_home_timer = node.create_timer(0.05, self._start_home_tick, callback_group=self.home_group)
        if self.preview_enabled:
            self.preview_thread = threading.Thread(target=self._preview_loop, name="rollout_preview", daemon=True)
            self.preview_thread.start()

        mode = "EXECUTE" if self.execute else "DRY-RUN"
        node.get_logger().info(
            f"SmolVLA rollout ready ({mode}). policy={args.policy_path}, fps={args.fps:.1f}, "
            f"command_hz={args.command_hz:.1f}, inference_mode={self.inference_mode}, "
            f"chunk_size={policy_config.chunk_size}, execution_horizon={execution_horizon}/{ckpt_action_horizon}, "
            f"replan_every_step={args.replan_every_step}, return_home_on_start={args.return_home_on_start}, "
            f"start_home_settle_s={args.start_home_settle_s:.2f}, "
            f"start_open_gripper_s={args.start_open_gripper_s:.2f}, "
            f"preview={self.preview_enabled}, sync_inference_hz={args.sync_inference_hz:.2f}, "
            f"action_position_mode={args.action_position_mode}, "
            f"action_orientation_source={args.action_orientation_source}, "
            f"action_repr_source={getattr(args, 'action_repr_source', 'manual')}"
        )

    def close(self) -> None:
        self.preview_stop.set()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)
        if self.preview_enabled and not self.preview_failed:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        if self.worker is not None:
            self.worker.close()

    def _set_latest_action(self, packet: ActionPacket) -> None:
        with self.latest_action_lock:
            self.latest_action = packet

    def _get_latest_action(self) -> ActionPacket | None:
        if self.worker is not None:
            return self.worker.get_latest()
        with self.latest_action_lock:
            if self.sync_action_queue:
                return self.sync_action_queue.popleft()
            return None

    def _push(self, key: str, stamp: float, value: Any) -> None:
        self.buffers[key].push(stamp, value)
        self.topic_counts[key] += 1
        self.topic_last[key] = now_sec()

    def _on_front(self, msg) -> None:
        self._push("front", now_sec(), msg)

    def _on_wrist(self, msg) -> None:
        self._push("wrist", now_sec(), msg)

    def _on_state(self, msg) -> None:
        try:
            payload = parse_robot_state(msg.data)
        except Exception as exc:
            self.node.get_logger().warn(f"Bad robot_state: {exc}")
            return
        self._push("state", float(payload.get("stamp", now_sec())), payload)

    def _sample_tick(self) -> None:
        if self.worker is not None and self.worker.error is not None:
            self.node.get_logger().error(f"SmolVLA inference failed: {self.worker.error}")
            return
        if not self.policy_ready:
            if now_sec() - self.last_log_time > 2.0:
                self.last_log_time = now_sec()
                self.node.get_logger().info("Waiting for startup home/open-gripper sequence before inference.")
            return
        stamp = now_sec()
        front, dt_front = self.buffers["front"].nearest(stamp, self.args.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.args.max_dt_wrist_image)
        state, dt_state = self.buffers["state"].nearest(stamp, self.args.max_dt_state)
        self.last_dt_map = {"front": dt_front, "wrist": dt_wrist, "state": dt_state}
        if front is None or wrist is None or state is None:
            if stamp - self.last_log_time > 2.0:
                self.last_log_time = stamp
                self.node.get_logger().warn(f"Waiting for aligned observation: {self._format_status()}")
            return
        try:
            batch = {
                "observation.images.cam_front": image_msg_to_tensor(front),
                "observation.images.cam_wrist": image_msg_to_tensor(wrist),
                "observation.state": self._robot_state_to_tensor(state, rl_mark=float(self.args.rl_mark)),
                "task": self.args.task,
            }
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build observation: {exc}")
            return
        packet = ObservationPacket(stamp=stamp, batch=batch, dt_map=self.last_dt_map.copy())
        if self.worker is not None:
            self.worker.submit(packet)
            return
        if self.inferencer is None:
            return
        with self.latest_action_lock:
            if self.sync_action_queue:
                return
        if self.args.replan_every_step and stamp - self.last_sync_infer_start < 1.0 / max(
            self.args.sync_inference_hz, 1e-6
        ):
            return
        if not self.sync_infer_lock.acquire(blocking=False):
            return
        self.sync_infer_busy = True
        self.last_sync_infer_start = stamp
        try:
            sequence_len = 1 if self.args.replan_every_step else self.execution_horizon
            action_packets = self.inferencer.infer_sequence(packet, sequence_len)
            with self.latest_action_lock:
                self.sync_action_queue.extend(action_packets)
                self.latest_action = action_packets[-1] if action_packets else None
            if stamp - self.last_log_time > 1.0 / max(self.args.dry_run_log_hz, 1e-6):
                self.last_log_time = stamp
                self.node.get_logger().info(
                    f"sync inference done actions={len(action_packets)} "
                    f"infer={(action_packets[0].inference_s if action_packets else 0.0) * 1000.0:.0f}ms "
                    f"{self._format_status()}"
                )
        except Exception as exc:
            self.node.get_logger().error(f"SmolVLA synchronous inference failed: {exc}")
        finally:
            self.sync_infer_busy = False
            self.sync_infer_lock.release()

    def _start_home_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        if not self.startup_sequence_enabled:
            if not self.execute and not self.policy_ready:
                self.node.get_logger().info("Dry-run: startup return_to_home skipped.")
            self._mark_policy_ready("startup sequence skipped")
            try:
                self.start_home_timer.cancel()
            except Exception:
                pass
            return

        elapsed = now_sec() - self.start_time
        if not self.home_request_done:
            if elapsed < self.args.start_home_delay_s:
                return
            msg = Float64MultiArray()
            msg.data = make_vr_command({"enable": False, "gripper": self._current_gripper(), "home": True})
            self.command_pub.publish(msg)
            self.home_request_done = True
            self.node.get_logger().info("Startup return_to_home requested once.")
            return

        if not self.home_release_done:
            if elapsed < self.args.start_home_delay_s + self.args.start_home_pulse_s:
                return
            msg = Float64MultiArray()
            msg.data = make_vr_command({"enable": False, "gripper": self._current_gripper(), "home": False})
            self.command_pub.publish(msg)
            self.home_release_done = True
            self.node.get_logger().info(
                f"Startup return_to_home released; waiting {self.args.start_home_settle_s:.2f}s before opening gripper."
            )
            return

        settle_end = self.args.start_home_delay_s + self.args.start_home_pulse_s + self.args.start_home_settle_s
        if not self.home_settle_done:
            if elapsed < settle_end:
                return
            self.home_settle_done = True
            self.node.get_logger().info("Startup home settle done; opening gripper before policy inference.")

        open_end = settle_end + self.args.start_open_gripper_s
        if not self.open_gripper_done:
            msg = Float64MultiArray()
            msg.data = make_vr_command(
                {"enable": False, "gripper": self.args.start_open_gripper_value, "home": False}
            )
            self.command_pub.publish(msg)
            if elapsed < open_end:
                return
            self.open_gripper_done = True
            self._mark_policy_ready("startup home and gripper-open sequence complete")
            try:
                self.start_home_timer.cancel()
            except Exception:
                pass
            return

    def _current_gripper(self) -> float:
        state, _ = self.buffers["state"].latest()
        if isinstance(state, dict):
            return float(np.clip(state.get("gripper", 0.0), 0.0, self.args.gripper_max))
        return 0.0

    def _mark_policy_ready(self, reason: str) -> None:
        if self.policy_ready:
            return
        with self.latest_action_lock:
            self.latest_action = None
        self.last_target_pos = None
        self.last_state_rotvec = None
        self.last_publish_state_rotvec = None
        self.last_sample_time = 0.0
        self.last_sync_infer_start = 0.0
        if self.inferencer is not None:
            self.inferencer.policy.reset()
        if self.worker is not None:
            while True:
                try:
                    self.worker.in_queue.get_nowait()
                except queue.Empty:
                    break
            with self.worker.latest_lock:
                self.worker.latest = None
        with self.latest_action_lock:
            self.sync_action_queue.clear()
        self.policy_ready = True
        self.node.get_logger().info(f"Policy inference enabled: {reason}.")

    def _preview_tick(self) -> None:
        if self.preview_failed:
            return
        front, front_age = self.buffers["front"].latest()
        wrist, wrist_age = self.buffers["wrist"].latest()
        try:
            frame = make_preview_frame(front, wrist, front_age, wrist_age)
            import cv2

            cv2.imshow("UR3e SmolVLA Rollout", frame)
            cv2.waitKey(1)
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV rollout preview disabled: {exc}")

    def _preview_loop(self) -> None:
        try:
            import cv2

            cv2.namedWindow("UR3e SmolVLA Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e SmolVLA Rollout", 1280, 480)
            self.node.get_logger().info("OpenCV rollout preview window started.")
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV rollout preview disabled: {exc}")
            return

        period = 1.0 / max(self.args.preview_hz, 1.0)
        next_t = now_sec()
        while not self.preview_stop.is_set() and not self.preview_failed:
            self._preview_tick()
            next_t += period
            time.sleep(max(0.001, next_t - now_sec()))

    def _robot_state_to_tensor(self, robot_state: dict[str, Any], *, rl_mark: float) -> torch.Tensor:
        tensor, rotvec = robot_state_to_tensor(
            robot_state,
            rl_mark=rl_mark,
            previous_rotvec=self.last_state_rotvec,
        )
        self.last_state_rotvec = rotvec.astype(float)
        return tensor

    def _publish_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        if not self.policy_ready:
            return
        packet = self._get_latest_action()
        stamp = now_sec()
        if packet is None:
            return
        action_age = stamp - packet.stamp
        if action_age > self.args.max_action_age_s:
            if stamp - self.last_log_time > 2.0:
                self.last_log_time = stamp
                self.node.get_logger().warn(f"Holding: stale model action age={action_age:.3f}s")
            return

        action = packet.action
        if action.size < 8:
            self.node.get_logger().warn(f"Bad model action shape: {action.shape}")
            return

        state, state_age = self.buffers["state"].latest()
        if not isinstance(state, dict):
            return
        if state_age is None or state_age > self.args.max_dt_state:
            if stamp - self.last_log_time > 2.0:
                self.last_log_time = stamp
                self.node.get_logger().warn(
                    f"Holding: stale robot state age={'none' if state_age is None else f'{state_age:.3f}s'}"
                )
            return

        current_pos = np.asarray(state["tcp_pos"], dtype=float)
        raw_pos = np.asarray(action[:3], dtype=float)
        if self.args.action_position_mode == "relative":
            pos = current_pos + raw_pos
        elif self.args.action_position_mode == "absolute":
            pos = raw_pos
        else:
            self.node.get_logger().warn(f"Bad action_position_mode: {self.args.action_position_mode!r}")
            return
        pos = self.safety.clamp_impedance_workspace(pos)
        pos = self._limit_position_step(pos)

        if self.args.action_orientation_source == "state":
            quat = np.asarray(state["tcp_quat"], dtype=float)
            rotvec = continuous_rotvec_from_quat(quat, self.last_publish_state_rotvec)
            self.last_publish_state_rotvec = rotvec.astype(float)
        elif self.args.action_orientation_source == "ik_target":
            rotvec = np.asarray(action[3:6], dtype=float)
            quat = R.from_rotvec(rotvec).as_quat()
        else:
            self.node.get_logger().warn(f"Bad action_orientation_source: {self.args.action_orientation_source!r}")
            return
        gripper = float(np.clip(action[6], 0.0, self.args.gripper_max))

        if self.execute:
            pose_msg = Float64MultiArray()
            pose_msg.data = make_ik_target(pos, quat)
            self.pose_pub.publish(pose_msg)

            command_msg = Float64MultiArray()
            command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
            self.command_pub.publish(command_msg)

        if stamp - self.last_log_time > 1.0 / max(self.args.dry_run_log_hz, 1e-6):
            self.last_log_time = stamp
            prefix = "publish" if self.execute else "dry-run"
            pos_label = "delta" if self.args.action_position_mode == "relative" else "raw_pos"
            self.node.get_logger().info(
                f"{prefix} {pos_label}={fmt_vec(raw_pos)} pos={fmt_vec(pos)} "
                f"rotvec={fmt_vec(rotvec)} gripper={gripper:.3f} "
                f"action_age={action_age * 1000.0:.0f}ms infer={packet.inference_s * 1000.0:.0f}ms "
                f"{self._format_status()}"
            )

    def _limit_position_step(self, pos: np.ndarray) -> np.ndarray:
        if self.last_target_pos is None:
            self.last_target_pos = pos.copy()
            return pos
        delta = pos - self.last_target_pos
        norm = float(np.linalg.norm(delta))
        limit = float(max(self.args.max_position_step_m, 1e-6))
        if norm > limit:
            pos = self.last_target_pos + delta / norm * limit
        self.last_target_pos = pos.copy()
        return pos

    def _format_status(self) -> str:
        dt = " ".join(
            f"{key}={'none' if value is None else f'{value * 1000.0:.0f}ms'}"
            for key, value in self.last_dt_map.items()
        )
        now = now_sec()
        topics = " ".join(
            f"{key}#{self.topic_counts[key]}@"
            f"{'none' if self.topic_counts[key] <= 0 else f'{(now - self.topic_last[key]) * 1000.0:.0f}ms'}"
            for key in ("front", "wrist", "state")
        )
        return f"sync[{dt}] topics[{topics}]"


def image_msg_to_tensor(msg) -> torch.Tensor:
    image = decode_image_msg(msg)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0


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


def robot_state_to_tensor(
    robot_state: dict[str, Any],
    *,
    rl_mark: float,
    previous_rotvec: np.ndarray | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    pos = np.asarray(robot_state["tcp_pos"], dtype=np.float32)
    quat = np.asarray(robot_state["tcp_quat"], dtype=float)
    rotvec = continuous_rotvec_from_quat(quat, previous_rotvec)
    gripper = np.float32(np.clip(robot_state.get("gripper", 0.0), 0.0, CFG.gripper_max))
    state = np.concatenate([pos, rotvec, [gripper, np.float32(rl_mark)]]).astype(np.float32)
    return torch.from_numpy(state), rotvec


def make_preview_frame(front_msg, wrist_msg, front_age: float | None, wrist_age: float | None) -> np.ndarray:
    import cv2

    front = preview_image(front_msg, "front D455", front_age)
    wrist = preview_image(wrist_msg, "wrist D405", wrist_age)
    height = min(front.shape[0], wrist.shape[0])
    front = resize_to_height(front, height)
    wrist = resize_to_height(wrist, height)
    return np.hstack([front, wrist])


def preview_image(msg, label: str, age: float | None) -> np.ndarray:
    import cv2

    if msg is None:
        image = np.zeros((360, 480, 3), dtype=np.uint8)
        status = "waiting"
    else:
        image = cv2.cvtColor(decode_image_msg(msg), cv2.COLOR_RGB2BGR)
        status = f"age={age * 1000.0:.0f}ms" if age is not None else "age=none"
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"{label}  {status}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    import cv2

    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v): .4f}" for v in vec) + "]"


def infer_action_representation(policy_path: Path) -> dict[str, str | None]:
    """Infer action decoding from the dataset used to train this checkpoint.

    Newer collectors write this explicitly in sync reports.  Older impedance
    datasets did not, so we fall back to simple stats: absolute TCP actions have
    position means/ranges comparable to observation.state, while relative
    actions stay close to zero.
    """
    result: dict[str, str | None] = {"position": None, "orientation": None, "source": None}
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        return result
    try:
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    except Exception:
        return result

    dataset_root = None
    dataset_cfg = train_config.get("dataset")
    if isinstance(dataset_cfg, dict):
        root_value = dataset_cfg.get("root")
        if root_value:
            dataset_root = Path(root_value).expanduser()
    if dataset_root is None or not dataset_root.exists():
        return result

    reports = sorted((dataset_root / "meta").glob("sync_report_episode_*.json"))
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rep = report.get("action_representation") or report.get("representation")
        if not isinstance(rep, dict):
            continue
        position = rep.get("position") or rep.get("ee_action_position_mode")
        orientation = rep.get("orientation") or rep.get("action_orientation_source")
        if position in {"relative", "absolute"}:
            result["position"] = str(position)
        if orientation in {"state", "ik_target"}:
            result["orientation"] = str(orientation)
        if result["position"] or result["orientation"]:
            result["source"] = str(report_path)
            return result

    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        return result
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return result

    action_stats = stats.get("action")
    state_stats = stats.get("observation.state")
    if not isinstance(action_stats, dict):
        return result
    try:
        action_mean = np.asarray(action_stats.get("mean"), dtype=float).reshape(-1)
        action_std = np.asarray(action_stats.get("std"), dtype=float).reshape(-1)
        action_min = np.asarray(action_stats.get("min"), dtype=float).reshape(-1)
        action_max = np.asarray(action_stats.get("max"), dtype=float).reshape(-1)
    except Exception:
        return result

    if action_mean.size >= 6 and action_std.size >= 6 and action_min.size >= 6 and action_max.size >= 6:
        pos_mean_norm = float(np.linalg.norm(action_mean[:3]))
        pos_abs_range = float(np.max(np.abs(np.concatenate([action_min[:3], action_max[:3]]))))
        if pos_mean_norm > 0.12 or pos_abs_range > 0.20:
            result["position"] = "absolute"
        else:
            result["position"] = "relative"

        rot_range = action_max[3:6] - action_min[3:6]
        rot_std_norm = float(np.linalg.norm(action_std[3:6]))
        result["orientation"] = "ik_target" if rot_std_norm < 0.02 and float(np.max(rot_range)) < 0.05 else "state"

        if isinstance(state_stats, dict):
            try:
                state_mean = np.asarray(state_stats.get("mean"), dtype=float).reshape(-1)
                if state_mean.size >= 6 and np.linalg.norm(action_mean[:3] - state_mean[:3]) < 0.08:
                    result["position"] = "absolute"
            except Exception:
                pass
        result["source"] = str(stats_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async SmolVLA rollout for UR3e impedance control.")
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--output-root", type=Path, default=CFG.output_root)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CFG.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=CFG.ik_target_topic)
    parser.add_argument("--vr-command-topic", default=CFG.vr_command_topic)
    parser.add_argument("--fps", type=float, default=CFG.fps)
    parser.add_argument("--command-hz", type=float, default=CFG.command_hz)
    parser.add_argument("--inference-mode", choices=["async", "sync"], default=CFG.inference_mode)
    parser.add_argument(
        "--replan-every-step",
        action="store_true",
        default=CFG.replan_every_step,
        help="Reset SmolVLA action queue before every inference and use the first action of a fresh chunk.",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=CFG.execution_horizon,
        help="Number of queued actions to execute from each SmolVLA chunk before replanning.",
    )
    parser.add_argument("--sync-inference-hz", type=float, default=CFG.sync_inference_hz)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=CFG.max_dt_state)
    parser.add_argument("--buffer-maxlen", type=int, default=CFG.buffer_maxlen)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--gripper-max", type=float, default=CFG.gripper_max)
    parser.add_argument("--max-action-age-s", type=float, default=CFG.max_action_age_s)
    parser.add_argument("--max-position-step-m", type=float, default=CFG.max_position_step_m)
    parser.add_argument(
        "--action-position-mode",
        choices=("relative", "absolute"),
        default=CFG.action_position_mode,
        help="Decode action[0:3] as current-state delta or absolute target TCP position. "
        "Default: infer from checkpoint training metadata.",
    )
    parser.add_argument(
        "--action-orientation-source",
        choices=("state", "ik_target"),
        default=CFG.action_orientation_source,
        help="Use current robot-state orientation or model action[3:6] as absolute target orientation. "
        "Default: infer from checkpoint training metadata.",
    )
    parser.add_argument("--dry-run-log-hz", type=float, default=CFG.dry_run_log_hz)
    parser.add_argument("--rl-mark", type=float, default=0.0)
    parser.add_argument("--return-home-on-start", action=argparse.BooleanOptionalAction, default=CFG.return_home_on_start)
    parser.add_argument("--start-home-delay-s", type=float, default=CFG.start_home_delay_s)
    parser.add_argument("--start-home-pulse-s", type=float, default=CFG.start_home_pulse_s)
    parser.add_argument("--start-home-settle-s", type=float, default=CFG.start_home_settle_s)
    parser.add_argument("--start-open-gripper-s", type=float, default=CFG.start_open_gripper_s)
    parser.add_argument("--start-open-gripper-value", type=float, default=CFG.start_open_gripper_value)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=CFG.preview)
    parser.add_argument("--preview-hz", type=float, default=CFG.preview_hz)
    parser.add_argument("--execute", action="store_true", help="Publish model actions to robot topics.")
    return parser.parse_args()


def run_pre_model_startup_sequence(node, args: argparse.Namespace) -> bool:
    """Return home and open the gripper before loading the policy.

    This intentionally runs before SmolVLARolloutNode is constructed, because
    that constructor loads the policy weights.
    """
    if not (bool(args.execute) and bool(args.return_home_on_start)):
        return False

    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    pub = node.create_publisher(Float64MultiArray, args.vr_command_topic, qos)

    def spin_sleep(duration_s: float) -> None:
        end = now_sec() + max(0.0, float(duration_s))
        while rclpy.ok() and now_sec() < end:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, end - now_sec())))

    def publish(*, home: bool, gripper: float) -> None:
        msg = Float64MultiArray()
        msg.data = make_vr_command({"enable": False, "gripper": float(gripper), "home": bool(home)})
        pub.publish(msg)

    open_gripper = float(np.clip(args.start_open_gripper_value, 0.0, args.gripper_max))
    node.get_logger().info(
        "Pre-model startup: return_to_home and open gripper before loading policy "
        f"(delay={args.start_home_delay_s:.2f}s, pulse={args.start_home_pulse_s:.2f}s, "
        f"settle={args.start_home_settle_s:.2f}s, open={args.start_open_gripper_s:.2f}s)."
    )

    # Give ROS discovery a brief chance before the first command.
    spin_sleep(min(0.25, max(0.0, args.start_home_delay_s)))
    spin_sleep(max(0.0, args.start_home_delay_s - 0.25))

    home_end = now_sec() + max(0.0, float(args.start_home_pulse_s))
    while rclpy.ok() and now_sec() < home_end:
        publish(home=True, gripper=open_gripper)
        spin_sleep(0.05)
    publish(home=False, gripper=open_gripper)
    node.get_logger().info("Pre-model startup: return_to_home released; waiting for home settle.")

    spin_sleep(args.start_home_settle_s)

    open_end = now_sec() + max(0.0, float(args.start_open_gripper_s))
    while rclpy.ok() and now_sec() < open_end:
        publish(home=False, gripper=open_gripper)
        spin_sleep(0.05)
    publish(home=False, gripper=open_gripper)
    node.get_logger().info("Pre-model startup: gripper opened; loading policy now.")
    return True


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(CFG.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CFG.hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.output_root = args.output_root.expanduser()
    if not args.output_root.is_absolute():
        args.output_root = (REPO_ROOT / args.output_root).resolve()
    args.policy_path = resolve_policy_path(args.policy_path, args.output_root)
    inferred_representation = infer_action_representation(args.policy_path)
    if args.action_position_mode is None:
        args.action_position_mode = inferred_representation.get("position") or CollectConfig.action_position_mode
    if args.action_orientation_source is None:
        args.action_orientation_source = (
            inferred_representation.get("orientation") or CollectConfig.action_orientation_source
        )
    args.action_repr_source = inferred_representation.get("source") or "config/default"

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("ur3e_smolvla_rollout")
    args.pre_model_startup_done = run_pre_model_startup_sequence(node, args)
    rollout = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        rollout = SmolVLARolloutNode(node, args)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rollout is not None:
            rollout.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
