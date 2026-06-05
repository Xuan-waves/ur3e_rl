#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.messages import make_joint_target, make_vr_command, parse_robot_state
from vr_servoj_test.rollout.config import CFG


def now_monotonic() -> float:
    return time.monotonic()


def stamp_to_sec(stamp: Any) -> float:
    return float(getattr(stamp, "sec", 0.0)) + float(getattr(stamp, "nanosec", 0.0)) * 1e-9


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

    def latest(self, *, now: float | None = None) -> tuple[Any | None, float | None]:
        with self._lock:
            if not self._data:
                return None, None
            sample_stamp, value = self._data[-1]
        return value, (now_monotonic() if now is None else float(now)) - float(sample_stamp)


@dataclass(slots=True)
class ObservationPacket:
    stamp: float
    batch: dict[str, Any]
    dt_map: dict[str, float | None]


@dataclass(slots=True)
class ActionPacket:
    ready_stamp: float
    obs_stamp: float
    action: np.ndarray
    inference_s: float


def resolve_policy_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()

    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    candidates.extend(sorted(path.glob("checkpoints/*/pretrained_model")))
    for candidate in candidates:
        if (candidate / "model.safetensors").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve LeRobot pretrained_model directory from: {path}")


class SmolVLASyncRunner:
    def __init__(
        self,
        policy_path: Path,
        device: str,
        *,
        execution_horizon: int,
        replan_every_step: bool,
        amp: bool,
    ) -> None:
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

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.amp = bool(amp and device == "cuda")
        self.replan_every_step = bool(replan_every_step)
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

    def infer_sequence(self, packet: ObservationPacket, count: int) -> list[ActionPacket]:
        start = now_monotonic()
        actions: list[np.ndarray] = []
        autocast = torch.autocast(device_type="cuda") if self.amp else nullcontext()
        with torch.inference_mode(), autocast:
            batch = self.preprocessor(packet.batch)
            batch = {key: value for key, value in batch.items() if key in self.input_keys}
            if self.replan_every_step:
                self.policy.reset()
            for _ in range(max(1, int(count))):
                raw_action = self.policy.select_action(batch)
                action = self.postprocessor(raw_action.detach().cpu())
                actions.append(np.asarray(action.detach().cpu(), dtype=np.float32).reshape(-1))
        inference_s = now_monotonic() - start
        ready_stamp = packet.stamp + inference_s
        return [ActionPacket(ready_stamp, packet.stamp, action, inference_s) for action in actions]


class ServoJSmolVLARollout:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.policy_path = resolve_policy_path(args.policy_path)
        self.sync_reference = str(args.sync_reference).lower()
        if self.sync_reference not in {"front", "wrist", "timer"}:
            raise ValueError("--sync-reference must be front, wrist, or timer")

        self.joint_limits = np.asarray(TeleopConfig().joint_limits, dtype=np.float32)
        self.pending_reference_stamps: queue.SimpleQueue[float] = queue.SimpleQueue()
        self.action_queue: deque[ActionPacket] = deque()
        self.action_lock = threading.Lock()
        self.current_action: ActionPacket | None = None
        self.last_action_step_time = 0.0
        self.last_published_q: np.ndarray | None = None
        self.last_published_gripper: float | None = None
        self.infer_lock = threading.Lock()
        self.infer_busy = False
        self.last_log_time = 0.0
        self.last_dt_map: dict[str, float | None] = {}
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
        self.joint_pub = node.create_publisher(Float64MultiArray, args.joint_target_topic, data_qos)
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
            f"UR3e servoJ SmolVLA rollout ready ({mode}). policy={self.policy_path}, "
            f"task={args.task!r}, fps={args.fps:.1f}, command_hz={args.command_hz:.1f}, "
            f"action_step_hz={args.action_step_hz:.1f}, prefetch_actions={args.prefetch_actions}, "
            f"sync_reference={self.sync_reference}, horizon={self.runner.execution_horizon}/"
            f"{self.runner.ckpt_action_horizon}, replan_every_step={args.replan_every_step}, "
            f"device={self.runner.device}, amp={self.runner.amp}, hf_home={args.hf_home}, "
            f"offline={args.offline}, preview={args.preview}"
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
            queue_len = len(self.action_queue)
            if queue_len > int(self.args.prefetch_actions):
                return
        with self.infer_lock:
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
        with self.infer_lock:
            self.infer_busy = True
        threading.Thread(
            target=self._run_inference,
            args=(packet, count),
            name="servoj_smolvla_sync_inference",
            daemon=True,
        ).start()

    def _run_inference(self, packet: ObservationPacket, count: int) -> None:
        try:
            actions = self.runner.infer_sequence(packet, count)
        except Exception as exc:
            self.node.get_logger().error(f"SmolVLA inference failed: {exc}")
            with self.infer_lock:
                self.infer_busy = False
            return

        with self.action_lock:
            if self.args.replace_queue_on_infer:
                self.action_queue.clear()
            self.action_queue.extend(actions)
            ready_stamp = self.now_sec()
            for action in self.action_queue:
                action.ready_stamp = ready_stamp
        with self.infer_lock:
            self.infer_busy = False
        self._log_info(
            f"infer actions={len(actions)} infer={actions[0].inference_s * 1000.0:.0f}ms "
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
            batch = {
                "observation.images.cam_front": image_msg_to_tensor(front),
                "observation.images.cam_wrist": image_msg_to_tensor(wrist),
                "observation.state": robot_state_to_joint_tensor(
                    state,
                    rl_mark=self.args.rl_mark,
                    gripper_max=self.args.gripper_max,
                ),
                "task": self.args.task,
            }
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build observation: {exc}")
            return None
        return ObservationPacket(stamp=stamp, batch=batch, dt_map=self.last_dt_map.copy())

    def _publish_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        stamp = self.now_sec()
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
        if action.size < 7:
            self.node.get_logger().warn(f"Bad action shape: {action.shape}")
            return

        q = np.clip(action[:6], self.joint_limits[:, 0], self.joint_limits[:, 1])
        gripper = float(np.clip(action[6], 0.0, self.args.gripper_max))
        q, gripper = self._smooth_action(q, gripper)
        if self.execute:
            msg = Float64MultiArray()
            msg.data = make_joint_target(tracking=True, q=q, gripper=gripper, reason="tracking", ok=True)
            self.joint_pub.publish(msg)

        prefix = "publish" if self.execute else "dry-run"
        self._log_info(
            f"{prefix} q={fmt_vec(q)} gripper={gripper:.3f} "
            f"queue={len(self.action_queue)} action_age={action_age * 1000.0:.0f}ms"
        )

    def _smooth_action(self, q: np.ndarray, gripper: float) -> tuple[np.ndarray, float]:
        q = np.asarray(q, dtype=np.float32).copy()
        if self.last_published_q is not None:
            alpha = float(np.clip(self.args.action_q_filter_alpha, 0.0, 1.0))
            q = alpha * q + (1.0 - alpha) * self.last_published_q
            max_step = float(max(self.args.max_action_joint_step, 1e-6))
            delta = np.clip(q - self.last_published_q, -max_step, max_step)
            q = self.last_published_q + delta
        q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1]).astype(np.float32)
        self.last_published_q = q.copy()

        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * float(gripper) + (1.0 - alpha_g) * self.last_published_gripper
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        self.last_published_gripper = gripper
        return q, gripper

    def _preview_loop(self) -> None:
        try:
            import cv2

            cv2.namedWindow("UR3e ServoJ SmolVLA Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e ServoJ SmolVLA Rollout", 1280, 480)
            self.node.get_logger().info("OpenCV rollout preview window started.")
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV rollout preview disabled: {exc}")
            return

        period = 1.0 / max(self.args.preview_hz, 1.0)
        while not self.preview_stop.is_set() and not self.preview_failed:
            try:
                import cv2

                now = self.now_sec()
                front, front_age = self.buffers["front"].latest(now=now)
                wrist, wrist_age = self.buffers["wrist"].latest(now=now)
                cv2.imshow("UR3e ServoJ SmolVLA Rollout", make_preview_frame(front, wrist, front_age, wrist_age))
                cv2.waitKey(1)
            except Exception as exc:
                self.preview_failed = True
                self.node.get_logger().warn(f"OpenCV rollout preview disabled: {exc}")
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


def image_msg_to_tensor(msg) -> torch.Tensor:
    image = decode_image_msg(msg)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0


def decode_image_msg(msg) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding == "mono8":
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        gray = flat[: int(msg.step) * int(msg.height)].reshape((int(msg.height), int(msg.step)))
        gray = gray[:, : int(msg.width)]
        return np.repeat(gray[:, :, None], 3, axis=2).copy()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
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


def robot_state_to_joint_tensor(robot_state: dict[str, Any], *, rl_mark: float, gripper_max: float) -> torch.Tensor:
    q = np.asarray(robot_state["q"], dtype=np.float32).reshape(6)
    gripper = np.float32(np.clip(robot_state.get("gripper", 0.0), 0.0, gripper_max))
    state = np.concatenate([q, [gripper, np.float32(rl_mark)]]).astype(np.float32)
    return torch.from_numpy(state)


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
    cv2.putText(image, f"{label}  {status}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return image


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    import cv2

    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v): .3f}" for v in vec) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous SmolVLA rollout for UR3e servoJ jointspace control.")
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CFG.robot_state_topic)
    parser.add_argument("--joint-target-topic", default=CFG.joint_target_topic)
    parser.add_argument("--vr-command-topic", default=CFG.vr_command_topic)
    parser.add_argument("--fps", type=float, default=CFG.fps)
    parser.add_argument("--command-hz", type=float, default=CFG.command_hz)
    parser.add_argument("--action-step-hz", type=float, default=CFG.action_step_hz)
    parser.add_argument("--execution-horizon", type=int, default=CFG.execution_horizon)
    parser.add_argument("--prefetch-actions", type=int, default=CFG.prefetch_actions)
    parser.add_argument("--replace-queue-on-infer", action=argparse.BooleanOptionalAction, default=CFG.replace_queue_on_infer)
    parser.add_argument("--replan-every-step", action="store_true", default=CFG.replan_every_step)
    parser.add_argument("--sync-reference", choices=("front", "wrist", "timer"), default=CFG.sync_reference)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=CFG.max_dt_state)
    parser.add_argument("--buffer-maxlen", type=int, default=CFG.buffer_maxlen)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=CFG.amp)
    parser.add_argument("--hf-home", type=Path, default=CFG.hf_home)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=CFG.offline)
    parser.add_argument("--rl-mark", type=float, default=CFG.rl_mark)
    parser.add_argument("--gripper-max", type=float, default=CFG.gripper_max)
    parser.add_argument("--max-action-age-s", type=float, default=CFG.max_action_age_s)
    parser.add_argument("--action-q-filter-alpha", type=float, default=CFG.action_q_filter_alpha)
    parser.add_argument("--action-gripper-filter-alpha", type=float, default=CFG.action_gripper_filter_alpha)
    parser.add_argument("--max-action-joint-step", type=float, default=CFG.max_action_joint_step)
    parser.add_argument("--log-hz", type=float, default=CFG.log_hz)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=CFG.preview)
    parser.add_argument("--preview-hz", type=float, default=CFG.preview_hz)
    parser.add_argument("--return-home-on-start", action=argparse.BooleanOptionalAction, default=CFG.return_home_on_start)
    parser.add_argument("--start-home-delay-s", type=float, default=CFG.start_home_delay_s)
    parser.add_argument("--start-home-pulse-s", type=float, default=CFG.start_home_pulse_s)
    parser.add_argument("--start-home-settle-s", type=float, default=CFG.start_home_settle_s)
    parser.add_argument("--start-open-gripper-s", type=float, default=CFG.start_open_gripper_s)
    parser.add_argument("--start-open-gripper-value", type=float, default=CFG.start_open_gripper_value)
    parser.add_argument("--execute", action="store_true", help="Publish model joint targets to the robot node.")
    return parser.parse_args()


def run_pre_model_startup_sequence(node, args: argparse.Namespace) -> bool:
    if not (args.execute and args.return_home_on_start):
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
        end = now_monotonic() + max(0.0, duration_s)
        while rclpy.ok() and now_monotonic() < end:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, end - now_monotonic())))

    def publish(*, home: bool, gripper: float) -> None:
        msg = Float64MultiArray()
        msg.data = make_vr_command({"enable": False, "gripper": float(gripper), "home": bool(home)})
        pub.publish(msg)

    node.get_logger().info("Pre-model startup: return_to_home before loading policy.")
    spin_sleep(args.start_home_delay_s)
    publish(home=True, gripper=args.start_open_gripper_value)
    spin_sleep(args.start_home_pulse_s)
    publish(home=False, gripper=args.start_open_gripper_value)
    spin_sleep(args.start_home_settle_s)
    node.get_logger().info("Pre-model startup: opening gripper before policy inference.")
    open_until = now_monotonic() + max(0.0, args.start_open_gripper_s)
    while rclpy.ok() and now_monotonic() < open_until:
        publish(home=False, gripper=args.start_open_gripper_value)
        spin_sleep(0.05)
    return True


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
    node = rclpy.create_node("ur3e_servoj_smolvla_rollout")
    rollout = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = ServoJSmolVLARollout(node, args)
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
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
