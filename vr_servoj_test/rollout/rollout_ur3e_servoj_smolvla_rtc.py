#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.messages import make_joint_target, parse_robot_state
from vr_servoj_test.rollout.config import CFG
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import (
    ObservationPacket,
    SyncBuffer,
    fmt_vec,
    image_msg_to_tensor,
    make_preview_frame,
    now_monotonic,
    resolve_policy_path,
    robot_state_to_joint_tensor,
    run_pre_model_startup_sequence,
    stamp_to_sec,
)


def _rtc_schedule(name: str):
    from lerobot.configs.types import RTCAttentionSchedule

    key = str(name).strip().upper()
    aliases = {
        "ZERO": "ZEROS",
        "0": "ZEROS",
        "ONE": "ONES",
        "1": "ONES",
        "LIN": "LINEAR",
        "EXPONENTIAL": "EXP",
    }
    key = aliases.get(key, key)
    try:
        return RTCAttentionSchedule[key]
    except KeyError as exc:
        choices = ", ".join(item.name.lower() for item in RTCAttentionSchedule)
        raise ValueError(f"Unknown RTC prefix attention schedule {name!r}. choices: {choices}") from exc


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected previous actions as [T, A], got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


class SmolVLARTCRunner:
    def __init__(
        self,
        policy_path: Path,
        device: str,
        *,
        execution_horizon: int,
        max_guidance_weight: float,
        prefix_attention_schedule: str,
        latency_window: int,
        idle_sleep_s: float,
        queue_refill_threshold: int,
        debug: bool,
        action_step_hz: float,
        log_fn,
        warn_fn,
    ) -> None:
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.policies.rtc.action_queue import ActionQueue
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.rtc.latency_tracker import LatencyTracker
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
        self.log_fn = log_fn
        self.warn_fn = warn_fn
        self.action_step_hz = float(max(action_step_hz, 1e-6))
        self.idle_sleep_s = float(max(idle_sleep_s, 0.0005))
        self.shutdown = threading.Event()
        self.active = threading.Event()
        self.latest_lock = threading.Lock()
        self.latest_packet: ObservationPacket | None = None
        self.stats_lock = threading.Lock()
        self.infer_count = 0
        self.last_infer_s = 0.0
        self.last_estimated_delay_steps = 0
        self.last_delay_steps = 0
        self.last_queue_size = 0
        self.last_error: str | None = None

        self.policy = SmolVLAPolicy.from_pretrained(
            policy_path,
            cli_overrides=[f"--device={device}"],
            local_files_only=True,
        )
        self.policy.eval()
        if hasattr(self.policy, "to"):
            self.policy.to(device)
        self.ckpt_action_horizon = int(getattr(self.policy.config, "n_action_steps", 1))
        self.chunk_size = int(getattr(self.policy.config, "chunk_size", self.ckpt_action_horizon))
        self.execution_horizon = max(1, min(int(execution_horizon), self.chunk_size))
        self.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=_rtc_schedule(prefix_attention_schedule),
            max_guidance_weight=float(max_guidance_weight),
            execution_horizon=self.execution_horizon,
            debug=bool(debug),
        )
        self.policy.config.rtc_config = self.rtc_config
        self.policy.init_rtc_processor()
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
        self.action_queue = ActionQueue(self.rtc_config)
        self.latency_tracker = LatencyTracker(maxlen=max(1, int(latency_window)))
        if int(queue_refill_threshold) < 0:
            queue_refill_threshold = max(0, self.chunk_size - self.execution_horizon)
        self.queue_refill_threshold = int(queue_refill_threshold)
        self.thread = threading.Thread(target=self._loop, name="servoj_smolvla_rtc_inference", daemon=True)
        self.thread.start()

    def start(self) -> None:
        self.active.set()

    def close(self) -> None:
        self.shutdown.set()
        self.active.clear()
        self.thread.join(timeout=2.0)

    def update_observation(self, packet: ObservationPacket) -> None:
        with self.latest_lock:
            self.latest_packet = packet

    def get_action(self) -> np.ndarray | None:
        action = self.action_queue.get()
        if action is None:
            return None
        return np.asarray(action.detach().cpu(), dtype=np.float32).reshape(-1)

    def qsize(self) -> int:
        return int(self.action_queue.qsize())

    def stats(self) -> dict[str, Any]:
        with self.stats_lock:
            return {
                "infer_count": self.infer_count,
                "last_infer_s": self.last_infer_s,
                "last_estimated_delay_steps": self.last_estimated_delay_steps,
                "last_delay_steps": self.last_delay_steps,
                "last_queue_size": self.last_queue_size,
                "last_error": self.last_error,
                "latency_max_s": self.latency_tracker.max() or 0.0,
                "latency_p95_s": self.latency_tracker.p95() or 0.0,
            }

    def _loop(self) -> None:
        while not self.shutdown.is_set():
            if not self.active.is_set():
                time.sleep(self.idle_sleep_s)
                continue
            if self.action_queue.qsize() > self.queue_refill_threshold:
                time.sleep(self.idle_sleep_s)
                continue
            with self.latest_lock:
                packet = self.latest_packet
            if packet is None:
                time.sleep(self.idle_sleep_s)
                continue
            try:
                self._produce_chunk(packet)
            except Exception as exc:
                with self.stats_lock:
                    self.last_error = str(exc)
                self.warn_fn(f"RTC inference failed: {exc}\n{traceback.format_exc(limit=2)}")
                time.sleep(0.2)

    def _produce_chunk(self, packet: ObservationPacket) -> None:
        start = now_monotonic()
        idx_before = self.action_queue.get_action_index()
        prev_actions = self.action_queue.get_left_over()
        latency = self.latency_tracker.max() or 0.0
        inference_delay = int(math.ceil(latency * self.action_step_hz)) if latency > 0.0 else 0
        if prev_actions is not None:
            prev_actions = _normalize_prev_actions_length(prev_actions, self.execution_horizon)
            prev_actions = prev_actions.to(self.device)

        batch = self.preprocessor(packet.batch)
        batch = {key: value for key, value in batch.items() if key in self.input_keys}
        actions = self.policy.predict_action_chunk(
            batch,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_actions,
            execution_horizon=self.execution_horizon,
        )
        original = actions.detach().squeeze(0).clone()
        processed = self.postprocessor(actions.detach().cpu()).squeeze(0).detach().cpu()

        elapsed = now_monotonic() - start
        estimated_delay = int(math.ceil(elapsed * self.action_step_hz))
        # In this ROS rollout the first inference can run while no actions are
        # consumed yet. Cropping by wall-clock latency would then drop a fresh
        # chunk. Merge by the queue index that was actually consumed.
        real_delay = max(0, int(self.action_queue.get_action_index()) - int(idx_before))
        self.latency_tracker.add(elapsed)
        self.action_queue.merge(original, processed, real_delay, idx_before)
        with self.stats_lock:
            self.infer_count += 1
            self.last_infer_s = elapsed
            self.last_estimated_delay_steps = estimated_delay
            self.last_delay_steps = real_delay
            self.last_queue_size = self.action_queue.qsize()
            self.last_error = None


class ServoJSmolVLARTCRollout:
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
        self.last_action_step_time = 0.0
        self.current_action: np.ndarray | None = None
        self.current_ready_stamp = 0.0
        self.last_published_q: np.ndarray | None = None
        self.last_published_gripper: float | None = None
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

        self.runner = SmolVLARTCRunner(
            self.policy_path,
            args.device,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            latency_window=args.rtc_latency_window,
            idle_sleep_s=args.rtc_idle_sleep_s,
            queue_refill_threshold=args.rtc_queue_refill_threshold,
            debug=args.rtc_debug,
            action_step_hz=args.action_step_hz,
            log_fn=node.get_logger().info,
            warn_fn=node.get_logger().warn,
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
        self.observe_timer = node.create_timer(
            1.0 / max(args.fps, 1.0),
            self._observe_tick,
            callback_group=self.callback_group,
        )
        self.publish_timer = node.create_timer(
            1.0 / max(args.command_hz, 1.0),
            self._publish_tick,
            callback_group=self.callback_group,
        )
        if args.preview:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()

        self.runner.start()
        mode = "EXECUTE" if self.execute else "DRY-RUN"
        node.get_logger().info(
            f"UR3e servoJ SmolVLA RTC rollout ready ({mode}). policy={self.policy_path}, "
            f"task={args.task!r}, fps={args.fps:.1f}, command_hz={args.command_hz:.1f}, "
            f"action_step_hz={args.action_step_hz:.1f}, sync_reference={self.sync_reference}, "
            f"chunk={self.runner.chunk_size}, rtc_horizon={self.runner.execution_horizon}, "
            f"rtc_refill={self.runner.queue_refill_threshold}, rtc_schedule={args.rtc_prefix_attention_schedule}, "
            f"rtc_guidance={args.rtc_max_guidance_weight:.2f}, device={self.runner.device}, "
            f"hf_home={args.hf_home}, offline={args.offline}, preview={args.preview}"
        )

    def close(self) -> None:
        self.runner.close()
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

    def _observe_tick(self) -> None:
        stamp = self._take_reference_stamp()
        if stamp is None:
            self._log_wait("waiting for reference image")
            return
        packet = self._build_observation(stamp)
        if packet is None:
            return
        self.runner.update_observation(packet)

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
        if stamp - self.last_action_step_time >= 1.0 / max(self.args.action_step_hz, 1e-6):
            action = self.runner.get_action()
            if action is not None:
                self.current_action = action
                self.current_ready_stamp = stamp
                self.last_action_step_time = stamp

        if self.current_action is None:
            self._log_wait(f"RTC queue warming up {self._format_status()} {self._format_rtc_status()}")
            return

        action = np.asarray(self.current_action, dtype=np.float32).reshape(-1)
        if action.size < 7:
            self.node.get_logger().warn(f"Bad action shape: {action.shape}")
            return

        q = np.clip(action[:6], self.joint_limits[:, 0], self.joint_limits[:, 1])
        gripper = float(np.clip(action[6], 0.0, self.args.gripper_max))
        q, gripper = self._smooth_action(q, gripper)
        if self.execute:
            msg = Float64MultiArray()
            msg.data = make_joint_target(tracking=True, q=q, gripper=gripper, reason="rtc_tracking", ok=True)
            self.joint_pub.publish(msg)

        prefix = "publish" if self.execute else "dry-run"
        self._log_info(
            f"{prefix} q={fmt_vec(q)} gripper={gripper:.3f} "
            f"age={(stamp - self.current_ready_stamp) * 1000.0:.0f}ms {self._format_rtc_status()}"
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

            cv2.namedWindow("UR3e ServoJ SmolVLA RTC Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e ServoJ SmolVLA RTC Rollout", 1280, 480)
            self.node.get_logger().info("OpenCV RTC rollout preview window started.")
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV RTC rollout preview disabled: {exc}")
            return

        period = 1.0 / max(self.args.preview_hz, 1.0)
        while not self.preview_stop.is_set() and not self.preview_failed:
            try:
                import cv2

                now = self.now_sec()
                front, front_age = self.buffers["front"].latest(now=now)
                wrist, wrist_age = self.buffers["wrist"].latest(now=now)
                cv2.imshow(
                    "UR3e ServoJ SmolVLA RTC Rollout",
                    make_preview_frame(front, wrist, front_age, wrist_age),
                )
                cv2.waitKey(1)
            except Exception as exc:
                self.preview_failed = True
                self.node.get_logger().warn(f"OpenCV RTC rollout preview disabled: {exc}")
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

    def _format_rtc_status(self) -> str:
        stats = self.runner.stats()
        error = stats["last_error"]
        err_text = "" if error is None else f" err={error[:80]!r}"
        return (
            f"rtc[queue={self.runner.qsize()} infer#{stats['infer_count']} "
            f"last={stats['last_infer_s'] * 1000.0:.0f}ms "
            f"delay={stats['last_delay_steps']}/{stats['last_estimated_delay_steps']} "
            f"lat_max={stats['latency_max_s'] * 1000.0:.0f}ms]{err_text}"
        )

    def _log_wait(self, text: str) -> None:
        if self.now_sec() - self.last_log_time > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_log_time = self.now_sec()
            self.node.get_logger().warn(text)

    def _log_info(self, text: str) -> None:
        if self.now_sec() - self.last_log_time > 1.0 / max(self.args.log_hz, 1e-6):
            self.last_log_time = self.now_sec()
            self.node.get_logger().info(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTC SmolVLA rollout for UR3e servoJ jointspace control.")
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
    parser.add_argument("--rtc-execution-horizon", type=int, default=CFG.rtc_execution_horizon)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=CFG.rtc_max_guidance_weight)
    parser.add_argument("--rtc-prefix-attention-schedule", default=CFG.rtc_prefix_attention_schedule)
    parser.add_argument("--rtc-latency-window", type=int, default=CFG.rtc_latency_window)
    parser.add_argument("--rtc-idle-sleep-s", type=float, default=CFG.rtc_idle_sleep_s)
    parser.add_argument("--rtc-queue-refill-threshold", type=int, default=CFG.rtc_queue_refill_threshold)
    parser.add_argument("--rtc-debug", action=argparse.BooleanOptionalAction, default=CFG.rtc_debug)
    parser.add_argument("--sync-reference", choices=("front", "wrist", "timer"), default=CFG.sync_reference)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--max-dt-state", type=float, default=CFG.max_dt_state)
    parser.add_argument("--buffer-maxlen", type=int, default=CFG.buffer_maxlen)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--hf-home", type=Path, default=CFG.hf_home)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=CFG.offline)
    parser.add_argument("--rl-mark", type=float, default=CFG.rl_mark)
    parser.add_argument("--gripper-max", type=float, default=CFG.gripper_max)
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
    parser.add_argument("--execute", action="store_true", help="Publish RTC model joint targets to the robot node.")
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
    node = rclpy.create_node("ur3e_servoj_smolvla_rtc_rollout")
    rollout = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = ServoJSmolVLARTCRollout(node, args)
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
