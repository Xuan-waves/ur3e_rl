#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.config import TeleopConfig
from real_teleop.kinematics import RobotKinematics
from real_teleop.messages import make_ik_target, make_vr_command, parse_robot_state
from real_teleop.rotation_repr import continuous_rotvec_from_quat
from real_teleop.safety import SafetyLimiter
from scripts.rollout_smolvla_no_rotvec.config import CFG
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import (
    ObservationPacket,
    SyncBuffer,
    fmt_vec,
    image_msg_to_tensor,
    make_preview_frame,
    resolve_policy_path,
    run_pre_model_startup_sequence,
    stamp_to_sec,
)
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla_rtc import SmolVLARTCRunner


def home_orientation() -> tuple[np.ndarray, np.ndarray]:
    cfg = TeleopConfig()
    _, quat = RobotKinematics(cfg).forward(cfg.hardware_home_q)
    quat = np.asarray(quat, dtype=float).reshape(4)
    return quat, continuous_rotvec_from_quat(quat).astype(float)


def robot_state_to_no_rotvec_tensor(
    robot_state: dict[str, Any],
    *,
    rl_mark: float,
    gripper_max: float,
) -> torch.Tensor:
    pos = np.asarray(robot_state["tcp_pos"], dtype=np.float32).reshape(3)
    gripper = np.float32(np.clip(robot_state.get("gripper", 0.0), 0.0, gripper_max))
    state = np.concatenate([pos, [gripper, np.float32(rl_mark)]]).astype(np.float32)
    return torch.from_numpy(state)


def infer_no_rotvec_position_mode(policy_path: Path) -> tuple[str | None, str | None]:
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        return None, None
    try:
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    dataset_cfg = train_config.get("dataset")
    if not isinstance(dataset_cfg, dict) or not dataset_cfg.get("root"):
        return None, None
    dataset_root = Path(dataset_cfg["root"]).expanduser()
    if not dataset_root.exists():
        return None, None

    for report_path in sorted((dataset_root / "meta").glob("sync_report_episode_*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("action_representation", "representation"):
            representation = report.get(key)
            if not isinstance(representation, dict):
                continue
            mode = representation.get("position") or representation.get("ee_action_position_mode")
            if mode in {"relative", "absolute"}:
                return str(mode), str(report_path)

    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        return None, None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        action = stats.get("action", {})
        state = stats.get("observation.state", {})
        action_mean = np.asarray(action.get("mean"), dtype=float).reshape(-1)
        action_min = np.asarray(action.get("min"), dtype=float).reshape(-1)
        action_max = np.asarray(action.get("max"), dtype=float).reshape(-1)
        state_mean = np.asarray(state.get("mean"), dtype=float).reshape(-1)
    except Exception:
        return None, None
    if action_mean.size < 3:
        return None, None
    pos_abs_range = float(np.max(np.abs(np.concatenate([action_min[:3], action_max[:3]]))))
    pos_mean_norm = float(np.linalg.norm(action_mean[:3]))
    if state_mean.size >= 3 and np.linalg.norm(action_mean[:3] - state_mean[:3]) < 0.08:
        return "absolute", str(stats_path)
    return ("relative" if pos_mean_norm < 0.12 and pos_abs_range < 0.20 else "absolute"), str(stats_path)


class NoRotvecSmolVLARTCRollout:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.policy_path = resolve_policy_path(args.policy_path)
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
        self.last_published_pos: np.ndarray | None = None
        self.last_published_gripper: float | None = None
        self.current_action: np.ndarray | None = None
        self.current_ready_stamp = 0.0
        self.last_action_step_time = 0.0
        self.last_log_time = 0.0
        self.last_dt_map: dict[str, float | None] = {}

        self.pending_reference_stamps: queue.SimpleQueue[float] = queue.SimpleQueue()
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
        self.observe_timer = node.create_timer(1.0 / max(args.fps, 1.0), self._observe_tick, callback_group=self.callback_group)
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
            f"UR3e no-rotvec SmolVLA RTC rollout ready ({mode}). policy={self.policy_path}, "
            f"task={args.task!r}, state/action=5D no-rotvec, action_position_mode={args.action_position_mode}, "
            f"action_mode_source={args.action_position_mode_source}, fps={args.fps:.1f}, "
            f"command_hz={args.command_hz:.1f}, action_step_hz={args.action_step_hz:.1f}, "
            f"sync_reference={self.sync_reference}, chunk={self.runner.chunk_size}, "
            f"rtc_horizon={self.runner.execution_horizon}, fixed_home_rotvec={fmt_vec(self.fixed_rotvec)}"
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
        if packet is not None:
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
                "observation.state": robot_state_to_no_rotvec_tensor(
                    state,
                    rl_mark=self.args.rl_mark,
                    gripper_max=self.args.gripper_max,
                ),
                "task": self.args.task,
            }
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build no-rotvec observation: {exc}")
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
        if action.size < 5:
            self.node.get_logger().warn(f"Bad no-rotvec action shape: {action.shape}")
            return

        state, state_age = self.buffers["state"].latest(now=stamp)
        if state is None or state_age is None or state_age > self.args.max_dt_state:
            self._log_wait(f"holding: no fresh robot state age={state_age}")
            return

        current_pos = np.asarray(state["tcp_pos"], dtype=float).reshape(3)
        target_pos, gripper = self._decode_and_smooth_action(action, current_pos=current_pos)
        if self.execute:
            pose_msg = Float64MultiArray()
            pose_msg.data = make_ik_target(target_pos, self.fixed_quat)
            self.ik_target_pub.publish(pose_msg)

            command_msg = Float64MultiArray()
            command_msg.data = make_vr_command({"enable": False, "gripper": gripper, "home": False})
            self.command_pub.publish(command_msg)

        prefix = "publish" if self.execute else "dry-run"
        self._log_info(
            f"{prefix} pos={fmt_vec(target_pos)} fixed_rot={fmt_vec(self.fixed_rotvec)} "
            f"gripper={gripper:.3f} age={(stamp - self.current_ready_stamp) * 1000.0:.0f}ms "
            f"{self._format_rtc_status()}"
        )

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

        if self.last_published_pos is not None:
            alpha = float(np.clip(self.args.action_pose_filter_alpha, 0.0, 1.0))
            pos = alpha * pos + (1.0 - alpha) * self.last_published_pos
            delta = pos - self.last_published_pos
            norm = float(np.linalg.norm(delta))
            max_step = float(max(self.args.max_action_pos_step, 1e-6))
            if norm > max_step:
                pos = self.last_published_pos + delta / norm * max_step
        self.last_published_pos = pos.copy()

        if self.last_published_gripper is not None:
            alpha_g = float(np.clip(self.args.action_gripper_filter_alpha, 0.0, 1.0))
            gripper = alpha_g * gripper + (1.0 - alpha_g) * self.last_published_gripper
        gripper = float(np.clip(gripper, 0.0, self.args.gripper_max))
        self.last_published_gripper = gripper
        return pos.astype(np.float32), gripper

    def _preview_loop(self) -> None:
        try:
            import cv2

            cv2.namedWindow("UR3e No-Rotvec SmolVLA RTC Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e No-Rotvec SmolVLA RTC Rollout", 1280, 480)
            self.node.get_logger().info("OpenCV no-rotvec RTC rollout preview window started.")
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
                    "UR3e No-Rotvec SmolVLA RTC Rollout",
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
    parser = argparse.ArgumentParser(description="RTC SmolVLA rollout for UR3e impedance no-rotvec control.")
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--robot-state-topic", default=CFG.robot_state_topic)
    parser.add_argument("--ik-target-topic", default=CFG.ik_target_topic)
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
    parser.add_argument(
        "--action-position-mode",
        choices=("auto", "relative", "absolute"),
        default=CFG.action_position_mode,
        help="Decode action[0:3] as relative delta or absolute TCP target. auto reads checkpoint metadata.",
    )
    parser.add_argument("--action-pose-filter-alpha", type=float, default=CFG.action_pose_filter_alpha)
    parser.add_argument("--max-action-pos-step", type=float, default=CFG.max_action_pos_step)
    parser.add_argument(
        "--min-action-z",
        type=float,
        default=CFG.min_action_z,
        help="Runtime lower bound for target TCP z in meters. Use nan to disable.",
    )
    parser.add_argument("--action-gripper-filter-alpha", type=float, default=CFG.action_gripper_filter_alpha)
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
    node = rclpy.create_node("ur3e_no_rotvec_smolvla_rtc_rollout")
    rollout = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = NoRotvecSmolVLARTCRollout(node, args)
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
