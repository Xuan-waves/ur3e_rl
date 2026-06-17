#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
if "QT_QPA_FONTDIR" not in os.environ:
    for font_dir in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"):
        if Path(font_dir).exists():
            os.environ["QT_QPA_FONTDIR"] = font_dir
            break

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import make_vr_command, parse_vr_command
from real_teleop.ros_qos import latest_qos
from scripts.rlt_gate.live_rlt_gate_monitor import decode_ros_image


@dataclass(frozen=True)
class ReadyGateCollectConfig:
    front_image_topic: str = "/camera/d455/color/image_raw"
    wrist_image_topic: str = "/camera/d405/color/image_raw"
    vr_raw_topic: str = "/ur3e_vr/vr_command_raw"
    vr_command_topic: str = "/ur3e_vr/vr_command"
    output_root: Path = REPO_ROOT / "datasets/ready_gate"
    fps: float = 30.0
    max_dt_front_image: float = 0.08
    max_dt_wrist_image: float = 0.08
    vr_stale_s: float = 0.25
    startup_home_pulse_s: float = 2.0
    manual_home_pulse_s: float = 1.5
    home_gripper_value: float = 0.0
    reset_impedance_during_home: bool = True
    jpeg_quality: int = 95
    preview: bool = True
    preview_hz: float = 20.0
    preview_width: int = 1280
    status_hz: float = 4.0


CFG = ReadyGateCollectConfig()


class LatestImage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.msg_stamp = 0.0
        self.recv_stamp = 0.0
        self.image: np.ndarray | None = None
        self.count = 0

    def set(self, msg_stamp: float, image: np.ndarray) -> None:
        with self.lock:
            self.msg_stamp = float(msg_stamp)
            self.recv_stamp = time.monotonic()
            self.image = np.ascontiguousarray(image)
            self.count += 1

    def get(self) -> tuple[np.ndarray | None, float, float, int]:
        with self.lock:
            image = None if self.image is None else self.image.copy()
            return image, self.msg_stamp, self.recv_stamp, self.count


class ButtonEdges:
    def __init__(self) -> None:
        self.prev: dict[str, bool] = {
            "a": False,
            "b": False,
            "x": False,
            "y": False,
        }

    def rising(self, name: str, value: bool) -> bool:
        was = bool(self.prev.get(name, False))
        self.prev[name] = bool(value)
        return bool(value) and not was


class ReadyGateCollector:
    def __init__(self, node, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        self.node = node
        self.args = args
        self.group = ReentrantCallbackGroup()
        self.front = LatestImage()
        self.wrist = LatestImage()
        self.buttons = ButtonEdges()
        self.record_lock = threading.Lock()
        self.vr_command: dict[str, Any] | None = None
        self.vr_recv_stamp = 0.0

        self.output_dir = self._make_output_dir(args.output_root)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.episode_dir: Path | None = None
        self.labels_file = None
        self.recording = False
        self.pending_gate = int(args.initial_gate)
        self.current_gate = int(args.initial_gate)
        self.episode_index = 0
        self.frame_index = 0
        self.total_frames = 0
        self.dropped_frames = 0
        self.record_start_stamp = 0.0
        self.last_sample_stamp = 0.0
        self.last_status_stamp = 0.0
        self.last_preview_stamp = 0.0
        self.stop_event = threading.Event()
        self.preview_thread: threading.Thread | None = None
        now = time.monotonic()
        self.home_pulse_until = now + float(args.startup_home_pulse_s)
        self.last_event = "startup return_home pulse"

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = latest_qos()
        self.command_pub = node.create_publisher(Float64MultiArray, args.vr_command_topic, command_qos)
        self.subs = [
            node.create_subscription(Image, args.front_image_topic, self._on_front, image_qos, callback_group=self.group),
            node.create_subscription(Image, args.wrist_image_topic, self._on_wrist, image_qos, callback_group=self.group),
            node.create_subscription(
                Float64MultiArray,
                args.vr_raw_topic,
                self._on_raw_vr,
                command_qos,
                callback_group=self.group,
            ),
        ]
        self.relay_timer = node.create_timer(1.0 / 100.0, self._relay_tick, callback_group=self.group)
        self.sample_timer = node.create_timer(1.0 / max(args.fps, 1e-6), self._sample_tick, callback_group=self.group)
        self.button_timer = node.create_timer(1.0 / 60.0, self._button_tick, callback_group=self.group)
        if args.preview:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
        else:
            self.preview_thread = None

        self._write_metadata()
        self.node.get_logger().info(
            "Ready-gate collector ready. X=start, B=stop/save, Y=toggle next gate, A=return_home. "
            f"fps={args.fps:.1f}, output={self.output_dir}, initial_next_gate={self.pending_gate}"
        )

    def _make_output_dir(self, root: Path) -> Path:
        root = root.expanduser()
        if not root.is_absolute():
            root = (REPO_ROOT / root).resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"ready_gate_{stamp}"

    def _write_metadata(self) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "semantics": {
                "ready_gate": (
                    "1 means the Ethernet plug is in the start holder/fixture and the scene is ready "
                    "for a new rollout; 0 means not ready."
                ),
                "Y": "toggles the label used by the next episode; current recording keeps its start label.",
            },
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @staticmethod
    def _msg_stamp(msg: Any) -> float:
        stamp = getattr(msg, "header", None).stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_front(self, msg: Any) -> None:
        try:
            self.front.set(self._msg_stamp(msg), decode_ros_image(msg))
        except Exception as exc:
            self.node.get_logger().warn(f"Bad front image: {exc}")

    def _on_wrist(self, msg: Any) -> None:
        try:
            self.wrist.set(self._msg_stamp(msg), decode_ros_image(msg))
        except Exception as exc:
            self.node.get_logger().warn(f"Bad wrist image: {exc}")

    def _on_raw_vr(self, msg: Any) -> None:
        try:
            self.vr_command = parse_vr_command(msg.data)
            self.vr_recv_stamp = time.monotonic()
        except Exception as exc:
            self.node.get_logger().warn(f"Bad raw VR command: {exc}")

    def _button_tick(self) -> None:
        if self.vr_command is None or time.monotonic() - self.vr_recv_stamp > self.args.vr_stale_s:
            return
        if self.buttons.rising("a", bool(self.vr_command.get("home", False))):
            self._request_home("A pressed")
        if self.buttons.rising("y", bool(self.vr_command.get("rl_toggle", False))):
            self.pending_gate = 1 - int(self.pending_gate)
            self.last_event = f"next ready_gate set to {self.pending_gate}"
            self.node.get_logger().info(self.last_event)
        if self.buttons.rising("x", bool(self.vr_command.get("record_start", False))):
            self._start_recording()
        if self.buttons.rising("b", bool(self.vr_command.get("record_stop", False))):
            self._stop_recording()

    def _request_home(self, reason: str) -> None:
        self.home_pulse_until = time.monotonic() + float(self.args.manual_home_pulse_s)
        self.last_event = f"return_home requested: {reason}"
        self.node.get_logger().info(self.last_event)

    def _relay_tick(self) -> None:
        from std_msgs.msg import Float64MultiArray

        stamp = time.monotonic()
        if stamp < self.home_pulse_until:
            payload = {
                "enable": False,
                "gripper": float(np.clip(self.args.home_gripper_value, 0.0, 1.0)),
                "home": True,
                "reset_impedance": bool(self.args.reset_impedance_during_home),
            }
        elif self.vr_command is not None and stamp - self.vr_recv_stamp <= self.args.vr_stale_s:
            payload = dict(self.vr_command)
        else:
            return
        msg = Float64MultiArray()
        msg.data = make_vr_command(payload)
        self.command_pub.publish(msg)

    def _start_recording(self) -> None:
        with self.record_lock:
            if self.recording:
                return
            self.current_gate = int(self.pending_gate)
            self.frame_index = 0
            self.record_start_stamp = time.monotonic()
            self.last_sample_stamp = 0.0
            self.episode_dir = self.output_dir / f"episode_{self.episode_index:06d}"
            (self.episode_dir / "front").mkdir(parents=True, exist_ok=False)
            (self.episode_dir / "wrist").mkdir(parents=True, exist_ok=False)
            self.labels_file = (self.episode_dir / "labels.jsonl").open("w", encoding="utf-8")
            episode_meta = {
                "episode_index": self.episode_index,
                "ready_gate": self.current_gate,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            (self.episode_dir / "metadata.json").write_text(json.dumps(episode_meta, indent=2), encoding="utf-8")
            self.recording = True
        self.last_event = f"recording started episode={self.episode_index:06d} ready_gate={self.current_gate}"
        self.node.get_logger().info(self.last_event)

    def _stop_recording(self) -> None:
        with self.record_lock:
            if not self.recording:
                self.last_event = "B pressed, but not recording"
                self.node.get_logger().warn(self.last_event)
                return
            self.recording = False
            if self.labels_file is not None:
                self.labels_file.close()
                self.labels_file = None
            meta_path = self.episode_dir / "metadata.json" if self.episode_dir else None
            frames = self.frame_index
            gate = self.current_gate
            episode = self.episode_index
            if meta_path is not None and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta.update(
                    {
                        "stopped_at": datetime.now().isoformat(timespec="seconds"),
                        "frames": frames,
                        "duration_s": time.monotonic() - self.record_start_stamp,
                    }
                )
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.episode_index += 1
            self.episode_dir = None
            self.frame_index = 0
        self.last_event = f"recording saved episode={episode:06d} frames={frames} ready_gate={gate}"
        self.node.get_logger().info(self.last_event)

    def _sample_tick(self) -> None:
        with self.record_lock:
            if not self.recording or self.episode_dir is None or self.labels_file is None:
                self._status_tick()
                return
            now_mono = time.monotonic()
            front, front_stamp, front_recv, front_count = self.front.get()
            wrist, wrist_stamp, wrist_recv, wrist_count = self.wrist.get()
            front_age = now_mono - front_recv if front is not None else float("inf")
            wrist_age = now_mono - wrist_recv if wrist is not None else float("inf")
            if (
                front is None
                or wrist is None
                or front_age > self.args.max_dt_front_image
                or wrist_age > self.args.max_dt_wrist_image
            ):
                self.dropped_frames += 1
                self._status_tick()
                return
            frame_index = self.frame_index
            episode_index = self.episode_index
            ready_gate = self.current_gate
            name = f"frame_{frame_index:06d}.jpg"
            front_path = self.episode_dir / "front" / name
            wrist_path = self.episode_dir / "wrist" / name
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)]
            ok_front = cv2.imwrite(str(front_path), front, params)
            ok_wrist = cv2.imwrite(str(wrist_path), wrist, params)
            if not ok_front or not ok_wrist:
                self.dropped_frames += 1
                self.node.get_logger().warn(f"Failed to write frame {frame_index}")
                self._status_tick()
                return
            row = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": time.time(),
                "monotonic_time": now_mono,
                "ready_gate": ready_gate,
                "front_path": str(front_path.relative_to(self.output_dir)),
                "wrist_path": str(wrist_path.relative_to(self.output_dir)),
                "front_stamp": front_stamp,
                "wrist_stamp": wrist_stamp,
                "front_age_ms": front_age * 1000.0,
                "wrist_age_ms": wrist_age * 1000.0,
                "front_count": front_count,
                "wrist_count": wrist_count,
            }
            self.labels_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.frame_index += 1
            self.total_frames += 1
            self.last_sample_stamp = now_mono
        self._status_tick()

    def _preview_loop(self) -> None:
        period = 1.0 / max(float(self.args.preview_hz), 1e-6)
        try:
            cv2.namedWindow("UR3e ready-gate collector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UR3e ready-gate collector", int(self.args.preview_width), 540)
        except cv2.error as exc:
            self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
            return
        while not self.stop_event.is_set():
            front, _, front_recv, front_count = self.front.get()
            wrist, _, wrist_recv, wrist_count = self.wrist.get()
            now_mono = time.monotonic()
            canvas = self._draw_preview(
                front,
                wrist,
                front_count=front_count,
                wrist_count=wrist_count,
                front_age_ms=(now_mono - front_recv) * 1000.0 if front is not None else float("inf"),
                wrist_age_ms=(now_mono - wrist_recv) * 1000.0 if wrist is not None else float("inf"),
            )
            try:
                cv2.imshow("UR3e ready-gate collector", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.node.get_logger().info("OpenCV preview closed by keyboard; collection continues.")
                    break
            except cv2.error as exc:
                self.node.get_logger().warn(f"OpenCV preview disabled: {exc}")
                break
            self.stop_event.wait(period)
        try:
            cv2.destroyWindow("UR3e ready-gate collector")
        except cv2.error:
            pass

    def _draw_preview(
        self,
        front: np.ndarray | None,
        wrist: np.ndarray | None,
        *,
        front_count: int,
        wrist_count: int,
        front_age_ms: float,
        wrist_age_ms: float,
    ) -> np.ndarray:
        width = int(self.args.preview_width)
        if front is None and wrist is None:
            canvas = np.zeros((360, width, 3), dtype=np.uint8)
            cv2.putText(canvas, "Waiting for camera topics", (24, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            return canvas
        if front is None:
            front = np.zeros((wrist.shape[0], wrist.shape[1], 3), dtype=np.uint8)
        if wrist is None:
            wrist = np.zeros((front.shape[0], front.shape[1], 3), dtype=np.uint8)
        half = max(width // 2, 160)
        front_small = self._resize_to_width(front, half)
        wrist_small = self._resize_to_width(wrist, half)
        height = max(front_small.shape[0], wrist_small.shape[0])
        front_small = self._pad_height(front_small, height)
        wrist_small = self._pad_height(wrist_small, height)
        canvas = np.concatenate([front_small, wrist_small], axis=1)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 104), (0, 0, 0), -1)
        mode = "RECORDING" if self.recording else "WAITING"
        color = (0, 220, 0) if self.current_gate else (80, 80, 255)
        cv2.putText(canvas, f"ReadyGate {mode} | X=start B=stop Y=next_gate A=home", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(canvas, f"episode={self.episode_index:06d} frame={self.frame_index} current={self.current_gate} next={self.pending_gate} drops={self.dropped_frames}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(canvas, f"front#{front_count} age={front_age_ms:.0f}ms wrist#{wrist_count} age={wrist_age_ms:.0f}ms", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 220, 180), 2)
        cv2.circle(canvas, (canvas.shape[1] - 34, 34), 16, color, -1)
        cv2.putText(canvas, "front", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "wrist", (front_small.shape[1] + 12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return canvas

    @staticmethod
    def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
        scale = float(width) / max(float(image.shape[1]), 1.0)
        height = max(1, int(round(float(image.shape[0]) * scale)))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
        if image.shape[0] >= height:
            return image
        pad = height - image.shape[0]
        top = pad // 2
        bottom = pad - top
        return np.pad(image, ((top, bottom), (0, 0), (0, 0)), mode="constant")

    def _status_tick(self) -> None:
        now_mono = time.monotonic()
        if now_mono - self.last_status_stamp < 1.0 / max(self.args.status_hz, 1e-6):
            return
        self.last_status_stamp = now_mono
        mode = "RECORDING" if self.recording else "WAITING"
        front, _, front_recv, front_count = self.front.get()
        wrist, _, wrist_recv, wrist_count = self.wrist.get()
        front_age = (now_mono - front_recv) * 1000.0 if front is not None else float("inf")
        wrist_age = (now_mono - wrist_recv) * 1000.0 if wrist is not None else float("inf")
        line = (
            f"\033[2K\rReadyGate {mode} | X=start B=stop Y=next_gate A=home | "
            f"ep={self.episode_index:06d} frame={self.frame_index:<5d} total={self.total_frames:<6d} "
            f"current={self.current_gate} next={self.pending_gate} drops={self.dropped_frames:<4d} "
            f"front#{front_count}@{front_age:.0f}ms wrist#{wrist_count}@{wrist_age:.0f}ms "
            f"event={self.last_event[:64]:<64}"
        )
        print(line, end="", flush=True)

    def close(self) -> None:
        self.stop_event.set()
        if self.preview_thread is not None and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=1.0)
        with self.record_lock:
            recording = self.recording
        if recording:
            self._stop_recording()
        with self.record_lock:
            if self.labels_file is not None:
                self.labels_file.close()
                self.labels_file = None
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect image labels for the UR3e ready gate.")
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--vr-raw-topic", default=CFG.vr_raw_topic)
    parser.add_argument("--vr-command-topic", default=CFG.vr_command_topic)
    parser.add_argument("--output-root", type=Path, default=CFG.output_root)
    parser.add_argument("--fps", type=float, default=CFG.fps)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--vr-stale-s", type=float, default=CFG.vr_stale_s)
    parser.add_argument("--startup-home-pulse-s", type=float, default=CFG.startup_home_pulse_s)
    parser.add_argument("--manual-home-pulse-s", type=float, default=CFG.manual_home_pulse_s)
    parser.add_argument("--home-gripper-value", type=float, default=CFG.home_gripper_value)
    parser.add_argument("--reset-impedance-during-home", action=argparse.BooleanOptionalAction, default=CFG.reset_impedance_during_home)
    parser.add_argument("--jpeg-quality", type=int, default=CFG.jpeg_quality)
    parser.add_argument("--initial-gate", type=int, choices=(0, 1), default=1)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=CFG.preview)
    parser.add_argument("--preview-hz", type=float, default=CFG.preview_hz)
    parser.add_argument("--preview-width", type=int, default=CFG.preview_width)
    parser.add_argument("--status-hz", type=float, default=CFG.status_hz)
    return parser.parse_args()


def main() -> int:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("ur3e_ready_gate_collector")
    collector = ReadyGateCollector(node, args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        collector.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
