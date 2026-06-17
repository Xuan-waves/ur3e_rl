#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
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
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_gate.eval_rlt_gate import load_checkpoint


class LatestImage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.image: np.ndarray | None = None
        self.count = 0
        self.stamp = 0.0

    def set(self, image: np.ndarray) -> None:
        with self.lock:
            self.image = np.ascontiguousarray(image)
            self.count += 1
            self.stamp = time.monotonic()

    def get(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            image = None if self.image is None else self.image.copy()
            return image, self.count, self.stamp


class FpsMeter:
    def __init__(self, window: float = 2.0) -> None:
        self.window = float(window)
        self.times: list[float] = []

    def tick(self) -> float:
        now = time.monotonic()
        self.times.append(now)
        cutoff = now - self.window
        self.times = [t for t in self.times if t >= cutoff]
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)

    def value(self) -> float:
        now = time.monotonic()
        cutoff = now - self.window
        self.times = [t for t in self.times if t >= cutoff]
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)


def decode_ros_image(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    if encoding == "mono8":
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(msg.step) * int(msg.height)
        gray = flat[:expected].reshape((int(msg.height), int(msg.step)))[:, : int(msg.width)]
        return np.repeat(gray[:, :, None], 3, axis=2)
    if encoding not in {"bgr8", "rgb8", "rgba8", "bgra8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")
    flat = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.step) * int(msg.height)
    image = flat[:expected].reshape((int(msg.height), int(msg.step)))[:, : int(msg.width) * channels]
    image = image.reshape((int(msg.height), int(msg.width), channels))
    if encoding == "bgr8":
        return image.copy()
    if encoding == "rgb8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, [2, 1, 0]].copy()
    return image[:, :, :3].copy()


def preprocess_bgr(image: np.ndarray, image_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.transpose(x, (2, 0, 1))


def compose_input(front: np.ndarray, wrist: np.ndarray, camera: str, image_size: int) -> torch.Tensor:
    if camera == "both":
        arr = np.concatenate([preprocess_bgr(front, image_size), preprocess_bgr(wrist, image_size)], axis=0)
    elif camera == "wrist":
        arr = preprocess_bgr(wrist, image_size)
    else:
        arr = preprocess_bgr(front, image_size)
    return torch.from_numpy(arr).unsqueeze(0)


def resize_half(image: np.ndarray, total_width: int) -> np.ndarray:
    target_width = max(int(total_width) // 2, 160)
    scale = target_width / max(int(image.shape[1]), 1)
    target_height = max(1, int(round(float(image.shape[0]) * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = height - image.shape[0]
    top = pad // 2
    bottom = pad - top
    return np.pad(image, ((top, bottom), (0, 0), (0, 0)), mode="constant")


def draw_preview(
    front: np.ndarray | None,
    wrist: np.ndarray | None,
    width: int,
    *,
    front_count: int,
    wrist_count: int,
    front_age_ms: float,
    wrist_age_ms: float,
    prob: float | None,
    raw_phase: int,
    hyst_phase: int,
    infer_hz: float,
    draw_hz: float,
    infer_ms: float,
    model_name: str,
    checkpoint_name: str,
) -> np.ndarray:
    if front is None and wrist is None:
        canvas = np.zeros((360, int(width), 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for camera topics", (24, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return canvas
    if front is None:
        front = np.zeros_like(wrist)
    if wrist is None:
        wrist = np.zeros_like(front)
    front_small = resize_half(front, width)
    wrist_small = resize_half(wrist, width)
    height = max(front_small.shape[0], wrist_small.shape[0])
    front_small = pad_to_height(front_small, height)
    wrist_small = pad_to_height(wrist_small, height)
    canvas = np.concatenate([front_small, wrist_small], axis=1)

    panel_h = 104
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_h), (0, 0, 0), -1)
    cv2.putText(canvas, "front", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "wrist", (front_small.shape[1] + 12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    prob_text = "none" if prob is None else f"{prob:.3f}"
    cv2.putText(
        canvas,
        f"RLT gate live | model={model_name} ckpt={checkpoint_name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"prob={prob_text} raw={raw_phase} hyst={hyst_phase} infer_hz={infer_hz:.1f} draw_hz={draw_hz:.1f} infer={infer_ms:.1f}ms",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"front#{front_count} age={front_age_ms:.0f}ms  wrist#{wrist_count} age={wrist_age_ms:.0f}ms",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 220, 180),
        2,
    )
    bar_x, bar_y, bar_w, bar_h = 12, 92, min(canvas.shape[1] - 24, 500), 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    if prob is not None:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * prob), bar_y + bar_h), (0, 210, 255), -1)
    color = (0, 220, 0) if hyst_phase else (80, 80, 255)
    cv2.circle(canvas, (canvas.shape[1] - 32, 34), 15, color, -1)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live ROS camera RLT gate monitor for UR3e VR impedance tests.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/rlt_gate/rlt_gate_20260610_172234/best.pt"))
    parser.add_argument("--front-topic", default="/camera/d455/color/image_raw")
    parser.add_argument("--wrist-topic", default="/camera/d405/color/image_raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-infer-hz", type=float, default=15.0)
    parser.add_argument("--max-display-hz", type=float, default=30.0)
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--negative-threshold", type=float, default=None)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--window-name", default="UR3e RLT gate live")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, cfg = load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    camera = str(cfg.get("camera", "both"))
    image_size = int(cfg.get("image_size", 128))
    pos_t = float(args.positive_threshold if args.positive_threshold is not None else cfg.get("positive_threshold", 0.6))
    neg_t = float(args.negative_threshold if args.negative_threshold is not None else cfg.get("negative_threshold", 0.4))

    latest_front = LatestImage()
    latest_wrist = LatestImage()
    infer_meter = FpsMeter()
    draw_meter = FpsMeter()
    phase = 0
    pos_count = 0
    neg_count = 0
    prob: float | None = None
    raw_phase = 0
    infer_ms = 0.0
    last_infer = 0.0
    last_draw = 0.0
    last_pair_counts = (-1, -1)

    rclpy.init()
    node = rclpy.create_node("ur3e_rlt_gate_live_monitor")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def on_front(msg: Image) -> None:
        try:
            latest_front.set(decode_ros_image(msg))
        except Exception as exc:
            node.get_logger().warn(f"Bad front image: {exc}")

    def on_wrist(msg: Image) -> None:
        try:
            latest_wrist.set(decode_ros_image(msg))
        except Exception as exc:
            node.get_logger().warn(f"Bad wrist image: {exc}")

    node.create_subscription(Image, args.front_topic, on_front, qos)
    node.create_subscription(Image, args.wrist_topic, on_wrist, qos)
    node.get_logger().info(
        f"RLT gate monitor ready. checkpoint={args.checkpoint}, model={cfg.get('model')}, camera={camera}, "
        f"device={device}, infer_hz<= {args.max_infer_hz}, front={args.front_topic}, wrist={args.wrist_topic}"
    )
    if not args.no_window:
        cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            front, front_count, front_stamp = latest_front.get()
            wrist, wrist_count, wrist_stamp = latest_wrist.get()
            can_infer = front is not None and wrist is not None and (front_count, wrist_count) != last_pair_counts
            if can_infer and now - last_infer >= 1.0 / max(float(args.max_infer_hz), 1.0):
                start = time.monotonic()
                x = compose_input(front, wrist, camera, image_size).to(device)
                with torch.no_grad():
                    p = float(torch.sigmoid(model(x).view(-1))[0].detach().cpu())
                infer_ms = (time.monotonic() - start) * 1000.0
                prob = p
                raw_phase = int(p >= float(cfg.get("threshold", 0.5)))
                if p >= pos_t:
                    pos_count += 1
                    neg_count = 0
                elif p <= neg_t:
                    neg_count += 1
                    pos_count = 0
                else:
                    pos_count = 0
                    neg_count = 0
                if phase == 0 and pos_count >= int(args.hold_frames):
                    phase = 1
                elif phase == 1 and neg_count >= int(args.hold_frames):
                    phase = 0
                infer_meter.tick()
                last_infer = now
                last_pair_counts = (front_count, wrist_count)

            if not args.no_window and now - last_draw >= 1.0 / max(float(args.max_display_hz), 1.0):
                draw_hz = draw_meter.tick()
                canvas = draw_preview(
                    front,
                    wrist,
                    int(args.width),
                    front_count=front_count,
                    wrist_count=wrist_count,
                    front_age_ms=(now - front_stamp) * 1000.0 if front_stamp > 0 else 999999.0,
                    wrist_age_ms=(now - wrist_stamp) * 1000.0 if wrist_stamp > 0 else 999999.0,
                    prob=prob,
                    raw_phase=raw_phase,
                    hyst_phase=phase,
                    infer_hz=infer_meter.value(),
                    draw_hz=draw_hz,
                    infer_ms=infer_ms,
                    model_name=str(cfg.get("model", "unknown")),
                    checkpoint_name=args.checkpoint.parent.name,
                )
                cv2.imshow(str(args.window_name), canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                last_draw = now
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
