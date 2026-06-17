#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
if "QT_QPA_FONTDIR" not in os.environ:
    for font_dir in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"):
        if Path(font_dir).exists():
            os.environ["QT_QPA_FONTDIR"] = font_dir
            break

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_gate.eval_rlt_gate import load_checkpoint
from scripts.rlt_gate.live_rlt_gate_monitor import (
    FpsMeter,
    LatestImage,
    compose_input,
    decode_ros_image,
    pad_to_height,
    resize_half,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/ready_gate/ready_gate_20260617_150953/best.pt"


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
    raw_ready: int,
    stable_ready: int,
    infer_hz: float,
    draw_hz: float,
    infer_ms: float,
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

    panel_h = 108
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_h), (0, 0, 0), -1)
    cv2.putText(canvas, "front", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "wrist", (front_small.shape[1] + 12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    prob_text = "none" if prob is None else f"{prob:.3f}"
    state_text = "READY" if stable_ready else "NOT READY"
    state_color = (0, 220, 0) if stable_ready else (80, 80, 255)
    cv2.putText(
        canvas,
        f"Ready gate live | ckpt={checkpoint_name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"prob={prob_text} raw={raw_ready} stable={stable_ready} {state_text} "
        f"infer_hz={infer_hz:.1f} draw_hz={draw_hz:.1f} infer={infer_ms:.1f}ms",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"front#{front_count} age={front_age_ms:.0f}ms  wrist#{wrist_count} age={wrist_age_ms:.0f}ms",
        (12, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 220, 180),
        2,
    )

    bar_x, bar_y, bar_w, bar_h = 12, 96, min(canvas.shape[1] - 72, 520), 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    if prob is not None:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * prob), bar_y + bar_h), (0, 210, 255), -1)
    cv2.circle(canvas, (canvas.shape[1] - 34, 36), 16, state_color, -1)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live ready-gate classifier monitor from ROS camera topics.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--front-topic", default="/camera/d455/color/image_raw")
    parser.add_argument("--wrist-topic", default="/camera/d405/color/image_raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-infer-hz", type=float, default=15.0)
    parser.add_argument("--max-display-hz", type=float, default=30.0)
    parser.add_argument("--positive-threshold", type=float, default=0.6)
    parser.add_argument("--negative-threshold", type=float, default=0.4)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--window-name", default="UR3e ready gate live")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--print-hz", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = args.checkpoint.expanduser().resolve()
    model, cfg = load_checkpoint(checkpoint, device)
    camera = str(cfg.get("camera", "both"))
    image_size = int(cfg.get("image_size", 128))

    latest_front = LatestImage()
    latest_wrist = LatestImage()
    infer_meter = FpsMeter()
    draw_meter = FpsMeter()
    stable_ready = 0
    raw_ready = 0
    pos_count = 0
    neg_count = 0
    prob: float | None = None
    infer_ms = 0.0
    last_infer = 0.0
    last_draw = 0.0
    last_print = 0.0
    last_pair_counts = (-1, -1)

    rclpy.init()
    node = rclpy.create_node("ur3e_ready_gate_live_eval")
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
        f"Ready-gate live eval ready. checkpoint={checkpoint}, model={cfg.get('model')}, camera={camera}, "
        f"device={device}, infer_hz<={args.max_infer_hz}, front={args.front_topic}, wrist={args.wrist_topic}"
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
                    prob = float(torch.sigmoid(model(x).view(-1))[0].detach().cpu())
                infer_ms = (time.monotonic() - start) * 1000.0
                raw_ready = int(prob >= float(cfg.get("threshold", 0.5)))
                if prob >= float(args.positive_threshold):
                    pos_count += 1
                    neg_count = 0
                elif prob <= float(args.negative_threshold):
                    neg_count += 1
                    pos_count = 0
                else:
                    pos_count = 0
                    neg_count = 0
                if stable_ready == 0 and pos_count >= int(args.hold_frames):
                    stable_ready = 1
                elif stable_ready == 1 and neg_count >= int(args.hold_frames):
                    stable_ready = 0
                infer_meter.tick()
                last_infer = now
                last_pair_counts = (front_count, wrist_count)

            if now - last_print >= 1.0 / max(float(args.print_hz), 0.1):
                prob_text = "none" if prob is None else f"{prob:.3f}"
                print(
                    f"\rready={stable_ready} raw={raw_ready} prob={prob_text} "
                    f"infer_hz={infer_meter.value():.1f} infer={infer_ms:.1f}ms "
                    f"front#{front_count} wrist#{wrist_count}",
                    end="",
                    flush=True,
                )
                last_print = now

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
                    raw_ready=raw_ready,
                    stable_ready=stable_ready,
                    infer_hz=infer_meter.value(),
                    draw_hz=draw_hz,
                    infer_ms=infer_ms,
                    checkpoint_name=checkpoint.parent.name,
                )
                cv2.imshow(str(args.window_name), canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                last_draw = now
    finally:
        print()
        if not args.no_window:
            cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
