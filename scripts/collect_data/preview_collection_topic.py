#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import numpy as np


class LatestImage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.image: np.ndarray | None = None
        self.count = 0
        self.last_stamp = 0.0

    def set(self, image: np.ndarray) -> None:
        with self.lock:
            self.image = np.ascontiguousarray(image)
            self.count += 1
            self.last_stamp = time.monotonic()

    def get(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            image = None if self.image is None else self.image.copy()
            return image, self.count, self.last_stamp


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV preview for UR3e collection camera ROS image topics.")
    parser.add_argument("--topic", default="/ur3e_vr/collection_preview")
    parser.add_argument("--front-topic", default="")
    parser.add_argument("--wrist-topic", default="")
    parser.add_argument("--window-name", default="UR3e collection preview")
    parser.add_argument("--max-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--status", action="store_true", default=True)
    parser.add_argument("--no-status", dest="status", action="store_false")
    return parser.parse_args()


def resize_to_half_width(image: np.ndarray, total_width: int) -> np.ndarray:
    import cv2

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


def compose_pair(
    front: tuple[np.ndarray | None, int, float],
    wrist: tuple[np.ndarray | None, int, float],
    width: int,
    status: str = "",
) -> np.ndarray:
    import cv2

    front_img, front_count, front_stamp = front
    wrist_img, wrist_count, wrist_stamp = wrist
    if front_img is None and wrist_img is None:
        canvas = np.zeros((360, int(width), 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for camera topics", (24, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if status:
            cv2.putText(canvas, status, (24, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
        return canvas
    if front_img is None:
        front_img = np.zeros_like(wrist_img)
    if wrist_img is None:
        wrist_img = np.zeros_like(front_img)
    front_small = resize_to_half_width(front_img, width)
    wrist_small = resize_to_half_width(wrist_img, width)
    height = max(front_small.shape[0], wrist_small.shape[0])
    front_small = pad_to_height(front_small, height)
    wrist_small = pad_to_height(wrist_small, height)
    canvas = np.concatenate([front_small, wrist_small], axis=1)
    now = time.monotonic()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (32, 32, 32), -1)
    cv2.putText(canvas, "front", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "wrist", (front_small.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        f"front#{front_count} age={(now-front_stamp)*1000:.0f}ms  wrist#{wrist_count} age={(now-wrist_stamp)*1000:.0f}ms",
        (12, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    return canvas


def main() -> int:
    import cv2
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    args = parse_args()
    latest = LatestImage()
    latest_front = LatestImage()
    latest_wrist = LatestImage()
    paired_mode = bool(args.front_topic and args.wrist_topic)

    rclpy.init()
    node = rclpy.create_node("ur3e_collection_opencv_preview")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def on_image(msg: Image) -> None:
        try:
            latest.set(decode_ros_image(msg))
        except Exception as exc:
            node.get_logger().warn(f"Bad preview image: {exc}")

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

    if paired_mode:
        node.create_subscription(Image, str(args.front_topic), on_front, qos)
        node.create_subscription(Image, str(args.wrist_topic), on_wrist, qos)
        node.get_logger().info(
            f"OpenCV preview waiting on front={args.front_topic}, wrist={args.wrist_topic}. Press q to close."
        )
    else:
        node.create_subscription(Image, str(args.topic), on_image, qos)
        node.get_logger().info(f"OpenCV preview waiting on {args.topic}. Press q in the window to close.")
    cv2.namedWindow(str(args.window_name), cv2.WINDOW_NORMAL)

    period = 1.0 / max(float(args.max_fps), 1.0)
    last_draw = 0.0
    last_count = -1
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            if paired_mode:
                front_state = latest_front.get()
                wrist_state = latest_wrist.get()
                status = (
                    f"publishers front={node.count_publishers(str(args.front_topic))} "
                    f"wrist={node.count_publishers(str(args.wrist_topic))} "
                    f"frames front={front_state[1]} wrist={wrist_state[1]}"
                )
                image = compose_pair(front_state, wrist_state, int(args.width), status=status)
                count = front_state[1] + wrist_state[1]
                stamp = now
            else:
                image, count, stamp = latest.get()
            if image is None and now - last_draw >= 0.2:
                image = np.zeros((360, int(args.width), 3), dtype=np.uint8)
                cv2.putText(
                    image,
                    f"Waiting for {args.topic}",
                    (24, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    image,
                    "collector is running but no preview images have arrived yet",
                    (24, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (180, 180, 180),
                    2,
                )
                cv2.imshow(str(args.window_name), image)
                last_draw = now
            elif image is not None and (count != last_count or now - last_draw >= period):
                if args.status and not paired_mode:
                    age_ms = (now - stamp) * 1000.0
                    cv2.putText(
                        image,
                        f"preview topic frame={count} age={age_ms:.0f}ms",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                cv2.imshow(str(args.window_name), image)
                last_draw = now
                last_count = count
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
