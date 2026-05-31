#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_D455_SERIAL = "151422253456"
DEFAULT_D405_SERIAL = "218722270648"
TOPICS = {
    "d455": "/camera/d455/color/image_raw",
    "d405": "/camera/d405/color/image_raw",
}


@dataclass(frozen=True)
class CameraSpec:
    key: str
    model: str
    serial: str
    topic: str
    usb_port: str = ""


@dataclass
class FrameStats:
    count: int = 0
    first_time: float | None = None
    last_time: float | None = None
    first_mean: float | None = None
    last_mean: float | None = None

    def add(self, now: float, mean: float | None) -> None:
        if self.count == 0:
            self.first_time = now
            self.first_mean = mean
        self.count += 1
        self.last_time = now
        self.last_mean = mean

    @property
    def rate(self) -> float:
        if self.count < 2 or self.first_time is None or self.last_time is None:
            return 0.0
        dt = self.last_time - self.first_time
        return 0.0 if dt <= 0 else (self.count - 1) / dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Layered RealSense diagnostics for the UR3e collection cameras. "
            "Default mode only enumerates devices; use --mode sdk or --mode ros "
            "to test frame delivery."
        )
    )
    parser.add_argument("--mode", choices=("detect", "sdk", "ros", "all"), default="detect")
    parser.add_argument("--camera", choices=("d455", "d405", "both"), default="both")
    parser.add_argument("--d455-serial", default="", help="Override D455 serial; auto-detected if empty.")
    parser.add_argument("--d405-serial", default="", help="Override D405 serial; auto-detected if empty.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=5.0, help="Frame counting duration after first frame.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for the first frame.")
    parser.add_argument("--cleanup", action="store_true", help="Stop stale RealSense ROS nodes before testing.")
    parser.add_argument("--no-launch", action="store_true", help="ROS mode: subscribe to existing topics without launching cameras.")
    parser.add_argument("--keep-running", action="store_true", help="ROS mode: keep launched camera nodes alive after the test.")
    parser.add_argument("--parallel-sdk", action="store_true", help="SDK mode with --camera both: stream both cameras at the same time.")
    parser.add_argument("--launch-mode", choices=("separate", "multi"), default="separate")
    parser.add_argument("--stagger", type=float, default=2.0, help="Delay between separate ROS launches.")
    parser.add_argument("--use-usb-port-id", action="store_true", help="Pass usb_port_id to ROS launch when available.")
    parser.add_argument("--ros-reliability", choices=("reliable", "best_effort"), default="reliable")
    parser.add_argument("--ros-durability", choices=("transient_local", "volatile"), default="transient_local")
    return parser.parse_args()


def selected_keys(camera: str) -> list[str]:
    return ["d455", "d405"] if camera == "both" else [camera]


def safe_info(dev, info) -> str:
    try:
        if dev.supports(info):
            return dev.get_info(info)
    except Exception:
        pass
    return ""


def detect_devices() -> dict[str, CameraSpec]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(
            "[detect] FAIL cannot import pyrealsense2. Run from the camera environment, e.g.\n"
            "  conda activate ur3e_rlt\n"
            f"original error: {exc}"
        )
        return {}

    specs: dict[str, CameraSpec] = {}
    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
    except Exception as exc:
        print(f"[detect] FAIL pyrealsense2 could not query devices: {exc}")
        return specs
    if not devices:
        print("[detect] no RealSense device found")
        return specs

    for dev in devices:
        name = safe_info(dev, rs.camera_info.name)
        serial = safe_info(dev, rs.camera_info.serial_number)
        physical_port = safe_info(dev, rs.camera_info.physical_port)
        usb_type = safe_info(dev, rs.camera_info.usb_type_descriptor)
        firmware = safe_info(dev, rs.camera_info.firmware_version)
        port_match = re.search(r"/usb\d+/([^/]+)/", physical_port)
        usb_port = port_match.group(1) if port_match else ""
        print(
            "[detect] "
            f"name={name or 'unknown'} serial={serial or 'unknown'} "
            f"usb={usb_type or 'unknown'} fw={firmware or 'unknown'} port={usb_port or 'unknown'}"
        )
        if "D455" in name:
            specs["d455"] = CameraSpec("d455", "D455", serial, TOPICS["d455"], usb_port)
        elif "D405" in name:
            specs["d405"] = CameraSpec("d405", "D405", serial, TOPICS["d405"], usb_port)
    return specs


def specs_from_args(args: argparse.Namespace) -> dict[str, CameraSpec]:
    detected = detect_devices()
    d455 = detected.get(
        "d455",
        CameraSpec("d455", "D455", DEFAULT_D455_SERIAL, TOPICS["d455"]),
    )
    d405 = detected.get(
        "d405",
        CameraSpec("d405", "D405", DEFAULT_D405_SERIAL, TOPICS["d405"]),
    )
    if args.d455_serial:
        d455 = CameraSpec("d455", "D455", args.d455_serial, TOPICS["d455"], d455.usb_port)
    if args.d405_serial:
        d405 = CameraSpec("d405", "D405", args.d405_serial, TOPICS["d405"], d405.usb_port)
    return {"d455": d455, "d405": d405}


def cleanup_realsense_ros() -> None:
    patterns = [
        "realsense2_camera_node.*__node:=d455.*__ns:=/camera",
        "realsense2_camera_node.*__node:=d405.*__ns:=/camera",
        "ros2 launch realsense2_camera rs_launch.py.*camera_name:=d455",
        "ros2 launch realsense2_camera rs_launch.py.*camera_name:=d405",
        "ros2 launch realsense2_camera rs_multi_camera_launch.py",
    ]
    print("[cleanup] stopping stale RealSense ROS launch/node processes")
    for pattern in patterns:
        subprocess.run(["pkill", "-INT", "-f", pattern], check=False)
    time.sleep(1.0)


def ros_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", "/tmp/ur3e_rlt_ros_logs")
    Path(env["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def launch_single_ros_camera(spec: CameraSpec, args: argparse.Namespace) -> subprocess.Popen:
    profile = f"{args.width},{args.height},{args.fps}"
    port_arg = f" usb_port_id:={spec.usb_port}" if args.use_usb_port_id and spec.usb_port else ""
    cmd = (
        "set +u; source /opt/ros/humble/setup.bash; set -u; "
        "ros2 launch realsense2_camera rs_launch.py "
        "camera_namespace:=camera "
        f"camera_name:={spec.key} "
        f"serial_no:=_{spec.serial} "
        f"{port_arg} "
        "enable_color:=true "
        f"rgb_camera.color_profile:={profile} "
        f"depth_module.color_profile:={profile} "
        "enable_depth:=false "
        "enable_infra:=false "
        "enable_infra1:=false "
        "enable_infra2:=false "
        "pointcloud.enable:=false "
        "align_depth.enable:=false "
        "publish_tf:=false "
        "output:=screen"
    )
    port_text = f" port={spec.usb_port}" if port_arg else ""
    print(f"[ros-launch:{spec.key}] serial={spec.serial}{port_text} profile={profile}")
    return subprocess.Popen(["bash", "-lc", cmd], env=ros_env(), preexec_fn=os.setsid)


def launch_multi_ros_camera(specs: dict[str, CameraSpec], args: argparse.Namespace) -> subprocess.Popen:
    d455 = specs["d455"]
    d405 = specs["d405"]
    profile = f"{args.width},{args.height},{args.fps}"
    port_args = ""
    if args.use_usb_port_id and d455.usb_port and d405.usb_port:
        port_args = f" usb_port_id1:={d455.usb_port} usb_port_id2:={d405.usb_port}"
    cmd = (
        "set +u; source /opt/ros/humble/setup.bash; set -u; "
        "ros2 launch realsense2_camera rs_multi_camera_launch.py "
        "camera_namespace1:=camera "
        "camera_namespace2:=camera "
        "camera_name1:=d455 "
        "camera_name2:=d405 "
        f"serial_no1:=_{d455.serial} "
        f"serial_no2:=_{d405.serial} "
        f"{port_args} "
        "enable_color1:=true "
        "enable_color2:=true "
        f"rgb_camera.color_profile1:={profile} "
        f"rgb_camera.color_profile2:={profile} "
        f"depth_module.color_profile1:={profile} "
        f"depth_module.color_profile2:={profile} "
        "enable_depth1:=false "
        "enable_depth2:=false "
        "enable_infra1:=false "
        "enable_infra2:=false "
        "enable_infra11:=false "
        "enable_infra12:=false "
        "enable_infra21:=false "
        "enable_infra22:=false "
        "pointcloud.enable1:=false "
        "pointcloud.enable2:=false "
        "align_depth.enable1:=false "
        "align_depth.enable2:=false "
        "publish_tf1:=false "
        "publish_tf2:=false "
        "output1:=screen "
        "output2:=screen"
    )
    print(f"[ros-launch:both] d455={d455.serial} d405={d405.serial} profile={profile}")
    return subprocess.Popen(["bash", "-lc", cmd], env=ros_env(), preexec_fn=os.setsid)


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5.0)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass


def run_sdk_stream(spec: CameraSpec, args: argparse.Namespace) -> bool:
    try:
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(
            "[sdk] FAIL cannot import pyrealsense2/numpy. Run from the camera environment, e.g.\n"
            "  conda activate ur3e_rlt\n"
            f"original error: {exc}"
        )
        return False

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(spec.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    stats = FrameStats()
    print(f"[sdk:{spec.key}] starting serial={spec.serial} {args.width}x{args.height}@{args.fps}")
    try:
        pipeline.start(config)
        deadline = time.monotonic() + args.timeout
        while stats.count == 0 and time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if color:
                image = np.asanyarray(color.get_data())
                stats.add(time.monotonic(), float(image.mean()))
        if stats.count == 0:
            print(f"[sdk:{spec.key}] FAIL no color frame within {args.timeout:.1f}s")
            return False

        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if color:
                image = np.asanyarray(color.get_data())
                stats.add(time.monotonic(), float(image.mean()))
    except Exception as exc:
        print(f"[sdk:{spec.key}] FAIL {exc}")
        return False
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    print(
        f"[sdk:{spec.key}] OK frames={stats.count} rate={stats.rate:.1f}Hz "
        f"mean(first,last)=({stats.first_mean:.1f},{stats.last_mean:.1f})"
    )
    return stats.count >= max(2, int(args.duration * args.fps * 0.5))


def run_parallel_sdk_stream(specs: dict[str, CameraSpec], args: argparse.Namespace) -> bool:
    try:
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(
            "[sdk:both] FAIL cannot import pyrealsense2/numpy. Run from the camera environment, e.g.\n"
            "  conda activate ur3e_rlt\n"
            f"original error: {exc}"
        )
        return False

    pipelines: dict[str, object] = {}
    stats = {"d455": FrameStats(), "d405": FrameStats()}
    try:
        for key in ("d455", "d405"):
            spec = specs[key]
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(spec.serial)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
            print(f"[sdk:{key}] starting parallel serial={spec.serial} {args.width}x{args.height}@{args.fps}")
            pipeline.start(config)
            pipelines[key] = pipeline

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and not all(item.count > 0 for item in stats.values()):
            for key, pipeline in pipelines.items():
                frames = pipeline.poll_for_frames()
                color = frames.get_color_frame() if frames else None
                if color:
                    image = np.asanyarray(color.get_data())
                    stats[key].add(time.monotonic(), float(image.mean()))
            time.sleep(0.001)

        missing = [key for key, item in stats.items() if item.count == 0]
        if missing:
            print(f"[sdk:both] FAIL no first frame for {missing} within {args.timeout:.1f}s")
            return False

        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            for key, pipeline in pipelines.items():
                frames = pipeline.poll_for_frames()
                color = frames.get_color_frame() if frames else None
                if color:
                    image = np.asanyarray(color.get_data())
                    stats[key].add(time.monotonic(), float(image.mean()))
            time.sleep(0.001)
    except Exception as exc:
        print(f"[sdk:both] FAIL {exc}")
        return False
    finally:
        for pipeline in pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass

    ok = True
    min_frames = max(2, int(args.duration * args.fps * 0.5))
    for key, item in stats.items():
        print(
            f"[sdk:{key}] parallel frames={item.count} rate={item.rate:.1f}Hz "
            f"mean(first,last)=({item.first_mean:.1f},{item.last_mean:.1f})"
        )
        ok = ok and item.count >= min_frames
    if not ok:
        print(f"[sdk:both] FAIL frame count below half of expected {args.fps:.1f}Hz")
    return ok


def run_sdk_mode(args: argparse.Namespace, specs: dict[str, CameraSpec]) -> bool:
    if args.camera == "both" and args.parallel_sdk:
        return run_parallel_sdk_stream(specs, args)
    ok = True
    for key in selected_keys(args.camera):
        ok = run_sdk_stream(specs[key], args) and ok
        time.sleep(0.5)
    return ok


def message_mean(msg) -> float | None:
    if not msg.data:
        return None
    # Sample instead of decoding the whole frame; this is enough to prove the
    # image data is changing and keeps diagnostics lightweight.
    step = max(1, len(msg.data) // 4096)
    return sum(msg.data[::step]) / len(msg.data[::step])


def run_ros_subscriber(topics: dict[str, str], args: argparse.Namespace) -> bool:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
    except ImportError as exc:
        print(
            "[ros-sub] FAIL cannot import ROS Python packages. Run this from the ROS/conda environment, e.g.\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  conda activate ur3e_rlt\n"
            f"original error: {exc}"
        )
        return False

    reliability = (
        ReliabilityPolicy.RELIABLE
        if args.ros_reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    durability = (
        DurabilityPolicy.TRANSIENT_LOCAL
        if args.ros_durability == "transient_local"
        else DurabilityPolicy.VOLATILE
    )
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=reliability,
        durability=durability,
    )

    class CounterNode(Node):
        def __init__(self) -> None:
            super().__init__("ur3e_realsense_layered_diagnostic")
            self.stats = {key: FrameStats() for key in topics}
            for key, topic in topics.items():
                self.create_subscription(Image, topic, lambda msg, k=key: self.on_image(k, msg), qos)

        def on_image(self, key: str, msg) -> None:
            self.stats[key].add(time.monotonic(), message_mean(msg))

    rclpy.init(args=None)
    node = CounterNode()
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if all(stat.count > 0 for stat in node.stats.values()):
                break
        missing = [key for key, stat in node.stats.items() if stat.count == 0]
        if missing:
            for key, topic in topics.items():
                print(
                    f"[ros-sub:{key}] topic={topic} publishers={node.count_publishers(topic)} "
                    f"frames={node.stats[key].count}"
                )
            print(f"[ros-sub] FAIL no first frame for {missing} within {args.timeout:.1f}s")
            return False

        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

        ok = True
        min_frames = max(2, int(args.duration * args.fps * 0.5))
        for key, topic in topics.items():
            stat = node.stats[key]
            print(
                f"[ros-sub:{key}] topic={topic} publishers={node.count_publishers(topic)} "
                f"frames={stat.count} rate={stat.rate:.1f}Hz "
                f"mean(first,last)=({stat.first_mean:.1f},{stat.last_mean:.1f})"
            )
            ok = ok and stat.count >= min_frames
        if not ok:
            print(f"[ros-sub] FAIL frame count below half of expected {args.fps:.1f}Hz")
        return ok
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_ros_mode(args: argparse.Namespace, specs: dict[str, CameraSpec]) -> bool:
    keys = selected_keys(args.camera)
    topics = {key: specs[key].topic for key in keys}
    procs: list[subprocess.Popen] = []
    if not args.no_launch:
        if args.launch_mode == "multi" and keys == ["d455", "d405"]:
            procs.append(launch_multi_ros_camera(specs, args))
        else:
            for index, key in enumerate(keys):
                if index > 0:
                    time.sleep(args.stagger)
                procs.append(launch_single_ros_camera(specs[key], args))
    try:
        ok = run_ros_subscriber(topics, args)
        if args.keep_running and procs:
            print("[ros] keeping launched camera nodes alive; press Ctrl+C to stop")
            while True:
                time.sleep(1.0)
        return ok
    finally:
        if not args.keep_running:
            for proc in procs:
                stop_process(proc)


def check_requested_serials(specs: dict[str, CameraSpec], keys: Iterable[str]) -> bool:
    ok = True
    for key in keys:
        spec = specs[key]
        if not spec.serial:
            print(f"[error] missing serial for {key}")
            ok = False
    return ok


def main() -> int:
    args = parse_args()
    if args.cleanup:
        cleanup_realsense_ros()

    if args.mode == "detect":
        detected = detect_devices()
        missing = [key for key in selected_keys(args.camera) if key not in detected]
        if missing:
            print(f"[detect] missing requested cameras: {missing}")
            return 1
        return 0

    specs = specs_from_args(args)
    keys = selected_keys(args.camera)
    if not check_requested_serials(specs, keys):
        return 2

    if args.mode == "sdk":
        return 0 if run_sdk_mode(args, specs) else 1

    if args.mode == "ros":
        return 0 if run_ros_mode(args, specs) else 1

    sdk_ok = run_sdk_mode(args, specs)
    if not sdk_ok:
        print("[all] SDK test failed; skipping ROS launch because the lower layer is already failing.")
        return 1
    if args.cleanup:
        cleanup_realsense_ros()
    ros_ok = run_ros_mode(args, specs)
    return 0 if ros_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
