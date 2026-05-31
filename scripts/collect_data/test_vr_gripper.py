#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import parse_joint_target, parse_robot_state, parse_vr_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect VR/robot gripper values published by the UR3e teleop stack.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--vr-topic", default="/ur3e_vr/vr_command")
    parser.add_argument("--state-topic", default="/ur3e_vr/robot_state")
    parser.add_argument("--joint-topic", default="/ur3e_vr/joint_target")
    parser.add_argument("--print-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    class GripperProbe(Node):
        def __init__(self) -> None:
            super().__init__("ur3e_vr_gripper_probe")
            self.vr_count = 0
            self.state_count = 0
            self.joint_count = 0
            self.vr_gripper = 0.0
            self.state_gripper = 0.0
            self.joint_gripper = 0.0
            self.vr_max = 0.0
            self.state_max = 0.0
            self.joint_max = 0.0
            self.enable = False
            self.left_trigger = 0.0
            self.left_grip = 0.0
            self.stop_collection = False
            self.last_vr_time = 0.0
            self.last_state_time = 0.0
            self.last_joint_time = 0.0
            self.create_subscription(Float64MultiArray, args.vr_topic, self._on_vr, qos)
            self.create_subscription(Float64MultiArray, args.state_topic, self._on_state, qos)
            self.create_subscription(Float64MultiArray, args.joint_topic, self._on_joint, qos)

        def _on_vr(self, msg: Float64MultiArray) -> None:
            payload = parse_vr_command(msg.data)
            self.vr_count += 1
            self.vr_gripper = float(payload.get("gripper", 0.0))
            self.vr_max = max(self.vr_max, self.vr_gripper)
            self.enable = bool(payload.get("enable", False))
            self.left_trigger = float(payload.get("left_trigger", 0.0))
            self.left_grip = float(payload.get("left_grip", 0.0))
            self.stop_collection = bool(payload.get("stop_collection", False))
            self.last_vr_time = time.monotonic()

        def _on_state(self, msg: Float64MultiArray) -> None:
            payload = parse_robot_state(msg.data)
            self.state_count += 1
            self.state_gripper = float(payload.get("gripper", 0.0))
            self.state_max = max(self.state_max, self.state_gripper)
            self.last_state_time = time.monotonic()

        def _on_joint(self, msg: Float64MultiArray) -> None:
            payload = parse_joint_target(msg.data)
            self.joint_count += 1
            self.joint_gripper = float(payload.get("gripper", 0.0))
            self.joint_max = max(self.joint_max, self.joint_gripper)
            self.last_joint_time = time.monotonic()

    rclpy.init()
    node = GripperProbe()
    start = time.monotonic()
    next_print = start
    try:
        while time.monotonic() - start < args.duration:
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now >= next_print:
                next_print = now + 1.0 / max(args.print_hz, 1e-6)
                vr_age = np.inf if node.last_vr_time <= 0 else now - node.last_vr_time
                state_age = np.inf if node.last_state_time <= 0 else now - node.last_state_time
                joint_age = np.inf if node.last_joint_time <= 0 else now - node.last_joint_time
                print(
                    "vr_gripper={:.3f} joint_gripper={:.3f} state_gripper={:.3f} "
                    "max(vr,joint,state)=({:.3f},{:.3f},{:.3f}) enable={} "
                    "left_trigger={:.3f} left_grip={:.3f} stop={} "
                    "count(vr,joint,state)=({},{},{}) age_ms(vr,joint,state)=({:.0f},{:.0f},{:.0f})".format(
                        node.vr_gripper,
                        node.joint_gripper,
                        node.state_gripper,
                        node.vr_max,
                        node.joint_max,
                        node.state_max,
                        int(node.enable),
                        node.left_trigger,
                        node.left_grip,
                        int(node.stop_collection),
                        node.vr_count,
                        node.joint_count,
                        node.state_count,
                        vr_age * 1000.0,
                        joint_age * 1000.0,
                        state_age * 1000.0,
                    )
                )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.vr_count <= 0:
        print(f"FAIL: no VR command received on {args.vr_topic}")
        return 1
    if node.vr_max <= 1e-3:
        print("WARN: VR gripper stayed near 0. Press the right index trigger while this probe is running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
