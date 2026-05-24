"""Quick connectivity test for UR3e + Robotiq gripper.

Usage::

    python test_connection.py [robot_ip]

Defaults to 192.168.5.1 when no IP is given.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ur3e_api import UR3eController


def main(robot_ip: str) -> int:
    print(f"Connecting to UR3e at {robot_ip} ...")
    robot = UR3eController(robot_ip, auto_connect=False)
    robot.connect()
    print("  RTDE interface: OK")

    # -- joint state --
    q = robot.get_joint_positions()
    print(f"  Actual Q (deg): {np.round(np.degrees(q), 2).tolist()}")
    print(f"  Actual Q (rad): {np.round(q, 4).tolist()}")

    # -- move to home --
    print("  Moving to home pose ...")
    robot.move_joints(robot.home_q)
    reached = robot.wait_until_joints_reached(robot.home_q, tolerance=0.02, timeout=10.0)
    print(f"    Home reached: {reached}")
    q_home = robot.get_joint_positions()
    print(f"    Home Q (deg): {np.round(np.degrees(q_home), 2).tolist()}")

    # -- servoJ --
    print("  Testing servoJ (2 second loop) ...")
    offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    servo_start = time.monotonic()
    while time.monotonic() - servo_start < 3.0:
        t = time.monotonic() - servo_start
        # sine wiggle on joint 5 (wrist roll), back and forth
        target = robot.home_q + offset * np.sin(2.0 * np.pi * 0.25 * t)
        robot.servo_joints(target)
        time.sleep(0.002)  # 500 Hz
    robot.servo_stop()
    q_after_servo = robot.get_joint_positions()
    print(f"    After servo Q (deg): {np.round(np.degrees(q_after_servo), 2).tolist()}")

    # -- gripper --
    try:
        robot.attach_gripper()
        print("  Gripper: connected & activated")

        trigger = robot.get_gripper_trigger()
        angle = robot.get_gripper_angle()
        print(f"  Gripper trigger : {trigger:.3f}  (0=open, 1=closed)")
        print(f"  Gripper angle   : {angle:.4f} rad")

        # open → close → open
        print("  Closing gripper ...")
        robot.move_gripper_by_trigger(0.93)
        time.sleep(2.0)
        print(f"    After close: trigger={robot.get_gripper_trigger():.3f}, angle={robot.get_gripper_angle():.4f}")

        print("  Opening gripper ...")
        robot.move_gripper_by_trigger(0.0)
        time.sleep(2.0)
        print(f"    After open:  trigger={robot.get_gripper_trigger():.3f}, angle={robot.get_gripper_angle():.4f}")

    except Exception as exc:
        print(f"  Gripper: skipped  ({exc})")

    robot.close()
    print("Disconnected OK.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="UR3e connection test")
    p.add_argument("ip", nargs="?", default="192.168.5.1", help="Robot IP (default: 192.168.5.1)")
    args = p.parse_args()
    rc = main(args.ip)
    sys.exit(rc)
