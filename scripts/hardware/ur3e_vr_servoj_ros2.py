#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_teleop.config import TOPIC_VR_COMMAND, TeleopConfig
from real_teleop.nodes import IkNode, RobotNode, VrNode


def _split_nodes(args) -> list[str]:
    nodes = ["vr", "ik", "robot"]
    return nodes


def _node_command(args, node_name: str) -> list[str]:
    if node_name == "twin":
        node_name = "ik"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--node", node_name]
    cmd += ["--robot-ip", args.robot_ip]
    cmd += ["--control-mode", args.control_mode]
    cmd += ["--impedance-profile", args.impedance_profile]
    if args.xml:
        cmd += ["--xml", args.xml]
    if args.dry_run and node_name == "robot":
        cmd.append("--dry-run")
    if args.no_twin and node_name == "ik":
        cmd.append("--no-twin")
    if node_name == "vr" and args.vr_output_topic != TOPIC_VR_COMMAND:
        cmd += ["--vr-output-topic", args.vr_output_topic]
    return cmd


def _shell_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _node_shell(args, node_name: str, *, keep_open: bool = True) -> str:
    parts = [f"cd {shlex.quote(str(ROOT))}"]
    if os.environ.get("ROS_LOG_DIR") is None:
        parts.append("export ROS_LOG_DIR=/tmp/ros_logs")
        parts.append("mkdir -p /tmp/ros_logs")
    parts.append(_shell_command(_node_command(args, node_name)))
    if keep_open:
        parts.append("exec bash")
    return "; ".join(parts)


def _launch_split_tabs(args) -> int:
    if args.split_launcher == "print":
        _print_split_commands(args, "Split launch commands:")
        return 0

    if args.split_launcher in ("auto", "tmux") and shutil.which("tmux") is not None:
        return _launch_tmux(args)

    if args.split_launcher == "tmux":
        _print_split_commands(args, "tmux not found.")
        return 1

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        _print_split_commands(args, "No graphical display found.")
        return 1

    if args.split_launcher in ("auto", "gnome-tabs"):
        return _launch_gnome_tabs(args)

    _print_split_commands(args, f"Unsupported launcher: {args.split_launcher}")
    return 1


def _launch_tmux(args) -> int:
    nodes = _split_nodes(args)
    session = f"ur3e_vr_{os.getpid()}"
    tmux = shutil.which("tmux")

    first = nodes[0]
    subprocess.run(
        [tmux, "new-session", "-d", "-s", session, "-n", first, "bash", "-lc", _node_shell(args, first)],
        cwd=str(ROOT),
        check=True,
    )
    for node_name in nodes[1:]:
        subprocess.run(
            [tmux, "new-window", "-t", session, "-n", node_name, "bash", "-lc", _node_shell(args, node_name)],
            cwd=str(ROOT),
            check=True,
        )

    attach_cmd = ["tmux", "attach-session", "-t", session]
    attach = " ".join(shlex.quote(part) for part in attach_cmd)
    print(f"Launching tmux session {session}: {', '.join(nodes)}")
    if sys.stdin.isatty():
        subprocess.run(attach_cmd, cwd=str(ROOT), check=False)
    else:
        print(f"Attach with: {attach}")
    return 0


def _launch_gnome_tabs(args) -> int:
    nodes = _split_nodes(args)
    terminal = shutil.which("gnome-terminal")
    if terminal is None:
        _print_split_commands(args, "No supported tabbed terminal found. Install gnome-terminal or run manually.")
        return 1

    launch_cmd = [terminal]
    for node_name in nodes:
        launch_cmd += ["--tab", "--title", f"ur3e-{node_name}", "--", "bash", "-lc", _node_shell(args, node_name)]

    print(f"Launching split tabs with {Path(terminal).name}: {', '.join(nodes)}")
    subprocess.Popen(launch_cmd, cwd=str(ROOT))
    return 0


def _print_split_commands(args, reason: str) -> None:
    print(reason)
    print("Run these commands in separate terminals:")
    for node_name in _split_nodes(args):
        print(_node_shell(args, node_name, keep_open=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="UR3e real-machine VR teleop over ROS2 topics.")
    parser.add_argument("--node", choices=["all", "all-tabs", "vr", "ik", "robot", "twin"], default="all")
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--xml", default=None, help="MuJoCo XML used for IK and digital twin.")
    parser.add_argument(
        "--control-mode",
        choices=["impedance", "servoj"],
        default=TeleopConfig().robot_control_mode,
        help="Robot execution mode. impedance consumes target TCP poses; servoj consumes IK joint targets.",
    )
    parser.add_argument(
        "--impedance-profile",
        default=TeleopConfig().impedance_profile,
        help="Profile name from real_teleop/impedance/config.py used by impedance mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not connect to the UR controller.")
    parser.add_argument("--no-twin", action="store_true", help="Do not start the MuJoCo twin viewer inside the IK node.")
    parser.add_argument(
        "--vr-output-topic",
        default=TOPIC_VR_COMMAND,
        help=(
            "Topic used by the VR node. Keep the default for normal teleop. "
            "For rollout/RLT intervention, publish VR to a raw topic such as /ur3e_vr/vr_command_raw."
        ),
    )
    parser.add_argument("--split-tabs", action="store_true", help="Launch vr/ik/robot in separate terminal tabs.")
    parser.add_argument(
        "--split-launcher",
        choices=["auto", "tmux", "gnome-tabs", "print"],
        default="auto",
        help="Launcher used by --node all-tabs / --split-tabs.",
    )
    args = parser.parse_args()

    if args.node == "all-tabs" or (args.node == "all" and args.split_tabs):
        return _launch_split_tabs(args)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    cfg = TeleopConfig(
        robot_ip=args.robot_ip,
        xml_path=args.xml or TeleopConfig().xml_path,
        robot_control_mode=args.control_mode,
        impedance_profile=args.impedance_profile,
    )
    rclpy.init()

    ros_nodes = []
    resources = []
    try:
        if args.node in ("all", "vr"):
            node = rclpy.create_node("ur3e_vr_input")
            resources.append(VrNode(node, cfg, output_topic=args.vr_output_topic))
            ros_nodes.append(node)
        if args.node in ("all", "ik", "twin"):
            node = rclpy.create_node("ur3e_mink_ik")
            resources.append(IkNode(node, cfg, enable_twin=True if args.node == "twin" else not args.no_twin))
            ros_nodes.append(node)
        if args.node in ("all", "robot"):
            node = rclpy.create_node("ur3e_robot")
            resources.append(RobotNode(node, cfg, dry_run=args.dry_run))
            ros_nodes.append(node)

        executor = MultiThreadedExecutor(num_threads=max(2, len(ros_nodes)))
        for node in ros_nodes:
            executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        for resource in reversed(resources):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
        for node in ros_nodes:
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
