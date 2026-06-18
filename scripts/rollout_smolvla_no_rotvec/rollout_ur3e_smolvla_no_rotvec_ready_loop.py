#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_teleop.messages import make_vr_command  # noqa: E402
from scripts.rlt_gate.eval_rlt_gate import load_checkpoint as load_gate_checkpoint  # noqa: E402
from scripts.rlt_gate.live_rlt_gate_monitor import compose_input, decode_ros_image  # noqa: E402
from scripts.rlt_train.config import CFG as RLT_CFG  # noqa: E402
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_rtc import (  # noqa: E402
    NoRotvecSmolVLARTCRollout,
    parse_args as parse_rtc_args,
)
from scripts.rollout_smolvla_no_rotvec.rollout_ur3e_smolvla_no_rotvec_sync import (  # noqa: E402
    NoRotvecSmolVLASyncRollout,
    parse_args as parse_sync_args,
    run_pre_model_startup_sequence,
)
from vr_servoj_test.rollout.rollout_ur3e_servoj_smolvla import resolve_policy_path  # noqa: E402


DEFAULT_POLICY_PATH = REPO_ROOT / "outputs/rlt_vla/ur3e_smolvla_0614/checkpoints/030000/pretrained_model"
DEFAULT_READY_GATE_CHECKPOINT = REPO_ROOT / "outputs/ready_gate/ready_gate_20260617_150953/best.pt"


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _argv_has(name: str) -> bool:
    prefix = f"{name}="
    return any(arg == name or arg.startswith(prefix) for arg in sys.argv[1:])


class ReadyLoopMixin:
    """Adds ready-gate controlled repeated trials to plain VLA rollouts."""

    def _init_ready_loop(self) -> None:
        self.ready_gate_checkpoint = _resolve(self.args.ready_gate_checkpoint)
        self.ready_model, ready_cfg = load_gate_checkpoint(self.ready_gate_checkpoint, torch.device(self.args.device))
        self.ready_device = next(self.ready_model.parameters()).device
        self.ready_camera = str(ready_cfg.get("camera", "both"))
        self.ready_image_size = int(ready_cfg.get("image_size", 128))
        self.ready_prob: float | None = None
        self.ready_phase = 0
        self.ready_pos_count = 0
        self.ready_neg_count = 0
        self.last_ready_infer = 0.0
        self.loop_active = False
        self.waiting_for_ready = bool(self.args.wait_ready_on_start)
        self.trial_start_s = 0.0
        self.trial_index = 0
        self.home_pulse_until = 0.0
        self.home_block_until = 0.0
        self.ready_start_allowed_after = self.now_sec() + float(self.args.ready_after_home_settle_s)
        self.ready_timer = self.node.create_timer(
            1.0 / max(self.args.ready_gate_infer_hz, 1.0),
            self._ready_tick,
            callback_group=self.callback_group,
        )
        if not self.waiting_for_ready:
            self._start_trial("startup")
        self.node.get_logger().info(
            f"Ready-loop enabled. mode={self.args.mode}, ready_gate={self.ready_gate_checkpoint}, "
            f"trial_duration={self.args.trial_duration_s:.1f}s, "
            f"reset_start={self.args.reset_impedance_on_trial_start}, reset_home={self.args.reset_impedance_during_home}, "
            f"start={'wait_ready' if self.waiting_for_ready else 'immediate'}"
        )

    def _ready_loop_allows_model(self) -> bool:
        now = self.now_sec()
        if now < self.home_block_until:
            self._log_info(f"WAIT home/reset {self.home_block_until - now:.2f}s")
            return False
        if not self.loop_active:
            self._log_info(f"WAIT ready={self.ready_phase}/{self._fmt_prob(self.ready_prob)}")
            return False
        return True

    def _ready_loop_publish_home_if_needed(self) -> bool:
        from std_msgs.msg import Float64MultiArray

        now = self.now_sec()
        if now >= self.home_pulse_until:
            return False
        if self.execute:
            msg = Float64MultiArray()
            msg.data = make_vr_command(
                {
                    "enable": False,
                    "gripper": float(np.clip(self.args.home_gripper_value, 0.0, self.args.gripper_max)),
                    "home": True,
                    "reset_impedance": bool(self.args.reset_impedance_during_home),
                }
            )
            self.command_pub.publish(msg)
        return True

    def _ready_loop_maybe_end_trial(self) -> bool:
        duration = float(self.args.trial_duration_s)
        if duration <= 0.0 or not self.loop_active:
            return False
        if self.now_sec() - self.trial_start_s < duration:
            return False
        self._end_trial("duration")
        return True

    def _start_trial(self, reason: str) -> None:
        from std_msgs.msg import Float64MultiArray

        self._drain_ready_loop_actions()
        self.last_published_pos = None
        self.last_published_gripper = None
        self.loop_active = True
        self.waiting_for_ready = False
        self.trial_start_s = self.now_sec()
        self.trial_index += 1
        if self.execute and bool(self.args.reset_impedance_on_trial_start):
            msg = Float64MultiArray()
            msg.data = make_vr_command({"enable": False, "home": False, "reset_impedance": True})
            self.command_pub.publish(msg)
        self.node.get_logger().info(f"VLA trial {self.trial_index:04d} started by {reason}; impedance reset sent.")

    def _end_trial(self, reason: str) -> None:
        self._drain_ready_loop_actions()
        self.loop_active = False
        self.waiting_for_ready = True
        now = self.now_sec()
        self.home_pulse_until = now + float(self.args.home_pulse_s)
        self.home_block_until = self.home_pulse_until if self.args.block_model_during_home else now
        self.ready_start_allowed_after = self.home_pulse_until + float(self.args.ready_after_home_settle_s)
        self.ready_phase = 0
        self.ready_pos_count = 0
        self.ready_neg_count = 0
        self.node.get_logger().info(
            f"VLA trial {self.trial_index:04d} ended by {reason}; return_home/reset then wait ready_gate."
        )

    def _drain_ready_loop_actions(self) -> None:
        self.current_action = None
        self.last_action_step_time = 0.0
        if hasattr(self, "action_queue"):
            with self.action_lock:
                self.action_queue.clear()
        if hasattr(self, "runner") and hasattr(self.runner, "get_action"):
            for _ in range(256):
                if self.runner.get_action() is None:
                    break

    def _ready_tick(self) -> None:
        if not self.waiting_for_ready:
            return
        now = self.now_sec()
        if now < self.ready_start_allowed_after:
            return
        if time.monotonic() - self.last_ready_infer < 1.0 / max(self.args.ready_gate_infer_hz, 1e-6):
            return
        self.last_ready_infer = time.monotonic()
        front, front_age = self.buffers["front"].latest(now=now)
        wrist, wrist_age = self.buffers["wrist"].latest(now=now)
        if front is None or wrist is None:
            return
        if front_age is not None and front_age > self.args.max_dt_front_image:
            return
        if wrist_age is not None and wrist_age > self.args.max_dt_wrist_image:
            return
        try:
            x = compose_input(decode_ros_image(front), decode_ros_image(wrist), self.ready_camera, self.ready_image_size).to(
                self.ready_device
            )
            with torch.no_grad():
                prob = float(torch.sigmoid(self.ready_model(x).view(-1))[0].detach().cpu())
        except Exception as exc:
            self.node.get_logger().warn(f"ready_gate inference failed: {exc}")
            return
        self.ready_prob = prob
        if prob >= self.args.ready_gate_positive_threshold:
            self.ready_pos_count += 1
            self.ready_neg_count = 0
        elif prob <= self.args.ready_gate_negative_threshold:
            self.ready_neg_count += 1
            self.ready_pos_count = 0
        else:
            self.ready_pos_count = 0
            self.ready_neg_count = 0
        old = self.ready_phase
        if self.ready_phase == 0 and self.ready_pos_count >= self.args.ready_gate_hold_frames:
            self.ready_phase = 1
        elif self.ready_phase == 1 and self.ready_neg_count >= self.args.ready_gate_hold_frames:
            self.ready_phase = 0
        if self.ready_phase != old:
            self.node.get_logger().info(f"ready_gate={self.ready_phase} prob={prob:.3f}")
        if self.args.auto_start_on_ready and self.ready_phase == 1:
            self._start_trial("ready_gate")

    @staticmethod
    def _fmt_prob(value: float | None) -> str:
        return "none" if value is None else f"{value:.3f}"


class ReadyLoopSyncRollout(ReadyLoopMixin, NoRotvecSmolVLASyncRollout):
    def __init__(self, node, args: argparse.Namespace) -> None:
        super().__init__(node, args)
        self._init_ready_loop()

    def _infer_tick(self) -> None:
        if not self._ready_loop_allows_model():
            return
        super()._infer_tick()

    def _publish_tick(self) -> None:
        if self._ready_loop_publish_home_if_needed():
            return
        if self._ready_loop_maybe_end_trial():
            return
        if not self.loop_active:
            return
        super()._publish_tick()


class ReadyLoopRTCRollout(ReadyLoopMixin, NoRotvecSmolVLARTCRollout):
    def __init__(self, node, args: argparse.Namespace) -> None:
        super().__init__(node, args)
        self._init_ready_loop()

    def _observe_tick(self) -> None:
        if not self._ready_loop_allows_model():
            return
        super()._observe_tick()

    def _publish_tick(self) -> None:
        if self._ready_loop_publish_home_if_needed():
            return
        if self._ready_loop_maybe_end_trial():
            return
        if not self.loop_active:
            return
        super()._publish_tick()


def _parse_base(mode: str, remaining: list[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *remaining]
        return parse_rtc_args() if mode == "rtc" else parse_sync_args()
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    loop_parser = argparse.ArgumentParser(add_help=False)
    loop_parser.add_argument("--mode", choices=("sync", "rtc"), default="sync")
    loop_parser.add_argument("--ready-gate-checkpoint", type=Path, default=DEFAULT_READY_GATE_CHECKPOINT)
    loop_parser.add_argument("--ready-gate-positive-threshold", type=float, default=0.6)
    loop_parser.add_argument("--ready-gate-negative-threshold", type=float, default=0.4)
    loop_parser.add_argument("--ready-gate-hold-frames", type=int, default=3)
    loop_parser.add_argument("--ready-gate-infer-hz", type=float, default=15.0)
    loop_parser.add_argument("--ready-after-home-settle-s", type=float, default=0.8)
    loop_parser.add_argument("--auto-start-on-ready", action=argparse.BooleanOptionalAction, default=True)
    loop_parser.add_argument("--wait-ready-on-start", action=argparse.BooleanOptionalAction, default=False)
    loop_parser.add_argument("--trial-duration-s", type=float, default=18.0)
    loop_parser.add_argument("--home-pulse-s", type=float, default=RLT_CFG.home_pulse_s)
    loop_parser.add_argument("--home-gripper-value", type=float, default=RLT_CFG.home_gripper_value)
    loop_parser.add_argument("--block-model-during-home", action=argparse.BooleanOptionalAction, default=RLT_CFG.block_model_during_home)
    loop_parser.add_argument(
        "--reset-impedance-on-trial-start",
        action=argparse.BooleanOptionalAction,
        default=RLT_CFG.reset_impedance_on_trial_start,
    )
    loop_parser.add_argument(
        "--reset-impedance-during-home",
        action=argparse.BooleanOptionalAction,
        default=RLT_CFG.reset_impedance_during_home,
    )
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Ready-loop options:")
        loop_parser.print_help()
        print("\nBase rollout options:")
    loop_args, remaining = loop_parser.parse_known_args()
    args = _parse_base(loop_args.mode, remaining)
    for key, value in vars(loop_args).items():
        setattr(args, key, value)
    if not _argv_has("--policy-path"):
        args.policy_path = DEFAULT_POLICY_PATH
    if not _argv_has("--vr-override") and not _argv_has("--no-vr-override"):
        args.vr_override = False
    return args


def main() -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    args = parse_args()
    args.policy_path = resolve_policy_path(args.policy_path)
    hf_home = _resolve(args.hf_home)
    args.hf_home = hf_home
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    rclpy.init()
    node = rclpy.create_node(f"ur3e_no_rotvec_smolvla_{args.mode}_ready_loop")
    rollout = None
    try:
        run_pre_model_startup_sequence(node, args)
        rollout = ReadyLoopRTCRollout(node, args) if args.mode == "rtc" else ReadyLoopSyncRollout(node, args)
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
